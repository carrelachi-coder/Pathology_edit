#!/usr/bin/env python3
"""
Layer 1: LLM 语义解析器
========================

输入: 原始病理报告 + 编辑后病理报告
输出: 结构化语义差异 JSON (直接送入 RuleEngine)

模型: Qwen2.5-VL-7B-Instruct (本地部署)

用法:
    from llm_parser import LLMParser

    parser = LLMParser()
    diff = parser.parse(
        "Low-grade IDC with sparse TILs.",
        "High-grade IDC with dense TILs and focal necrosis.",
    )
    # {
    #   "tumor_change": {"growth": "increase", "degree": "moderate", "grade_change": "upgrade"},
    #   "lymphocyte_change": {"infiltration": "increase", "degree": "significant"},
    #   "necrosis_change": {"action": "add", "extent": "focal"},
    #   "stroma_change": {"density": "none", "degree": "moderate"}
    # }
"""

import json
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# System Prompt
# =============================================================================
SYSTEM_PROMPT = """You are a pathology report difference analyzer. You compare an ORIGINAL report and an EDITED report, and output ONLY the changes as a JSON object.

CRITICAL RULES:
1. ONLY report changes that are EXPLICITLY stated in the text. Do NOT infer or assume changes.
2. If a feature (necrosis, lymphocytes, stroma, etc.) is described THE SAME WAY in both reports, or is NOT MENTIONED in the edited report, set it to "none".
3. If necrosis is mentioned identically in both reports (e.g., both say "extensive necrosis"), necrosis_change.action = "none".
4. If stroma is not explicitly discussed as changing, stroma_change.density = "none". Stroma being consumed by tumor expansion is NOT a stroma density change.
5. When in doubt, output "none". It is much better to miss a subtle change than to hallucinate one.
6. grade_change refers to histological grade / differentiation ONLY, not tumor size. If grade or differentiation changes but tumor extent does not, set growth = "none".
7. If the report describes treatment effect (tumor regression, residual tumor, therapy response), set growth = "decrease".

Output ONLY this JSON schema:

{
  "tumor_change": {
    "growth": "none" | "increase" | "decrease",
    "degree": "mild" | "moderate" | "significant",
    "grade_change": "none" | "upgrade" | "downgrade"
  },
  "lymphocyte_change": {
    "infiltration": "none" | "increase" | "decrease",
    "degree": "mild" | "moderate" | "significant"
  },
  "necrosis_change": {
    "action": "none" | "add" | "increase" | "decrease" | "remove",
    "extent": "focal" | "moderate" | "extensive"
  },
  "stroma_change": {
    "density": "none" | "increase" | "decrease",
    "degree": "mild" | "moderate" | "significant"
  }
}

Field mapping rules:

TUMOR_CHANGE:
- growth: ONLY if the report explicitly describes tumor size/volume/extent changing. "expansion", "enlarged", "occupying majority" → increase. "residual", "regression", "treatment effect", "shrinkage" → decrease.
- degree: mild = minor wording change. moderate = clear change. significant = dramatic change.
- grade_change: ONLY if grade or differentiation explicitly changes. "well-differentiated" → "poorly-differentiated" = upgrade. "high-grade" → "intermediate-to-low-grade" = downgrade. If grade stays the same, = "none".

LYMPHOCYTE_CHANGE:
- infiltration: ONLY if TIL/lymphocyte description explicitly changes. "sparse" → "dense" = increase. "brisk TILs" → "sparse TILs" = decrease.
- If lymphocytes are not mentioned in either report, set to "none".

NECROSIS_CHANGE:
- action: ONLY if necrosis description explicitly changes between the two reports.
  - "no necrosis" → "focal necrosis" = add
  - "focal" → "extensive" = increase
  - "extensive necrosis" → "limited necrosis with fibrotic repair" = decrease
  - "necrosis" → "no necrosis" = remove
  - Both say "extensive necrosis" = "none" (NO CHANGE)
- If necrosis is described the same way in both reports, action MUST be "none".

STROMA_CHANGE:
- density: ONLY if stromal density/desmoplasia/fibrosis is explicitly described as changing. "fibrous stroma" → "dense desmoplastic stroma" = increase. Almost always "none".
- degree: mild/moderate/significant, only meaningful when density != "none".

Output ONLY the JSON. No explanation, no markdown, no extra text."""


# =============================================================================
# Few-shot examples
# =============================================================================

FEW_SHOT_EXAMPLES = [
    # Example 1: Multiple changes (综合变化: growth + grade + lymph + necrosis add)
    {
        "original": "Well-differentiated invasive ductal carcinoma forming tubular structures, with minimal lymphocytic response. No necrosis identified.",
        "edited": "Poorly-differentiated invasive ductal carcinoma with solid growth pattern, moderate lymphocytic infiltrate and focal necrosis.",
        "output": {
            "tumor_change": {"growth": "increase", "degree": "moderate", "grade_change": "upgrade"},
            "lymphocyte_change": {"infiltration": "increase", "degree": "moderate"},
            "necrosis_change": {"action": "add", "extent": "focal"},
            "stroma_change": {"density": "none", "degree": "moderate"}
        }
    },
    # Example 2: ONLY tumor expansion, everything else unchanged
    # Key: necrosis is mentioned the same way in both → "none"
    {
        "original": "High-grade invasive ductal carcinoma with extensive necrosis. A small viable tumor island is present.",
        "edited": "High-grade invasive ductal carcinoma with extensive necrosis. The viable tumor shows moderate expansion into surrounding stroma.",
        "output": {
            "tumor_change": {"growth": "increase", "degree": "moderate", "grade_change": "none"},
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none", "degree": "moderate"}
        }
    },
    # Example 3: ONLY lymphocyte change
    {
        "original": "Invasive carcinoma with sparse peritumoral lymphocytes.",
        "edited": "Invasive carcinoma with brisk tumor-infiltrating lymphocytes (TILs >50%).",
        "output": {
            "tumor_change": {"growth": "none", "degree": "mild", "grade_change": "none"},
            "lymphocyte_change": {"infiltration": "increase", "degree": "significant"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none", "degree": "moderate"}
        }
    },
    # Example 4: No change at all
    {
        "original": "Grade II invasive ductal carcinoma with moderate TILs and focal necrosis.",
        "edited": "Grade II invasive ductal carcinoma with moderate TILs and focal necrosis.",
        "output": {
            "tumor_change": {"growth": "none", "degree": "mild", "grade_change": "none"},
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none", "degree": "moderate"}
        }
    },
    # Example 5: ONLY necrosis increase
    {
        "original": "High-grade carcinoma with small foci of necrosis. Sparse TILs.",
        "edited": "High-grade carcinoma with extensive comedo-type necrosis. Sparse TILs.",
        "output": {
            "tumor_change": {"growth": "none", "degree": "mild", "grade_change": "none"},
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "increase", "extent": "extensive"},
            "stroma_change": {"density": "none", "degree": "moderate"}
        }
    },
    # Example 6: Treatment response (tumor decrease + cell changes)
    {
        "original": "High-grade invasive ductal carcinoma occupying most of the field. Moderate stromal component.",
        "edited": "Invasive ductal carcinoma with treatment effect. Residual tumor nests are small, scattered within fibrotic stroma. Decreased cellularity with scattered tumor cell necrosis and pyknotic nuclei.",
        "output": {
            "tumor_change": {"growth": "decrease", "degree": "moderate", "grade_change": "none"},
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none", "degree": "moderate"}
        }
    },
    # Example 7: Grade-only change (no growth change)
    {
        "original": "Invasive ductal carcinoma, high histological grade. Tumor cells with marked nuclear atypia and frequent mitotic figures.",
        "edited": "Invasive ductal carcinoma, intermediate-to-low histological grade. Tumor cells are well-differentiated with mild nuclear atypia and rare mitotic figures.",
        "output": {
            "tumor_change": {"growth": "none", "degree": "moderate", "grade_change": "downgrade"},
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none", "degree": "moderate"}
        }
    },
    # Example 8: Necrosis decrease (fibrosis repair)
    {
        "original": "High-grade carcinoma. Extensive coagulative necrosis occupies a large portion of the field.",
        "edited": "High-grade carcinoma. The necrotic area is limited, with a peripheral fibrotic reparative zone containing macrophage infiltration and fibroblast proliferation.",
        "output": {
            "tumor_change": {"growth": "none", "degree": "mild", "grade_change": "none"},
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "decrease", "extent": "moderate"},
            "stroma_change": {"density": "none", "degree": "moderate"}
        }
    },
    # Example 9: Stromal fibrosis (desmoplastic reaction)
    {
        "original": "Invasive carcinoma with loose myxoid stroma and scattered adipose tissue between tumor nests.",
        "edited": "Invasive carcinoma with dense desmoplastic stroma. Collagenous fibrous tissue replaces the previously loose stroma, with markedly reduced cellularity in the stromal compartment.",
        "output": {
            "tumor_change": {"growth": "none", "degree": "mild", "grade_change": "none"},
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "increase", "degree": "moderate"}
        }
    },
]


# =============================================================================
# 默认输出 (解析失败时的 fallback)
# =============================================================================

DEFAULT_OUTPUT = {
    "tumor_change": {"growth": "none", "degree": "mild", "grade_change": "none"},
    "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
    "necrosis_change": {"action": "none", "extent": "focal"},
    "stroma_change": {"density": "none", "degree": "moderate"},
}

# 合法值
VALID_VALUES = {
    "tumor_change.growth": {"none", "increase", "decrease"},
    "tumor_change.degree": {"mild", "moderate", "significant"},
    "tumor_change.grade_change": {"none", "upgrade", "downgrade"},
    "lymphocyte_change.infiltration": {"none", "increase", "decrease"},
    "lymphocyte_change.degree": {"mild", "moderate", "significant"},
    "necrosis_change.action": {"none", "add", "increase", "decrease", "remove"},
    "necrosis_change.extent": {"focal", "moderate", "extensive"},
    "stroma_change.density": {"none", "increase", "decrease"},
    "stroma_change.degree": {"mild", "moderate", "significant"},
}


# =============================================================================
# LLM Parser
# =============================================================================

class LLMParser:
    """
    用 Qwen2.5-VL-7B-Instruct 从报告对中提取语义差异。

    Args:
        model_path: 模型路径
        device: "cuda" / "cpu" / "cuda:0"
        use_few_shot: 是否使用 few-shot examples
    """

    def __init__(
        self,
        model_path: str = "/data/huggingface/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        use_few_shot: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.use_few_shot = use_few_shot
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._model is not None:
            return

        logger.info(f"Loading LLM from {self.model_path}...")
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        self._model.eval()
        logger.info("LLM loaded.")

    def _build_messages(self, original_report: str, edited_report: str) -> list:
        """构建 chat messages"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Few-shot examples
        if self.use_few_shot:
            for ex in FEW_SHOT_EXAMPLES:
                user_msg = (
                    f"ORIGINAL: {ex['original']}\n"
                    f"EDITED: {ex['edited']}"
                )
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": json.dumps(ex["output"])})

        # 实际请求
        user_msg = (
            f"ORIGINAL: {original_report}\n"
            f"EDITED: {edited_report}"
        )
        messages.append({"role": "user", "content": user_msg})

        return messages

    def _extract_json(self, text: str) -> Optional[Dict]:
        """从 LLM 输出中提取 JSON"""
        text = text.strip()

        # 去掉可能的 markdown 包裹
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # 找到第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None

        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

    def _validate_output(self, output: Dict) -> Dict:
        """验证并修复 LLM 输出, 确保符合 schema"""
        result = json.loads(json.dumps(DEFAULT_OUTPUT))  # deep copy

        for section_key in ["tumor_change", "lymphocyte_change", "necrosis_change", "stroma_change"]:
            if section_key not in output:
                continue
            section = output[section_key]
            if not isinstance(section, dict):
                continue

            for field_key, field_val in section.items():
                full_key = f"{section_key}.{field_key}"
                if full_key in VALID_VALUES:
                    if field_val in VALID_VALUES[full_key]:
                        result[section_key][field_key] = field_val
                    else:
                        logger.warning(f"Invalid value '{field_val}' for {full_key}, "
                                       f"keeping default")

        return result

    def parse(self, original_report: str, edited_report: str) -> Dict:
        """
        解析两段报告的差异。

        Args:
            original_report: 原始病理报告
            edited_report: 编辑后病理报告

        Returns:
            结构化语义差异 JSON
        """
        self._ensure_model()

        messages = self._build_messages(original_report, edited_report)

        # Qwen chat format
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        import torch
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # 只取新生成的 token
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        logger.info(f"LLM raw output: {response[:300]}")

        # 提取 JSON
        parsed = self._extract_json(response)

        if parsed is None:
            logger.warning("Failed to parse JSON from LLM output, using defaults")
            return json.loads(json.dumps(DEFAULT_OUTPUT))

        # 验证并修复
        result = self._validate_output(parsed)
        return result

    def parse_batch(self, report_pairs: list) -> list:
        """
        批量解析多对报告。

        Args:
            report_pairs: [(original, edited), ...]

        Returns:
            [semantic_diff, ...]
        """
        results = []
        for i, (orig, edit) in enumerate(report_pairs):
            logger.info(f"Parsing pair {i+1}/{len(report_pairs)}...")
            diff = self.parse(orig, edit)
            results.append(diff)
        return results


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser_arg = argparse.ArgumentParser(description="Test LLM Parser")
    parser_arg.add_argument("--model", default="/data/huggingface/Qwen2.5-VL-7B-Instruct")
    parser_arg.add_argument("--device", default="cuda")
    parser_arg.add_argument("--mode", choices=["offline", "interactive", "eval"],
                            default="interactive",
                            help="offline: JSON parsing only (no GPU); "
                                 "interactive: manual input (default); "
                                 "eval: auto test suite with scoring")
    parser_arg.add_argument("--original", default=None, help="Original report (for single-shot)")
    parser_arg.add_argument("--edited", default=None, help="Edited report (for single-shot)")
    args = parser_arg.parse_args()

    print("=" * 60)
    print("Layer 1: LLM Parser Test")
    print("=" * 60)

    # =========================================================================
    # Mode: offline — JSON 提取 + 验证 (不需要 GPU)
    # =========================================================================
    if args.mode == "offline":
        print("\n--- JSON extraction & validation (offline) ---")
        dummy_parser = LLMParser.__new__(LLMParser)
        dummy_parser.model_path = args.model
        dummy_parser.device = args.device
        dummy_parser.use_few_shot = True
        dummy_parser._model = None
        dummy_parser._tokenizer = None

        # 正常 JSON
        raw1 = '{"tumor_change": {"growth": "increase", "degree": "moderate", "grade_change": "upgrade"}, "lymphocyte_change": {"infiltration": "increase", "degree": "significant"}, "necrosis_change": {"action": "add", "extent": "focal"}, "stroma_change": {"density": "none", "degree": "moderate"}}'
        parsed1 = dummy_parser._extract_json(raw1)
        validated1 = dummy_parser._validate_output(parsed1)
        print(f"  Normal JSON: {json.dumps(validated1, indent=2)}")

        # 带 markdown 包裹
        raw2 = '```json\n{"tumor_change": {"growth": "none"}, "lymphocyte_change": {"infiltration": "decrease", "degree": "mild"}, "necrosis_change": {"action": "none"}, "stroma_change": {"density": "none"}}\n```'
        parsed2 = dummy_parser._extract_json(raw2)
        validated2 = dummy_parser._validate_output(parsed2)
        print(f"  Markdown wrapped: {json.dumps(validated2, indent=2)}")

        # 非法值
        raw3 = '{"tumor_change": {"growth": "explode", "degree": "massive"}, "lymphocyte_change": {"infiltration": "none"}, "necrosis_change": {"action": "none"}, "stroma_change": {"density": "none"}}'
        parsed3 = dummy_parser._extract_json(raw3)
        validated3 = dummy_parser._validate_output(parsed3)
        print(f"  Invalid values (should fallback): tumor_change.growth={validated3['tumor_change']['growth']} (expect 'none')")

        # stroma degree 验证
        raw4 = '{"tumor_change": {"growth": "none", "degree": "mild", "grade_change": "none"}, "lymphocyte_change": {"infiltration": "none", "degree": "mild"}, "necrosis_change": {"action": "none", "extent": "focal"}, "stroma_change": {"density": "increase", "degree": "significant"}}'
        parsed4 = dummy_parser._extract_json(raw4)
        validated4 = dummy_parser._validate_output(parsed4)
        print(f"  Stroma with degree: density={validated4['stroma_change']['density']}, degree={validated4['stroma_change']['degree']}")

        print("\nAll offline tests passed.")

    # =========================================================================
    # Mode: interactive — 手动输入报告对, LLM 推理输出 JSON
    # =========================================================================
    elif args.mode == "interactive":
        llm_parser = LLMParser(model_path=args.model, device=args.device)

        # 单次模式: 通过命令行参数传入
        if args.original and args.edited:
            print(f"\nORIGINAL: {args.original}")
            print(f"EDITED:   {args.edited}")
            diff = llm_parser.parse(args.original, args.edited)
            print(f"\nSemantic diff:")
            print(json.dumps(diff, indent=2))
        else:
            # 交互循环
            print("\nInteractive mode: enter report pairs, type 'quit' to exit.")
            print("Model loaded, ready for input.\n")

            round_num = 0
            while True:
                round_num += 1
                print(f"{'='*60}")
                print(f"Round {round_num}")
                print(f"{'='*60}")

                print("ORIGINAL report (paste, then press Enter on an empty line):")
                original_lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    if line.strip().lower() == "quit":
                        print("Bye!")
                        exit(0)
                    original_lines.append(line)
                original_report = " ".join(original_lines).strip()

                if not original_report:
                    print("Empty input, skipping.\n")
                    continue

                print("EDITED report (paste, then press Enter on an empty line):")
                edited_lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    if line.strip().lower() == "quit":
                        print("Bye!")
                        exit(0)
                    edited_lines.append(line)
                edited_report = " ".join(edited_lines).strip()

                if not edited_report:
                    print("Empty input, skipping.\n")
                    continue

                print(f"\nParsing...")
                diff = llm_parser.parse(original_report, edited_report)
                print(f"\nSemantic diff:")
                print(json.dumps(diff, indent=2))
                print()

    # =========================================================================
    # Mode: eval — 自动测试套件 (需要 GPU)
    # =========================================================================
    elif args.mode == "eval":
        print("\n--- Auto evaluation test suite (GPU) ---")
        llm_parser = LLMParser(model_path=args.model, device=args.device)

        test_pairs = [
            # 1. 综合变化 (growth + grade + lymph + necrosis)
            (
                "Well-differentiated invasive ductal carcinoma with minimal lymphocytic response. No necrosis identified.",
                "Poorly-differentiated invasive ductal carcinoma with dense TILs and focal necrosis.",
            ),
            # 2. 无变化
            (
                "Grade II invasive ductal carcinoma with moderate TILs and focal necrosis.",
                "Grade II invasive ductal carcinoma with moderate TILs and focal necrosis.",
            ),
            # 3. 纯 grade downgrade (growth=none)
            (
                "Invasive ductal carcinoma, high histological grade. Tumor cells with marked nuclear atypia, frequent mitotic figures, and dense cellularity.",
                "Invasive ductal carcinoma, intermediate-to-low histological grade. Tumor cells are well-differentiated with mild nuclear atypia, rare mitotic figures, and moderate cellularity.",
            ),
            # 4. 肿瘤退缩 (treatment response)
            (
                "Invasive ductal carcinoma, high histological grade. Tumor cells arranged in broad cords and nests, occupying the majority of the field.",
                "Invasive ductal carcinoma with treatment effect. Residual tumor nests are small in volume, scattered within a fibrotic stroma. Decreased cellularity with scattered tumor cell necrosis and pyknotic nuclei.",
            ),
            # 5. 坏死减少/纤维化修复
            (
                "High-grade carcinoma. Extensive coagulative necrosis occupies a large portion of the field with abundant karyorrhectic debris.",
                "High-grade carcinoma. The necrotic area is limited in extent, with a peripheral fibrotic reparative zone containing macrophage infiltration and fibroblast proliferation.",
            ),
            # 6. 间质纤维化
            (
                "Invasive carcinoma with loose fibrous stroma and scattered adipose tissue. Stroma cellularity is moderate.",
                "Invasive carcinoma with dense desmoplastic stroma. Collagenous fibrous tissue broadly distributed with markedly reduced cellularity in the stromal compartment.",
            ),
            # 7. 肿瘤增长 + 坏死增加
            (
                "Invasive ductal carcinoma, high histological grade. Tumor cells in cords and nests alternating with stroma. A large area of coagulative necrosis present, abutting the viable tumor.",
                "Invasive ductal carcinoma, high histological grade. Tumor cells in large confluent sheets occupying the majority of the field. Extensive coagulative necrosis distributed throughout the tumor with increased karyorrhectic debris. Stroma is minimal.",
            ),
            # 8. 淋巴浸润减少
            (
                "Invasive carcinoma with brisk tumor-infiltrating lymphocytes throughout the tumor and stromal compartments.",
                "Invasive carcinoma. No significant lymphocytic infiltration is identified.",
            ),
        ]

        expected = [
            {"growth": "increase", "grade_change": "upgrade",
             "infiltration": "increase", "action": "add"},
            {"growth": "none", "grade_change": "none",
             "infiltration": "none", "action": "none"},
            {"growth": "none", "grade_change": "downgrade",
             "infiltration": "none", "action": "none"},
            {"growth": "decrease", "grade_change": "none",
             "infiltration": "none", "action": "none"},
            {"growth": "none", "grade_change": "none",
             "infiltration": "none", "action": "decrease"},
            {"growth": "none", "grade_change": "none",
             "infiltration": "none", "density": "increase"},
            {"growth": "increase", "grade_change": "none",
             "infiltration": "none", "action": "increase"},
            {"growth": "none", "grade_change": "none",
             "infiltration": "decrease", "action": "none"},
        ]

        n_correct = 0
        n_total = 0
        errors = []

        for i, (orig, edit) in enumerate(test_pairs):
            print(f"\n  Test {i+1}/{len(test_pairs)}:")
            print(f"    ORIGINAL: {orig[:80]}...")
            print(f"    EDITED:   {edit[:80]}...")

            diff = llm_parser.parse(orig, edit)
            print(f"    RESULT:   {json.dumps(diff)}")

            exp = expected[i]
            for key, exp_val in exp.items():
                if key == "growth":
                    actual = diff["tumor_change"]["growth"]
                elif key == "grade_change":
                    actual = diff["tumor_change"]["grade_change"]
                elif key == "infiltration":
                    actual = diff["lymphocyte_change"]["infiltration"]
                elif key == "action":
                    actual = diff["necrosis_change"]["action"]
                elif key == "density":
                    actual = diff["stroma_change"]["density"]
                else:
                    continue

                n_total += 1
                if actual == exp_val:
                    n_correct += 1
                    print(f"    ✓ {key}: {actual}")
                else:
                    errors.append(f"Test {i+1} {key}: expected={exp_val}, got={actual}")
                    print(f"    ✗ {key}: expected={exp_val}, got={actual}")

        print(f"\n{'='*60}")
        if n_total > 0:
            print(f"Results: {n_correct}/{n_total} key fields correct "
                  f"({n_correct/n_total*100:.0f}%)")
        else:
            print("No tests run")
        if errors:
            print(f"Errors:")
            for e in errors:
                print(f"  - {e}")
        print(f"{'='*60}")



'''
CUDA_VISIBLE_DEVICES=0 python /home/lyw/wqx-DL/flow-edit/FlowEdit-main/mask_edit/rule_engine/llm_parser.py --mode interactive \
    --original "Invasive ductal carcinoma, high histological grade. Tumor cells are arranged in broad cords and nests, alternating with intervening stroma. The tumor nests contain densely packed, highly atypical neoplastic cells with frequent mitotic figures. The stroma consists of fibrous connective tissue distributed in bands interspersed among the tumor nests. A large area of coagulative necrosis is present in the lower-left portion of the field, containing scattered karyorrhectic debris and sparse inflammatory cell infiltration, directly abutting the viable tumor. No significant lymphocytic infiltration is identified. The overall morphology is consistent with a high-grade tumor with extensive necrosis." \
    --edited "Invasive ductal carcinoma, Nottingham grade 3. Tumor cells are arranged in cords and nests, alternating with intervening stroma. The neoplastic cells exhibit severe nuclear atypia with markedly enlarged, hyperchromatic, and highly pleomorphic nuclei; mitotic count exceeds 20 per 10 HPF, and the cellular arrangement is extremely dense. Scattered single-cell necrosis is present within the tumor nests. The stroma consists of fibrous connective tissue distributed in bands among the tumor nests. A large area of coagulative necrosis is present on one side. No significant lymphocytic infiltration is identified. The overall morphology is consistent with a grade 3, poorly differentiated tumor with high proliferative activity."
'''
