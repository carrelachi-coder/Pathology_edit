#!/usr/bin/env python3
"""
Layer 2: 规则引擎
==================

将 LLM 解析出的结构化语义差异 → Layer 3 可执行的操作序列。

输入: semantic_diff (LLM 输出的 JSON)
     original_mask (numpy array, 用于检查当前状态)

输出: ops (操作序列, Layer 3 直接执行)

设计原则:
  1. 操作顺序: tumor_dilate → necrosis_replace → lymph_dilate (符合病理进程)
  2. 参数从 prior_db 范围内采样, 不会超出生物学合理范围
  3. 每个操作会检查前置条件 (比如坏死替换需要已有坏死区域)
  4. 支持反向操作: "decrease" 通过对调训练对自然实现

用法:
    from rule_engine import RuleEngine

    engine = RuleEngine(prior_db_path="prior_db.json")

    semantic_diff = {
        "tumor_change": {"growth": "increase", "degree": "moderate", "grade_change": "upgrade"},
        "lymphocyte_change": {"infiltration": "increase", "degree": "significant"},
        "necrosis_change": {"action": "add", "extent": "focal"},
        "stroma_change": {"density": "none"}
    }

    ops = engine.plan(semantic_diff, original_mask)
    # [
    #   {"op": "tumor_dilate", "params": {"target_delta": 0.12}},
    #   {"op": "necrosis_replace", "params": {"n_pick": 1}},
    #   {"op": "lymph_dilate", "params": {"target_delta": 0.10}},
    # ]
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

TISSUE_NAME_MAP = {
    0: "outside_roi", 1: "tumor", 2: "stroma", 3: "lymphocytic_infiltrate",
    4: "necrosis_or_debris", 5: "glandular_secretions", 6: "blood", 7: "exclude",
    8: "metaplasia_NOS", 9: "fat", 10: "plasma_cells", 11: "other_immune_infiltrate",
    12: "mucoid_material", 13: "normal_acinus_or_duct", 14: "lymphatics",
    15: "undetermined", 16: "nerve", 17: "skin_adnexa", 18: "blood_vessel",
    19: "angioinvasion", 20: "dcis", 21: "other"
}

TUMOR_ID = 1
STROMA_ID = 2
LYMPH_ID = 3
NECROSIS_ID = 4

# 执行顺序 (符合病理进程)
# 间质纤维化排在最后: 先完成肿瘤/坏死/淋巴变化, 最后做基质重塑
EXECUTION_ORDER = [
    "necrosis_replace",   # 先在现有肿瘤内建立坏死
    "necrosis_fibrosis",
    "tumor_dilate",       # 再扩张肿瘤
    "tumor_shrink",
    "lymph_dilate",
    "stromal_fibrosis",
]

# degree → target_delta 映射范围
# 这些范围和你的 Transform 代码中的参数范围对齐
TUMOR_DELTA_MAP = {
    "mild":        (0.15, 0.25),
    "moderate":    (0.25, 0.35),
    "significant": (0.35, 0.45),
}

LYMPH_DELTA_MAP = {
    "mild":        (0.2, 0.3),
    "moderate":    (0.3, 0.4),
    "significant": (0.4, 0.5),
}

# necrosis extent → n_pick 映射
NECROSIS_PICK_MAP = {
    "focal":     1,
    "moderate":  2,  # 1个连通域, 但validator会控制合理性
    "extensive": 3,
}

# necrosis fibrosis: extent → degree 映射 + delta 范围
NECROSIS_EXTENT_TO_DEGREE = {
    "focal":     "mild",
    "moderate":  "moderate",
    "extensive": "significant",
}

FIBROSIS_DELTA_MAP = {
    "mild":        (0.1, 0.2),
    "moderate":    (0.2, 0.3),
    "significant": (0.3, 0.4),
}

STROMAL_DELTA_MAP = {
    "mild":        (0.1, 0.2),
    "moderate":    (0.2, 0.3),
    "significant": (0.3, 0.4),
}


# =============================================================================
# Mask 状态分析
# =============================================================================

class MaskAnalyzer:
    """分析当前 mask 的组织状态, 用于判断操作的前置条件和可行性"""

    def __init__(self, mask: np.ndarray):
        self.mask = mask
        self.total = mask.size
        self._ratios = {}
        for tid in np.unique(mask):
            tid = int(tid)
            self._ratios[tid] = (mask == tid).sum() / self.total

    def ratio(self, tissue_id: int) -> float:
        return self._ratios.get(tissue_id, 0.0)

    def has_tissue(self, tissue_id: int, min_ratio: float = 0.01) -> bool:
        return self.ratio(tissue_id) >= min_ratio

    def summary(self) -> Dict[str, float]:
        result = {}
        for tid, r in sorted(self._ratios.items()):
            name = TISSUE_NAME_MAP.get(tid, f"unknown_{tid}")
            if r >= 0.005:
                result[name] = round(r, 4)
        return result


# =============================================================================
# 规则引擎
# =============================================================================

class RuleEngine:
    """
    将语义差异映射到操作序列。

    核心逻辑:
      1. 解析每种变化类型, 决定是否需要对应操作
      2. 根据 degree/extent 确定参数
      3. 检查前置条件 (当前mask状态)
      4. 按固定顺序排列操作
    """

    def __init__(self, prior_db_path: Optional[str] = None, seed: int = 42):
        if prior_db_path:
            with open(prior_db_path, "r") as f:
                self.db = json.load(f)
                self.db.pop("_meta", None)
        else:
            self.db = None
        self.rng = np.random.default_rng(seed)

    def plan(
        self,
        semantic_diff: Dict,
        original_mask: np.ndarray,
    ) -> List[Dict]:
        """
        根据语义差异和当前 mask 状态, 生成操作序列。

        Args:
            semantic_diff: LLM 解析出的结构化语义差异
            original_mask: 当前的纯组织 mask (H, W), 值 0-21

        Returns:
            ops: 操作序列, 每个元素是 {"op": str, "params": dict}
        """
        analyzer = MaskAnalyzer(original_mask)
        logger.info(f"Current mask state: {analyzer.summary()}")

        # 收集所有可能的操作
        candidate_ops = {}

        # --- 肿瘤变化 ---
        tumor_op = self._plan_tumor(semantic_diff.get("tumor_change", {}), analyzer)
        if tumor_op:
            candidate_ops[tumor_op["op"]] = tumor_op

        # --- 坏死变化 ---
        necrosis_op = self._plan_necrosis(semantic_diff.get("necrosis_change", {}), analyzer)
        if necrosis_op:
            candidate_ops[necrosis_op["op"]] = necrosis_op

        # --- 淋巴浸润变化 ---
        lymph_op = self._plan_lymph(semantic_diff.get("lymphocyte_change", {}), analyzer)
        if lymph_op:
            candidate_ops[lymph_op["op"]] = lymph_op

        # --- 间质变化 ---
        stroma_op = self._plan_stroma(semantic_diff.get("stroma_change", {}), analyzer)
        if stroma_op:
            candidate_ops[stroma_op["op"]] = stroma_op

        # 按固定顺序排列
        ops = []
        for op_name in EXECUTION_ORDER:
            if op_name in candidate_ops:
                ops.append(candidate_ops[op_name])

        if not ops:
            logger.info("No operations needed based on semantic diff.")
        else:
            logger.info(f"Planned {len(ops)} operations:")
            for op in ops:
                logger.info(f"  {op['op']}: {op['params']}")

        return ops

    # -----------------------------------------------------------------
    # 细胞层调整: 从 semantic_diff 推导 type_bias + density_scale
    # -----------------------------------------------------------------
    def compute_cell_adjustments(
        self, semantic_diff: Dict,
    ) -> Tuple[Dict[int, Dict[str, float]], Dict[int, float]]:
        """
        从 semantic_diff 推导:
          1. type_bias_map:    {tissue_id: {type_name: multiplier}} — 核类型概率偏移
          2. density_scale_map: {tissue_id: scale} — 核间距缩放 (< 1.0 = 密度↑)

        type_name: "neoplastic"(101), "inflammatory"(102),
                   "connective"(103), "dead"(104), "epithelial"(105)

        density_scale 作用在 poisson disk sampling 的 min_distance 上:
          min_distance *= density_scale
          scale < 1.0 → 间距缩小 → 密度增大
          scale > 1.0 → 间距增大 → 密度减小
        """
        bias = {}    # {tissue_id: {type_name: multiplier}}
        density = {} # {tissue_id: min_distance_scale}

        # --- Grade change → tumor 区域 ---
        tumor_change = semantic_diff.get("tumor_change", {})
        grade = tumor_change.get("grade_change", "none")
        growth = tumor_change.get("growth", "none")
        degree = tumor_change.get("degree", "mild")

        GRADE_BIAS = {
            "mild":        {"neoplastic": 1.2, "dead": 1.1},
            "moderate":    {"neoplastic": 1.4, "dead": 1.2},
            "significant": {"neoplastic": 1.6, "dead": 1.4},
        }
        GRADE_DOWN_BIAS = {
            "mild":        {"neoplastic": 0.9, "epithelial": 1.2},
            "moderate":    {"neoplastic": 0.7, "epithelial": 1.4},
            "significant": {"neoplastic": 0.5, "epithelial": 1.6},
        }
        GRADE_DENSITY = {
            "mild": 0.90, "moderate": 0.82, "significant": 0.75,
        }
        GRADE_DOWN_DENSITY = {
            "mild": 1.10, "moderate": 1.20, "significant": 1.30,
        }
        TREATMENT_BIAS = {
            "mild":        {"dead": 1.3, "neoplastic": 0.8},
            "moderate":    {"dead": 1.6, "neoplastic": 0.6},
            "significant": {"dead": 2.0, "neoplastic": 0.5},
        }
        TREATMENT_DENSITY = {
            "mild": 1.10, "moderate": 1.25, "significant": 1.35,
        }

        if grade == "upgrade":
            bias.setdefault(1, {}).update(GRADE_BIAS.get(degree, {}))
            density.setdefault(1, GRADE_DENSITY.get(degree, 1.0))
        elif grade == "downgrade":
            bias.setdefault(1, {}).update(GRADE_DOWN_BIAS.get(degree, {}))
            density.setdefault(1, GRADE_DOWN_DENSITY.get(degree, 1.0))

        if growth == "decrease":
            treatment = TREATMENT_BIAS.get(degree, {})
            tumor_bias = bias.setdefault(1, {})
            for k, v in treatment.items():
                if k in tumor_bias:
                    tumor_bias[k] = v if abs(v - 1.0) > abs(tumor_bias[k] - 1.0) else tumor_bias[k]
                else:
                    tumor_bias[k] = v
            # 治疗后密度下降, 取更稀疏的值
            treat_d = TREATMENT_DENSITY.get(degree, 1.0)
            density[1] = max(density.get(1, 1.0), treat_d)

        # --- 淋巴浸润 → tumor + stroma 区域 inflammatory 偏移 ---
        lymph_change = semantic_diff.get("lymphocyte_change", {})
        infiltration = lymph_change.get("infiltration", "none")
        lymph_degree = lymph_change.get("degree", "mild")

        LYMPH_CELL_BIAS = {
            "mild": 1.3, "moderate": 1.6, "significant": 2.0,
        }

        if infiltration == "increase":
            inflam_mult = LYMPH_CELL_BIAS.get(lymph_degree, 1.3)
            for tid in [1, 2]:
                bias.setdefault(tid, {})
                existing = bias[tid].get("inflammatory", 1.0)
                bias[tid]["inflammatory"] = max(existing, inflam_mult)
        elif infiltration == "decrease":
            inflam_mult = 1.0 / LYMPH_CELL_BIAS.get(lymph_degree, 1.3)
            for tid in [1, 2]:
                bias.setdefault(tid, {}).setdefault("inflammatory", inflam_mult)

        # --- 间质纤维化 → stroma 区域 ---
        stroma_change = semantic_diff.get("stroma_change", {})
        stroma_density = stroma_change.get("density", "none")
        stroma_degree = stroma_change.get("degree", "moderate")

        STROMA_FIBROSIS_BIAS = {
            "mild":        {"connective": 0.85},
            "moderate":    {"connective": 0.7},
            "significant": {"connective": 0.55},
        }
        STROMA_FIBROSIS_DENSITY = {
            "mild": 1.10, "moderate": 1.20, "significant": 1.30,
        }

        if stroma_density == "increase":
            fib_bias = STROMA_FIBROSIS_BIAS.get(stroma_degree, {})
            stroma_bias = bias.setdefault(2, {})
            for k, v in fib_bias.items():
                stroma_bias.setdefault(k, v)
            density.setdefault(2, STROMA_FIBROSIS_DENSITY.get(stroma_degree, 1.0))

        # --- 坏死变化 → necrosis 区域 (tid=4) 细胞组成调整 ---
        necrosis_change = semantic_diff.get("necrosis_change", {})
        nec_action = necrosis_change.get("action", "none")
        nec_extent = necrosis_change.get("extent", "focal")

        NECROSIS_ADD_BIAS = {
            "focal":     {"dead": 1.3},
            "moderate":  {"dead": 1.5},
            "extensive": {"dead": 1.8},
        }
        NECROSIS_REPAIR_BIAS = {
            "focal":     {"inflammatory": 1.3, "connective": 1.2, "dead": 0.8},
            "moderate":  {"inflammatory": 1.5, "connective": 1.4, "dead": 0.6},
            "extensive": {"inflammatory": 1.8, "connective": 1.6, "dead": 0.5},
        }

        if nec_action in ("add", "increase"):
            nec_bias = NECROSIS_ADD_BIAS.get(nec_extent, {})
            bias.setdefault(4, {}).update(nec_bias)
        elif nec_action in ("decrease", "remove"):
            repair_extent = "extensive" if nec_action == "remove" else nec_extent
            nec_bias = NECROSIS_REPAIR_BIAS.get(repair_extent, {})
            bias.setdefault(4, {}).update(nec_bias)

        # 过滤空条目
        bias = {tid: b for tid, b in bias.items() if b}

        if bias:
            logger.info(f"Type bias map: {bias}")
        if density:
            logger.info(f"Density scale map: {density}")

        return bias, density

    # -----------------------------------------------------------------
    # prior_db 查表工具
    # -----------------------------------------------------------------
    def _get_tissue_stats(self, tissue_name: str) -> Optional[Dict]:
        """从 prior_db 获取组织统计"""
        if self.db and tissue_name in self.db:
            return self.db[tissue_name]
        return None

    def _calibrate_delta_from_db(
        self,
        tissue_name: str,
        current_ratio: float,
        degree: str,
    ) -> Optional[float]:
        stats = self._get_tissue_stats(tissue_name)
        if stats is None or "area" not in stats:
            return None

        area_stats = stats["area"]
        mean_area = area_stats["mean"]
        std_area = area_stats["std"]
        max_observed = area_stats.get("max_observed", 1.0)

        if degree == "mild":
            target_area = current_ratio + (mean_area - current_ratio) * 0.5
        elif degree == "moderate":
            target_area = mean_area + 0.3 * std_area
        elif degree == "significant":
            target_area = mean_area + 0.7 * std_area
        else:
            target_area = mean_area

        target_area += float(self.rng.uniform(-0.02, 0.02))
        target_area = min(target_area, max_observed * 0.8)

        delta = target_area - current_ratio
        if delta < 0.02:
            return None

        logger.info(f"  prior_db [{tissue_name}]: current={current_ratio:.3f}, "
                    f"target={target_area:.3f}, delta={delta:.3f} "
                    f"(db: mean={mean_area:.3f}, std={std_area:.3f})")
        return delta

    def _calibrate_shrink_delta_from_db(
        self,
        tissue_name: str,
        current_ratio: float,
        degree: str,
    ) -> Optional[float]:
        stats = self._get_tissue_stats(tissue_name)
        if stats is None or "area" not in stats:
            return None

        area_stats = stats["area"]
        mean_area = area_stats["mean"]
        std_area = area_stats["std"]

        if degree == "mild":
            target_area = current_ratio - (current_ratio - mean_area) * 0.3
        elif degree == "moderate":
            target_area = mean_area
        elif degree == "significant":
            target_area = max(mean_area - 0.3 * std_area, 0.01)
        else:
            target_area = mean_area

        target_area += float(self.rng.uniform(-0.01, 0.01))
        target_area = max(target_area, 0.005)

        delta = current_ratio - target_area
        if delta < 0.01:
            return None

        logger.info(f"  prior_db [{tissue_name} shrink]: current={current_ratio:.3f}, "
                    f"target={target_area:.3f}, delta={delta:.3f} "
                    f"(db: mean={mean_area:.3f}, std={std_area:.3f})")
        return delta

    # -----------------------------------------------------------------
    # 肿瘤变化 → tumor_dilate / tumor_shrink
    # -----------------------------------------------------------------
    def _plan_tumor(self, tumor_change: Dict, analyzer: MaskAnalyzer) -> Optional[Dict]:
        growth = tumor_change.get("growth", "none")
        degree = tumor_change.get("degree", "mild")
        grade_change = tumor_change.get("grade_change", "none")

        if growth == "none":
            return None

        if growth == "decrease":
            if not analyzer.has_tissue(TUMOR_ID, min_ratio=0.05):
                logger.warning("Tumor too small to shrink, skipping")
                return None

            current_ratio = analyzer.ratio(TUMOR_ID)

            stats = self._get_tissue_stats("tumor")
            if stats and "area" in stats:
                mean_area = stats["area"]["mean"]
                std_area = stats["area"]["std"]

                if degree == "mild":
                    target_area = current_ratio - (current_ratio - mean_area) * 0.3
                elif degree == "moderate":
                    target_area = mean_area
                elif degree == "significant":
                    target_area = max(mean_area - 0.3 * std_area, 0.03)
                else:
                    target_area = mean_area

                target_delta = current_ratio - target_area
                target_delta = max(target_delta, 0.03)
                target_delta = min(target_delta, current_ratio - 0.03)

                logger.info(f"  prior_db [tumor shrink]: current={current_ratio:.3f}, "
                            f"target={target_area:.3f}, delta={target_delta:.3f}")
            else:
                SHRINK_DELTA_MAP = {
                    "mild":        (0.10, 0.20),
                    "moderate":    (0.20, 0.30),
                    "significant": (0.30, 0.40),
                }
                delta_range = SHRINK_DELTA_MAP.get(degree, (0.1, 0.4))
                target_delta = float(self.rng.uniform(*delta_range))

            # 确保 shrink delta 至少达到 degree 对应范围的中点
            SHRINK_RANGE = {
                "mild":        (0.10, 0.20),
                    "moderate":    (0.20, 0.30),
                    "significant": (0.30, 0.40),
            }
            shrink_range = SHRINK_RANGE.get(degree, (0.1, 0.4))
            min_delta = (shrink_range[0] + shrink_range[1]) / 2
            if target_delta < min_delta:
                logger.info(f"  Tumor shrink: delta={target_delta:.3f} < "
                            f"mid for {degree}={min_delta:.3f}, using mid")
                target_delta = min_delta

            target_delta = float(np.clip(target_delta, 0.1, 0.4))
            # 不让 tumor 缩到 3% 以下
            target_delta = min(target_delta, current_ratio - 0.03)

            return {
                "op": "tumor_shrink",
                "params": {"target_delta": round(target_delta, 3)},
                "direction": "expand",
                "reason": f"tumor decrease, {degree}",
            }

        if growth == "increase":
            if not analyzer.has_tissue(TUMOR_ID, min_ratio=0.02):
                logger.warning("No tumor in mask, skipping tumor_dilate")
                return None

            current_ratio = analyzer.ratio(TUMOR_ID)

            # 优先用 prior_db 校准
            target_delta = self._calibrate_delta_from_db("tumor", current_ratio, degree)

            if target_delta is None:
                delta_range = TUMOR_DELTA_MAP.get(degree, (0.05, 0.10))
                target_delta = float(self.rng.uniform(*delta_range))
                logger.info(f"  Tumor: fallback delta={target_delta:.3f}")

            # [修复1] 确保 delta 至少达到 degree 对应范围的中点
            # (prior_db 校准可能因为 current 已接近 mean 而给出过小的 delta)
            delta_range = TUMOR_DELTA_MAP.get(degree, (0.10, 0.18))
            min_delta = (delta_range[0] + delta_range[1]) / 2
            if target_delta < min_delta:
                logger.info(f"  Tumor: prior_db delta={target_delta:.3f} < "
                            f"mid for {degree}={min_delta:.3f}, using mid")
                target_delta = min_delta

            # growth=increase 同时 grade_change=upgrade 时, 膨胀量加成
            if grade_change == "upgrade":
                target_delta = target_delta * 1.2

            target_delta = float(np.clip(target_delta, 0.10, 0.40))
            if current_ratio + target_delta > 0.90:
                target_delta = max(0.90 - current_ratio, 0.05)

            return {
                "op": "tumor_dilate",
                "params": {"target_delta": round(target_delta, 3)},
                "direction": "expand",
                "reason": f"tumor {growth}, {degree}, grade {grade_change}",
            }

        return None

    # -----------------------------------------------------------------
    # 坏死变化 → necrosis_replace / necrosis_fibrosis
    # -----------------------------------------------------------------
    def _plan_necrosis(self, necrosis_change: Dict, analyzer: MaskAnalyzer) -> Optional[Dict]:
        action = necrosis_change.get("action", "none")
        extent = necrosis_change.get("extent", "focal")

        if action == "none":
            return None

        if action in ("add", "increase"):
            has_necrosis = analyzer.has_tissue(NECROSIS_ID, min_ratio=0.005)
            has_tumor = analyzer.has_tissue(TUMOR_ID, min_ratio=0.05)

            if not has_necrosis:
                logger.warning("Skipping necrosis_replace: no existing necrosis as seed")
                return None
            if not has_tumor:
                logger.warning("Skipping necrosis_replace: no tumor to replace")
                return None

            current_necrosis = analyzer.ratio(NECROSIS_ID)
            nec_stats = self._get_tissue_stats("necrosis_or_debris")

            if nec_stats and "area" in nec_stats:
                mean_nec = nec_stats["area"]["mean"]
                std_nec = nec_stats["area"]["std"]
                upper = mean_nec + std_nec

                if current_necrosis > upper:
                    logger.info(f"  Necrosis already high ({current_necrosis:.3f} > "
                                f"mean+std={upper:.3f}), capping n_pick=1")
                    n_pick = 1
                else:
                    n_pick = NECROSIS_PICK_MAP.get(extent, 1)

                logger.info(f"  Necrosis: current={current_necrosis:.3f}, "
                            f"n_pick={n_pick} (db: mean={mean_nec:.3f})")
            else:
                n_pick = NECROSIS_PICK_MAP.get(extent, 1)

            return {
                "op": "necrosis_replace",
                "params": {"n_pick": n_pick},
                "direction": "expand",
                "reason": f"necrosis {action}, extent={extent}",
            }

        if action in ("decrease", "remove"):
            current_necrosis = analyzer.ratio(NECROSIS_ID)

            if current_necrosis < 0.02:
                logger.warning(f"Necrosis too small to fibrosis ({current_necrosis:.2%}), skipping")
                return None

            if action == "remove":
                degree = "significant"
            else:
                degree = NECROSIS_EXTENT_TO_DEGREE.get(extent, "moderate")

            target_delta = self._calibrate_shrink_delta_from_db(
                "necrosis_or_debris", current_necrosis, degree)

            if target_delta is None:
                delta_range = FIBROSIS_DELTA_MAP.get(degree, (0.03, 0.06))
                target_delta = float(self.rng.uniform(*delta_range))
                logger.info(f"  Necrosis fibrosis: fallback delta={target_delta:.3f}")

            # remove 时全部纤维化, decrease 时保留部分
            if action == "remove":
                max_delta = current_necrosis  # 全部去掉
            else:
                max_delta = current_necrosis * 0.90
            target_delta = float(np.clip(target_delta, 0.02, max_delta))

            logger.info(f"  Necrosis fibrosis: current={current_necrosis:.3f}, "
                        f"degree={degree}, delta={target_delta:.3f}")

            return {
                "op": "necrosis_fibrosis",
                "params": {
                    "target_delta": round(target_delta, 3),
                    "remove_all": (action == "remove"),
                },
                "direction": "expand",
                "reason": f"necrosis {action}, extent={extent} → fibrosis ({degree})",
            }

        return None

    # -----------------------------------------------------------------
    # 淋巴浸润变化 → lymph_dilate
    # -----------------------------------------------------------------
    def _plan_lymph(self, lymph_change: Dict, analyzer: MaskAnalyzer) -> Optional[Dict]:
        infiltration = lymph_change.get("infiltration", "none")
        degree = lymph_change.get("degree", "mild")

        if infiltration == "none":
            return None

        if infiltration == "decrease":
            logger.info("Lymphocyte decrease not yet supported, skipping")
            return None

        if infiltration == "increase":
            if not analyzer.has_tissue(LYMPH_ID, min_ratio=0.01):
                logger.warning("No lymphocytic infiltrate in mask, skipping")
                return None

            current_ratio = analyzer.ratio(LYMPH_ID)

            target_delta = self._calibrate_delta_from_db(
                "lymphocytic_infiltrate", current_ratio, degree)

            if target_delta is None:
                delta_range = LYMPH_DELTA_MAP.get(degree, (0.03, 0.06))
                target_delta = float(self.rng.uniform(*delta_range))
                logger.info(f"  Lymph: fallback delta={target_delta:.3f}")

            target_delta = float(np.clip(target_delta, 0.03, 0.5))

            expandable = analyzer.ratio(2) + analyzer.ratio(9) + analyzer.ratio(10)
            if target_delta > expandable * 0.8:
                target_delta = max(expandable * 0.8, 0.02)

            return {
                "op": "lymph_dilate",
                "params": {"target_delta": round(target_delta, 3)},
                "direction": "expand",
                "reason": f"lymphocyte infiltration {infiltration}, {degree}",
            }

        return None

    # -----------------------------------------------------------------
    # 间质变化 → stromal_fibrosis
    # -----------------------------------------------------------------
    def _plan_stroma(self, stroma_change: Dict, analyzer: MaskAnalyzer) -> Optional[Dict]:
        density = stroma_change.get("density", "none")
        degree = stroma_change.get("degree", "moderate")

        if density == "none":
            return None

        if density == "increase":
            if not analyzer.has_tissue(STROMA_ID, min_ratio=0.05):
                logger.warning("Stroma too small for fibrosis, skipping")
                return None

            current_ratio = analyzer.ratio(STROMA_ID)

            expandable = sum(analyzer.ratio(tid) for tid in [3,9, 10, 12, 13, 14, 18])
            if expandable < 0.02:
                logger.warning(f"No expandable tissue for stromal fibrosis ({expandable:.2%})")
                return None

            target_delta = self._calibrate_delta_from_db("stroma", current_ratio, degree)

            if target_delta is None:
                delta_range = STROMAL_DELTA_MAP.get(degree, (0.03, 0.08))
                target_delta = float(self.rng.uniform(*delta_range))
                logger.info(f"  Stroma fibrosis: fallback delta={target_delta:.3f}")

            target_delta = float(np.clip(target_delta, 0.03, 0.4))

            if target_delta > expandable * 0.8:
                target_delta = max(expandable * 0.8, 0.02)

            if current_ratio + target_delta > 0.95:
                target_delta = max(0.95 - current_ratio, 0.02)

            logger.info(f"  Stromal fibrosis: current={current_ratio:.3f}, "
                        f"expandable={expandable:.3f}, delta={target_delta:.3f}")

            return {
                "op": "stromal_fibrosis",
                "params": {"target_delta": round(target_delta, 3)},
                "direction": "expand",
                "reason": f"stroma density increase, {degree}",
            }

        if density == "decrease":
            logger.info("Stroma decrease not yet supported, skipping")
            return None

        return None


# =============================================================================
# Layer 3 执行器 (串联 规则引擎 → 组织编辑 → ProbNet)
# =============================================================================

class MaskEditor:
    """
    完整的 mask 编辑执行器。
    """

    def __init__(
        self,
        prior_db_path: str,
        project_root: Optional[str] = None,
        prob_net_ckpt: Optional[str] = None,
        nuclei_library_path: Optional[str] = None,
        seed: int = 42,
    ):
        self.prior_db_path = prior_db_path
        self.seed = seed

        if project_root is not None:
            self.project_root = project_root
        else:
            from pathlib import Path
            p = Path(__file__).resolve().parent
            for _ in range(5):
                if (p / "BCSS_dataset").exists() or (p / "inpaint_cells").exists():
                    break
                p = p.parent
            self.project_root = str(p)

        self.rule_engine = RuleEngine(prior_db_path=prior_db_path, seed=seed)

        self._tumor_transform = None
        self._lymph_transform = None
        self._necrosis_transform = None
        self._necrosis_fibrosis_transform = None
        self._stromal_fibrosis_transform = None
        self._validator = None

        self.prob_net_ckpt = prob_net_ckpt
        self.nuclei_library_path = nuclei_library_path
        self._prob_net = None
        self._prob_net_device = None
        self._nuclei_library = None

    def _ensure_transforms(self):
        if self._validator is not None:
            return

        import sys
        from pathlib import Path

        mask_gen_dir = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/mask_edit/mask_data_generate"

        if str(mask_gen_dir) not in sys.path:
            sys.path.insert(0, str(mask_gen_dir))

        from mask_validator import MaskValidator
        from boundary_deform import TumorBoundaryTransform
        from lymphocyte_infiltration import LymphocyteInfiltrationTransform
        from tumor_to_necrosis import NecrosisReplacementTransform
        from tumor_shrink import TumorShrinkTransform
        from necrosis_fibrosis import NecrosisFibrosisTransform
        from stromal_fibrosis import StromalFibrosisTransform

        self._validator = MaskValidator(self.prior_db_path)
        self._tumor_transform = TumorBoundaryTransform(
            self.prior_db_path, self._validator, seed=self.seed)
        self._lymph_transform = LymphocyteInfiltrationTransform(
            self.prior_db_path, self._validator, seed=self.seed)
        self._necrosis_transform = NecrosisReplacementTransform(
            self.prior_db_path, self._validator, seed=self.seed)
        self._tumor_shrink_transform = TumorShrinkTransform(
            self.prior_db_path, self._validator, seed=self.seed)
        self._necrosis_fibrosis_transform = NecrosisFibrosisTransform(
            self.prior_db_path, self._validator, seed=self.seed)
        self._stromal_fibrosis_transform = StromalFibrosisTransform(
            self.prior_db_path, self._validator, seed=self.seed)

    def _get_tissue_only(self, combined_mask: np.ndarray) -> np.ndarray:
        """
        从 combined_mask 中去除所有细胞, 得到纯组织层 (0-21)。

        两步:
          1. 最近邻填充 (处理孤立像素)
          2. 按核实例统一: 每个核的质心处是什么组织, 整个核区域就填什么
             避免边界上的核被拆成两种组织
        """
        from scipy.ndimage import distance_transform_edt, label

        tissue = combined_mask.copy()
        nuc_mask = combined_mask >= 100

        if not nuc_mask.any():
            return tissue

        # Step 1: 最近邻填充 (基础)
        _, nearest_idx = distance_transform_edt(nuc_mask, return_indices=True)
        tissue[nuc_mask] = combined_mask[nearest_idx[0][nuc_mask], nearest_idx[1][nuc_mask]]
        tissue = np.clip(tissue, 0, 21)

        # Step 2: 按核实例统一组织类型 (质心处的类型覆盖整个核)
        H, W = tissue.shape
        for nuc_val in [101, 102, 103, 104, 105]:
            binary = (combined_mask == nuc_val).astype(np.uint8)
            if not binary.any():
                continue
            labeled, n = label(binary)
            for lbl in range(1, n + 1):
                ys, xs = np.where(labeled == lbl)
                if len(ys) < 2:
                    continue
                cy = min(max(int(np.mean(ys)), 0), H - 1)
                cx = min(max(int(np.mean(xs)), 0), W - 1)
                tissue[ys, xs] = tissue[cy, cx]

        return tissue

    def _get_cell_instances(self, combined_mask: np.ndarray) -> list:
        from scipy.ndimage import label

        NUCLEI_CLASSES = [101, 102, 103, 104, 105]
        instances = []

        for nuc_val in NUCLEI_CLASSES:
            binary = (combined_mask == nuc_val).astype(np.uint8)
            if not np.any(binary):
                continue

            labeled, n = label(binary)
            for lbl in range(1, n + 1):
                ys, xs = np.where(labeled == lbl)
                if len(ys) < 2:
                    continue
                pixels = set(zip(ys.tolist(), xs.tolist()))
                instances.append({"type": nuc_val, "pixels": pixels})

        return instances

    def _retain_cells_outside_change(
        self,
        cell_instances: list,
        change_region: np.ndarray,
        edited_tissue: np.ndarray,
    ) -> np.ndarray:
        """
        以质心判断保留/丢弃:
          - 质心不在 change_region → 整个保留 (完整粘贴)
          - 质心在 change_region → 整个丢弃
        """
        H, W = edited_tissue.shape
        nuclei_layer = np.zeros((H, W), dtype=np.int64)

        retained = 0
        discarded = 0

        for inst in cell_instances:
            pixels = list(inst["pixels"])
            cy = int(np.mean([p[0] for p in pixels]))
            cx = int(np.mean([p[1] for p in pixels]))
            cy = min(max(cy, 0), H - 1)
            cx = min(max(cx, 0), W - 1)

            if change_region[cy, cx]:
                discarded += 1
                continue

            for y, x in inst["pixels"]:
                nuclei_layer[y, x] = inst["type"]
            retained += 1

        logger.info(f"  Cell instances: {retained} retained, {discarded} discarded")
        return nuclei_layer

    def _merge_tissue_and_cells(
        self,
        tissue: np.ndarray,
        nuclei_layer: np.ndarray,
    ) -> np.ndarray:
        combined = tissue.copy()
        cell_mask = nuclei_layer > 0
        combined[cell_mask] = nuclei_layer[cell_mask]
        return combined

    # -----------------------------------------------------------------
    # 纯细胞层 in-place 调整
    # -----------------------------------------------------------------
    def _apply_cell_only_adjustments(
        self,
        cell_instances: list,
        edited_tissue: np.ndarray,
        type_bias_map: Dict[int, Dict[str, float]],
        density_scale_map: Dict[int, float],
    ) -> np.ndarray:
        import sys
        from pathlib import Path

        project_root = Path(self.project_root)
        inpaint_dir = project_root / "inpaint_cells" / "DDPM+Cell_inpaint"
        if str(inpaint_dir) not in sys.path:
            sys.path.insert(0, str(inpaint_dir))

        INPAINT_DIR = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/inpaint_cells/DDPM+Cell_inpaint"
        if INPAINT_DIR not in sys.path:
            sys.path.insert(0, INPAINT_DIR)

        from generate_nuclei import poisson_disk_sampling, place_nucleus

        H, W = edited_tissue.shape
        NUCLEI_CLASSES = [101, 102, 103, 104, 105]

        _TYPE_NAME_TO_VAL = {
            "neoplastic": 101, "inflammatory": 102,
            "connective": 103, "dead": 104, "epithelial": 105,
        }
        _VAL_TO_NAME = {v: k for k, v in _TYPE_NAME_TO_VAL.items()}

        rng = np.random.default_rng(self.seed)

        instances_by_tissue = {}
        for inst in cell_instances:
            pixels = list(inst["pixels"])
            cy = int(np.mean([p[0] for p in pixels]))
            cx = int(np.mean([p[1] for p in pixels]))
            cy, cx = np.clip(cy, 0, H - 1), np.clip(cx, 0, W - 1)
            tid = int(edited_tissue[cy, cx])
            inst["_tissue_id"] = tid
            instances_by_tissue.setdefault(tid, []).append(inst)

        affected_tids = set()
        if type_bias_map:
            affected_tids.update(type_bias_map.keys())
        if density_scale_map:
            affected_tids.update(density_scale_map.keys())

        adjusted_instances = [
            {"type": inst["type"], "pixels": inst["pixels"], "_tissue_id": inst.get("_tissue_id")}
            for inst in cell_instances
        ]
        adjusted_by_tissue = {}
        for inst in adjusted_instances:
            tid = inst.get("_tissue_id")
            if tid is not None:
                adjusted_by_tissue.setdefault(tid, []).append(inst)

        total_relabeled = 0
        total_removed = 0
        total_added = 0

        added_cells_layer = np.zeros((H, W), dtype=np.int64)

        for tid in affected_tids:
            tid_instances = adjusted_by_tissue.get(tid, [])
            if not tid_instances:
                continue

            bias = type_bias_map.get(tid, {})
            scale = density_scale_map.get(tid, 1.0)

            # Step 1: Re-label (删除旧实例, 从 library 采样新形状粘贴到质心)
            if bias:
                self._ensure_library()
                library = self._nuclei_library

                type_counts = {}
                for inst in tid_instances:
                    t = inst["type"]
                    type_counts[t] = type_counts.get(t, 0) + 1
                total_count = sum(type_counts.values())

                if total_count > 0:
                    current_probs = {t: c / total_count for t, c in type_counts.items()}
                    target_probs = {}
                    for nuc_val in NUCLEI_CLASSES:
                        name = _VAL_TO_NAME.get(nuc_val, "")
                        base_p = current_probs.get(nuc_val, 0.0)
                        mult = bias.get(name, 1.0)
                        target_probs[nuc_val] = base_p * mult

                    total_p = sum(target_probs.values())
                    if total_p > 0:
                        target_probs = {t: p / total_p for t, p in target_probs.items()}

                    target_counts = {t: int(round(target_probs.get(t, 0) * total_count))
                                     for t in NUCLEI_CLASSES}
                    delta_counts = {t: target_counts[t] - type_counts.get(t, 0)
                                    for t in NUCLEI_CLASSES}

                    # 收集需要转换的: donor (需减少的类型) → receiver (需增加的类型)
                    donors = []
                    receivers = []
                    for t in NUCLEI_CLASSES:
                        if delta_counts[t] < 0:
                            donors.append((t, -delta_counts[t]))
                        elif delta_counts[t] > 0:
                            receivers.append((t, delta_counts[t]))

                    # 从 donor 类型随机选实例, 记录质心, 从 adjusted_instances 中删除
                    relabel_tasks = []  # [(cy, cx, target_type), ...]
                    for donor_type, n_donate in donors:
                        candidates = [inst for inst in tid_instances if inst["type"] == donor_type]
                        rng.shuffle(candidates)
                        for inst in candidates[:n_donate]:
                            pixels = list(inst["pixels"])
                            cy = int(np.mean([p[0] for p in pixels]))
                            cx = int(np.mean([p[1] for p in pixels]))
                            cy = min(max(cy, 0), H - 1)
                            cx = min(max(cx, 0), W - 1)
                            relabel_tasks.append({"cy": cy, "cx": cx, "inst_id": id(inst)})

                    # 从 adjusted_instances 中删除这些实例
                    remove_ids = {task["inst_id"] for task in relabel_tasks}
                    adjusted_instances = [
                        inst for inst in adjusted_instances
                        if id(inst) not in remove_ids
                    ]

                    # 分配目标类型
                    rng.shuffle(relabel_tasks)
                    task_idx = 0
                    for recv_type, n_recv in receivers:
                        for _ in range(n_recv):
                            if task_idx >= len(relabel_tasks):
                                break
                            relabel_tasks[task_idx]["target_type"] = recv_type
                            task_idx += 1

                    # 从 library 采样新形状, 粘贴到质心位置
                    for task in relabel_tasks:
                        if "target_type" not in task:
                            continue
                        instance = library.sample_instance(tid, task["target_type"])
                        if instance is None:
                            continue
                        if place_nucleus(added_cells_layer, task["cy"], task["cx"],
                                        instance, augment=True):
                            total_relabeled += 1

                    logger.info(f"  Tissue {tid}: relabeled {total_relabeled} instances "
                                f"(delete+replace), type counts: {type_counts} → {target_counts}")

            # Step 2: Remove
            if scale > 1.0:
                keep_ratio = 1.0 / (scale ** 2)
                keep_ratio = np.clip(keep_ratio, 0.3, 1.0)

                n_current = len(tid_instances)
                n_keep = max(int(n_current * keep_ratio), 1)
                n_remove = n_current - n_keep

                if n_remove > 0:
                    remove_indices = rng.choice(len(tid_instances), size=n_remove, replace=False)
                    to_remove = set()
                    for idx in remove_indices:
                        to_remove.add(id(tid_instances[idx]))

                    adjusted_instances = [
                        inst for inst in adjusted_instances
                        if id(inst) not in to_remove
                    ]
                    total_removed += n_remove

                    logger.info(f"  Tissue {tid}: removed {n_remove}/{n_current} instances "
                                f"(scale={scale:.2f}, keep_ratio={keep_ratio:.2f})")

            # Step 3: Add
            if scale < 1.0:
                self._ensure_library()
                library = self._nuclei_library

                add_ratio = 1.0 / (scale ** 2) - 1.0
                add_ratio = np.clip(add_ratio, 0.0, 2.0)

                n_current = len(tid_instances)
                n_add = max(int(n_current * add_ratio), 1)

                occupied = np.zeros((H, W), dtype=bool)
                for inst in adjusted_instances:
                    for y, x in inst["pixels"]:
                        occupied[y, x] = True

                tissue_region = (edited_tissue == tid) & ~occupied

                if tissue_region.sum() < 50:
                    continue

                type_counts_after = {}
                remaining_tid = [inst for inst in adjusted_instances
                                 if inst.get("_tissue_id") == tid]
                for inst in remaining_tid:
                    t = inst["type"]
                    type_counts_after[t] = type_counts_after.get(t, 0) + 1

                total_after = sum(type_counts_after.values())
                if total_after == 0:
                    continue

                type_vals = []
                type_probs_after = []
                for nuc_val in NUCLEI_CLASSES:
                    c = type_counts_after.get(nuc_val, 0)
                    if c > 0:
                        type_probs_after.append(c / total_after)
                        type_vals.append(nuc_val)

                if not type_vals:
                    continue

                type_probs_after = np.array(type_probs_after)
                type_probs_after /= type_probs_after.sum()

                stats = library.stats.get(str(tid), {})
                mean_areas = [info['mean_area'] for info in stats.get('nuclei_types', {}).values()
                             if info.get('mean_area', 0) > 0]
                avg_area = np.mean(mean_areas) if mean_areas else 100
                min_distance = max(np.sqrt(avg_area / np.pi) * 2.5, 8)
                min_distance *= scale
                min_distance = max(min_distance, 5)

                centers = poisson_disk_sampling(tissue_region, min_distance)
                if len(centers) > n_add:
                    import random
                    random.shuffle(centers)
                    centers = centers[:n_add]

                placed = 0
                for cy, cx in centers:
                    nuc_type = int(rng.choice(type_vals, p=type_probs_after))
                    instance = library.sample_instance(tid, nuc_type)
                    if instance is None:
                        continue
                    if place_nucleus(added_cells_layer, cy, cx, instance, augment=True):
                        placed += 1
                        total_added += 1

                if placed > 0:
                    logger.info(f"  Tissue {tid}: added {placed} instances "
                                f"(scale={scale:.2f}, add_ratio={add_ratio:.2f})")

        logger.info(f"Cell-only adjustments total: {total_relabeled} relabeled, "
                    f"{total_removed} removed, {total_added} added")

        nuclei_layer = np.zeros((H, W), dtype=np.int64)
        for inst in adjusted_instances:
            for y, x in inst["pixels"]:
                nuclei_layer[y, x] = inst["type"]

        nuclei_layer[added_cells_layer > 0] = added_cells_layer[added_cells_layer > 0]

        return nuclei_layer

    # -----------------------------------------------------------------
    # 主编辑流程
    # -----------------------------------------------------------------
    def edit(
        self,
        original_combined_mask: np.ndarray,
        semantic_diff: Dict,
        max_attempts: int = 5,
    ) -> Dict:
        self._ensure_transforms()

        # Step 1: 拆分
        original_tissue = self._get_tissue_only(original_combined_mask)
        cell_instances = self._get_cell_instances(original_combined_mask)
        logger.info(f"Extracted {len(cell_instances)} cell instances")

        # Step 2: 规则引擎 → 组织层操作
        ops = self.rule_engine.plan(semantic_diff, original_tissue)

        # Step 2.5: 预计算细胞层调整
        type_bias_map, density_scale_map = self.rule_engine.compute_cell_adjustments(semantic_diff)
        has_cell_adjustments = bool(type_bias_map or density_scale_map)

        # Step 3: 过滤组织层操作
        active_ops = [op for op in ops if op.get("direction", "expand") == "expand"]

        if not active_ops and not has_cell_adjustments:
            logger.info("No operations needed (no tissue edits, no cell adjustments).")
            return {
                "src_mask": original_combined_mask.copy(),
                "tar_mask": original_combined_mask.copy(),
                "edited_tissue": original_tissue.copy(),
                "change_region": np.zeros_like(original_tissue, dtype=bool),
                "ops_log": [],
                "original_tissue": original_tissue.copy(),
                "has_shrink": False,
            }

        # Step 4: 在纯组织层上执行编辑
        current_tissue = original_tissue.copy()
        ops_log = []

        for op_spec in active_ops:
            op_name = op_spec["op"]
            params = op_spec["params"]
            direction = op_spec.get("direction", "expand")

            logger.info(f"Executing: {op_name} ({direction}) with {params}")

            success = False
            for attempt in range(max_attempts):
                if op_name == "tumor_dilate":
                    delta = abs(params.get("target_delta", 0.10))
                    new_tissue, log = self._tumor_transform.apply(
                        current_tissue, target_delta=delta)
                elif op_name == "tumor_shrink":
                    delta = abs(params.get("target_delta", 0.08))
                    new_tissue, log = self._tumor_shrink_transform.apply(
                        current_tissue, target_delta=delta)
                elif op_name == "lymph_dilate":
                    delta = abs(params.get("target_delta", 0.05))
                    new_tissue, log = self._lymph_transform.apply(
                        current_tissue, target_delta=delta)
                elif op_name == "necrosis_replace":
                    new_tissue, log = self._necrosis_transform.apply(
                        current_tissue, n_pick=params.get("n_pick", 1))
                elif op_name == "necrosis_fibrosis":
                    delta = abs(params.get("target_delta", 0.08))
                    # remove_all 模式: 临时降低最低保留量
                    if params.get("remove_all", False):
                        
        
                        # remove_all: 直接把所有坏死替换成 stroma, 不走 SDF
                        new_tissue = current_tissue.copy()
                        necrosis_pixels = (new_tissue == 4)
                        n_necrosis = necrosis_pixels.sum()
                        new_tissue[necrosis_pixels] = 2  # stroma
                        actual_delta = n_necrosis / new_tissue.size

                        from dataclasses import dataclass as _dc
                        log = type('Log', (), {
                            'accepted': True,
                            'rejection_reason': '',
                            'area_change': {
                                'necrosis_or_debris': -round(actual_delta, 4),
                                'stroma': round(actual_delta, 4),
                            },
                            'to_dict': lambda self: {
                                'type': 'necrosis_fibrosis',
                                'tissue_id': 4,
                                'tissue_name': 'necrosis_or_debris',
                                'params': {
                                    'target_delta': round(delta, 3),
                                    'actual_delta': round(actual_delta, 4),
                                    'remove_all': True,
                                    'pixels_released': int(n_necrosis),
                                },
                                'area_change': self.area_change,
                                'accepted': True,
                                'rejection_reason': '',
                            },
                        })()
                        logger.info(f"  remove_all: replaced {n_necrosis} necrosis pixels "
                                f"with stroma ({actual_delta:.1%})")
                    else:
                        new_tissue, log = self._necrosis_fibrosis_transform.apply(
                            current_tissue, target_delta=delta)
                elif op_name == "stromal_fibrosis":
                    delta = abs(params.get("target_delta", 0.08))
                    new_tissue, log = self._stromal_fibrosis_transform.apply(
                        current_tissue, target_delta=delta)
                else:
                    logger.warning(f"Unknown operation: {op_name}")
                    break

                if log.accepted:
                    current_tissue = new_tissue
                    success = True
                    ops_log.append({
                        "op": op_name,
                        "direction": direction,
                        "params": params,
                        "log": log.to_dict(),
                        "attempt": attempt + 1,
                    })
                    logger.info(f"  Accepted on attempt {attempt + 1}: "
                                f"{log.area_change}")
                    break
                else:
                    logger.info(f"  Attempt {attempt + 1} rejected: "
                                f"{log.rejection_reason}")

            if not success:
                logger.warning(f"  {op_name} failed after {max_attempts} attempts")
                ops_log.append({
                    "op": op_name, "direction": direction, "params": params,
                    "log": {"accepted": False, "reason": "max_attempts_exceeded"},
                })

        # Step 5: 计算组织层变化区域
        tissue_change_region = (current_tissue != original_tissue)
        has_tissue_change = tissue_change_region.any()
        logger.info(f"Change region (tissue edits): {tissue_change_region.sum()} pixels "
                     f"({tissue_change_region.sum() / tissue_change_region.size * 100:.1f}%)")

        # =====================================================================
        # Step 5.5: 分支 — cell_only_mode vs 常规模式
        # =====================================================================
        cell_only_mode = has_cell_adjustments and not has_tissue_change

        if cell_only_mode:
            logger.info("Cell-only mode: tissue unchanged, adjusting cells in-place")
            adjusted_nuclei = self._apply_cell_only_adjustments(
                cell_instances, current_tissue, type_bias_map, density_scale_map)

            edited_combined = self._merge_tissue_and_cells(current_tissue, adjusted_nuclei)

            original_nuclei = self._retain_cells_outside_change(
                cell_instances, np.zeros_like(original_tissue, dtype=bool), original_tissue)
            original_combined = self._merge_tissue_and_cells(original_tissue, original_nuclei)

            cell_change_region = (original_combined != edited_combined)
            logger.info(f"Cell-only change region: {cell_change_region.sum()} pixels "
                        f"({cell_change_region.sum() / cell_change_region.size * 100:.1f}%)")

            return {
                "src_mask": original_combined,
                "tar_mask": edited_combined,
                "edited_tissue": current_tissue,
                "change_region": cell_change_region,
                "ops_log": ops_log,
                "original_tissue": original_tissue,
                "has_shrink": False,
            }

        # ---- 常规模式: 组织层有变化 ----

        # Step 6: 质心在 tissue_change_region 外的细胞 → 完整粘贴到 edited_tissue 上
        retained_nuclei = self._retain_cells_outside_change(
            cell_instances, tissue_change_region, current_tissue)

        # Step 6.9: 粘贴保留细胞后, 计算需要生成细胞的区域
        #   = 当前状态 (edited_tissue + retained_nuclei) vs 原始完整 mask 不一样的区域
        current_combined = self._merge_tissue_and_cells(current_tissue, retained_nuclei)
        fill_region = (current_combined != original_combined_mask)
        logger.info(f"Fill region (need new cells): {fill_region.sum()} pixels "
                    f"({fill_region.sum() / fill_region.size * 100:.1f}%)")

        # Step 7: 在 fill_region 中生成新细胞 (带 type_bias 和 density_scale)
        if self.nuclei_library_path and fill_region.any():
            logger.info("Filling nuclei: ProbNet → location/type, Library → shape...")
            new_cells = self._fill_cells_probnet_and_library(
                current_tissue, retained_nuclei, fill_region,
                type_bias_map=type_bias_map,
                density_scale_map=density_scale_map)
            retained_nuclei[new_cells > 0] = new_cells[new_cells > 0]
        elif fill_region.any():
            logger.info("No nuclei library configured, fill region will have no cells")

        # Step 8: 合并得到完整 tar_mask
        edited_combined = self._merge_tissue_and_cells(current_tissue, retained_nuclei)

        # Step 8.5: 在完整的 tar_mask 上做细胞调整 (type bias / density)
        #   此时所有细胞都是完整的实例, 不存在跨边界切割问题
        if has_cell_adjustments:
            logger.info("Applying cell adjustments on final mask...")
            # 从完整的 tar_mask 重新提取所有细胞实例
            final_cell_instances = self._get_cell_instances(edited_combined)
            logger.info(f"  Final mask has {len(final_cell_instances)} cell instances")

            # 在 edited_tissue 上做调整
            adjusted_nuclei = self._apply_cell_only_adjustments(
                final_cell_instances, current_tissue, type_bias_map, density_scale_map)

            # 用调整后的结果重建 tar_mask
            edited_combined = self._merge_tissue_and_cells(current_tissue, adjusted_nuclei)

        # Step 9: 构建 src_mask
        original_nuclei = self._retain_cells_outside_change(
            cell_instances, np.zeros_like(original_tissue, dtype=bool), original_tissue)
        original_combined = self._merge_tissue_and_cells(original_tissue, original_nuclei)

        logger.info("Expand mode: src=original, tar=edited")

        # 最终 change_region: 返回给 test_pipeline 决定 inpaint 范围
        full_change_region = (original_combined != edited_combined)
        logger.info(f"Full change region (tissue+cell): {full_change_region.sum()} pixels "
                    f"({full_change_region.sum() / full_change_region.size * 100:.1f}%) "
                    f"[tissue only was {tissue_change_region.sum()} pixels]")

        return {
            "src_mask": original_combined,
            "tar_mask": edited_combined,
            "edited_tissue": current_tissue,
            "change_region": full_change_region,
            "ops_log": ops_log,
            "original_tissue": original_tissue,
            "has_shrink": False,
        }

    def _run_probnet(self, tissue, nuclei, change_region):
        logger.warning("ProbNet inference not yet connected, "
                       "returning nuclei with empty change region")
        return nuclei

    def _ensure_library(self):
        if self._nuclei_library is not None:
            return

        import sys
        from pathlib import Path

        project_root = Path(self.project_root)
        inpaint_dir = project_root / "inpaint_cells" / "DDPM+Cell_inpaint"

        if str(inpaint_dir) not in sys.path:
            sys.path.insert(0, str(inpaint_dir))

        from generate_nuclei import NucleiLibrary
        self._nuclei_library = NucleiLibrary(self.nuclei_library_path)
        logger.info(f"Loaded NucleiLibrary from {self.nuclei_library_path}")

    def _ensure_probnet(self):
        if self._prob_net is not None:
            return
        if self.prob_net_ckpt is None:
            return

        import sys
        import torch
        from pathlib import Path

        project_root = Path(self.project_root)
        inpaint_dir = project_root / "inpaint_cells" / "DDPM+Cell_inpaint"

        if str(inpaint_dir) not in sys.path:
            sys.path.insert(0, str(inpaint_dir))

        from train_prob_net import ProbUNet, NUM_TISSUE, NUM_NUCLEI

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ProbUNet(in_ch=NUM_TISSUE + NUM_NUCLEI + 1, out_ch=NUM_NUCLEI, base_ch=64)
        ckpt = torch.load(self.prob_net_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()

        self._prob_net = model
        self._prob_net_device = device
        logger.info(f"Loaded ProbNet from {self.prob_net_ckpt}")

    def _fill_cells_probnet_and_library(
        self,
        edited_tissue: np.ndarray,
        retained_nuclei: np.ndarray,
        change_region: np.ndarray,
        type_bias_map: Optional[Dict] = None,
        density_scale_map: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        import sys
        from pathlib import Path

        project_root = Path(self.project_root)
        inpaint_dir = project_root / "inpaint_cells" / "DDPM+Cell_inpaint"
        if str(inpaint_dir) not in sys.path:
            sys.path.insert(0, str(inpaint_dir))

        INPAINT_DIR = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/inpaint_cells/DDPM+Cell_inpaint"
        if INPAINT_DIR not in sys.path:
            sys.path.insert(0, INPAINT_DIR)

        from generate_nuclei import poisson_disk_sampling, place_nucleus

        H, W = edited_tissue.shape
        new_cells = np.zeros((H, W), dtype=np.int64)

        self._ensure_library()
        library = self._nuclei_library

        # === Step A: 获取概率图 ===
        prob_map = None
        self._ensure_probnet()

        if self._prob_net is not None:
            import torch
            import torch.nn.functional as F
            from train_prob_net import to_onehot, NUM_TISSUE, NUM_NUCLEI

            tissue_oh = to_onehot(edited_tissue.astype(np.int64), NUM_TISSUE)

            NUCLEI_CLASSES = [101, 102, 103, 104, 105]
            nuclei_idx = np.zeros((H, W), dtype=np.int64)
            for i, nuc_val in enumerate(NUCLEI_CLASSES):
                nuclei_idx[retained_nuclei == nuc_val] = i + 1
            nuclei_idx[change_region] = 0
            nuclei_oh = to_onehot(nuclei_idx, NUM_NUCLEI)

            mask_ch = change_region.astype(np.float32)[np.newaxis, :, :]

            device = self._prob_net_device
            with torch.no_grad():
                t_tissue = torch.from_numpy(tissue_oh).unsqueeze(0).to(device)
                t_nuclei = torch.from_numpy(nuclei_oh).unsqueeze(0).to(device)
                t_mask = torch.from_numpy(mask_ch).unsqueeze(0).to(device)

                logits = self._prob_net(t_tissue, t_nuclei, t_mask)
                prob_map = F.softmax(logits, dim=1)[0].cpu().numpy()

            logger.info("  ProbNet inference done")

        # === Step B: 用概率图 + Library 填充 ===
        NUCLEI_CLASSES = [101, 102, 103, 104, 105]
        total_placed = 0

        for tissue_id in np.unique(edited_tissue[change_region]):
            tissue_id = int(tissue_id)
            tissue_region = change_region & (edited_tissue == tissue_id)

            if tissue_region.sum() < 50:
                continue

            if prob_map is not None:
                nuc_prob = 1.0 - prob_map[0]
                avg_nuc_prob = nuc_prob[tissue_region].mean()
                region_area = tissue_region.sum()
                num_nuclei = int(avg_nuc_prob * region_area / 80)
            else:
                density = library.get_density(tissue_id)
                region_area = tissue_region.sum()
                num_nuclei = int(density * region_area / 10000.0)

            num_nuclei = max(0, int(num_nuclei * np.random.uniform(0.8, 1.2)))
            if num_nuclei == 0:
                continue

            stats = library.stats.get(str(tissue_id), {})
            mean_areas = [info['mean_area'] for info in stats.get('nuclei_types', {}).values()
                         if info.get('mean_area', 0) > 0]
            avg_area = np.mean(mean_areas) if mean_areas else 100
            min_distance = max(np.sqrt(avg_area / np.pi) * 3, 10)

            if density_scale_map and tissue_id in density_scale_map:
                min_distance *= density_scale_map[tissue_id]
                min_distance = max(min_distance, 5)

            centers = poisson_disk_sampling(tissue_region, min_distance)
            if len(centers) > num_nuclei:
                import random
                random.shuffle(centers)
                centers = centers[:num_nuclei]

            for cy, cx in centers:
                if prob_map is not None:
                    type_probs = prob_map[1:, cy, cx].copy()
                    if type_probs.sum() < 0.05:
                        continue
                    if type_bias_map and tissue_id in type_bias_map:
                        _BIAS_NAME_TO_IDX = {
                            "neoplastic": 0, "inflammatory": 1,
                            "connective": 2, "dead": 3, "epithelial": 4,
                        }
                        for tname, mult in type_bias_map[tissue_id].items():
                            if tname in _BIAS_NAME_TO_IDX:
                                type_probs[_BIAS_NAME_TO_IDX[tname]] *= mult
                    type_probs = type_probs / type_probs.sum()
                    nuc_type_idx = np.random.choice(5, p=type_probs)
                else:
                    type_dist = library.get_type_distribution(tissue_id)
                    if not type_dist:
                        continue
                    types = list(type_dist.keys())
                    probs = np.array([type_dist[t] for t in types])
                    if type_bias_map and tissue_id in type_bias_map:
                        _BIAS_TYPE_TO_VAL = {
                            "neoplastic": 101, "inflammatory": 102,
                            "connective": 103, "dead": 104, "epithelial": 105,
                        }
                        for tname, mult in type_bias_map[tissue_id].items():
                            val = _BIAS_TYPE_TO_VAL.get(tname)
                            if val is not None and val in types:
                                idx = types.index(val)
                                probs[idx] *= mult
                    probs = probs / probs.sum()
                    nuc_type_idx = NUCLEI_CLASSES.index(np.random.choice(types, p=probs))

                nuc_type = NUCLEI_CLASSES[nuc_type_idx]

                instance = library.sample_instance(tissue_id, nuc_type)
                if instance is None:
                    continue

                if place_nucleus(new_cells, cy, cx, instance, augment=True):
                    total_placed += 1

        logger.info(f"  Placed {total_placed} nuclei in change region")
        return new_cells


# =============================================================================
# 便捷接口
# =============================================================================

class SemanticEditor:
    def __init__(
        self,
        llm_path: str,
        prior_db_path: str,
        prob_net_ckpt: Optional[str] = None,
        nuclei_library_path: Optional[str] = None,
        llm_device: str = "cuda",
        seed: int = 42,
    ):
        self.llm_path = llm_path
        self.llm_device = llm_device
        self._llm_parser = None

        self.mask_editor = MaskEditor(
            prior_db_path=prior_db_path,
            prob_net_ckpt=prob_net_ckpt,
            nuclei_library_path=nuclei_library_path,
            seed=seed,
        )

    def _ensure_llm(self):
        if self._llm_parser is None:
            from llm_parser import LLMParser
            self._llm_parser = LLMParser(self.llm_path, device=self.llm_device)

    def edit_from_reports(self, original_mask, original_report, edited_report):
        self._ensure_llm()
        semantic_diff = self._llm_parser.parse(original_report, edited_report)
        logger.info(f"Semantic diff: {json.dumps(semantic_diff, indent=2)}")
        result = self.mask_editor.edit(original_mask, semantic_diff)
        result["semantic_diff"] = semantic_diff
        result["original_report"] = original_report
        result["edited_report"] = edited_report
        return result

    def edit_from_diff(self, original_mask, semantic_diff):
        return self.mask_editor.edit(original_mask, semantic_diff)


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Layer 2: Rule Engine Test (with cell-only adjustments)")
    print("=" * 60)

    mask = np.full((256, 256), 2, dtype=np.int16)
    yy, xx = np.ogrid[:256, :256]
    mask[((yy-128)**2/60**2 + (xx-128)**2/80**2) < 1] = 1
    mask[((yy-50)**2/25**2 + (xx-50)**2/30**2) < 1] = 3
    mask[200:256, 0:60] = 4

    print(f"Mask tissues: {MaskAnalyzer(mask).summary()}")

    print("\n--- Test 1: tumor growth + lymph increase ---")
    engine = RuleEngine(seed=42)
    diff1 = {
        "tumor_change": {"growth": "increase", "degree": "moderate", "grade_change": "none"},
        "lymphocyte_change": {"infiltration": "increase", "degree": "significant"},
        "necrosis_change": {"action": "none", "extent": "focal"},
        "stroma_change": {"density": "none"},
    }
    ops1 = engine.plan(diff1, mask)
    bias1, dens1 = engine.compute_cell_adjustments(diff1)
    for op in ops1:
        print(f"  {op['op']}: {op['params']} ({op.get('reason', '')})")
    print(f"  Cell adjustments: bias={bias1}, density={dens1}")

    print("\n--- Test 2: necrosis decrease (fibrosis) ---")
    diff2 = {
        "tumor_change": {"growth": "none", "degree": "mild", "grade_change": "none"},
        "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
        "necrosis_change": {"action": "decrease", "extent": "moderate"},
        "stroma_change": {"density": "none"},
    }
    ops2 = engine.plan(diff2, mask)
    for op in ops2:
        print(f"  {op['op']}: {op['params']} ({op.get('reason', '')})")

    print("\n--- Test 3: grade upgrade (cell-only) ---")
    diff3 = {
        "tumor_change": {"growth": "none", "degree": "moderate", "grade_change": "upgrade"},
        "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
        "necrosis_change": {"action": "none", "extent": "focal"},
        "stroma_change": {"density": "none"},
    }
    ops3 = engine.plan(diff3, mask)
    bias3, dens3 = engine.compute_cell_adjustments(diff3)
    print(f"  Tissue ops: {len(ops3)} (expected 0)")
    print(f"  Cell adjustments: bias={bias3}, density={dens3}")

    print("\n--- Test 4: tumor growth moderate (delta min guarantee) ---")
    diff4 = {
        "tumor_change": {"growth": "increase", "degree": "moderate", "grade_change": "none"},
        "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
        "necrosis_change": {"action": "none", "extent": "focal"},
        "stroma_change": {"density": "none"},
    }
    ops4 = engine.plan(diff4, mask)
    for op in ops4:
        print(f"  {op['op']}: {op['params']} ({op.get('reason', '')})")
        if op['op'] == 'tumor_dilate':
            delta = op['params']['target_delta']
            print(f"  → delta={delta:.3f} (should be >= 0.10 for moderate)")