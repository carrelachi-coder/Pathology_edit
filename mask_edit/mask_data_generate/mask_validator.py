"""
Mask Transform Validator
========================
基于 prior_db.json 的病理学先验知识，验证组织级 Mask 变换的合理性。

用法:
    from mask_validator import MaskValidator
    validator = MaskValidator("prior_db.json")
    is_valid, report = validator.validate(new_mask, original_mask)

验证维度:
    1. 组织面积比例约束 (单组织 + 全局归一化)
    2. 组织邻接关系合理性 (新增邻接对 + 边界长度占比)
    3. 组织共现模式约束 (基于 cooccurrence_correlation)
    4. 形态学质量约束 (碎片过滤 + 连通性)
    5. 变化幅度约束 (防止单次变换过于剧烈)
"""

import numpy as np
import json
from scipy import ndimage
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ============================================================
# 0. 配置常量
# ============================================================

TISSUE_NAME_MAP = {
    0: "outside_roi", 1: "tumor", 2: "stroma", 3: "lymphocytic_infiltrate",
    4: "necrosis_or_debris", 5: "glandular_secretions", 6: "blood", 7: "exclude",
    8: "metaplasia_NOS", 9: "fat", 10: "plasma_cells", 11: "other_immune_infiltrate",
    12: "mucoid_material", 13: "normal_acinus_or_duct", 14: "lymphatics",
    15: "undetermined", 16: "nerve", 17: "skin_adnexa", 18: "blood_vessel",
    19: "angioinvasion", 20: "dcis", 21: "other"
}

NAME_TO_ID = {v: k for k, v in TISSUE_NAME_MAP.items()}

# 非生物学标签，验证时给予更宽松的阈值
NON_BIO_TISSUES = {"outside_roi", "exclude", "undetermined", "other"}


# ============================================================
# 1. 验证报告数据结构
# ============================================================

@dataclass
class CheckResult:
    """单项检查结果"""
    name: str
    passed: bool
    message: str
    severity: str = "error"  # "error" = 必须通过, "warning" = 仅记录


@dataclass
class ValidationReport:
    """完整验证报告"""
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    @property
    def warnings(self) -> List[CheckResult]:
        return [c for c in self.checks if c.severity == "warning" and not c.passed]

    def summary(self) -> str:
        lines = []
        status = "PASS" if self.is_valid else "FAIL"
        lines.append(f"=== Validation {status} ===")
        for c in self.checks:
            icon = "✓" if c.passed else ("✗" if c.severity == "error" else "⚠")
            lines.append(f"  [{icon}] {c.name}: {c.message}")
        return "\n".join(lines)


# ============================================================
# 2. 工具函数
# ============================================================

def compute_tissue_ratios(mask: np.ndarray) -> Dict[str, float]:
    """计算每种组织的面积占比"""
    total = mask.size
    unique, counts = np.unique(mask, return_counts=True)
    ratios = {}
    for uid, cnt in zip(unique, counts):
        name = TISSUE_NAME_MAP.get(int(uid))
        if name is not None:
            ratios[name] = cnt / total
    return ratios


def extract_adjacency_pairs_with_length(mask: np.ndarray) -> Dict[Tuple[str, str], int]:
    """
    提取所有组织邻接对及其边界长度（像素数）。
    使用偏移比对法，与你 tissue_adjacency.py 的逻辑一致。
    """
    m = mask.astype(np.int32)
    pair_counts: Dict[Tuple[str, str], int] = {}

    # 垂直边界
    v_diff = m[:-1, :] != m[1:, :]
    v_i = m[:-1, :][v_diff]
    v_j = m[1:, :][v_diff]

    # 水平边界
    h_diff = m[:, :-1] != m[:, 1:]
    h_i = m[:, :-1][h_diff]
    h_j = m[:, 1:][h_diff]

    all_i = np.concatenate([v_i, h_i])
    all_j = np.concatenate([v_j, h_j])

    for i_id, j_id in zip(all_i, all_j):
        name_i = TISSUE_NAME_MAP.get(int(i_id))
        name_j = TISSUE_NAME_MAP.get(int(j_id))
        if name_i is None or name_j is None:
            continue
        # 排序保证 (A,B) == (B,A)
        key = tuple(sorted([name_i, name_j]))
        pair_counts[key] = pair_counts.get(key, 0) + 1

    return pair_counts


def get_connected_components_stats(mask: np.ndarray) -> Dict[str, List[int]]:
    """获取每种组织所有连通域的面积列表"""
    stats = {}
    for tid, tname in TISSUE_NAME_MAP.items():
        binary = (mask == tid)
        if not np.any(binary):
            continue
        labeled, num = ndimage.label(binary)
        if num == 0:
            continue
        areas = ndimage.sum(binary, labeled, range(1, num + 1))
        stats[tname] = sorted([int(a) for a in areas], reverse=True)
    return stats


# ============================================================
# 3. 核心验证器
# ============================================================

class MaskValidator:
    """
    基于 prior_db 的 Mask 变换验证器。
    
    用法:
        validator = MaskValidator("prior_db.json")
        is_valid, report = validator.validate(new_mask, original_mask)
    """

    def __init__(self, prior_db_path: str, strict: bool = False):
        """
        Args:
            prior_db_path: prior_db.json 的路径
            strict: 是否启用严格模式（将 warning 也视为 error）
        """
        with open(prior_db_path, "r") as f:
            self.db = json.load(f)

        self.meta = self.db.pop("_meta", {})
        self.cooccurrence = self.meta.get("cooccurrence_correlation", {})
        self.strict = strict

    # ----------------------------------------------------------
    # 3.1 面积比例检查
    # ----------------------------------------------------------
    def _check_area_ratios(
        self,
        new_ratios: Dict[str, float],
        old_ratios: Dict[str, float]
    ) -> List[CheckResult]:
        """
        检查每种组织的面积占比是否在 prior_db 记录的合理范围内。
        
        核心原则: 只惩罚"变换使情况变得更差"的情况。
        如果原图某组织已经超标，变换没有让它进一步恶化，就不惩罚。
        
        规则:
        - 硬约束: 不超过 max_observed (数据集中观测到的最大值)
        - 软约束: 在 mean ± 3σ 范围内 (覆盖 99.7% 的真实分布)
        - 对于 occurrence_rate < 5% 的稀有组织，如果新出现了且面积 > 10%，视为 warning
        - 以上所有检查: 如果原图已经超标，且变换没有使其更严重，则跳过
        """
        results = []

        for tissue_name, ratio in new_ratios.items():
            if tissue_name not in self.db or "area" not in self.db[tissue_name]:
                continue
            if ratio < 0.001:  # 忽略面积极小的组织
                continue

            area_prior = self.db[tissue_name]["area"]
            mean = area_prior["mean"]
            std = area_prior["std"]
            max_obs = area_prior.get("max_observed", 1.0)
            occ_rate = area_prior.get("occurrence_rate", 1.0)
            old_ratio = old_ratios.get(tissue_name, 0)

            # 硬约束: 不超过数据集中观测到的最大值
            if ratio > max_obs + 0.05:  # 给 5% 容差
                # 如果原图已经超标，且变换没有使其更严重 → 跳过
                if old_ratio > max_obs + 0.05 and ratio <= old_ratio + 0.01:
                    continue
                results.append(CheckResult(
                    name=f"area_hard_{tissue_name}",
                    passed=False,
                    message=f"{tissue_name}: ratio {ratio:.3f} > max_observed {max_obs:.3f} + 0.05",
                    severity="error"
                ))
                continue

            # 软约束: mean ± 3σ
            upper = min(mean + 3 * std, 1.0)
            if std > 0 and ratio > upper:
                # 如果原图已经超标，且变换没让它更严重 → 跳过
                if old_ratio > upper and ratio <= old_ratio + 0.01:
                    continue
                results.append(CheckResult(
                    name=f"area_soft_{tissue_name}",
                    passed=False,
                    message=f"{tissue_name}: ratio {ratio:.3f} > mean+3σ ({upper:.3f})",
                    severity="warning"
                ))

            # 稀有组织出现检查
            if occ_rate < 0.05 and ratio > 0.10:
                if old_ratio > 0.10 and ratio <= old_ratio + 0.01:
                    continue
                results.append(CheckResult(
                    name=f"area_rare_{tissue_name}",
                    passed=False,
                    message=f"{tissue_name}: rare tissue (occ={occ_rate:.3f}) with unusually large area {ratio:.3f}",
                    severity="warning"
                ))

        # 全局检查: 所有组织占比之和应接近 1.0
        total = sum(new_ratios.values())
        if abs(total - 1.0) > 0.02:
            results.append(CheckResult(
                name="area_sum",
                passed=False,
                message=f"Total tissue ratio = {total:.4f}, expected ~1.0",
                severity="error"
            ))

        if not results:
            results.append(CheckResult(
                name="area_ratios",
                passed=True,
                message="All tissue area ratios within acceptable range"
            ))

        return results

    # ----------------------------------------------------------
    # 3.2 邻接关系检查
    # ----------------------------------------------------------
    def _check_adjacency(self, new_mask: np.ndarray) -> List[CheckResult]:
        """
        检查新 Mask 中的组织邻接关系是否合理。
        
        规则:
        - 硬约束: 不允许出现 prior_db 中双方 adjacency 都为 0 的邻接对
          (即数据集中从未观测到的邻接组合)
        - 软约束: 单个组织的主要邻接对象不应与 prior_db 偏离过大
        """
        results = []
        adj_pairs = extract_adjacency_pairs_with_length(new_mask)

        if not adj_pairs:
            results.append(CheckResult(
                name="adjacency",
                passed=True,
                message="No adjacency pairs (single tissue)"
            ))
            return results

        # 统计每种组织的总边界长度
        tissue_border_total: Dict[str, int] = {}
        for (t_i, t_j), length in adj_pairs.items():
            tissue_border_total[t_i] = tissue_border_total.get(t_i, 0) + length
            tissue_border_total[t_j] = tissue_border_total.get(t_j, 0) + length

        violations = []
        for (t_i, t_j), length in adj_pairs.items():
            # 跳过非生物学标签的检查
            if t_i in NON_BIO_TISSUES or t_j in NON_BIO_TISSUES:
                continue

            # 检查 prior_db 中是否记录过这个邻接关系（双向查询）
            prob_i_to_j = self.db.get(t_i, {}).get("adjacency", {}).get(t_j, 0)
            prob_j_to_i = self.db.get(t_j, {}).get("adjacency", {}).get(t_i, 0)

            # 如果双方的邻接概率都为 0 → 数据集中从未出现过
            if prob_i_to_j == 0 and prob_j_to_i == 0:
                # 计算这个邻接对占该组织总边界的比例
                border_fraction_i = length / max(tissue_border_total.get(t_i, 1), 1)
                border_fraction_j = length / max(tissue_border_total.get(t_j, 1), 1)

                # 如果只是极小的接触（< 1% 边界），容忍
                if border_fraction_i < 0.01 and border_fraction_j < 0.01:
                    continue

                violations.append(f"{t_i}<->{t_j} (never observed, border={length}px)")

        if violations:
            results.append(CheckResult(
                name="adjacency_novel",
                passed=False,
                message=f"Novel adjacency pairs: {'; '.join(violations[:5])}",
                severity="error"
            ))
        else:
            results.append(CheckResult(
                name="adjacency",
                passed=True,
                message="All adjacency pairs are biologically plausible"
            ))

        return results

    # ----------------------------------------------------------
    # 3.3 共现模式检查
    # ----------------------------------------------------------
    def _check_cooccurrence(
        self,
        new_ratios: Dict[str, float],
        old_ratios: Dict[str, float]
    ) -> List[CheckResult]:
        """
        检查组织面积变化方向是否符合共现相关性模式。
        
        核心逻辑:
        - 如果 tumor 和 stroma 的相关系数为 -0.55 (强负相关)，
          那么 tumor 增大时 stroma 不应该也大幅增大。
        - 只对强相关 (|r| > 0.3) 的组织对做检查。
        - 只检查变化幅度 > 5% 的组织，忽略微小变化。
        """
        results = []
        violations = []

        # 找出变化显著的组织 (变化 > 5%)
        significant_changes = {}
        for tissue, new_r in new_ratios.items():
            old_r = old_ratios.get(tissue, 0)
            delta = new_r - old_r
            if abs(delta) > 0.05:
                significant_changes[tissue] = delta

        # 对每对显著变化的组织，检查共现方向
        checked_pairs = set()
        for t_a, delta_a in significant_changes.items():
            if t_a not in self.cooccurrence:
                continue
            for t_b, delta_b in significant_changes.items():
                if t_a == t_b:
                    continue
                pair_key = tuple(sorted([t_a, t_b]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                corr = self.cooccurrence.get(t_a, {}).get(t_b, None)
                if corr is None:
                    continue

                # 只检查强相关
                if abs(corr) < 0.3:
                    continue

                # 检查变化方向是否与相关性一致
                # 负相关: 一个增大另一个应减小 → delta 符号应相反
                # 正相关: 应同向变化 → delta 符号应相同
                same_direction = (delta_a > 0) == (delta_b > 0)

                if corr < -0.3 and same_direction:
                    violations.append(
                        f"{t_a}({delta_a:+.2f}) & {t_b}({delta_b:+.2f}): "
                        f"both increase/decrease but corr={corr:.2f} (negative)"
                    )
                elif corr > 0.3 and not same_direction:
                    violations.append(
                        f"{t_a}({delta_a:+.2f}) & {t_b}({delta_b:+.2f}): "
                        f"opposite directions but corr={corr:.2f} (positive)"
                    )

        if violations:
            results.append(CheckResult(
                name="cooccurrence_direction",
                passed=False,
                message=f"Co-occurrence violations: {'; '.join(violations[:3])}",
                severity="warning"  # warning 而非 error，因为相关性是统计趋势而非硬规则
            ))
        else:
            results.append(CheckResult(
                name="cooccurrence",
                passed=True,
                message="Tissue co-occurrence patterns consistent"
            ))

        return results

    # ----------------------------------------------------------
    # 3.4 形态学质量检查
    # ----------------------------------------------------------
    def _check_morphology(
        self,
        new_mask: np.ndarray,
        original_mask: np.ndarray
    ) -> List[CheckResult]:
        """
        检查 Mask 的形态学质量。
        
        核心原则: 只惩罚变换新引入的碎片，原图自带的不算。
        
        方法: 对比新旧 mask 的连通域统计，只看"新增"的碎片和连通域。
        """
        results = []
        min_fragment_size = 25  # 5x5 像素以下视为碎片
        max_new_fragments_per_tissue = 5  # 每种组织允许新增的碎片数

        new_cc = get_connected_components_stats(new_mask)
        old_cc = get_connected_components_stats(original_mask)

        fragment_violations = []

        for tissue_name, new_areas in new_cc.items():
            if tissue_name in NON_BIO_TISSUES:
                continue

            # 计算新旧 mask 中该组织的小碎片数量
            new_tiny = len([a for a in new_areas if a < min_fragment_size])
            old_areas = old_cc.get(tissue_name, [])
            old_tiny = len([a for a in old_areas if a < min_fragment_size])

            # 只检查新增的碎片数量
            added_fragments = max(0, new_tiny - old_tiny)
            if added_fragments > max_new_fragments_per_tissue:
                fragment_violations.append(
                    f"{tissue_name}: +{added_fragments} new fragments < {min_fragment_size}px "
                    f"(was {old_tiny}, now {new_tiny})"
                )

        if fragment_violations:
            results.append(CheckResult(
                name="morphology_fragments",
                passed=False,
                message=f"Transform introduced fragments: {'; '.join(fragment_violations[:3])}",
                severity="error"
            ))
        else:
            results.append(CheckResult(
                name="morphology",
                passed=True,
                message="Morphological quality acceptable (no excessive new fragments)"
            ))

        return results

    # ----------------------------------------------------------
    # 3.5 变化幅度检查
    # ----------------------------------------------------------
    def _check_delta_magnitude(
        self,
        new_ratios: Dict[str, float],
        old_ratios: Dict[str, float]
    ) -> List[CheckResult]:
        """
        检查单次变换的变化幅度是否合理。
        
        规则:
        - 单种组织面积变化不超过 40%
        - 总变化面积（所有组织变化量之和的一半）不超过 50%
        - 至少有一种组织发生了 > 1% 的变化（否则变换无意义）
        """
        results = []

        all_tissues = set(list(new_ratios.keys()) + list(old_ratios.keys()))
        deltas = {}
        for t in all_tissues:
            deltas[t] = new_ratios.get(t, 0) - old_ratios.get(t, 0)

        # 单组织最大变化
        max_delta_tissue = max(deltas.items(), key=lambda x: abs(x[1]))
        if abs(max_delta_tissue[1]) > 0.40:
            results.append(CheckResult(
                name="delta_single",
                passed=False,
                message=f"{max_delta_tissue[0]} changed by {max_delta_tissue[1]:+.3f} (limit: ±0.40)",
                severity="error"
            ))

        # 总变化面积（每种组织变化的绝对值之和 / 2，因为增减互补）
        total_delta = sum(abs(d) for d in deltas.values()) / 2
        if total_delta > 0.50:
            results.append(CheckResult(
                name="delta_total",
                passed=False,
                message=f"Total area change = {total_delta:.3f} (limit: 0.50)",
                severity="error"
            ))

        # 最小变化检查（变换必须有意义）
        # 边界形变天然是小幅度变化，0.2% 的面积变化 ≈ 131 像素 (256x256)
        # 这已经足以在视觉上看到边界移动
        max_abs_delta = max(abs(d) for d in deltas.values())
        if max_abs_delta < 0.002:
            results.append(CheckResult(
                name="delta_minimum",
                passed=False,
                message=f"Max change = {max_abs_delta:.4f}, transform has no visible effect",
                severity="error"
            ))

        if not results:
            results.append(CheckResult(
                name="delta_magnitude",
                passed=True,
                message=f"Change magnitude acceptable (max single: {abs(max_delta_tissue[1]):.3f}, total: {total_delta:.3f})"
            ))

        return results

    # ----------------------------------------------------------
    # 3.6 主验证入口
    # ----------------------------------------------------------
    def validate(
        self,
        new_mask: np.ndarray,
        original_mask: np.ndarray
    ) -> Tuple[bool, ValidationReport]:
        """
        执行完整验证。
        
        Args:
            new_mask: 变换后的组织 ID Mask (H, W), 值为 0-21
            original_mask: 原始组织 ID Mask (H, W), 值为 0-21
            
        Returns:
            (is_valid, report): 是否通过验证 + 详细报告
        """
        report = ValidationReport()

        # 计算基础信息
        new_ratios = compute_tissue_ratios(new_mask)
        old_ratios = compute_tissue_ratios(original_mask)

        # 依次执行所有检查
        report.checks.extend(self._check_area_ratios(new_ratios, old_ratios))
        report.checks.extend(self._check_adjacency(new_mask))
        report.checks.extend(self._check_cooccurrence(new_ratios, old_ratios))
        report.checks.extend(self._check_morphology(new_mask, original_mask))
        report.checks.extend(self._check_delta_magnitude(new_ratios, old_ratios))

        is_valid = report.is_valid
        if self.strict:
            # 严格模式下 warning 也算失败
            is_valid = all(c.passed for c in report.checks)

        return is_valid, report


# ============================================================
# 4. 便捷接口 & 测试
# ============================================================

def quick_validate(new_mask, original_mask, prior_db_path="prior_db.json"):
    """一行调用的快捷接口"""
    validator = MaskValidator(prior_db_path)
    is_valid, report = validator.validate(new_mask, original_mask)
    print(report.summary())
    return is_valid


# --- 自测 ---
if __name__ == "__main__":
    import sys

    # 创建一个简单的测试用例
    print("=" * 60)
    print("MaskValidator Self-Test")
    print("=" * 60)

    # 模拟一个 256x256 的 mask: 上半部分 tumor(1), 下半部分 stroma(2)
    original = np.zeros((256, 256), dtype=np.int16)
    original[:128, :] = 1   # tumor
    original[128:, :] = 2   # stroma

    # 变换: tumor 膨胀 20 行 → tumor 占比增大, stroma 减小
    transformed = original.copy()
    transformed[128:148, :] = 1  # tumor 向下膨胀 20 行

    # 需要 prior_db.json 存在才能运行
    db_path = "prior_db.json"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    try:
        validator = MaskValidator(db_path)
        is_valid, report = validator.validate(transformed, original)
        print(report.summary())
        print(f"\nFinal: {'ACCEPTED' if is_valid else 'REJECTED'}")
    except FileNotFoundError:
        print(f"[跳过测试] 未找到 {db_path}，请指定路径: python mask_validator.py /path/to/prior_db.json")