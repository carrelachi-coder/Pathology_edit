"""
Lymphocyte Infiltration Transform — 淋巴浸润扩展
=================================================

病理意义:
    免疫反应增强时，淋巴细胞从基质中向肿瘤方向聚集浸润。
    这是独立于肿瘤生长的免疫学过程。
    TIL（肿瘤浸润淋巴细胞）密度是乳腺癌重要的预后指标。

    淋巴浸润扩展 = 淋巴区域吃掉相邻的基质（免疫细胞在基质中聚集）
    反向（淋巴消退/免疫抑制）通过训练时对调获得。

设计:
    和 tumor_dilate 共用 SDF+噪声框架。
    淋巴浸润的噪声频谱: 中频为主（散在浸润，不像肿瘤那样成团块）
    只能吃基质和脂肪（淋巴细胞在间质中聚集，不会侵入肿瘤细胞）

用法:
    from mask_validator import MaskValidator
    from lymphocyte_infiltration import LymphocyteInfiltrationTransform

    validator = MaskValidator("prior_db.json")
    transform = LymphocyteInfiltrationTransform("prior_db.json", validator)
    results = transform.generate_variants(mask, n_variants=3)
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import json
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 配置
# ============================================================

TISSUE_NAME_MAP = {
    0: "outside_roi", 1: "tumor", 2: "stroma", 3: "lymphocytic_infiltrate",
    4: "necrosis_or_debris", 5: "glandular_secretions", 6: "blood", 7: "exclude",
    8: "metaplasia_NOS", 9: "fat", 10: "plasma_cells", 11: "other_immune_infiltrate",
    12: "mucoid_material", 13: "normal_acinus_or_duct", 14: "lymphatics",
    15: "undetermined", 16: "nerve", 17: "skin_adnexa", 18: "blood_vessel",
    19: "angioinvasion", 20: "dcis", 21: "other"
}

LYMPH_ID = 3
NON_BIO_TISSUES = {"outside_roi", "exclude", "undetermined", "other"}

# 淋巴浸润可以吃的组织（淋巴细胞在间质中聚集）
LYMPH_EXPANSION_TARGETS = [2, 9, 10]  # stroma, fat, plasma_cells

# 淋巴浸润噪声频谱: 中频主导（散在浸润模式）
# 不像肿瘤（低频团块）也不像坏死（高频碎裂）
LYMPH_NOISE_OCTAVES = [
    (30.0, 0.35),   # 低频: 大致方向
    (12.0, 0.45),   # 中频: 散在浸润（主导）
    (5.0,  0.20),   # 高频: 细微不规则
]

LYMPH_ALPHA_RANGE = (8, 18)       # 噪声幅度(px)
LYMPH_DELTA_RANGE = (0.03, 0.15)  # 面积变化范围
LYMPH_INFLUENCE_RADIUS = 35       # 边界影响半径(px)


@dataclass
class TransformLog:
    transform_type: str
    tissue_id: int
    tissue_name: str
    params: dict
    area_change: dict
    accepted: bool
    rejection_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.transform_type, "tissue_id": self.tissue_id,
            "tissue_name": self.tissue_name, "params": self.params,
            "area_change": self.area_change, "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


# ============================================================
# 核心算法
# ============================================================

def compute_sdf(mask: np.ndarray, tissue_id: int) -> np.ndarray:
    """Signed Distance Field: 内部正, 外部负, 边界零"""
    binary = (mask == tissue_id).astype(np.float64)
    if not np.any(binary):
        return -ndimage.distance_transform_edt(np.ones_like(binary))
    if np.all(binary):
        return ndimage.distance_transform_edt(binary)
    return ndimage.distance_transform_edt(binary) - ndimage.distance_transform_edt(1 - binary)


def compute_boundary_weight(mask: np.ndarray, tissue_id: int, radius: float) -> np.ndarray:
    """基于主连通域的距离衰减权重"""
    binary = (mask == tissue_id).astype(np.uint8)
    if not np.any(binary) or np.all(binary):
        return np.zeros(mask.shape, dtype=np.float64)

    labeled, n = ndimage.label(binary)
    if n > 1:
        areas = ndimage.sum(binary, labeled, range(1, n + 1))
        binary = (labeled == (np.argmax(areas) + 1)).astype(np.float64)
    else:
        binary = binary.astype(np.float64)

    dist_in = ndimage.distance_transform_edt(binary)
    dist_out = ndimage.distance_transform_edt(1 - binary)
    return np.clip(1.0 - np.minimum(dist_in, dist_out) / radius, 0.0, 1.0)


def generate_noise(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """淋巴浸润特征噪声: 中频主导，散在浸润"""
    noise = np.zeros(shape, dtype=np.float64)
    for sigma, weight in LYMPH_NOISE_OCTAVES:
        raw = rng.standard_normal(shape)
        smoothed = gaussian_filter(raw, sigma=sigma)
        vmax = np.abs(smoothed).max()
        if vmax > 0:
            smoothed /= vmax
        noise += weight * smoothed

    vmax = np.abs(noise).max()
    if vmax > 0:
        noise /= vmax
    return noise


def dilate_lymphocytes(
    mask: np.ndarray,
    noise: np.ndarray,
    alpha: float,
    beta: float,
    weight: np.ndarray,
) -> np.ndarray:
    """淋巴浸润膨胀: 只吃基质/脂肪（免疫细胞在间质中聚集）"""
    sdf = compute_sdf(mask, LYMPH_ID)
    sdf_new = sdf + weight * (alpha * noise + beta)
    new_region = sdf_new > 0

    original_lymph = (mask == LYMPH_ID)

    # 保留影响范围外的小碎片
    new_region = new_region | (original_lymph & (weight == 0))
    new_region = new_region & ~((weight == 0) & ~original_lymph)

    # 只吃允许的组织
    allowed = original_lymph.copy()
    for tid in LYMPH_EXPANSION_TARGETS:
        allowed |= (mask == tid)
    new_region = (new_region & allowed) | (original_lymph & new_region)

    new_mask = mask.copy()
    new_mask[new_region & ~original_lymph] = LYMPH_ID
    return new_mask


def calibrate_beta(
    mask: np.ndarray,
    noise: np.ndarray,
    alpha: float,
    weight: np.ndarray,
    target_delta: float,
) -> float:
    """二分搜索精确控制面积增量"""
    total = mask.size
    target_change = target_delta * total
    original_area = (mask == LYMPH_ID).sum()
    sdf = compute_sdf(mask, LYMPH_ID)
    original_lymph = (mask == LYMPH_ID)

    beta_lo, beta_hi = 0.0, 80.0
    best_beta, best_err = 0.0, float('inf')

    for _ in range(30):
        beta_mid = (beta_lo + beta_hi) / 2.0
        sdf_shifted = sdf + weight * (alpha * noise + beta_mid)
        new_region = sdf_shifted > 0

        new_region = new_region | (original_lymph & (weight == 0))
        new_region = new_region & ~((weight == 0) & ~original_lymph)

        allowed = original_lymph.copy()
        for tid in LYMPH_EXPANSION_TARGETS:
            allowed |= (mask == tid)
        new_region = (new_region & allowed) | (original_lymph & new_region)

        actual_change = new_region.sum() - original_area
        err = abs(actual_change - target_change)
        if err < best_err:
            best_err = err
            best_beta = beta_mid
        if actual_change < target_change:
            beta_lo = beta_mid
        else:
            beta_hi = beta_mid

    return best_beta


def clean_topology(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """移除小碎片孤岛"""
    result = mask.copy()
    for tid in np.unique(mask):
        name = TISSUE_NAME_MAP.get(int(tid), "")
        if name in NON_BIO_TISSUES:
            continue
        binary = (result == tid).astype(np.uint8)
        labeled, n = ndimage.label(binary)
        if n <= 1:
            continue
        areas = ndimage.sum(binary, labeled, range(1, n + 1))
        main_label = np.argmax(areas) + 1
        for lbl in range(1, n + 1):
            if lbl == main_label or areas[lbl - 1] >= min_size:
                continue
            frag = (labeled == lbl)
            non_self = (result != tid)
            if np.any(non_self):
                _, fi = ndimage.distance_transform_edt(~non_self, return_indices=True)
                result[frag] = result[fi[0], fi[1]][frag]
    return result


# ============================================================
# 主接口
# ============================================================

class LymphocyteInfiltrationTransform:
    """
    淋巴浸润扩展: 模拟免疫反应增强。
    反向（淋巴消退）通过训练时对调获得。

    前提: mask 中必须已存在淋巴浸润区域。

    用法:
        validator = MaskValidator("prior_db.json")
        transform = LymphocyteInfiltrationTransform("prior_db.json", validator)
        results = transform.generate_variants(mask, n_variants=3)
    """

    def __init__(self, prior_db_path: str, validator, seed: int = 42):
        with open(prior_db_path, "r") as f:
            self.db = json.load(f)
        self.db.pop("_meta", None)
        self.validator = validator
        self.rng = np.random.default_rng(seed)
        self._history: List[bool] = []

    def _compute_area_change(self, old_mask, new_mask) -> Dict[str, float]:
        total = old_mask.size
        change = {}
        for tid in set(np.unique(old_mask)) | set(np.unique(new_mask)):
            name = TISSUE_NAME_MAP.get(int(tid))
            if name is None:
                continue
            delta = (new_mask == tid).sum() / total - (old_mask == tid).sum() / total
            if abs(delta) > 0.001:
                change[name] = round(float(delta), 4)
        return change

    def apply(
        self,
        mask: np.ndarray,
        target_delta: Optional[float] = None,
    ) -> Tuple[np.ndarray, TransformLog]:
        """
        淋巴浸润向基质方向扩展。

        Args:
            mask: 纯组织 Mask (H, W), 值 0-21
            target_delta: 面积增加量(占全图), None随机
        """
        lymph_ratio = (mask == LYMPH_ID).sum() / mask.size

        if lymph_ratio < 0.01:
            return mask, TransformLog(
                "lymph_dilate", LYMPH_ID, "lymphocytic_infiltrate",
                {"error": "no_lymph"}, {}, False,
                "No existing lymphocytic infiltrate"
            )

        # 可扩展空间
        expandable = sum((mask == tid).sum() for tid in LYMPH_EXPANSION_TARGETS) / mask.size
        if expandable < 0.03:
            return mask, TransformLog(
                "lymph_dilate", LYMPH_ID, "lymphocytic_infiltrate",
                {"error": "no_room"}, {}, False,
                "Not enough stroma/fat to expand into"
            )

        if target_delta is None:
            max_delta = min(LYMPH_DELTA_RANGE[1], expandable * 0.5)
            lo = min(LYMPH_DELTA_RANGE[0], max_delta)
            target_delta = float(self.rng.uniform(lo, max_delta))

        if target_delta < 0.005:
            return mask, TransformLog(
                "lymph_dilate", LYMPH_ID, "lymphocytic_infiltrate",
                {"error": "delta_too_small"}, {}, False,
                f"Delta {target_delta:.3f} too small"
            )

        # 噪声 + 权重
        noise = generate_noise(mask.shape, self.rng)
        alpha = float(self.rng.uniform(*LYMPH_ALPHA_RANGE))
        weight = compute_boundary_weight(mask, LYMPH_ID, LYMPH_INFLUENCE_RADIUS)

        # 校准beta
        beta = calibrate_beta(mask, noise, alpha, weight, target_delta)

        # 膨胀
        new_mask = dilate_lymphocytes(mask, noise, alpha, beta, weight)

        # 拓扑清理
        new_mask = clean_topology(new_mask, min_size=100)

        # 验证
        area_change = self._compute_area_change(mask, new_mask)
        actual_delta = area_change.get("lymphocytic_infiltrate", 0)
        is_valid, report = self.validator.validate(new_mask, mask)

        self._history.append(is_valid)
        if len(self._history) > 20:
            self._history.pop(0)

        log = TransformLog(
            transform_type="lymph_dilate",
            tissue_id=LYMPH_ID,
            tissue_name="lymphocytic_infiltrate",
            params={
                "target_delta": round(float(target_delta), 3),
                "actual_delta": round(float(actual_delta), 4),
                "alpha": round(alpha, 1),
                "beta": round(float(beta), 2),
            },
            area_change=area_change,
            accepted=is_valid,
            rejection_reason="" if is_valid else report.summary(),
        )

        return (new_mask if is_valid else mask), log

    def generate_variants(
        self,
        mask: np.ndarray,
        n_variants: int = 3,
        max_attempts: int = 10,
    ) -> List[Tuple[np.ndarray, TransformLog]]:
        """生成淋巴浸润变体，增量从小到大。"""
        lymph_ratio = (mask == LYMPH_ID).sum() / mask.size
        if lymph_ratio < 0.01:
            logger.info("[LymphDilate] Skipping: no lymphocytic infiltrate.")
            return []

        expandable = sum((mask == tid).sum() for tid in LYMPH_EXPANSION_TARGETS) / mask.size
        if expandable < 0.03:
            logger.info("[LymphDilate] Skipping: no room to expand.")
            return []

        max_delta = min(LYMPH_DELTA_RANGE[1], expandable * 0.5)
        lo = min(LYMPH_DELTA_RANGE[0], max_delta)
        deltas = np.linspace(lo, max_delta, n_variants).tolist()

        accepted = []
        total = 0
        for idx, td in enumerate(deltas):
            found = False
            for _ in range(max_attempts):
                total += 1
                new_mask, log = self.apply(mask, target_delta=td)
                if log.accepted:
                    accepted.append((new_mask, log))
                    found = True
                    break
            if not found:
                logger.info(f"  V{idx+1}/{n_variants} lymph (Δ=+{td:.2f}): failed")

        rate = sum(self._history) / len(self._history) if self._history else 0
        logger.info(
            f"[LymphDilate] {len(accepted)}/{n_variants} variants, "
            f"{total} attempts, rate={rate:.0%}"
        )
        return accepted


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    import sys, os

    print("=" * 50)
    print("LymphocyteInfiltrationTransform Test")
    print("=" * 50)

    mask = np.full((256, 256), 2, dtype=np.int16)  # stroma
    yy, xx = np.ogrid[:256, :256]
    mask[((yy-128)**2/60**2 + (xx-128)**2/80**2) < 1] = 1  # tumor
    mask[((yy-60)**2/30**2 + (xx-60)**2/40**2) < 1] = 3    # lymph
    mask[200:256, 180:256] = 9  # fat

    print(f"Tissues: { {TISSUE_NAME_MAP[t]: f'{(mask==t).sum()/mask.size:.1%}' for t in np.unique(mask)} }")

    db_path = sys.argv[1] if len(sys.argv) > 1 else "prior_db.json"
    if not os.path.exists(db_path):
        print(f"[跳过] 未找到 {db_path}")
        sys.exit(0)

    from mask_validator import MaskValidator
    validator = MaskValidator(db_path)
    transform = LymphocyteInfiltrationTransform(db_path, validator, seed=42)

    print("\n--- Generate 3 lymph infiltration variants ---")
    variants = transform.generate_variants(mask, n_variants=3)
    for i, (vm, vl) in enumerate(variants):
        p = vl.params
        print(f"  V{i+1}: Δ_target=+{p['target_delta']:.2f} "
              f"Δ_actual=+{p['actual_delta']:.4f} "
              f"α={p['alpha']:.0f} β={p['beta']:+.1f}")
        print(f"        {vl.area_change}")