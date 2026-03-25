"""
Tumor Boundary Transform v7 — 只做膨胀
========================================

核心简化:
    只实现肿瘤膨胀（浸润性生长）。
    收缩效果通过训练时对调 (mask_A, mask_B) 获得，无需单独实现。

    膨胀 = SDF + W·(α·noise + β)
    - 噪声场产生自然的浸润性边界
    - 距离衰减权重防止远处孤岛
    - 只吃基质/淋巴浸润等独立组织，不吃坏死

用法:
    from mask_validator import MaskValidator
    from boundary_deform import TumorBoundaryTransform

    validator = MaskValidator("prior_db.json")
    transform = TumorBoundaryTransform("prior_db.json", validator)
    results = transform.generate_variants(mask, n_variants=6)
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

TUMOR_ID = 1
NON_BIO_TISSUES = {"outside_roi", "exclude", "undetermined", "other"}

# 肿瘤噪声频谱: 低频团块 + 中频起伏 + 高频浸润指突
TUMOR_NOISE_OCTAVES = [
    (40.0, 0.55),
    (15.0, 0.30),
    (5.0,  0.15),
]

TUMOR_ALPHA_RANGE = (10, 25)      # 噪声幅度(px)
TUMOR_DELTA_RANGE = (0.05, 0.22)  # 面积变化范围
TUMOR_INFLUENCE_RADIUS = 45       # 边界影响半径(px) 

TUMOR_EXPANSION_TARGETS = [2, 3, 4, 9, 10, 13, 18]  # stroma, lymph, fat, plasma, blood_vessel


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
    """基于主连通域的距离衰减权重。边界处=1, radius外=0。"""
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
    """生成肿瘤特征的多尺度噪声场, 归一化到 [-1, 1]"""
    noise = np.zeros(shape, dtype=np.float64)
    for sigma, weight in TUMOR_NOISE_OCTAVES:
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


def dilate_tumor(
    mask: np.ndarray,
    noise: np.ndarray,
    alpha: float,
    beta: float,
    weight: np.ndarray,
) -> np.ndarray:
    """
    SDF偏移膨胀肿瘤。只吃 TUMOR_EXPANSION_TARGETS 中的组织。
    """
    sdf = compute_sdf(mask, TUMOR_ID)
    sdf_new = sdf + weight * (alpha * noise + beta)
    new_region = sdf_new > 0

    original_tumor = (mask == TUMOR_ID)

    # 保留不在影响范围内的小碎片
    new_region = new_region | (original_tumor & (weight == 0))
    # 影响范围外不创造新区域
    new_region = new_region & ~((weight == 0) & ~original_tumor)

    # 只能吃掉允许的组织
    allowed = original_tumor.copy()
    for tid in TUMOR_EXPANSION_TARGETS:
        allowed |= (mask == tid)
    new_region = (new_region & allowed) | (original_tumor & new_region)

    new_mask = mask.copy()
    new_mask[new_region & ~original_tumor] = TUMOR_ID
    return new_mask


def calibrate_beta(
    mask: np.ndarray,
    noise: np.ndarray,
    alpha: float,
    weight: np.ndarray,
    target_delta: float,
) -> float:
    """二分搜索找精确控制面积变化的beta"""
    total = mask.size
    target_change = target_delta * total
    original_area = (mask == TUMOR_ID).sum()
    sdf = compute_sdf(mask, TUMOR_ID)
    original_tumor = (mask == TUMOR_ID)

    beta_lo, beta_hi = 0.0, 80.0
    best_beta, best_err = 0.0, float('inf')

    for _ in range(30):
        beta_mid = (beta_lo + beta_hi) / 2.0
        sdf_shifted = sdf + weight * (alpha * noise + beta_mid)
        new_region = sdf_shifted > 0

        new_region = new_region | (original_tumor & (weight == 0))
        new_region = new_region & ~((weight == 0) & ~original_tumor)

        allowed = original_tumor.copy()
        for tid in TUMOR_EXPANSION_TARGETS:
            allowed |= (mask == tid)
        new_region = (new_region & allowed) | (original_tumor & new_region)

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
    """移除所有组织的小碎片孤岛"""
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
            else:
                result[frag] = 2
    return result


# ============================================================
# 主接口
# ============================================================

class TumorBoundaryTransform:
    """
    肿瘤膨胀变换器。收缩效果通过对调训练对获得。

    用法:
        validator = MaskValidator("prior_db.json")
        transform = TumorBoundaryTransform("prior_db.json", validator)
        results = transform.generate_variants(mask, n_variants=6)
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
        对肿瘤执行一次膨胀。

        Args:
            mask: 纯组织 Mask (H, W), 值 0-21
            target_delta: 肿瘤面积增加量 (0.10=增10%), None随机
        """
        if not np.any(mask == TUMOR_ID):
            return mask, TransformLog(
                "tumor_dilate", TUMOR_ID, "tumor",
                {"error": "no_tumor"}, {}, False, "No tumor in mask"
            )

        current_ratio = (mask == TUMOR_ID).sum() / mask.size

        if target_delta is None:
            target_delta = float(self.rng.uniform(*TUMOR_DELTA_RANGE))

        # 安全检查: 膨胀后不超95%
        if current_ratio + target_delta > 0.95:
            target_delta = 0.95 - current_ratio

        if target_delta < 0.01:
            return mask, TransformLog(
                "tumor_dilate", TUMOR_ID, "tumor",
                {"error": "no_room"}, {}, False,
                f"Not enough room: current={current_ratio:.2f}"
            )

        # 噪声 + 权重
        noise = generate_noise(mask.shape, self.rng)
        alpha = float(self.rng.uniform(*TUMOR_ALPHA_RANGE))
        weight = compute_boundary_weight(mask, TUMOR_ID, TUMOR_INFLUENCE_RADIUS)

        # 校准beta
        beta = calibrate_beta(mask, noise, alpha, weight, target_delta)

        # 膨胀
        new_mask = dilate_tumor(mask, noise, alpha, beta, weight)

        # 拓扑清理
        new_mask = clean_topology(new_mask, min_size=100)

        # 验证
        area_change = self._compute_area_change(mask, new_mask)
        actual_delta = area_change.get("tumor", 0)
        is_valid, report = self.validator.validate(new_mask, mask)

        self._history.append(is_valid)
        if len(self._history) > 20:
            self._history.pop(0)

        log = TransformLog(
            transform_type="tumor_dilate",
            tissue_id=TUMOR_ID,
            tissue_name="tumor",
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
        n_variants: int = 6,
        max_attempts: int = 10,
    ) -> List[Tuple[np.ndarray, TransformLog]]:
        """生成多样化的肿瘤膨胀变体，面积变化从小到大。"""
        if not np.any(mask == TUMOR_ID):
            logger.warning("No tumor in mask, skipping.")
            return []

        current_ratio = (mask == TUMOR_ID).sum() / mask.size
        max_possible = min(0.95 - current_ratio, TUMOR_DELTA_RANGE[1])

        if max_possible < 0.02:
            logger.warning(f"Tumor already at {current_ratio:.1%}, no room to dilate.")
            return []

        # 从小到大的膨胀量
        deltas = np.linspace(
            TUMOR_DELTA_RANGE[0],
            max_possible,
            n_variants
        ).tolist()

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
                logger.info(f"  V{idx+1}/{n_variants} tumor (Δ=+{td:.2f}): failed")

        rate = sum(self._history) / len(self._history) if self._history else 0
        logger.info(
            f"[TumorDilate] {len(accepted)}/{n_variants} variants, "
            f"{total} attempts, rate={rate:.0%}"
        )
        return accepted


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    import sys, os

    print("=" * 50)
    print("TumorBoundaryTransform v7 (dilate only)")
    print("=" * 50)

    mask = np.full((256, 256), 2, dtype=np.int16)
    yy, xx = np.ogrid[:256, :256]
    mask[((yy-128)**2/60**2 + (xx-128)**2/80**2) < 1] = 1
    mask[200:256, 0:60] = 4
    mask[0:40, 180:256] = 3

    print(f"Tissues: { {TISSUE_NAME_MAP[t]: f'{(mask==t).sum()/mask.size:.1%}' for t in np.unique(mask)} }")

    db_path = sys.argv[1] if len(sys.argv) > 1 else "prior_db.json"
    if not os.path.exists(db_path):
        print(f"[跳过] 未找到 {db_path}")
        sys.exit(0)

    from mask_validator import MaskValidator
    validator = MaskValidator(db_path)
    transform = TumorBoundaryTransform(db_path, validator, seed=42)

    print("\n--- Generate 6 tumor dilate variants ---")
    variants = transform.generate_variants(mask, n_variants=6)
    for i, (vm, vl) in enumerate(variants):
        p = vl.params
        print(f"  V{i+1}: Δ_target=+{p['target_delta']:.2f} Δ_actual=+{p['actual_delta']:.4f} "
              f"α={p['alpha']:.0f} β={p['beta']:+.1f}")
        print(f"        {vl.area_change}")