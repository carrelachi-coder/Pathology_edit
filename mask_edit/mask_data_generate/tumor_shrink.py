"""
Tumor Shrink Transform — 肿瘤收缩（治疗后退缩）
================================================

病理学背景:
    肿瘤收缩对应治疗后反应 (treatment response / tumor regression)。
    肿瘤细胞消退后，空出的区域被纤维化间质替代（伤口修复过程）。

算法:
    1. SDF 负偏移: sdf - W·(α·noise + β) → 边界从外向内侵蚀
    2. Edge fade mask: boundary_weight 乘以边缘渐变遮罩，patch 边界处不收缩
    3. 释放区域回填: nearest-neighbor，坏死不参与回填
    4. 碎片清除: 小于阈值的肿瘤残余岛回填
    5. beta 二分校准: 精确控制面积缩小量

用法:
    from tumor_shrink import TumorShrinkTransform
    from mask_validator import MaskValidator

    validator = MaskValidator("prior_db.json")
    transform = TumorShrinkTransform("prior_db.json", validator)
    new_mask, log = transform.apply(mask, target_delta=0.10)
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
NECROSIS_ID = 4
STROMA_ID = 2
NON_BIO_TISSUES = {"outside_roi", "exclude", "undetermined", "other"}

# 回填时排除的组织
BACKFILL_EXCLUDE = {TUMOR_ID, NECROSIS_ID, 0, 7, 15, 21}

# 噪声参数
SHRINK_NOISE_OCTAVES = [
    (45.0, 0.60),
    (18.0, 0.28),
    (6.0,  0.12),
]

SHRINK_ALPHA_RANGE = (10, 25)
SHRINK_DELTA_RANGE = (0.05, 0.30)
SHRINK_INFLUENCE_RADIUS = 45
MIN_FRAGMENT_SIZE = 80

# Patch 边缘渐变遮罩: 距离边缘 EDGE_FADE_MARGIN 像素内收缩量线性衰减到 0
# 防止 patch 边界处肿瘤出现方形截断
EDGE_FADE_MARGIN = 40


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


def compute_edge_fade_mask(shape: Tuple[int, int], margin: int = EDGE_FADE_MARGIN) -> np.ndarray:
    """
    Patch 边缘渐变遮罩。

    距离 patch 边缘 0px 处 = 0（不收缩）
    距离 patch 边缘 margin px 处 = 1（正常收缩）
    中间线性插值。

    乘到 weight 上后，边缘处的肿瘤不参与收缩，
    向内逐渐过渡到正常收缩，不会产生方形截断或空心边框。
    """
    H, W = shape
    fade = np.ones((H, W), dtype=np.float64)

    # 上下左右四条边的距离
    rows = np.arange(H, dtype=np.float64)
    cols = np.arange(W, dtype=np.float64)

    # 距离上边缘
    fade_top = np.clip(rows / margin, 0.0, 1.0)
    fade *= fade_top[:, np.newaxis]

    # 距离下边缘
    fade_bottom = np.clip((H - 1 - rows) / margin, 0.0, 1.0)
    fade *= fade_bottom[:, np.newaxis]

    # 距离左边缘
    fade_left = np.clip(cols / margin, 0.0, 1.0)
    fade *= fade_left[np.newaxis, :]

    # 距离右边缘
    fade_right = np.clip((W - 1 - cols) / margin, 0.0, 1.0)
    fade *= fade_right[np.newaxis, :]

    return fade


def generate_noise(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """生成收缩特征的多尺度噪声场, 归一化到 [-1, 1]"""
    noise = np.zeros(shape, dtype=np.float64)
    for sigma, w in SHRINK_NOISE_OCTAVES:
        raw = rng.standard_normal(shape)
        smoothed = gaussian_filter(raw, sigma=sigma)
        vmax = np.abs(smoothed).max()
        if vmax > 0:
            smoothed /= vmax
        noise += w * smoothed

    vmax = np.abs(noise).max()
    if vmax > 0:
        noise /= vmax
    return noise


def backfill_nearest_neighbor(mask: np.ndarray, released_region: np.ndarray) -> np.ndarray:
    """
    用最近的合法邻居组织回填释放区域。
    坏死不参与回填。
    """
    new_mask = mask.copy()

    if not np.any(released_region):
        return new_mask

    backfill_source = np.ones_like(mask, dtype=bool)
    for tid in BACKFILL_EXCLUDE:
        backfill_source &= (mask != tid)

    if not np.any(backfill_source):
        new_mask[released_region] = STROMA_ID
        return new_mask

    _, nearest_indices = ndimage.distance_transform_edt(
        ~backfill_source, return_indices=True
    )

    new_mask[released_region] = mask[nearest_indices[0][released_region],
                                     nearest_indices[1][released_region]]

    return new_mask


def shrink_tumor(
    mask: np.ndarray,
    noise: np.ndarray,
    alpha: float,
    beta: float,
    weight: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SDF负偏移收缩肿瘤。
    weight 已经乘过 edge_fade_mask，patch 边缘处 weight≈0，肿瘤不被收缩。
    """
    sdf = compute_sdf(mask, TUMOR_ID)
    original_tumor = (mask == TUMOR_ID)

    sdf_new = sdf - weight * (alpha * noise + beta)
    # 高斯平滑 SDF，消除锯齿边缘
    sdf_new = gaussian_filter(sdf_new, sigma=10.0)
    remaining_tumor = (sdf_new > 0) & original_tumor

    # 影响范围外的肿瘤保持不变
    remaining_tumor = remaining_tumor | (original_tumor & (weight == 0))

    released_region = original_tumor & ~remaining_tumor

    return remaining_tumor, released_region


def remove_small_fragments(
    mask: np.ndarray,
    tissue_id: int,
    min_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """移除指定组织的小碎片，用 nearest-neighbor 回填。"""
    binary = (mask == tissue_id).astype(np.uint8)
    labeled, n = ndimage.label(binary)

    if n <= 1:
        return mask, np.zeros_like(mask, dtype=bool)

    areas = ndimage.sum(binary, labeled, range(1, n + 1))
    fragment_region = np.zeros_like(mask, dtype=bool)

    for lbl in range(1, n + 1):
        if areas[lbl - 1] < min_size:
            fragment_region |= (labeled == lbl)

    if not np.any(fragment_region):
        return mask, fragment_region

    new_mask = backfill_nearest_neighbor(mask, fragment_region)
    return new_mask, fragment_region


def calibrate_beta_shrink(
    mask: np.ndarray,
    noise: np.ndarray,
    alpha: float,
    weight: np.ndarray,
    target_delta: float,
) -> float:
    """二分搜索找精确控制面积收缩量的beta。"""
    total = mask.size
    target_shrink = target_delta * total
    original_area = (mask == TUMOR_ID).sum()
    sdf = compute_sdf(mask, TUMOR_ID)
    original_tumor = (mask == TUMOR_ID)

    beta_lo, beta_hi = 0.0, 150.0
    best_beta, best_err = 0.0, float('inf')

    for _ in range(30):
        beta_mid = (beta_lo + beta_hi) / 2.0

        sdf_new = sdf - weight * (alpha * noise + beta_mid)
        remaining = (sdf_new > 0) & original_tumor
        remaining = remaining | (original_tumor & (weight == 0))

        actual_shrink = original_area - remaining.sum()
        err = abs(actual_shrink - target_shrink)

        if err < best_err:
            best_err = err
            best_beta = beta_mid

        if actual_shrink < target_shrink:
            beta_lo = beta_mid
        else:
            beta_hi = beta_mid

    return best_beta


# ============================================================
# 主接口
# ============================================================

class TumorShrinkTransform:
    """
    肿瘤收缩变换器（治疗后退缩）。

    使用 edge fade mask 解决 patch 边界截断问题:
      - boundary_weight 乘以从边缘到中心的渐变遮罩
      - patch 边缘 EDGE_FADE_MARGIN 像素内收缩量线性衰减到 0
      - 边界处肿瘤保持不变，向内平滑过渡到正常收缩

    用法:
        validator = MaskValidator("prior_db.json")
        transform = TumorShrinkTransform("prior_db.json", validator)
        new_mask, log = transform.apply(mask, target_delta=0.10)
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
        对肿瘤执行一次收缩。

        Args:
            mask: 纯组织 Mask (H, W), 值 0-21
            target_delta: 肿瘤面积减少量 (0.10=减少10%), None随机
        """
        if not np.any(mask == TUMOR_ID):
            return mask, TransformLog(
                "tumor_shrink", TUMOR_ID, "tumor",
                {"error": "no_tumor"}, {}, False, "No tumor in mask"
            )

        current_ratio = (mask == TUMOR_ID).sum() / mask.size

        if target_delta is None:
            target_delta = float(self.rng.uniform(*SHRINK_DELTA_RANGE))

        # 安全: 至少保留 2%
        max_shrink = current_ratio - 0.02
        if target_delta > max_shrink:
            target_delta = max_shrink

        if target_delta < 0.01:
            return mask, TransformLog(
                "tumor_shrink", TUMOR_ID, "tumor",
                {"error": "tumor_too_small"}, {}, False,
                f"Tumor too small to shrink: current={current_ratio:.2%}"
            )

        # 1. 计算 SDF / weight / noise，weight 乘以 edge fade mask
        sdf_raw_weight = compute_boundary_weight(mask, TUMOR_ID, SHRINK_INFLUENCE_RADIUS)
        edge_fade = compute_edge_fade_mask(mask.shape)
        weight = sdf_raw_weight * edge_fade  # 边缘处 weight→0，不收缩

        noise = generate_noise(mask.shape, self.rng)
        alpha = float(self.rng.uniform(*SHRINK_ALPHA_RANGE))

        # 2. 校准 beta
        beta = calibrate_beta_shrink(mask, noise, alpha, weight, target_delta)

        # 3. 收缩
        remaining_tumor, released_region = shrink_tumor(
            mask, noise, alpha, beta, weight)
        
        
        released_region = (mask == TUMOR_ID) & ~remaining_tumor


        # 4. 回填释放区域
        new_mask = backfill_nearest_neighbor(mask, released_region)
        new_mask[remaining_tumor] = TUMOR_ID

        # 5. 碎片清除
        new_mask, fragment_region = remove_small_fragments(
            new_mask, TUMOR_ID, MIN_FRAGMENT_SIZE)

        # 6. 验证
        area_change = self._compute_area_change(mask, new_mask)
        actual_delta = -area_change.get("tumor", 0)
        is_valid, report = self.validator.validate(new_mask, mask)

        n_released = released_region.sum()
        n_fragments = fragment_region.sum()

        self._history.append(is_valid)
        if len(self._history) > 20:
            self._history.pop(0)

        log = TransformLog(
            transform_type="tumor_shrink",
            tissue_id=TUMOR_ID,
            tissue_name="tumor",
            params={
                "target_delta": round(float(target_delta), 3),
                "actual_delta": round(float(actual_delta), 4),
                "alpha": round(alpha, 1),
                "beta": round(float(beta), 2),
                "pixels_released": int(n_released),
                "pixels_fragments_removed": int(n_fragments),
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
        """生成多样化的肿瘤收缩变体。"""
        if not np.any(mask == TUMOR_ID):
            logger.warning("No tumor in mask, skipping.")
            return []

        current_ratio = (mask == TUMOR_ID).sum() / mask.size
        max_shrink = current_ratio - 0.02

        if max_shrink < 0.02:
            logger.warning(f"Tumor too small ({current_ratio:.1%}) to shrink.")
            return []

        deltas = np.linspace(
            SHRINK_DELTA_RANGE[0],
            min(max_shrink, SHRINK_DELTA_RANGE[1]),
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
                logger.info(f"  V{idx+1}/{n_variants} shrink (Δ=-{td:.2f}): failed")

        rate = sum(self._history) / len(self._history) if self._history else 0
        logger.info(
            f"[TumorShrink] {len(accepted)}/{n_variants} variants, "
            f"{total} attempts, rate={rate:.0%}"
        )
        return accepted


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    import sys, os

    print("=" * 50)
    print("TumorShrinkTransform (with edge fade mask)")
    print("=" * 50)

    # 构造测试 mask: 肿瘤贴着 patch 边界 (模拟真实情况)
    mask = np.full((256, 256), STROMA_ID, dtype=np.int16)
    yy, xx = np.ogrid[:256, :256]

    # 肿瘤贴着左边和上边边界
    tumor_region = (yy < 150) & (xx < 160)
    tumor_region &= (((yy - 60)**2 / 80**2 + (xx - 60)**2 / 90**2) < 1.5)
    mask[tumor_region] = TUMOR_ID

    # 坏死 (肿瘤内部)
    necrosis_region = ((yy - 50)**2 / 15**2 + (xx - 50)**2 / 20**2) < 1
    mask[necrosis_region] = NECROSIS_ID

    # fat (右下角)
    mask[200:256, 180:256] = 9

    tissues = {TISSUE_NAME_MAP[t]: f'{(mask==t).sum()/mask.size:.1%}'
               for t in np.unique(mask)}
    print(f"Tissues: {tissues}")

    db_path = sys.argv[1] if len(sys.argv) > 1 else "prior_db.json"
    if not os.path.exists(db_path):
        print(f"[跳过] 未找到 {db_path}")
        sys.exit(0)

    from mask_validator import MaskValidator
    validator = MaskValidator(db_path)
    transform = TumorShrinkTransform(db_path, validator, seed=42)

    print(f"\n--- Single shrink (Δ=0.15) ---")
    new_mask, log = transform.apply(mask, target_delta=0.15)
    print(f"  Accepted: {log.accepted}")
    print(f"  Params: {log.params}")
    print(f"  Area change: {log.area_change}")

    if log.accepted:
        released = (mask == TUMOR_ID) & (new_mask != TUMOR_ID)
        backfill_types = {}
        for tid in np.unique(new_mask[released]):
            name = TISSUE_NAME_MAP.get(int(tid), f"unknown_{tid}")
            count = (new_mask[released] == tid).sum()
            backfill_types[name] = count
        print(f"  Backfill types: {backfill_types}")

        # 检查 patch 边界处
        border_before = (mask[:3, :] == TUMOR_ID).sum() + (mask[:, :3] == TUMOR_ID).sum()
        border_after = (new_mask[:3, :] == TUMOR_ID).sum() + (new_mask[:, :3] == TUMOR_ID).sum()
        print(f"  Border tumor pixels: before={border_before}, after={border_after}")
        print(f"  (should shrink smoothly at borders, no rectangular artifacts)")

    print(f"\n--- Generate 6 shrink variants ---")
    variants = transform.generate_variants(mask, n_variants=6)
    for i, (vm, vl) in enumerate(variants):
        p = vl.params
        print(f"  V{i+1}: Δ=-{p['target_delta']:.2f} actual=-{p['actual_delta']:.4f} "
              f"released={p['pixels_released']} fragments={p['pixels_fragments_removed']}")