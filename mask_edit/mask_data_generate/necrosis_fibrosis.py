"""
Necrosis Fibrosis Transform — 坏死纤维化
=========================================

病理背景:
    坏死区域的纤维化是经典的组织修复过程。
    肿瘤内坏死灶失去血供后，周围间质中的成纤维细胞从边缘向中心迁入，
    分泌胶原，将坏死组织替换为纤维化基质 (fibrotic stroma)。
    常见于治疗后反应（化疗/放疗后），也可自发于慢性缺血区域。

算法:
    1. 对坏死区域计算 SDF，做负偏移使边界从外向内收缩
    2. 方向约束: 只有与活组织 (stroma, tumor, lymph 等) 接壤的边界
       才有成纤维细胞来源，才参与纤维化。构建 fibrosis_source_weight
       限制收缩方向
    3. Edge fade mask: patch 边缘处不收缩，防止方形截断
    4. 释放区域一律回填为 stroma (id=2) — 纤维化产物
    5. 碎片清除: 小坏死残余岛也回填为 stroma
    6. beta 二分校准: 精确控制面积缩小量

    噪声频谱: 低频主导（成纤维细胞均匀推进，边界较平滑）

用法:
    from mask_validator import MaskValidator
    from necrosis_fibrosis import NecrosisFibrosisTransform

    validator = MaskValidator("prior_db.json")
    transform = NecrosisFibrosisTransform("prior_db.json", validator)
    results = transform.generate_variants(mask, n_variants=4)
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

NECROSIS_ID = 4
STROMA_ID = 2
NON_BIO_TISSUES = {"outside_roi", "exclude", "undetermined", "other"}

# 纤维化源组织: 这些组织有血供、有成纤维细胞来源
# 只有坏死与这些组织接壤的边界才参与纤维化
FIBROSIS_SOURCE_TISSUES = {1, 2, 3, 9, 10, 13, 18}
# tumor, stroma, lymph, fat, plasma, normal_acinus, blood_vessel

# 纤维化噪声频谱: 低频主导（成纤维细胞均匀推进，边界较平滑）
FIBROSIS_NOISE_OCTAVES = [
    (50.0, 0.65),   # 低频: 整体推进方向
    (20.0, 0.25),   # 中频: 局部不均匀
    (8.0,  0.10),   # 高频: 微小起伏
]

FIBROSIS_ALPHA_RANGE = (10, 22)       # 加大噪声幅度，纤维化推进更深
FIBROSIS_DELTA_RANGE = (0.05, 0.35)   # 上限提到35%，允许大面积纤维化
FIBROSIS_INFLUENCE_RADIUS = 60        # 加大影响半径，让坏死中心也能被纤维化到
FIBROSIS_MIN_RETAIN = 0.03            # 只保留3%，几乎可以全部纤维化
MIN_FRAGMENT_SIZE = 150               # 更激进清除碎片，避免残留小坏死岛
# Patch 边缘渐变: 距边缘 margin 像素内收缩量线性衰减到 0
EDGE_FADE_MARGIN = 40

# 方向约束: 纤维化源膨胀半径
# 只有坏死像素在纤维化源组织 FIBROSIS_SOURCE_DILATE_RADIUS 像素范围内，
# 才认为该处有成纤维细胞可以到达
FIBROSIS_SOURCE_DILATE_RADIUS = 8


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
    距离 patch 边缘 0px = 0（不收缩）, margin px = 1（正常收缩）。
    """
    H, W = shape
    fade = np.ones((H, W), dtype=np.float64)

    rows = np.arange(H, dtype=np.float64)
    cols = np.arange(W, dtype=np.float64)

    fade *= np.clip(rows / margin, 0.0, 1.0)[:, np.newaxis]
    fade *= np.clip((H - 1 - rows) / margin, 0.0, 1.0)[:, np.newaxis]
    fade *= np.clip(cols / margin, 0.0, 1.0)[np.newaxis, :]
    fade *= np.clip((W - 1 - cols) / margin, 0.0, 1.0)[np.newaxis, :]

    return fade


def compute_fibrosis_source_weight(
    mask: np.ndarray,
    dilate_radius: int = FIBROSIS_SOURCE_DILATE_RADIUS,
) -> np.ndarray:
    """
    方向约束权重: 只有靠近纤维化源组织的坏死边界才参与纤维化。

    1. 构建纤维化源组织的 binary mask
    2. 计算每个像素到最近纤维化源的距离
    3. 距离 <= dilate_radius 的区域权重=1, 之外线性衰减到 0

    效果: 坏死区域中远离任何活组织的中心部分不会被纤维化，
    只有边缘与活组织接壤的部分才被侵蚀 — 符合成纤维细胞
    从有血供的组织向坏死区迁入的病理过程。
    """
    source_binary = np.zeros(mask.shape, dtype=bool)
    for tid in FIBROSIS_SOURCE_TISSUES:
        source_binary |= (mask == tid)

    if not np.any(source_binary):
        return np.zeros(mask.shape, dtype=np.float64)

    dist_to_source = ndimage.distance_transform_edt(~source_binary)

    # 在 dilate_radius 内权重=1, 2*dilate_radius 外权重=0, 中间线性
    weight = np.clip(1.0 - (dist_to_source - dilate_radius) / dilate_radius, 0.0, 1.0)

    return weight


def generate_noise(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """纤维化噪声: 低频主导, 边界较平滑"""
    noise = np.zeros(shape, dtype=np.float64)
    for sigma, w in FIBROSIS_NOISE_OCTAVES:
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


def shrink_necrosis(
    mask: np.ndarray,
    noise: np.ndarray,
    alpha: float,
    beta: float,
    weight: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SDF 负偏移收缩坏死。
    weight 已经融合了 boundary_weight * edge_fade * fibrosis_source_weight。

    返回: (remaining_necrosis, released_region)
    """
    sdf = compute_sdf(mask, NECROSIS_ID)
    original_necrosis = (mask == NECROSIS_ID)

    sdf_new = sdf - weight * (alpha * noise + beta)
    remaining = (sdf_new > 0) & original_necrosis

    # 影响范围外的坏死保持不变
    remaining = remaining | (original_necrosis & (weight == 0))

    released_region = original_necrosis & ~remaining

    return remaining, released_region


def calibrate_beta_fibrosis(
    mask: np.ndarray,
    noise: np.ndarray,
    alpha: float,
    weight: np.ndarray,
    target_delta: float,
) -> float:
    """二分搜索: 精确控制坏死面积收缩量"""
    total = mask.size
    target_shrink = target_delta * total
    original_area = (mask == NECROSIS_ID).sum()
    sdf = compute_sdf(mask, NECROSIS_ID)
    original_necrosis = (mask == NECROSIS_ID)

    beta_lo, beta_hi = 0.0, 150.0
    best_beta, best_err = 0.0, float('inf')

    for _ in range(30):
        beta_mid = (beta_lo + beta_hi) / 2.0

        sdf_new = sdf - weight * (alpha * noise + beta_mid)
        remaining = (sdf_new > 0) & original_necrosis
        remaining = remaining | (original_necrosis & (weight == 0))

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


def remove_small_fragments(
    mask: np.ndarray,
    tissue_id: int,
    min_size: int,
) -> Tuple[np.ndarray, int]:
    """移除指定组织的小碎片, 回填为 stroma。返回 (new_mask, n_pixels_removed)。"""
    binary = (mask == tissue_id).astype(np.uint8)
    labeled, n = ndimage.label(binary)

    if n <= 1:
        return mask, 0

    areas = ndimage.sum(binary, labeled, range(1, n + 1))
    new_mask = mask.copy()
    n_removed = 0

    for lbl in range(1, n + 1):
        if areas[lbl - 1] < min_size:
            frag = (labeled == lbl)
            new_mask[frag] = STROMA_ID
            n_removed += int(frag.sum())

    return new_mask, n_removed


# ============================================================
# 主接口
# ============================================================

class NecrosisFibrosisTransform:
    """
    坏死纤维化: 坏死区域从边缘向中心被纤维化基质替代。

    特点:
      - 方向约束: 只有与活组织接壤的边界才纤维化
      - Edge fade: patch 边缘处不收缩
      - 回填一律为 stroma (纤维化产物)
      - 至少保留原坏死面积的 FIBROSIS_MIN_RETAIN

    用法:
        validator = MaskValidator("prior_db.json")
        transform = NecrosisFibrosisTransform("prior_db.json", validator)
        results = transform.generate_variants(mask, n_variants=4)
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
        对坏死执行一次纤维化收缩。

        Args:
            mask: 纯组织 Mask (H, W), 值 0-21
            target_delta: 坏死面积减少量 (占全图), None 随机
        """
        necrosis_area = (mask == NECROSIS_ID).sum()
        necrosis_ratio = necrosis_area / mask.size

        if necrosis_ratio < 0.02:
            return mask, TransformLog(
                "necrosis_fibrosis", NECROSIS_ID, "necrosis_or_debris",
                {"error": "no_necrosis"}, {}, False,
                f"Necrosis too small: {necrosis_ratio:.2%}"
            )

        if target_delta is None:
            target_delta = float(self.rng.uniform(*FIBROSIS_DELTA_RANGE))

        # 安全: 至少保留原坏死面积的 FIBROSIS_MIN_RETAIN
        max_shrink = necrosis_ratio * (1.0 - FIBROSIS_MIN_RETAIN)
        if target_delta > max_shrink:
            target_delta = max_shrink

        if target_delta < 0.005:
            return mask, TransformLog(
                "necrosis_fibrosis", NECROSIS_ID, "necrosis_or_debris",
                {"error": "delta_too_small"}, {}, False,
                f"Delta too small after clamping: {target_delta:.4f}"
            )

        # 1. 构建复合权重 = boundary_weight * edge_fade * fibrosis_source_weight
        bw = compute_boundary_weight(mask, NECROSIS_ID, FIBROSIS_INFLUENCE_RADIUS)
        edge_fade = compute_edge_fade_mask(mask.shape)
        source_weight = compute_fibrosis_source_weight(mask)
        weight = bw * edge_fade * source_weight

        # 2. 噪声
        noise = generate_noise(mask.shape, self.rng)
        alpha = float(self.rng.uniform(*FIBROSIS_ALPHA_RANGE))

        # 3. 校准 beta
        beta = calibrate_beta_fibrosis(mask, noise, alpha, weight, target_delta)

        # 4. 收缩
        remaining_necrosis, released_region = shrink_necrosis(
            mask, noise, alpha, beta, weight)

        # 5. 回填: 释放区域一律变 stroma
        new_mask = mask.copy()
        new_mask[released_region] = STROMA_ID
        new_mask[remaining_necrosis] = NECROSIS_ID  # 确保残留坏死不被覆盖

        # 6. 碎片清除
        new_mask, n_fragments = remove_small_fragments(
            new_mask, NECROSIS_ID, MIN_FRAGMENT_SIZE)

        # 7. 验证
        area_change = self._compute_area_change(mask, new_mask)
        actual_delta = -area_change.get("necrosis_or_debris", 0)
        is_valid, report = self.validator.validate(new_mask, mask)

        n_released = int(released_region.sum())

        self._history.append(is_valid)
        if len(self._history) > 20:
            self._history.pop(0)

        log = TransformLog(
            transform_type="necrosis_fibrosis",
            tissue_id=NECROSIS_ID,
            tissue_name="necrosis_or_debris",
            params={
                "target_delta": round(float(target_delta), 3),
                "actual_delta": round(float(actual_delta), 4),
                "alpha": round(alpha, 1),
                "beta": round(float(beta), 2),
                "pixels_released": n_released,
                "pixels_fragments_removed": n_fragments,
            },
            area_change=area_change,
            accepted=is_valid,
            rejection_reason="" if is_valid else report.summary(),
        )

        return (new_mask if is_valid else mask), log

    def generate_variants(
        self,
        mask: np.ndarray,
        n_variants: int = 4,
        max_attempts: int = 10,
    ) -> List[Tuple[np.ndarray, TransformLog]]:
        """生成多样化的坏死纤维化变体, 收缩量从小到大。"""
        necrosis_ratio = (mask == NECROSIS_ID).sum() / mask.size
        if necrosis_ratio < 0.02:
            logger.info("[NecrosisFibrosis] Skipping: necrosis too small.")
            return []

        max_shrink = necrosis_ratio * (1.0 - FIBROSIS_MIN_RETAIN)
        if max_shrink < 0.01:
            logger.info("[NecrosisFibrosis] Skipping: no room to fibrosis.")
            return []

        lo = min(FIBROSIS_DELTA_RANGE[0], max_shrink)
        hi = min(FIBROSIS_DELTA_RANGE[1], max_shrink)
        deltas = np.linspace(lo, hi, n_variants).tolist()

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
                logger.info(f"  V{idx+1}/{n_variants} fibrosis (Δ=-{td:.2f}): failed")

        rate = sum(self._history) / len(self._history) if self._history else 0
        logger.info(
            f"[NecrosisFibrosis] {len(accepted)}/{n_variants} variants, "
            f"{total} attempts, rate={rate:.0%}"
        )
        return accepted


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    import sys, os

    print("=" * 50)
    print("NecrosisFibrosisTransform")
    print("=" * 50)

    # 构造测试 mask: 肿瘤 + 坏死核心 + 基质背景
    mask = np.full((256, 256), STROMA_ID, dtype=np.int16)
    yy, xx = np.ogrid[:256, :256]

    # 肿瘤
    tumor_region = ((yy - 128)**2 / 70**2 + (xx - 128)**2 / 90**2) < 1
    mask[tumor_region] = 1

    # 坏死 (肿瘤内部, 贴着上边界测试 edge fade)
    necrosis_region = ((yy - 60)**2 / 40**2 + (xx - 128)**2 / 50**2) < 1
    necrosis_region &= tumor_region
    mask[necrosis_region] = NECROSIS_ID

    # 额外坏死 (不与活组织相邻, 靠近边角 — 测试方向约束)
    mask[0:30, 0:30] = NECROSIS_ID

    # fat
    mask[220:256, 200:256] = 9

    tissues = {TISSUE_NAME_MAP[t]: f'{(mask==t).sum()/mask.size:.1%}'
               for t in np.unique(mask)}
    print(f"Tissues: {tissues}")

    db_path = sys.argv[1] if len(sys.argv) > 1 else "prior_db.json"
    if not os.path.exists(db_path):
        print(f"[跳过] 未找到 {db_path}")
        sys.exit(0)

    from mask_validator import MaskValidator
    validator = MaskValidator(db_path)
    transform = NecrosisFibrosisTransform(db_path, validator, seed=42)

    print(f"\n--- Single fibrosis (Δ=0.08) ---")
    new_mask, log = transform.apply(mask, target_delta=0.08)
    print(f"  Accepted: {log.accepted}")
    print(f"  Params: {log.params}")
    print(f"  Area change: {log.area_change}")

    if log.accepted:
        # 检查方向约束: 边角处的坏死 (0:30, 0:30) 应该较少被纤维化
        corner_before = (mask[0:30, 0:30] == NECROSIS_ID).sum()
        corner_after = (new_mask[0:30, 0:30] == NECROSIS_ID).sum()
        print(f"  Corner necrosis: before={corner_before}, after={corner_after}")
        print(f"  (corner necrosis borders outside_roi, should fibrosis less)")

        # 检查 patch 边界
        border_before = (mask[:3, :] == NECROSIS_ID).sum() + (mask[:, :3] == NECROSIS_ID).sum()
        border_after = (new_mask[:3, :] == NECROSIS_ID).sum() + (new_mask[:, :3] == NECROSIS_ID).sum()
        print(f"  Border necrosis: before={border_before}, after={border_after}")
        print(f"  (should not shrink at patch borders due to edge fade)")

    print(f"\n--- Generate 4 fibrosis variants ---")
    variants = transform.generate_variants(mask, n_variants=4)
    for i, (vm, vl) in enumerate(variants):
        p = vl.params
        print(f"  V{i+1}: Δ=-{p['target_delta']:.2f} actual=-{p['actual_delta']:.4f} "
              f"released={p['pixels_released']} fragments={p['pixels_fragments_removed']}")
        print(f"        {vl.area_change}")