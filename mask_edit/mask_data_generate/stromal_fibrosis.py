"""
Stromal Fibrosis Transform — 间质纤维化
========================================

病理背景:
    间质纤维化是乳腺癌中常见的基质重塑过程 (desmoplastic reaction)。
    肿瘤微环境中的癌相关成纤维细胞 (CAF) 被激活后大量分泌胶原纤维，
    导致间质致密化，逐渐替代脂肪组织、正常腺泡/导管、淋巴管等疏松结构。

算法:
    和 tumor_dilate / lymph_dilate 共用 SDF + 噪声框架。
    1. SDF 正偏移膨胀 stroma 边界
    2. 只能吃掉疏松组织 (fat, normal_acinus, lymphatics, blood_vessel 等)
    3. 不能吃 tumor, necrosis, lymphocytic_infiltrate
    4. 噪声频谱: 低频强主导 (纤维沉积弥漫均匀，边界很平滑)
    5. beta 二分校准: 精确控制面积增量
    6. 拓扑清理: 小碎片回填 stroma

    不需要 edge fade mask (膨胀操作不会在 patch 边界产生方形截断)。
    不需要方向约束 (stroma 本身就是纤维化的执行者，每条边界都可向外扩展)。

用法:
    from mask_validator import MaskValidator
    from stromal_fibrosis import StromalFibrosisTransform

    validator = MaskValidator("prior_db.json")
    transform = StromalFibrosisTransform("prior_db.json", validator)
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

STROMA_ID = 2
NON_BIO_TISSUES = {"outside_roi", "exclude", "undetermined", "other"}

# 间质纤维化可以吃掉的组织 (疏松、缺乏抵抗力的结构)
# fat, plasma_cells, mucoid_material, normal_acinus_or_duct, lymphatics, blood_vessel
STROMA_EXPANSION_TARGETS = [9, 10, 12, 13, 14, 18]

# 间质纤维化噪声频谱: 低频强主导 (胶原沉积弥漫均匀，前沿很平滑)
STROMA_NOISE_OCTAVES = [
    (55.0, 0.70),   # 低频: 整体推进方向
    (22.0, 0.22),   # 中频: 局部密度差异
    (8.0,  0.08),   # 高频: 微小起伏
]

STROMA_ALPHA_RANGE = (5, 12)        # 噪声幅度(px), 偏小 → 边界很平滑
STROMA_DELTA_RANGE = (0.03, 0.25)   # 面积增量 (占全图)
STROMA_INFLUENCE_RADIUS = 50        # 弥漫性纤维化，影响半径大
MIN_FRAGMENT_SIZE = 100              # 小于此的被吃组织碎片回填 stroma


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
    """间质纤维化噪声: 低频强主导, 边界非常平滑"""
    noise = np.zeros(shape, dtype=np.float64)
    for sigma, w in STROMA_NOISE_OCTAVES:
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


def dilate_stroma(
    mask: np.ndarray,
    noise: np.ndarray,
    alpha: float,
    beta: float,
    weight: np.ndarray,
) -> np.ndarray:
    """
    SDF 正偏移膨胀 stroma。只吃 STROMA_EXPANSION_TARGETS 中的组织。
    """
    sdf = compute_sdf(mask, STROMA_ID)
    sdf_new = sdf + weight * (alpha * noise + beta)
    new_region = sdf_new > 0

    original_stroma = (mask == STROMA_ID)

    # 保留影响范围外的小碎片
    new_region = new_region | (original_stroma & (weight == 0))
    # 影响范围外不创造新区域
    new_region = new_region & ~((weight == 0) & ~original_stroma)

    # 只能吃掉允许的组织
    allowed = original_stroma.copy()
    for tid in STROMA_EXPANSION_TARGETS:
        allowed |= (mask == tid)
    new_region = (new_region & allowed) | (original_stroma & new_region)

    new_mask = mask.copy()
    new_mask[new_region & ~original_stroma] = STROMA_ID
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
    original_area = (mask == STROMA_ID).sum()
    sdf = compute_sdf(mask, STROMA_ID)
    original_stroma = (mask == STROMA_ID)

    beta_lo, beta_hi = 0.0, 80.0
    best_beta, best_err = 0.0, float('inf')

    for _ in range(30):
        beta_mid = (beta_lo + beta_hi) / 2.0
        sdf_shifted = sdf + weight * (alpha * noise + beta_mid)
        new_region = sdf_shifted > 0

        new_region = new_region | (original_stroma & (weight == 0))
        new_region = new_region & ~((weight == 0) & ~original_stroma)

        allowed = original_stroma.copy()
        for tid in STROMA_EXPANSION_TARGETS:
            allowed |= (mask == tid)
        new_region = (new_region & allowed) | (original_stroma & new_region)

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


def clean_topology(mask: np.ndarray, min_size: int = MIN_FRAGMENT_SIZE) -> np.ndarray:
    """移除所有组织的小碎片孤岛, 回填为最近邻组织"""
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
                result[frag] = STROMA_ID
    return result


# ============================================================
# 主接口
# ============================================================

class StromalFibrosisTransform:
    """
    间质纤维化: stroma 膨胀吃掉 fat / 正常腺泡 / 淋巴管等疏松组织。

    特点:
      - 只吃疏松组织, 不吃 tumor / necrosis / lymphocytic_infiltrate
      - 噪声低频主导, 边界非常平滑 (胶原沉积弥漫均匀)
      - 不需要 edge fade (膨胀不会产生方形截断)
      - 不需要方向约束 (stroma 本身就是纤维化执行者)

    用法:
        validator = MaskValidator("prior_db.json")
        transform = StromalFibrosisTransform("prior_db.json", validator)
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
        对 stroma 执行一次纤维化膨胀。

        Args:
            mask: 纯组织 Mask (H, W), 值 0-21
            target_delta: stroma 面积增加量 (占全图), None 随机
        """
        stroma_ratio = (mask == STROMA_ID).sum() / mask.size

        if stroma_ratio < 0.05:
            return mask, TransformLog(
                "stromal_fibrosis", STROMA_ID, "stroma",
                {"error": "no_stroma"}, {}, False,
                f"Stroma too small: {stroma_ratio:.2%}"
            )

        # 可扩展空间
        expandable = sum(
            (mask == tid).sum() for tid in STROMA_EXPANSION_TARGETS
        ) / mask.size

        if expandable < 0.02:
            return mask, TransformLog(
                "stromal_fibrosis", STROMA_ID, "stroma",
                {"error": "no_room"}, {}, False,
                f"Not enough expandable tissue: {expandable:.2%}"
            )

        if target_delta is None:
            max_delta = min(STROMA_DELTA_RANGE[1], expandable * 0.7)
            lo = min(STROMA_DELTA_RANGE[0], max_delta)
            target_delta = float(self.rng.uniform(lo, max_delta))

        # 安全: stroma 扩展后不超 95%
        max_allowed = 0.95 - stroma_ratio
        if target_delta > max_allowed:
            target_delta = max_allowed

        # 不超过可扩展空间的 80%
        if target_delta > expandable * 0.8:
            target_delta = expandable * 0.8

        if target_delta < 0.01:
            return mask, TransformLog(
                "stromal_fibrosis", STROMA_ID, "stroma",
                {"error": "delta_too_small"}, {}, False,
                f"Delta too small after clamping: {target_delta:.4f}"
            )

        # 噪声 + 权重
        noise = generate_noise(mask.shape, self.rng)
        alpha = float(self.rng.uniform(*STROMA_ALPHA_RANGE))
        weight = compute_boundary_weight(mask, STROMA_ID, STROMA_INFLUENCE_RADIUS)

        # 校准 beta
        beta = calibrate_beta(mask, noise, alpha, weight, target_delta)

        # 膨胀
        new_mask = dilate_stroma(mask, noise, alpha, beta, weight)

        # 拓扑清理
        new_mask = clean_topology(new_mask, min_size=MIN_FRAGMENT_SIZE)

        # 验证
        area_change = self._compute_area_change(mask, new_mask)
        actual_delta = area_change.get("stroma", 0)
        is_valid, report = self.validator.validate(new_mask, mask)

        self._history.append(is_valid)
        if len(self._history) > 20:
            self._history.pop(0)

        log = TransformLog(
            transform_type="stromal_fibrosis",
            tissue_id=STROMA_ID,
            tissue_name="stroma",
            params={
                "target_delta": round(float(target_delta), 3),
                "actual_delta": round(float(actual_delta), 4),
                "alpha": round(alpha, 1),
                "beta": round(float(beta), 2),
                "expandable_ratio": round(float(expandable), 3),
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
        """生成多样化的间质纤维化变体, 面积增量从小到大。"""
        stroma_ratio = (mask == STROMA_ID).sum() / mask.size
        if stroma_ratio < 0.05:
            logger.info("[StromalFibrosis] Skipping: stroma too small.")
            return []

        expandable = sum(
            (mask == tid).sum() for tid in STROMA_EXPANSION_TARGETS
        ) / mask.size

        if expandable < 0.02:
            logger.info("[StromalFibrosis] Skipping: no expandable tissue.")
            return []

        max_delta = min(STROMA_DELTA_RANGE[1], expandable * 0.7)
        lo = min(STROMA_DELTA_RANGE[0], max_delta)
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
                logger.info(f"  V{idx+1}/{n_variants} stromal_fibrosis (Δ=+{td:.2f}): failed")

        rate = sum(self._history) / len(self._history) if self._history else 0
        logger.info(
            f"[StromalFibrosis] {len(accepted)}/{n_variants} variants, "
            f"{total} attempts, rate={rate:.0%}"
        )
        return accepted


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    import sys, os

    print("=" * 50)
    print("StromalFibrosisTransform")
    print("=" * 50)

    # 构造测试 mask: stroma 背景 + 各种组织
    mask = np.full((256, 256), STROMA_ID, dtype=np.int16)
    yy, xx = np.ogrid[:256, :256]

    # 肿瘤 (不可被吃)
    tumor_region = ((yy - 128)**2 / 50**2 + (xx - 128)**2 / 60**2) < 1
    mask[tumor_region] = 1

    # 脂肪 (主要目标)
    mask[0:60, 180:256] = 9
    mask[200:256, 0:80] = 9

    # 正常腺泡
    acinus_region = ((yy - 40)**2 / 20**2 + (xx - 40)**2 / 25**2) < 1
    mask[acinus_region] = 13

    # 淋巴浸润 (不可被吃)
    lymph_region = ((yy - 180)**2 / 25**2 + (xx - 200)**2 / 30**2) < 1
    mask[lymph_region] = 3

    # 血管
    mask[100:110, 200:240] = 18

    tissues = {TISSUE_NAME_MAP[t]: f'{(mask==t).sum()/mask.size:.1%}'
               for t in np.unique(mask)}
    print(f"Tissues: {tissues}")

    db_path = sys.argv[1] if len(sys.argv) > 1 else "prior_db.json"
    if not os.path.exists(db_path):
        print(f"[跳过] 未找到 {db_path}")
        sys.exit(0)

    from mask_validator import MaskValidator
    validator = MaskValidator(db_path)
    transform = StromalFibrosisTransform(db_path, validator, seed=42)

    print(f"\n--- Single stromal fibrosis (Δ=0.08) ---")
    new_mask, log = transform.apply(mask, target_delta=0.08)
    print(f"  Accepted: {log.accepted}")
    print(f"  Params: {log.params}")
    print(f"  Area change: {log.area_change}")

    if log.accepted:
        # 检查 tumor 和 lymph 没被吃
        tumor_before = (mask == 1).sum()
        tumor_after = (new_mask == 1).sum()
        lymph_before = (mask == 3).sum()
        lymph_after = (new_mask == 3).sum()
        print(f"  Tumor pixels: before={tumor_before}, after={tumor_after} (should be equal)")
        print(f"  Lymph pixels: before={lymph_before}, after={lymph_after} (should be equal)")

        # 检查 fat 被吃了多少
        fat_before = (mask == 9).sum()
        fat_after = (new_mask == 9).sum()
        print(f"  Fat pixels: before={fat_before}, after={fat_after} (should decrease)")

    print(f"\n--- Generate 4 stromal fibrosis variants ---")
    variants = transform.generate_variants(mask, n_variants=4)
    for i, (vm, vl) in enumerate(variants):
        p = vl.params
        print(f"  V{i+1}: Δ_target=+{p['target_delta']:.2f} "
              f"Δ_actual=+{p['actual_delta']:.4f} "
              f"α={p['alpha']:.0f} β={p['beta']:+.1f} "
              f"expandable={p['expandable_ratio']:.3f}")
        print(f"        {vl.area_change}")