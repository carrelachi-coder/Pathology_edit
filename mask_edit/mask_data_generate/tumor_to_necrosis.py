"""
Necrosis Replacement v6 — 肿瘤连通域整体替换
=============================================

最简方案:
    1. 找到所有肿瘤连通域
    2. 随机选 1-2 个连通域
    3. 连通域内所有像素 → 坏死
    4. 连通域中间夹着的其他组织（如基质碎片）一并坏死

优势:
    - 形状天然合理（就是真实肿瘤的形状）
    - 不需要种子、噪声、SDF、平滑
    - 没有咬痕、触角、条纹等artifact
    - 代码极简，运行极快

病理意义:
    一块肿瘤区域因缺血整体坏死，
    该区域内夹杂的基质/淋巴等也一起坏死。

用法:
    from mask_validator import MaskValidator
    from necrosis_expansion import NecrosisReplacementTransform

    validator = MaskValidator("prior_db.json")
    transform = NecrosisReplacementTransform("prior_db.json", validator)
    results = transform.generate_variants(mask, n_variants=3)
"""

import numpy as np
import cv2
from scipy import ndimage
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
NECROSIS_FORBIDDEN = {0, 7, 15, 21}

# 连通域选择参数
MIN_COMPONENT_AREA = 200   # 太小的连通域不选（像素数）
MAX_COMPONENTS_PICK = 2    # 最多选几个连通域替换


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
def reclaim_all_trapped_islands(mask: np.ndarray) -> np.ndarray:
    """
    全局拓扑陷落逻辑：
    寻找所有非坏死区域，如果它们无法“到达”图像边缘（即被坏死围死），则强制转为坏死。
    """
    h, w = mask.shape
    new_mask = mask.copy()
    
    # 1. 创建"非坏死"掩码
    not_necrosis = (mask != NECROSIS_ID).astype(np.uint8)
    
    # 2. 识别哪些区域可以连接到外界（图像边缘）
    # 我们用 Flood Fill 从图像的四个边缘尝试向内填充
    # 能够被填充到的像素 = 具有外部血供通道的区域
    # 无法被填充到的像素 = 被坏死完全封锁的陷落区
    
    # 创建一个带 padding 的画布以便于从外部边缘填充
    canvas = np.pad(not_necrosis, 1, mode='constant', constant_values=1)
    flood_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)
    
    # 从左上角(0,0)开始填充，只要能连通，就说明没被围死
    cv2.floodFill(canvas, flood_mask, (0, 0), 2)
    
    # 获取填充结果并去掉 padding
    # 值为 2 的代表与外部连通，值为 1 的代表被坏死包围的孤岛
    trapped_mask = (canvas[1:-1, 1:-1] == 1)
    
    # 3. 过滤掉禁区（如 Outside ROI 即使被围住也不能变紫）
    for fid in NECROSIS_FORBIDDEN:
        trapped_mask &= (mask != fid)
        
    # 4. 执行回收：将陷落区全部转为坏死
    new_mask[trapped_mask] = NECROSIS_ID
    
    return new_mask

def find_replaceable_tumor_components(mask: np.ndarray) -> List[Tuple[np.ndarray, int]]:
    """
    找到所有可以被替换为坏死的肿瘤连通域。

    约束:
        1. 面积 >= MIN_COMPONENT_AREA
        2. 必须和已有坏死区域相邻（膨胀3像素内有接触）

    返回 [(binary_mask, area), ...] 按面积降序。
    """
    tumor_binary = (mask == TUMOR_ID).astype(np.uint8)
    labeled, n = ndimage.label(tumor_binary)

    if not np.any(mask == NECROSIS_ID):
        return []

    # 坏死区域膨胀3像素，作为"邻接检测区"
    necrosis_binary = (mask == NECROSIS_ID).astype(np.uint8)
    necrosis_dilated = cv2.dilate(necrosis_binary, np.ones((3, 3), np.uint8), iterations=3)
    necrosis_zone = necrosis_dilated.astype(bool)

    components = []
    for lbl in range(1, n + 1):
        comp = (labeled == lbl)
        area = comp.sum()

        # 约束1: 面积够大
        if area < MIN_COMPONENT_AREA:
            continue

        # 约束2: 和坏死相邻
        if not np.any(comp & necrosis_zone):
            continue

        components.append((comp, area))

    components.sort(key=lambda x: -x[1])
    return components


def fill_enclosed_holes(region: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    把选中区域内部"包围"的其他组织也一并纳入。

    方法: 对选中区域做凸包或形态学填充，
    把被该区域包围的小洞也填上。

    简化实现: 用 flood fill 从图像边缘出发，
    没被flood到的区域 = 被选中区域完全包围的洞。
    """
    h, w = region.shape

    # 从边缘flood fill: 把所有能从边缘到达的非region像素标记为"外部"
    # 剩下的非region像素 = 被region包围的"内部洞"
    padded = np.pad(~region, 1, mode='constant', constant_values=True).astype(np.uint8)
    # flood fill 从 (0,0) 开始
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 0)
    # padded 中剩余的 True = 被包围的洞
    holes = padded[1:-1, 1:-1].astype(bool)

    # 洞中不能包含禁区
    for fid in NECROSIS_FORBIDDEN:
        holes &= (mask != fid)

    return region | holes


def replace_components_with_necrosis(
    mask: np.ndarray,
    components_to_replace: List[np.ndarray],
) -> np.ndarray:
    """
    把选中的肿瘤连通域（及其内部包围的组织）全部变成坏死。
    """
    new_mask = mask.copy()

    for comp in components_to_replace:
        # 填充内部洞
        filled = fill_enclosed_holes(comp, mask)
        new_mask[filled] = NECROSIS_ID

    return new_mask


# ============================================================
# 主接口
# ============================================================

class NecrosisReplacementTransform:
    """
    肿瘤连通域整体替换为坏死。

    用法:
        validator = MaskValidator("prior_db.json")
        transform = NecrosisReplacementTransform("prior_db.json", validator)
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
        n_pick: Optional[int] = None,
    ) -> Tuple[np.ndarray, TransformLog]:
        """
        随机选 1-2 个与坏死相邻的肿瘤连通域替换为坏死。

        约束:
            1. 原图必须已有坏死
            2. 选中的连通域必须和坏死相邻
            3. 替换后必须保留至少一个肿瘤连通域（不能全变坏死）
            4. 面积合理性由 validator 兜底
        """
        # 约束1: 必须已有坏死
        necrosis_ratio = (mask == NECROSIS_ID).sum() / mask.size
        if necrosis_ratio < 0.01:
            return mask, TransformLog(
                "necrosis_replace", TUMOR_ID, "tumor",
                {"error": "no_necrosis"}, {}, False,
                "No existing necrosis to expand from"
            )

        # 找到可替换的连通域（已过滤：面积够大 + 和坏死相邻）
        components = find_replaceable_tumor_components(mask)

        if len(components) == 0:
            return mask, TransformLog(
                "necrosis_replace", TUMOR_ID, "tumor",
                {"error": "no_adjacent_tumor"}, {}, False,
                "No tumor components adjacent to necrosis"
            )

        # 约束3: 替换后至少保留一个肿瘤连通域
        # 统计所有肿瘤连通域（包括不和坏死相邻的）
        all_tumor = (mask == TUMOR_ID).astype(np.uint8)
        _, total_tumor_components = ndimage.label(all_tumor)

        max_replaceable = max(total_tumor_components - 1, 1)  # 至少保留1个

        if n_pick is None:
            n_pick = int(self.rng.integers(1, min(MAX_COMPONENTS_PICK, len(components)) + 1))

        n_pick = min(n_pick, len(components), max_replaceable)

        if n_pick == 0:
            return mask, TransformLog(
                "necrosis_replace", TUMOR_ID, "tumor",
                {"error": "would_remove_all_tumor"}, {}, False,
                "Cannot replace: would remove all tumor"
            )

        # 随机选择连通域（用面积作权重：大的更可能被选中，但小的也有机会）
        areas = np.array([a for _, a in components], dtype=np.float64)
        probs = areas / areas.sum()
        chosen_idx = self.rng.choice(len(components), size=n_pick, replace=False, p=probs)
        chosen_components = [components[i][0] for i in chosen_idx]
        chosen_areas = [components[i][1] for i in chosen_idx]

        # 替换
        new_mask = replace_components_with_necrosis(mask, chosen_components)
        new_mask = reclaim_all_trapped_islands(new_mask)
        # 验证
        area_change = self._compute_area_change(mask, new_mask)
        actual_delta = area_change.get("necrosis_or_debris", 0)
        is_valid, report = self.validator.validate(new_mask, mask)

        self._history.append(is_valid)
        if len(self._history) > 20:
            self._history.pop(0)

        log = TransformLog(
            transform_type="necrosis_replace",
            tissue_id=TUMOR_ID,
            tissue_name="tumor",
            params={
                "n_components_available": len(components),
                "n_picked": n_pick,
                "picked_areas": [int(a) for a in chosen_areas],
                "actual_necrosis_delta": round(float(actual_delta), 4),
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
        """
        生成多样化的坏死替换变体。
        每次随机选不同的连通域组合。
        """
        components = find_replaceable_tumor_components(mask)
        if len(components) == 0:
            logger.info("[NecrosisReplace] No replaceable tumor components, skipping.")
            return []

        accepted = []
        total = 0
        for idx in range(n_variants):
            found = False
            for _ in range(max_attempts):
                total += 1
                new_mask, log = self.apply(mask)
                if log.accepted:
                    accepted.append((new_mask, log))
                    found = True
                    break
            if not found:
                logger.info(f"  V{idx+1}/{n_variants} necrosis replace: failed")

        rate = sum(self._history) / len(self._history) if self._history else 0
        logger.info(
            f"[NecrosisReplace] {len(accepted)}/{n_variants} variants, "
            f"{total} attempts, rate={rate:.0%}"
        )
        return accepted


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    import sys, os

    print("=" * 50)
    print("NecrosisReplacementTransform v6")
    print("=" * 50)

    # 构造: 多个肿瘤连通域 + 基质背景
    mask = np.full((256, 256), 2, dtype=np.int16)  # stroma
    yy, xx = np.ogrid[:256, :256]
    mask[((yy-80)**2/40**2 + (xx-80)**2/50**2) < 1] = 1   # tumor 1
    mask[((yy-180)**2/30**2 + (xx-180)**2/40**2) < 1] = 1  # tumor 2
    mask[((yy-60)**2/20**2 + (xx-200)**2/25**2) < 1] = 1   # tumor 3
    mask[200:256, 0:60] = 4  # existing necrosis

    print(f"Tissues: { {TISSUE_NAME_MAP[t]: f'{(mask==t).sum()/mask.size:.1%}' for t in np.unique(mask)} }")

    db_path = sys.argv[1] if len(sys.argv) > 1 else "prior_db.json"
    if not os.path.exists(db_path):
        print(f"[跳过] 未找到 {db_path}")
        sys.exit(0)

    from mask_validator import MaskValidator
    validator = MaskValidator(db_path)
    transform = NecrosisReplacementTransform(db_path, validator, seed=42)

    print(f"\nReplaceable tumor components: {len(find_replaceable_tumor_components(mask))}")
    print("\n--- Generate 3 necrosis variants ---")
    variants = transform.generate_variants(mask, n_variants=3)
    for i, (vm, vl) in enumerate(variants):
        p = vl.params
        print(f"  V{i+1}: picked={p['n_picked']}/{p['n_components_available']} "
              f"areas={p['picked_areas']} Δ={p['actual_necrosis_delta']:+.4f}")
        print(f"        {vl.area_change}")