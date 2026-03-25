import json
import os

def merge_json_files(output_name="pathology_prior.json"):
    # 1. 定义文件名
    files = {
        "distribution": "distribution_prior_db.json",
        "spatial": "spatial_prior_knowledge_per_type.json",
        "adjacency": "tissue_adjacency_prior.json",
        "area": "tissue_area_prior.json" # 如果还没跑完，脚本会跳过它
    }

    # 加载已存在的 JSON
    data_sources = {}
    for key, filename in files.items():
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                data_sources[key] = json.load(f)
            print(f"已加载: {filename}")
        else:
            print(f"跳过: {filename} (文件不存在)")

    # 2. 获取所有出现过的组织名称（作为一级 Key）
    all_tissues = set()
    for src in data_sources.values():
        all_tissues.update([k for k in src.keys() if not k.startswith('_')])

    master_db = {}
    meta_info = {
        "cooccurrence_correlation": {},
        "pathology_rules": {}
    }

    # 3. 核心合并逻辑
    for tissue in all_tissues:
        master_db[tissue] = {}

        # A. 整合面积信息 (来自 tissue_area_prior.json)
        if "area" in data_sources and tissue in data_sources["area"]:
            master_db[tissue]["area"] = data_sources["area"][tissue]

        # B. 整合邻接信息 (来自 tissue_adjacency_prior.json)
        if "adjacency" in data_sources and tissue in data_sources["adjacency"]:
            master_db[tissue]["adjacency"] = data_sources["adjacency"][tissue]

        # C. 整合细胞分布和总体密度 (来自 pathology_prior_db.json)
        if "distribution" in data_sources and tissue in data_sources["distribution"]:
            dist_src = data_sources["distribution"][tissue]
            
            # 处理分布：转为人类可读字符串 (或保留原始列表，建议保留原始数据)
            master_db[tissue]["cell_distribution"] = {
                c_name: f"{val[0]:.3f}±{val[1]:.3f}" 
                for c_name, val in dist_src.get("cell_type_dist", {}).items()
            }
            
            # 处理密度
            d = dist_src.get("density", {})
            master_db[tissue]["cell_density"] = f"{d.get('mean', 0):.6f}±{d.get('std', 0):.6f}"
            # 同时保留一份原始数值供计算用
            master_db[tissue]["cell_density_raw"] = d

        # D. 整合空间排列 NND (来自 spatial_prior_knowledge_per_type.json)
        if "spatial" in data_sources and tissue in data_sources["spatial"]:
            master_db[tissue]["spatial_nnd"] = data_sources["spatial"][tissue]

    # 4. 提取全局 Meta 信息 (通常存储在 area 文件中)
    if "area" in data_sources:
        area_src = data_sources["area"]
        if "_tissue_cooccurrence" in area_src:
            meta_info["cooccurrence_correlation"] = area_src["_tissue_cooccurrence"]
        if "_pathology_rules" in area_src:
            meta_info["pathology_rules"] = area_src["_pathology_rules"]

    master_db["_meta"] = meta_info

    # 5. 保存最终文件
    with open(output_name, 'w', encoding='utf-8') as f:
        json.dump(master_db, f, indent=4, ensure_ascii=False)

    print(f"\n[成功] 所有先验已整合至: {output_name}")
    print(f"整合后的组织数量: {len(all_tissues)}")

if __name__ == "__main__":
    merge_json_files()