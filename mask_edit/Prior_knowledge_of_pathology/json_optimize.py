
import json
import os

def normalize_pathology_prior(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        db = json.load(f)

    new_db = {}

    for key, content in db.items():
        # 1. 处理全局元数据 _meta
        if key == "_meta":
            new_db[key] = content
            continue

        # 2. 处理组织层级
        tissue_data = content.copy()
        
        # --- A. 处理 cell_distribution (字符串转数值对象 + 小写化) ---
        if "cell_distribution" in tissue_data:
            old_dist = tissue_data["cell_distribution"]
            new_dist = {}
            for cell_type, val_str in old_dist.items():
                try:
                    # 分割 "mean±std"
                    parts = val_str.split('±')
                    mean_val = float(parts[0])
                    std_val = float(parts[1])
                    
                    # Key 转为全小写 (Neoplastic -> neoplastic)
                    new_dist[cell_type.lower()] = {
                        "mean": mean_val,
                        "std": std_val
                    }
                except (ValueError, IndexError):
                    print(f"警告：无法解析 {key} 中的 {cell_type}: {val_str}")
            
            tissue_data["cell_distribution"] = new_dist

        # --- B. 清理冗余的 cell_density 字符串 ---
        if "cell_density" in tissue_data:
            # 既然已有 cell_density_raw，直接删除这个带±号的字符串字段
            del tissue_data["cell_density"]

        # --- C. 确保 spatial_nnd 的 key 也是全小写 (防患未然) ---
        if "spatial_nnd" in tissue_data:
            old_nnd = tissue_data["spatial_nnd"]
            tissue_data["spatial_nnd"] = {k.lower(): v for k, v in old_nnd.items()}

        new_db[key] = tissue_data

    # 保存规范化后的 JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_db, f, indent=4, ensure_ascii=False)

    print(f"\n[成功] 数据已规范化！")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("-" * 30)
    print("改进点回顾：")
    print("1. cell_distribution 已转为 {'mean': x, 'std': y} 格式")
    print("2. 所有细胞类型 Key 已统一为全小写")
    print("3. 已移除冗余的字符串格式 cell_density")

if __name__ == "__main__":
    # 执行处理
    normalize_pathology_prior("pathology_prior.json", "final_pathology_prior_cleaned.json")
