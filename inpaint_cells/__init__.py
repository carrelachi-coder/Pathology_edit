# inpaint_cells — 细胞核填充模块
#
# 功能：对 tissue mask 编辑后的区域，预测细胞核概率分布并从实例库采样填充
#
# 子模块：
#   models/         模型定义 (ProbUNet)
#   data/           数据集 + 训练数据准备
#   losses/         Loss 函数 (FocalDiceLoss)
#   nuclei_library/ 细胞实例库 (建库 + 库类)
#   utils/          共享工具函数
#
# 入口脚本：
#   train.py        ProbNet 训练
#   generate.py     细胞核填充推理
