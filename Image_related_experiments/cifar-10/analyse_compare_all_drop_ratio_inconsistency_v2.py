import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- 1. 数据整理 --------------------------
dropout_ratios = [10.0, 30.0, 50.0, 70.0]
# 丢弃最重要特征
most_important_data = {
    "Dropout Ratio": dropout_ratios,
    "LIME": [0.1763, 0.5144, 0.7701, 0.8883],
    "SHAP": [0.2385, 0.6332, 0.7345, 0.7877],
    "Grad-CAM": [0.2438, 0.6231, 0.7715, 0.8389],
    "IG": [0.2496, 0.5588, 0.7224, 0.8293],
    "Dropout Type": ["Most Important"] * 4
}
# 丢弃最不重要特征
least_important_data = {
    "Dropout Ratio": dropout_ratios,
    "LIME": [0.1115, 0.2977, 0.5761, 0.8120],
    "SHAP": [0.0447, 0.1340, 0.3147, 0.5382],
    "Grad-CAM": [0.0324, 0.0957, 0.2559, 0.5734],
    "IG": [0.1636, 0.4252, 0.6820, 0.8073],
    "Dropout Type": ["Least Important"] * 4
}

# 转换为长格式DataFrame
df_most = pd.DataFrame(most_important_data).melt(
    id_vars=["Dropout Ratio", "Dropout Type"],
    var_name="Method",
    value_name="Inconsistency Score"
)
df_least = pd.DataFrame(least_important_data).melt(
    id_vars=["Dropout Ratio", "Dropout Type"],
    var_name="Method",
    value_name="Inconsistency Score"
)
df = pd.concat([df_most, df_least], ignore_index=True)

# -------------------------- 2. 绘图配置 --------------------------
plt.style.use('seaborn-v0_8-ticks')
#sns.set_theme(style='whitegrid')
plt.rcParams["font.family"] = ["SimSun", "Times New Roman"]
#plt.rcParams['font.size'] = 14
#plt.rcParams['axes.labelsize'] = 14
#plt.rcParams['legend.fontsize'] = 14
#plt.rcParams['xtick.labelsize'] = 14
#plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.unicode_minus'] = False

# 核心修改：拆分两种线型定义（适配不同API）
colors = {
    "LIME": "#FF6B6B",       # 红色
    "SHAP": "#4ECDC4",        # 蓝色
    "Grad-CAM": "#45B7D1",    # 绿色
    "IG": "#FFA07A"           # 橙色
}
markers = {'LIME': 'o', 'SHAP': 's', 'Grad-CAM': '^', 'IG': 'D'}
dropout_types = ["Most Important", "Least Important"]
linestyles = {"Most Important": "--", "Least Important": "-"}

# -------------------------- 3. 绘制图表 --------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for method in colors.keys():
    for dt in ["Most Important", "Least Important"]:
        d = df[(df.Method == method) & (df["Dropout Type"] == dt)]
        ax.plot(d["Dropout Ratio"], d["Inconsistency Score"],
                color=colors[method], marker=markers[method], linestyle=linestyles[dt],
                linewidth=2, markersize=8, label=method)

# -------------------------- 4. 手动创建图例 --------------------------
# 图例1：解释方法
method_handles = [
    plt.Line2D([], [], color=colors[m], marker=markers[m], linestyle="-",
               linewidth=2, markersize=8, label=m)
    for m in colors.keys()
]
# 图例2：丢弃类型（仅区分线型+标记，用matplotlib兼容的字符串）
type_handles = [
    plt.Line2D([], [], color="black", linestyle="--", linewidth=2, markersize=8, label="丢弃推测信息性最强特征"),
    plt.Line2D([], [], color="black", linestyle="-", linewidth=2, markersize=8, label="丢弃推测信息性最弱特征")
]

# 添加分组合图例
legend1 = ax.legend(handles=method_handles, title="解释器", loc="lower right", fontsize=12, title_fontsize=12, frameon=True)
ax.add_artist(legend1)
ax.legend(handles=type_handles, title="丢弃策略", loc="upper left", fontsize=12, title_fontsize=12, frameon=True)

ax.set_xlabel('特征丢弃率（%）', fontsize=14, fontweight='bold')
ax.set_ylabel('不一致性得分', fontsize=14, fontweight='bold')
ax.set_xlim(5, 75)
ax.set_ylim(0, 1.0)
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# -------------------------- 6. 保存和显示 --------------------------
plt.savefig("ablation_study_inconsistency_v2.svg", dpi=600, bbox_inches='tight')
plt.show()