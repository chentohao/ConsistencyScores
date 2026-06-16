import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- 1. 数据整理 --------------------------
dropout_ratios = [10.0, 30.0, 50.0, 70.0]
# 丢弃最重要特征
most_important_data = {
    "Dropout Ratio": dropout_ratios,
    "LIME": [0.8237, 0.4856, 0.2299, 0.1117],
    "SHAP": [0.7615, 0.3668, 0.2655, 0.2123],
    "Grad-CAM": [0.7562, 0.3769, 0.2285, 0.1611],
    "IG": [0.7504, 0.4412, 0.2776, 0.1707],
    "Dropout Type": ["Most Important"] * 4
}
# 丢弃最不重要特征
least_important_data = {
    "Dropout Ratio": dropout_ratios,
    "LIME": [0.8885, 0.7023, 0.4239, 0.1880],
    "SHAP": [0.9553, 0.8660, 0.6853, 0.4618],
    "Grad-CAM": [0.9676, 0.9043, 0.7441, 0.4266],
    "IG": [0.8364, 0.5748, 0.3180, 0.1927],
    "Dropout Type": ["Least Important"] * 4
}

# 转换为长格式DataFrame
df_most = pd.DataFrame(most_important_data).melt(
    id_vars=["Dropout Ratio", "Dropout Type"],
    var_name="Method",
    value_name="Consistency Score"
)
df_least = pd.DataFrame(least_important_data).melt(
    id_vars=["Dropout Ratio", "Dropout Type"],
    var_name="Method",
    value_name="Consistency Score"
)
df = pd.concat([df_most, df_least], ignore_index=True)

# -------------------------- 2. 绘图配置 --------------------------
plt.style.use('ggplot')
sns.set_theme(style='whitegrid')
plt.rcParams["font.family"] = ["SimSun", "Times New Roman"]
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 核心修改：拆分两种线型定义（适配不同API）
color_palette = {
    "LIME": "#E74C3C",       # 红色
    "SHAP": "#3498DB",        # 蓝色
    "Grad-CAM": "#2ECC71",    # 绿色
    "IG": "#F39C12"           # 橙色
}
dropout_types = ["Most Important", "Least Important"]
# 1. seaborn用的dashes列表
seaborn_dashes = {
    "Most Important": [5, 2],   # 虚线：丢弃最重要特征
    "Least Important": []       # 实线：丢弃最不重要特征
}
# 2. matplotlib Line2D用的线型字符串
matplotlib_linestyles = {
    "Most Important": "--",     # 虚线（对应[5,2]）
    "Least Important": "-"      # 实线（对应[]）
}
marker_styles = {
    "Most Important": "o",    # 圆形标记
    "Least Important": "s"    # 方形标记
}

# -------------------------- 3. 绘制图表（关闭自动图例） --------------------------
fig, ax = plt.subplots(figsize=(10, 6))

sns.lineplot(
    data=df,
    x="Dropout Ratio",
    y="Consistency Score",
    hue="Method",
    style="Dropout Type",
    ax=ax,
    palette=color_palette,
    dashes=[seaborn_dashes[t] for t in dropout_types],  # 用seaborn专属的列表
    markers=[marker_styles[t] for t in dropout_types],
    markersize=8,
    linewidth=2.5,
    markeredgecolor="black",
    markeredgewidth=0.8,
    legend=False  # 关闭自动图例，避免混杂
)

# -------------------------- 4. 手动创建图例 --------------------------
# 图例1：解释方法（仅区分颜色）
method_legend_elements = [
    plt.Line2D([0], [0], color=color_palette["LIME"], lw=2.5, label="LIME"),
    plt.Line2D([0], [0], color=color_palette["SHAP"], lw=2.5, label="SHAP"),
    plt.Line2D([0], [0], color=color_palette["Grad-CAM"], lw=2.5, label="Grad-CAM"),
    plt.Line2D([0], [0], color=color_palette["IG"], lw=2.5, label="IG")
]
# 图例2：丢弃类型（仅区分线型+标记，用matplotlib兼容的字符串）
type_legend_elements = [
    plt.Line2D([0], [0], color="gray", 
               linestyle=matplotlib_linestyles["Most Important"],  # 用字符串--
               marker=marker_styles["Most Important"], markersize=8,
               markeredgecolor="black", markeredgewidth=0.8, lw=2.5, label="丢弃推测信息性最强的特征"),
    plt.Line2D([0], [0], color="gray", 
               linestyle=matplotlib_linestyles["Least Important"], # 用字符串-
               marker=marker_styles["Least Important"], markersize=8,
               markeredgecolor="black", markeredgewidth=0.8, lw=2.5, label="丢弃推测信息性最弱的特征")
]

# 添加分组合图例
legend1 = ax.legend(handles=method_legend_elements, title="解释器",
                    loc="upper right", bbox_to_anchor=(1.0, 0.95), frameon=True)
ax.add_artist(legend1)
legend2 = ax.legend(handles=type_legend_elements, title="丢弃策略",
                    loc="lower left", bbox_to_anchor=(0.0, 0.05), frameon=True)

ax.set_xlabel("特征丢弃率（%）", fontweight='bold')
ax.set_ylabel("一致性得分", fontweight='bold')
#ax.set_title("Consistency Scores Under Different Feature Dropout Strategies", 
#             fontsize=14, fontweight='bold', pad=20)

ax.set_xlim(5, 75)
ax.set_ylim(0, 1.05)

# 优化网格和边框
ax.grid(alpha=0.3, linestyle="-", linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 调整布局
plt.subplots_adjust(right=0.85, bottom=0.15)

# -------------------------- 6. 保存和显示 --------------------------
plt.savefig("ablation_study_consistency_v2.svg", dpi=600, bbox_inches='tight')
plt.show()