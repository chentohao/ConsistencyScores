import matplotlib.pyplot as plt
import numpy as np

# 实验数据
dropout_ratios = [10.0, 30.0, 50.0, 70.0]  # 特征丢弃比例
methods = {
    "LIME": [0.8237, 0.4856, 0.2299, 0.1117],
    "SHAP": [0.7615, 0.3668, 0.2655, 0.2123],
    "CAM": [0.7562, 0.3769, 0.2285, 0.1611],
    "IG": [0.7504, 0.4412, 0.2776, 0.1707]
}

# 设置参数
bar_width = 15  # 柱状图宽度
x = np.arange(len(dropout_ratios)) * (bar_width * 5)  # 各组别x轴位置（避免重叠）

# 绘图
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 6))

# 定义颜色（与折线图保持一致，便于对照）
colors = {'LIME': '#FF6B6B', 'SHAP': '#4ECDC4', 'CAM': '#45B7D1', 'IG': '#FFA07A'}

# 绘制各组柱状图
for i, (method, scores) in enumerate(methods.items()):
    ax.bar(
        x + i * bar_width,  # 错开x轴位置，避免重叠
        scores,
        width=bar_width,
        label=method,
        color=colors[method],
        edgecolor='black'  # 边框线，增强区分度
    )

# 设置x轴刻度和标签（居中对齐每组柱子）
ax.set_xticks(x + bar_width * 1.5)  # 调整刻度位置至每组中间
ax.set_xticklabels([f'{r}%' for r in dropout_ratios], fontsize=10)

# 坐标轴和标题
ax.set_xlabel('Feature Dropout Ratio', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Consistency Score', fontsize=12, fontweight='bold')
ax.set_title('Consistency of Explanation Methods Under Different Dropout Ratios', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.9)  # y轴范围与折线图一致，便于对比
ax.grid(axis='y', linestyle='--', alpha=0.7)  # 仅显示水平网格线
ax.legend(fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig('consistency_barplot.svg', dpi=300, bbox_inches='tight')
plt.show()