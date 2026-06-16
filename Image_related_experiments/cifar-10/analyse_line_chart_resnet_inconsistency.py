import matplotlib.pyplot as plt
import numpy as np

# 实验数据
dropout_ratios = [10.0, 30.0, 50.0, 70.0]  # 特征丢弃比例
methods = {
    "LIME": [0.1763, 0.5144, 0.7701, 0.8883],
    "SHAP": [0.2385, 0.6332, 0.7345, 0.7877],
    "Grad-CAM": [0.2438, 0.6231, 0.7715, 0.8389],
    "IG": [0.2496, 0.5588, 0.7224, 0.8293]
}

# 设置绘图风格
plt.style.use('seaborn-v0_8-ticks')
plt.rcParams["font.family"] = ["SimSun", "Times New Roman"]
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(8, 5))

# 定义颜色和标记
colors = {'LIME': '#FF6B6B', 'SHAP': '#4ECDC4', 'Grad-CAM': '#45B7D1', 'IG': '#FFA07A'}
markers = {'LIME': 'o', 'SHAP': 's', 'Grad-CAM': '^', 'IG': 'D'}

# 绘制折线图
for method, scores in methods.items():
    ax.plot(
        dropout_ratios, 
        scores, 
        label=method, 
        color=colors[method], 
        marker=markers[method], 
        markersize=8, 
        linewidth=2
    )

# 设置坐标轴标签和标题
ax.set_xlabel('特征丢弃率（%）', fontsize=14, fontweight='bold')
ax.set_ylabel('不一致性得分', fontsize=14, fontweight='bold')

# 设置坐标轴范围
ax.set_xlim(5, 75)
ax.set_ylim(0, 1.0)

# 添加网格线（辅助阅读数据）
ax.grid(True, linestyle='--', alpha=0.7)

# 添加图例
ax.legend(fontsize=12, title="解释器", loc='lower right', title_fontsize=12, frameon=True)

# 优化布局
plt.tight_layout()

# 保存图像
plt.savefig('inconsistency_under_dropout_resnet_v2.svg', dpi=600, bbox_inches='tight')
plt.show()