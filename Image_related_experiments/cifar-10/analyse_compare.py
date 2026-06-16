import matplotlib.pyplot as plt
import numpy as np

# 实验数据
methods = ["LIME", "SHAP", "Grad-CAM", "IG"]  # 方法名称
drop_important = [0.2299, 0.2655, 0.2285, 0.2776]  # 丢弃最重要50%特征的分数
drop_unimportant = [0.4239, 0.6853, 0.7441, 0.3180]  # 丢弃最不重要50%特征的分数

# 设置参数
bar_width = 0.35  # 柱状图宽度
x = np.arange(len(methods))  # 方法的x轴位置

# 绘图
plt.style.use('default')
fig, ax = plt.subplots(figsize=(9, 6))

# 绘制两组柱状图（区分颜色和标签）
ax.bar(
    x - bar_width/2,  # 第一组柱子位置（左移）
    drop_important,
    width=bar_width,
    label='Drop 50% Most Important Features',
    color='#FF6B6B',  # 红色系：表示丢弃重要特征
    edgecolor='black'
)
ax.bar(
    x + bar_width/2,  # 第二组柱子位置（右移）
    drop_unimportant,
    width=bar_width,
    label='Drop 50% Least Important Features',
    color='#4ECDC4',  # 蓝色系：表示丢弃不重要特征
    edgecolor='black'
)

# 设置x轴刻度和标签
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)

# 坐标轴和标题
ax.set_xlabel('Interpreters', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Consistency Score', fontsize=12, fontweight='bold')
ax.set_title('Consistency Scores Under Different 50% Feature Dropout Strategies', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.8)  # 适当扩大y轴范围，避免顶部拥挤
ax.grid(axis='y', linestyle='--', alpha=0.7)  # 水平网格线辅助读值
ax.legend(fontsize=10, loc='upper left')

# 在柱子顶部标注具体数值（保留4位小数）
for i in range(len(methods)):
    ax.text(x[i] - bar_width/2, drop_important[i] + 0.02, f'{drop_important[i]:.4f}', 
            ha='center', fontsize=9)
    ax.text(x[i] + bar_width/2, drop_unimportant[i] + 0.02, f'{drop_unimportant[i]:.4f}', 
            ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('consistency_drop_strategies.svg', dpi=300, bbox_inches='tight')
plt.show()