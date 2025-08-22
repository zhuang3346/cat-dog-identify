import matplotlib.pyplot as plt
import numpy as np

def plot(loss_history: np.ndarray, acc_history: np.ndarray, epoch):

    epochs = np.arange(1, epoch + 1)  # 创建一个从1到epoch的数组

    # 创建画布和子图（1行2列）
    plt.figure(figsize=(12, 5))  # 画布尺寸（宽12英寸，高5英寸）

    # ---------------------------
    # 子图1：Loss曲线
    # ---------------------------
    plt.subplot(1, 2, 1)  # (行数, 列数, 子图索引)
    plt.plot(epochs, loss_history, 'r-', label='Training Loss', linewidth=2)
    plt.title('Training Loss Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
    plt.legend(fontsize=10)

    # ---------------------------
    # 子图2：Accuracy曲线
    # ---------------------------
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_history, 'b-', label='Accuracy', linewidth=2)
    plt.title('Validation Accuracy Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)

    # 调整子图间距
    plt.tight_layout(pad=3)  # 防止标题重叠

    # 保存图片（可选）
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')

    # 显示图像
    plt.show()
