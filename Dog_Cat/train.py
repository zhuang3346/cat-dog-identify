# from models.cnn_model import CNNModel
from models.inception_model import InceptionModel

from utils.data_loader import load_datasets
from utils.trainer import train_model
from config import Config
from torch.utils.data import DataLoader

def main(save_best=False):
    # 初始化
    print(f"Using device: {Config.device}")

    # 加载数据
    train_data, val_data, test_data, classes = load_datasets(
        Config.data_root, Config.train_ratio
    )

    train_loader = DataLoader(train_data, Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, Config.batch_size)

    # 创建模型
    # model = CNNModel(len(classes)).to(Config.device)
    model = InceptionModel(len(classes)).to(Config.device)

    # 训练
    metrics = train_model(
        model, train_loader, val_loader, Config.device,
        num_epochs=Config.num_epochs, lr=Config.lr, save_best=save_best
    )

    print(f"\nTraining completed in {metrics['time']:.1f}s")
    print(f"Best Validation Acc: {metrics['best_acc']:.3f}")
    print(f"Memory Usage: +{metrics['mem_usage']}MB")


if __name__ == '__main__':

    main(save_best=Config.save_best)
