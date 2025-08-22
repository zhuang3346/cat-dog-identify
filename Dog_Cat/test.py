# from models.cnn_model import CNNModel
from models.inception_model import InceptionModel

from utils.data_loader import load_datasets
from utils.evaluator import test_model
from config import Config
from torch.utils.data import DataLoader

def main(model_path=None):
    # 加载数据
    _, _, test_data, classes = load_datasets(Config.data_root)
    test_loader = DataLoader(test_data, Config.batch_size)

    # 加载模型
    # model = CNNModel(len(classes)).to(Config.device)
    model = InceptionModel(len(classes)).to(Config.device)

    test_model(model, test_loader, Config.device, model_path)


if __name__ == '__main__':

    main(model_path=Config.model_path)