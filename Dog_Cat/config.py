import torch

class Config:
    data_root = "data"
    batch_size = 32 #批量大小
    num_epochs = 100 #训练轮数
    lr = 0.001
    train_ratio = 0.8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_best = True
    model_path = 'best_model.pth'
    image_path = "test.jpg"
    num_classes = 2