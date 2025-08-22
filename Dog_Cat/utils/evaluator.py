import torch
from .trainer import evaluate

def test_model(model, test_loader, device, model_path=None):
    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))  # 显式启用安全模式
    model.eval()
    acc = evaluate(model, test_loader, device)
    print(f'Test Accuracy: {acc:.4f}')
    return acc
