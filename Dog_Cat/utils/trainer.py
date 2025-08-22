import torch
import time
import psutil
import numpy as np

def train_model(model, train_loader, val_loader, device, num_epochs=20, lr=0.001, save_best=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    process = psutil.Process()
    start_time = time.time()
    start_mem = process.memory_info().rss // 1024 // 1024

    max_batches = num_epochs  # 预分配空间（根据训练总批次数调整）
    loss_history = np.zeros(max_batches, dtype=np.float32)
    acc_history = np.zeros(max_batches, dtype=np.float32)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # loss = criterion(outputs, labels)

            if isinstance(outputs, tuple):  # 检查是否是元组（训练模式）
                main_output, aux1, aux2 = outputs
                loss_main = criterion(main_output, labels)  # 主损失
                loss_aux = 0.3 * criterion(aux1, labels) + 0.3 * criterion(aux2, labels)  # 辅助损失
                loss = loss_main + loss_aux  # 总损失
            else:  # 非元组（推理模式或非Inception模型）
                loss = criterion(outputs, labels)  # 直接计算损失

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        val_acc = evaluate(model, val_loader, device)

        print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.3f} | Val Acc: {val_acc:.3f}')

        loss_history[epoch] = epoch_loss
        acc_history[epoch] = val_acc

        # Save best model
        if save_best and val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saved best model with acc: {best_acc:.3f}')

    from utils.plot import plot
    plot(loss_history, acc_history, num_epochs)

    # Calculate metrics
    end_time = time.time()
    end_mem = process.memory_info().rss // 1024 // 1024
    return {
        'time': end_time - start_time,
        'mem_usage': end_mem - start_mem,
        'best_acc': best_acc
    }


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
