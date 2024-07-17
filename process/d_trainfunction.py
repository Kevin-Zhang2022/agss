import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from process.b_get_dataloaders import get_dataloaders
from process.c_get_models import get_models
import pandas as pd
import sys
import copy
import torchvision.transforms.functional as TF

def sequence_forward(model, inputs, total,correct, labels):
    outputs = []
    for t in range(inputs.size(2)):  # 对每个时间步进行分类
        time_step_input = inputs[:, :, t]
        output = model(time_step_input)
        outputs.append(output)
    outputs = torch.stack(outputs, dim=2)
    # 找到 c 维度上的最大值位置，形状为 [b, t]
    max_indices = torch.argmax(outputs, dim=1)
    one_hot = torch.zeros_like(outputs)
    # 使用 scatter_ 方法将最大值位置赋值为 1
    one_hot.scatter_(1, max_indices.unsqueeze(1), 1)
    p = torch.softmax(torch.sum(one_hot, dim=-1), dim=1)
    _, predicted = torch.max(p, 1)
    total += predicted.size(0)
    correct += (predicted == labels).sum().item()

    outputs = outputs.permute(0, 2, 1).contiguous()  # 先转换为 [20, 1120, 10]
    outputs = outputs.view(-1, outputs.size(-1))
    # 将标签扩展以匹配输出的形状
    labels = labels.unsqueeze(1).expand(-1, inputs.size(2)).contiguous()  # 先转换为 [20, 1120]
    labels = labels.view(-1)

    return outputs, total, correct, labels

def train_model(model, train_loader, model_name, dataset_name, num_epochs=10, initial_lr=0.1, step_size=5, gamma=0.1, test_train=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        batch_count = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if hasattr(model, 'sequence'):
                outputs, total, correct, labels = sequence_forward(model, inputs, total, correct, labels)
            else:
                outputs = model(inputs.unsqueeze(1))
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            # outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            batch_count += 1
            if test_train and batch_count == 5:
                break

        train_loss = running_loss / total
        train_accuracy = correct / total

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Model: {model_name}, Dataset: {dataset_name}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        sys.stdout.flush()

        # 更新学习率
        scheduler.step()


def test_model(model, test_loader, model_name, dataset_name, test_train=False):
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device, test_train=test_train)

    print(
        f"Model: {model_name}, Dataset: {dataset_name}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    sys.stdout.flush()
    return test_loss, test_accuracy


def evaluate_model(model, data_loader, criterion, device, test_train=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        batch_count=0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if hasattr(model, 'sequence'):
                outputs, total, correct, labels = sequence_forward(model, inputs, total, correct, labels)
            else:
                outputs = model(inputs.unsqueeze(1))
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            batch_count+=1
            if test_train and batch_count== 3:
                break
    loss = running_loss / total
    accuracy = correct / total
    return loss, accuracy


def train(datasets_paths, trials, path2tab, path2trained,
          batch_size=20, num_epochs=10, initial_lr=0.1, step_size=5, test_train=False):
    results = []
    best_models = {}  # 存储每个模型在每个数据集上的最佳模型和精度

    for trial in range(trials):
        dataloaders = get_dataloaders(datasets_paths, batch_size=batch_size)
        for path, loaders in dataloaders.items():
            train_loader = loaders['train_loader']
            test_loader = loaders['test_loader']
            models = get_models(in_channels=1, dataset=train_loader.dataset.file_paths[0].split('/')[1])  # 1
            for model_name, model in models.items():
                print(f"Training {model_name} on data from {path} in trial_{trial:d}")
                sys.stdout.flush()
                train_model(model, train_loader, model_name, path, num_epochs=num_epochs,
                            initial_lr=initial_lr, step_size=step_size, gamma=0.1, test_train=test_train)
                print(f"Testing {model_name} on data from {path} in trial_{trial:d}")
                sys.stdout.flush()
                test_loss, test_accuracy = test_model(model, test_loader, model_name, path, test_train=test_train)
                # torch.save(model, 'bestmodel' + f"/{model_name}_{path.split('/')[-2]}.pth")
                # 检查并更新当前模型在当前数据集上的最佳模型和精度
                key = (model_name, path)
                if key not in best_models or test_accuracy > best_models[key]['accuracy']:
                    best_models[key] = {
                        'model': model.state_dict(),
                        'accuracy': test_accuracy
                    }

                results.append({
                    "Model": model_name,
                    "Dataset": path,
                    "Test Accuracy": test_accuracy
                })

    # Calculate average test accuracy for each model on each dataset
    df = pd.DataFrame(results)
    avg_results = df.groupby(["Model", "Dataset"])["Test Accuracy"].mean().reset_index()
    avg_results.to_excel(path2tab+"/average_test_accuracies.xlsx", index=False)
    print("Average results saved to model_test_accuracies.xlsx")
    sys.stdout.flush()

    best_data = [{'model_name': k[0], 'path': k[1], 'accuracy': v['accuracy']} for k, v in best_models.items()]
    df = pd.DataFrame(best_data)
    # avg_results = df.groupby(["model_name", "path"])
    df.to_excel(path2tab+'/best_models_accuracy.xlsx', index=False)

    # Save the best models
    for (model_name, path), data in best_models.items():
        torch.save(data['model'], path2trained+f"/{model_name}_{path.split('/')[1]}_{path.split('/')[-1]}.pth")
    print("Best models saved")
    sys.stdout.flush()



if __name__ == "__main__":
    pass
    # datasets_paths = ["../data/esc10/npy/an_out", "../data/esc10/npy/stella_c",
    #                   "../data/us8k/npy/an_out/", "../data/us8k/npy/stella_c/"]
    # datasets_paths = ["../data/esc10/npy/an_out", "../data/esc10/npy/stella_c"]
    # # train(datasets_paths, path2tab='tab', path2trained='bestmodel',
    # #       trials=5, batch_size=20, num_epochs=10, initial_lr=0.1, step_size=5)
    # train(datasets_paths, path2tab='tab', path2trained='bestmodel',
    #       trials=2, batch_size=2, num_epochs=2, initial_lr=0.1, step_size=1, test_train=True)



