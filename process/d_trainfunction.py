import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from process.b_get_dataloaders import get_dataloaders
from process.c_get_models import get_models
import pandas as pd
import sys
import random
import numpy as np
import os
import copy
# import torchvision.transforms.functional as TF


def evaluate_model(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss_total += loss.item() * inputs.size(0)

    loss_avg = loss_total / total
    accuracy = correct / total
    return loss_avg, accuracy


def train_model(model, train_loader, val_loader, test_loader,
                model_name, dataset_name,
                num_epochs=10, initial_lr=0.1, step_size=5, gamma=0.1, test_train=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, weight_decay=5e-3)
    # optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    # optimizer = optim.SGD([
    #     {'params': model.a.linear.parameters(), 'lr': initial_lr*10},  # Larger LR for linear layer
    #     {'params': model.a.w, 'lr': initial_lr*10},
    #     # {'params': [p for name, p in model.named_parameters() if name not in ["a.linear.weight", "a.w"]], 'lr': initial_lr}# Larger LR for self.w
    # ], lr=initial_lr, weight_decay=5e-3)

    # for name, p in model.named_parameters():
    #     print(name)


    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_score = 0.0
    best_model_state = None
    best_accuracies = {}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0


        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            batch_count += 1
            if test_train and batch_count == 5:
                break

        train_loss, train_accuracy = running_loss / total, correct / total

        scheduler.step()
        # Compute validation and test accuracy if epochs exceed 100
        if epoch >= 90:
            val_loss, val_accuracy = evaluate_model(model, val_loader)
            test_loss, test_accuracy = evaluate_model(model, test_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Model: {model_name}, Dataset: {dataset_name}, "
                  f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
            sys.stdout.flush()
            total_score = train_accuracy + val_accuracy
            if total_score > best_score:
                best_score = total_score
                best_model_state = model.state_dict()
                best_accuracies = {
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                    "test_accuracy": test_accuracy
                }
        else:
            print(f"Epoch {epoch + 1}/{num_epochs}, Model: {model_name}, Dataset: {dataset_name}, "
                  f"Train Acc: {train_accuracy:.4f}")
            sys.stdout.flush()

    return best_model_state, best_accuracies

# def test_model(model, test_loader, model_name, dataset_name, test_train=False):
#     criterion = nn.CrossEntropyLoss()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#
#     test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device, test_train=test_train)
#
#     print(
#         f"Model: {model_name}, Dataset: {dataset_name}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
#     sys.stdout.flush()
#     return test_loss, test_accuracy


# def validate_model(model, validate_loader, model_name, dataset_name, test_train=False):
#     criterion = nn.CrossEntropyLoss()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#
#     validate_loss, validate_accuracy = evaluate_model(model, validate_loader, criterion, device, test_train=test_train)
#
#     print(
#         f"Model: {model_name}, Dataset: {dataset_name}, Validate Loss: {validate_loss:.4f}, Validate Accuracy: {validate_accuracy:.4f}")
#     sys.stdout.flush()
#     return validate_loss, validate_accuracy


# def get_train_accuracy(model, train_loader, model_name, dataset_name, test_train=False):
#     criterion = nn.CrossEntropyLoss()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#
#     train_loss, train_accuracy = evaluate_model(model, train_loader, criterion, device, test_train=test_train)
#
#     print(
#         f"Model: {model_name}, Dataset: {dataset_name}, train Loss: {train_loss:.4f}, train Accuracy: {train_accuracy:.4f}")
#     sys.stdout.flush()
#     return train_loss, train_accuracy

# def evaluate_model(model, data_loader, criterion, device, test_train=False):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         batch_count = 0
#         for inputs, labels in data_loader:
#             inputs = torch.tensor(inputs, dtype=torch.float32)
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#             loss = criterion(outputs, labels)
#             running_loss += loss.item() * inputs.size(0)
#             batch_count += 1
#             if test_train and batch_count == 3:
#                 break
#     loss = running_loss / total
#     accuracy = correct / total
#     return loss, accuracy


def train(datasets_paths, trials, path2tab, path2trained,
          batch_size=20, num_epochs=10, initial_lr=0.1, step_size=5, test_train=False, **kwargs):
    results = []
    best_models = {}  # 存储每个模型在每个数据集上的最佳模型和精度

    for trial in range(trials):
        dataloaders = get_dataloaders(datasets_paths, batch_size=batch_size)
        for path, loaders in dataloaders.items():
            train_loader = loaders['train_loader']
            val_loader = loaders['val_loader']
            test_loader = loaders['test_loader']

            dataset_name = path.split('/')[1]
            # labelmap = loaders['label_map']
            models = get_models(in_channels=1, dataset=train_loader.dataset.file_paths[0].split('/')[1], method= kwargs['method'])  # 1

            for model_name, model in models.items():
                print(f"Training {model_name} on data from {dataset_name} in trial_{trial:d}")
                sys.stdout.flush()
                best_model_state, best_accuracies = train_model(model, train_loader, val_loader, test_loader,
                                                               model_name, dataset_name, num_epochs=num_epochs,
                                                               initial_lr = initial_lr, step_size=step_size, gamma=0.1,
                                                                test_train=test_train)

                # Compute sum accuracy
                current_sum_accuracy = sum(best_accuracies.values())
                # Ensure dataset entry exists
                if dataset_name not in best_models:
                    best_models[dataset_name] = {}

                # If model is not stored yet OR the new accuracy is higher, update best model
                if (model_name not in best_models[dataset_name] or
                        current_sum_accuracy > best_models[dataset_name][model_name]['sum_accuracy']):
                    best_models[dataset_name][model_name] = {
                        'state': best_model_state,
                        'accuracies': best_accuracies,
                        'sum_accuracy': current_sum_accuracy
                    }

    # Save the best models' weights and results after all trials
    # model_save_path = os.path.join(path2trained, "best_models.pth")
    # torch.save(best_models, model_save_path)  # Save the entire best_models dictionary
    # print(f"Best models dictionary saved to {model_save_path}")
    #
    # for (model_name, path), data in best_models.items():
    #     torch.save(data['model'], path2trained+f"/{model_name}_{path.split('/')[1]}_{path.split('/')[-1]}.pth")
    # print("Best models saved")
    # sys.stdout.flush()

    # Prepare results for Excel
    excel_data = []
    for dataset_name, models in best_models.items():
        for model_name, model_info in models.items():
            excel_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Train Accuracy': round(model_info['accuracies'].get('train_accuracy', 0), 4),
                'Validation Accuracy': round(model_info['accuracies'].get('val_accuracy', 0), 4),
                'Test Accuracy': round(model_info['accuracies'].get('test_accuracy', 0), 4),
                'Sum Accuracy': round(model_info['sum_accuracy'], 4)
            })
            # Save model weights
            weight_save_path = f"bestmodel/{model_name}_{dataset_name}.pth"
            torch.save(model_info['state'], weight_save_path)

    # Save to Excel
    df_results = pd.DataFrame(excel_data)
    excel_save_path = "tab/best_model_results.xlsx"
    df_results.to_excel(excel_save_path, index=False)
    print(f"Results saved to {excel_save_path}")

def set_random_seed(seed=42):
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def test_ensemble(datasets_paths, trials, path2tab, path2trained, batch_size=20):
#     set_random_seed(777)
#     # esc10 123(99.51) 1024(1) 777(0.9854) 999 (0.9902) max1
#     # us8k 123-93.89(+-0.27) 1024-93.8(+-0.25) 777(0.9854) 999 (0.9902) max1
#
#     average_testacc = []
#     for trial in range(trials):
#         dataloaders = get_dataloaders(datasets_paths, batch_size=batch_size)
#         for path, loaders in dataloaders.items():
#             # train_loader = loaders['train_loader']
#             # loaders['label_map']
#             test_loader = loaders['test_loader']
#             dataset = path.split('/')[1]
#             # model = get_ensemblebncp(dataset)
#             # model_name = 'ensemblebncp'
#
#             sys.stdout.flush()
#             test_loss, test_accuracy = evaluate_model(model, test_loader)
#             average_testacc.append(test_accuracy)
#             print(f"Testing ensemble on data from {path} in trial_{trial:d},Tes acc: {test_accuracy:.4f}")
#     print(f'Average test accuracy on esc10 ensemble Average test accuracy: {np.mean(average_testacc):.4f}')


def test_normal(dataset, trials, batch_size=32, method='agss'):
    set_random_seed(40)
    # random_seed_list = [123,777,999,42,1024]
    # esc10 agss 123(97.56+-2.44) cgss 123(98.54+-1.46)
    # us8k agss 123(98.63+-0.68)   cgss 123(98.58+-0.62)
    dataset_name = dataset.split('/')[1]
    model = get_models(method=method, dataset=dataset_name)
    key = next(iter(model))
    model = copy.deepcopy(model[key])
    model.load_state_dict(torch.load(f'bestmodel/shufflenet_{method}_{dataset_name}.pth'), strict=False)

    average_testacc = []
    for trial in range(trials):
        # set_random_seed(random_seed_list[trial])
        dataloaders = get_dataloaders([dataset], batch_size=batch_size)
        for path, loaders in dataloaders.items():
            # train_loader = loaders['train_loader']
            # loaders['label_map']
            test_loader = loaders['test_loader']
            # dataset = path.split('/')[1]


            # torch.load()
            # model = get_ensemblebncp(dataset)
            # model_name = 'ensemblebncp'

            sys.stdout.flush()
            test_loss, test_accuracy = evaluate_model(model, test_loader)
            average_testacc.append(test_accuracy)
            print(f"Testing on data from {path} in trial_{trial:d},Tes acc: {test_accuracy:.4f}")
    print(f'Average test accuracy on {dataset_name}: {np.mean(average_testacc):.4f}')

if __name__ == "__main__":
    pass
    # datasets_paths = ["../data/esc10/npy/an_out", "../data/esc10/npy/stella_out",
    #                   "../data/us8k/npy/an_out/", "../data/us8k/npy/stella_out/"]
    # datasets_paths = ["../data/esc10/npy/an_out", "../data/esc10/npy/stella_out"]
    # # train(datasets_paths, path2tab='tab', path2trained='bestmodel',
    # #       trials=5, batch_size=20, num_epochs=10, initial_lr=0.1, step_size=5)
    # train(datasets_paths, path2tab='tab', path2trained='bestmodel',
    #       trials=2, batch_size=2, num_epochs=2, initial_lr=0.1, step_size=1, test_train=True)



