import os
import numpy as np
from natsort import natsorted

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# 定义处理.npy文件的自定义数据集类
class NpyDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        npy_file = np.load(self.file_paths[idx])
        label = self.labels[idx]
        return npy_file, label

# 数据集分割函数
def split_dataset(file_paths, labels, train_ratio=0.8):
    test_ratio = 1 - train_ratio
    X_train, X_test, y_train, y_test = train_test_split(file_paths, labels, test_size=test_ratio, stratify=labels)
    return X_train, X_test, y_train, y_test

# 创建特定路径的数据加载器
def create_dataloaders_for_path(path, train_ratio=0.8, batch_size=32):
    file_paths = []
    labels = []
    label_map = {}
    current_label = 0

    # 遍历路径中的子文件夹
    for root, dirs, files in os.walk(path):
        dirs[:] = natsorted(dirs)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            npy_files = [os.path.join(dir_path, f) for f in natsorted(os.listdir(dir_path)) if f.endswith('.npy')]
            file_paths.extend(npy_files)
            labels.extend([current_label] * len(npy_files))
            label_map[current_label] = dir_name
            current_label += 1

    # 将数据集分割为训练和测试集
    X_train, X_test, y_train, y_test = split_dataset(file_paths, labels, train_ratio)

    # 创建数据集对象
    train_dataset = NpyDataset(X_train, y_train)
    test_dataset = NpyDataset(X_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, label_map

# 获取多个路径的数据加载器
def get_dataloaders(paths, train_ratio=0.8, batch_size=32):
    loaders = {}
    for path in paths:
        train_loader, test_loader, label_map = create_dataloaders_for_path(path, train_ratio, batch_size)
        loaders[path] = {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'label_map': label_map
        }
    return loaders

if __name__ == "__main__":
    # 使用示例
    paths = ["../data/us8k/npy/an_out/", "../data/us8k/npy/stella_c/",
             "../data/esc10/npy/an_out", "../data/esc10/npy/stella_c"]
    dataloaders = get_dataloaders(paths, batch_size=20)

    for path, loaders in dataloaders.items():
        print(f"Path: {path}")
        print(f"Number of training samples: {len(loaders['train_loader'].dataset)}")
        print(f"Number of test samples: {len(loaders['test_loader'].dataset)}")
        print(f"Label map: {loaders['label_map']}")
