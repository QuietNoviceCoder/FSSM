import torch
import tarfile
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset
import os
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

transform = transforms.Compose([
    transforms.ToTensor(),
])
data_path = "D:/data/lra_release/lra_release/lra_release/pathfinder128/curv_baseline/"
path_32 = []

def get_data(meta_path):
    all_files = os.listdir(meta_path)
    save_data = []
    label = []
    for file in all_files:
        meta_path_obj = Path(meta_path)
        full_path_obj = meta_path_obj / file
        full_path = full_path_obj.as_posix()
        with open(full_path, 'rb') as f:
            lines = f.readlines()
        for line in lines:
            decoded_line = line.decode('utf-8').strip()
            decoded = decoded_line.split()
            path = data_path + decoded[0] + "/" + decoded[1]
            target = int(decoded[3])
            try:
                image = Image.open(path)
                data = np.array(image)
                data = torch.from_numpy(data).view(-1)
                save_data.append(data)
                label.append(target)
            except:break
    return save_data,label

# path32_path = "D:/data/lra_release/lra_release/lra_release/pathfinder32/curv_baseline/metadata"
path128_path = "D:/data/lra_release/lra_release/lra_release/pathfinder128/curv_baseline/metadata"
# pathfinder32,labels1 = get_data(path32_path)
pathfinder128,labels2 = get_data(path128_path)
# train_data1 = torch.stack(pathfinder32,dim=0)
train_data2 = torch.stack(pathfinder128,dim=0)
# train_dataset1 = TensorDataset(train_data1,torch.tensor(labels1))
train_dataset2 = TensorDataset(train_data2,torch.tensor(labels2))
# torch.save(train_dataset1, "../data/pathfinder/pathfinder32.pt")
torch.save(train_dataset2, "../../data/pathfinder/pathfinder128.pt")








