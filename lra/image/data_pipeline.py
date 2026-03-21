import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
# train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=transform)
# test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=transform)
#
# train_data = [train_dataset[i][0] for i in range(len(train_dataset))]
# test_data = [test_dataset[i][0] for i in range(len(test_dataset))]
# train_data = torch.stack(train_data,dim=0).reshape([50000, -1])
# test_data = torch.stack(test_data,dim=0).reshape([10000, -1])
# train_labels = train_dataset.targets
# test_labels = test_dataset.targets
# train_labels = torch.tensor(train_labels)
# test_labels = torch.tensor(test_labels)
# train_dataset = TensorDataset(train_data, train_labels)
# test_dataset = TensorDataset(test_data, test_labels)
# torch.save({'train':train_dataset,
#             'test':test_dataset,
#             }, "../../data/image/data.pt")
data = torch.load('../../data/image/data.pt',weights_only=False)
train_dataset = data['train'].tensors[0]
test_dataset = data['test'].tensors[0]
train_labels = data['train'].tensors[1]
test_labels = data['test'].tensors[1]
def trans_gray(data,size):
    images = data.reshape(-1,3,32,32)
    gray_images = (0.299 * images[:,0] + 0.587 * images[:,1] + 0.114 * images[:,2])
    return gray_images.reshape(size,-1,1)
train_data = trans_gray(train_dataset,50000)
test_data = trans_gray(test_dataset,10000)
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)
torch.save({'train':train_dataset,
            'test':test_dataset,
            }, "../../data/image/data.pt")




