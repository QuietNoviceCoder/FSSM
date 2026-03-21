import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import fssm
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import random
import numpy as np
import SSM_function as sf

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#加载数据
Idmb = torch.load("../../data/aclImdb/data.pt", weights_only=False)
#参数设置，以及修改内容
'''''
depth = 4，d=64,feture = 64,drop=0,epochs = 20,余弦调度器lr=0.001,wd=0
一个大反馈
'''''
batch_size = 50
hidden_size = 64
step = 0.001
activation = 'tanh'
emb_dim = 64
text_len = 4096
test_samples = 25000
test_loader = DataLoader(Idmb['test'].select(range(test_samples)), batch_size, shuffle=True)
stoi = Idmb['stoi']
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        '''
        网络结构:
        输出层
        '''
        self.pad_id = 0
        self.embedding = nn.Embedding(num_embeddings=len(stoi),embedding_dim=emb_dim,padding_idx=self.pad_id)
        self.ssm1 = fssm.FSSM_model(hidden_size,step,activation,text_len,emb_dim,mid_layers=2,skip = True,
                              dropout=0.0,norm = 'BN',
                              # final_act='tanh',
                              # feed_model='attention',
                              input_size=[batch_size,text_len,emb_dim],
                              feed_size=[batch_size,text_len,emb_dim],
                              feed_act=None
                                    )
        self.fc = nn.Linear(emb_dim,2)
    def forward(self,x):
        mask = (x!=self.pad_id).float()
        r = self.embedding(x)
        y1 ,H= self.ssm1(r)
        mask = mask.unsqueeze(-1)
        y4 = y1 * mask
        # mean pooling
        sum_y = y4.sum(dim=1)
        len_y = mask.sum(dim=1)  # [batch, 1]
        mean_y = sum_y / (len_y + 1e-8)
        logits = self.fc(mean_y)
        return logits,H
model = model()
model.load_state_dict(torch.load('../model/text/epoch_fc0_1.pth'))
model.to(device)
criterion = nn.CrossEntropyLoss()

pbar = tqdm(
    enumerate(test_loader),  # 用 enumerate 同时获取 batch 索引和数据
    total = test_samples/batch_size,  # 总 batch 数
    unit = "batch"  # 单位显示
)
total_correct , total_loss = 0,0
with torch.no_grad():
    for batch_idx, test in pbar:
        batch_x, batch_y = test['ids'].to(device), test['label'].to(device).long()
        # 前向
        logits, H = model(batch_x)
        loss = criterion(logits, batch_y) + fssm.loss_h(H, 0.1)
        total_loss += loss.item() * batch_x.size(0)
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",  # 保留4位小数
            "Batch": f"{batch_idx + 1}/{test_samples / batch_size}"  # 当前 batch 进度
        })

total_loss /= test_samples
test_acc = total_correct / test_samples
print(f"test Loss: {total_loss:.4f} Acc: {test_acc:.4f}")

