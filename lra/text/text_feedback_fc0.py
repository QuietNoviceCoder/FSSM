import torch
from sympy.physics.units import current
from torch.utils.data import DataLoader
import torch.nn as nn
import fssm
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR,ReduceLROnPlateau
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed=44):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(44)
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
text_len = 4000
total_samples = 5000
text_samples = 25000
def get_samples_for_epoch(epoch, max_samples=25000, min_samples=5000):
    # # 渐进式训练
    # if epoch < 2:
    #     return min_samples
    # elif epoch < 5:
    #     return min_samples + (max_samples - min_samples) * 0.3
    # elif epoch < 15:
    #     return min_samples + (max_samples - min_samples) * 0.6
    # else:
        return max_samples
Idmb['train'].set_format(type='torch', columns=['input_ids','label'])
Idmb['test'].set_format(type='torch', columns=['input_ids','label'])
train_loader = DataLoader(Idmb['train'].select(range(total_samples)), batch_size, shuffle=True)
test_loader = DataLoader(Idmb['test'].select(range(text_samples)), batch_size, shuffle=True)
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        '''
        网络结构:
        输出层
        '''
        self.pad_id = 0
        self.embedding = nn.Embedding(256,embedding_dim=emb_dim)
        self.ssm1 = fssm.FSSM_model(hidden_size,step,activation,text_len,emb_dim,mid_layers=2,skip = True,
                              dropout=0.0,norm = 'BN',h=0.2,
                              final_act='tanh',
                              input_size=[batch_size,text_len,emb_dim],
                              feed_size=[batch_size,text_len,emb_dim],
                              feed_act=None,
                              use_flash=True,
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
# model.load_state_dict(torch.load('../../model/text/epoch_fc0_8.pth'))
#设置优化器
model.to(device)
criterion = nn.CrossEntropyLoss()
epochs = 20
ssm_params = []
other_params = []
for name, param in model.named_parameters():
    if 'fssm' in name:ssm_params.append(param)
    else:other_params.append(param)

optimizer = optim.AdamW([
    {'params': ssm_params, 'lr': 0.0005},
    {'params': other_params, 'lr': 0.005},
    ],weight_decay=0.05)
# scheduler = CosineAnnealingLR(optimizer,epochs)
warm_scheduler = LinearLR(
    optimizer,
    start_factor=1e-6,
    end_factor=1.0,
    total_iters=3
)
Cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=0,
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warm_scheduler,Cosine_scheduler],
    milestones=[3]
)
# scheduler = ReduceLROnPlateau(optimizer,
#                               mode='max',
#                               factor=0.5,
#                               threshold=0.03,
#                               patience=4,
#                               verbose=True)
history = {'train Loss': [], 'train goat': [],
           'test Loss': [], 'test goat': []
           }

for epoch in range(epochs):
    model.train()
    total_loss ,total_correct = 0, 0
    current_train_samples = int(get_samples_for_epoch(epoch))
    if current_train_samples != int(len(train_loader)*batch_size):
        train_loader = DataLoader(Idmb['train'].select(range(current_train_samples)), batch_size, shuffle=True)
    pbar = tqdm(
        enumerate(train_loader),  # 用 enumerate 同时获取 batch 索引和数据
        total = current_train_samples/batch_size,  # 总 batch 数
        desc = f"Epoch {epoch + 1}/{epochs}",  # 进度条前缀（如 "Epoch 1/10"）
        unit = "batch"  # 单位显示
    )
    for batch_idx,train in pbar:
        batch_x, batch_y = train['input_ids'].to(device), train['label'].to(device)
        # 前向
        logits, H = model(batch_x)
        loss = criterion(logits, batch_y) + fssm.loss_h(H, 0.2)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # if epoch > 20:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",  # 保留4位小数
            "Batch": f"{batch_idx + 1}/{total_samples/batch_size}"  # 当前 batch 进度
        })
    scheduler.step()
    train_loss = total_loss / current_train_samples
    train_acc = total_correct / current_train_samples
    history['train Loss'].append(train_loss)
    history['train goat'].append(train_acc)
    model.eval()
    test_loss, test_correct = 0, 0
    with torch.no_grad():
        for test in test_loader:
            batch_x, batch_y = test['input_ids'].to(device), test['label'].to(device)
            logits ,H = model(batch_x)
            loss1 = criterion(logits, batch_y)
            loss2 = fssm.loss_h(H, 0.2)
            loss = loss1+loss2
            test_loss += loss.item() * batch_x.size(0)
            test_correct += (logits.argmax(dim=1) == batch_y).sum().item()

    test_loss /= text_samples
    test_acc = test_correct / text_samples

    # scheduler.step(test_acc)

    history['test Loss'].append(test_loss)
    history['test goat'].append(test_acc)
    save_path = os.path.join('../../model/text', f'epoch_fc0_{epoch + 1}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Epoch [{epoch + 1}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
save_path = os.path.join('../../model/text', f'epoch_fc0_{epochs}.pth')
loss_path = os.path.join('../../model/text', f'fc0_loss.pth')
torch.save(model.state_dict(), save_path)
torch.save(history, loss_path)

