import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import SSM_function as sf
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR,ReduceLROnPlateau
import random
import numpy as np
import fssm
from torch.amp import GradScaler,autocast

def set_seed(seed=44):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(40)
#加载数据

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device_str = "cuda:2" if torch.cuda.is_available() else "cpu"
data = torch.load('../../data/image/data.pt',weights_only=False)
train_dataset = data['train']
test_dataset = data['test']
'''
加入掩码，step = 0.01 ,lr = 0.01余弦
加入了梯度裁剪
'''
batch_size = 50
hidden_size = 512
step = 0.001
emb_dim = 512
len = 1024
epochs = 200
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
activation = 'tanh'

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        '''
        6层SSM
        '''
        self.pad_id = 0
        self.embeddind = nn.Linear(1, emb_dim)
        self.layer1 = sf.S4D_Block(hidden_size, step, activation, emb_dim, dropout=0.2, skip=True, norm='LN')
        self.layer2 = sf.S4D_Block(hidden_size, step, activation, emb_dim, dropout=0.2, skip=True, norm='LN')
        self.layer3 = sf.S4D_Block(hidden_size, step, activation, emb_dim, dropout=0.2, skip=True,norm='LN')
        self.layer4 = fssm.FS4Ddeq_model(hidden_size, step, activation, emb_dim, layers=3,
                                         final_act='gelu', skip=False, norm='LN', dropout=0.0,
                                         input_size=[batch_size, len, emb_dim],
                                         feed_size=[batch_size, len, emb_dim],
                                         feed_act='gelu',
                                         )
        self.fc = nn.Linear(emb_dim,10)
        self.dropout = nn.Dropout(0.2)
    def forward(self,x):
        r = self.dropout(self.embeddind(x))
        y1 = self.layer1(r)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4, loss, h, n_steps= self.layer4(y3)

        mean_y = torch.mean(y4,dim=1)
        logits = self.fc(mean_y)
        return logits,loss,h,n_steps
model = model()
# model.load_state_dict(torch.load('../../model/listops/S4_fs4d1_epoch_35.pth'),strict=False)
scaler = GradScaler(device_str)
#设置优化器
ssm_params = []
other_params = []
for name, param in model.named_parameters():
    if 'ssm' in name:ssm_params.append(param)
    else:other_params.append(param)
optimizer = optim.AdamW([
    {'params': ssm_params, 'lr': 4e-4},
    {'params': other_params, 'lr': 4e-3},
    ],weight_decay=1e-2)
model.to(device)
criterion = nn.CrossEntropyLoss()
# scheduler = CosineAnnealingLR(optimizer, epochs)
warm_scheduler = LinearLR(
    optimizer,
    start_factor=1e-6,
    end_factor=1,
    total_iters=20
)
Cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=0,
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warm_scheduler,Cosine_scheduler],
    milestones=[20]
)
# scheduler = ReduceLROnPlateau(optimizer,
#                               mode='max',
#                               factor=0.5,
#                               patience=4,
#                               verbose=True)
history = {'train Loss': [], 'train goat': [],
           'val Loss': [], 'val goat': []
           }

for epoch in range(epochs):
    model.train()
    total_loss ,total_correct = 0, 0
    total_samples = 50000

    pbar = tqdm(
        enumerate(train_loader),  # 用 enumerate 同时获取 batch 索引和数据
        total=total_samples/batch_size,  # 总 batch 数
        desc=f"Epoch {epoch + 1}/{epochs}",  # 进度条前缀（如 "Epoch 1/10"）
        unit="batch",  # 单位显示
        ncols=120
    )
    for batch_idx,(batch_x, batch_y) in pbar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        # 前向
        logits, h_loss, h, n_steps = model(batch_x.unsqueeze(-1))
        loss = criterion(logits, batch_y) + 0.1 * h_loss
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",  # 保留4位小数
            "loss_h": f"{h_loss.item():.4f}",
            "n_steps": f"{torch.mean(n_steps).item():.4f}"
        })
    scheduler.step()
    train_loss = total_loss / total_samples
    train_acc = total_correct / total_samples
    # scheduler.step(train_acc)
    history['train Loss'].append(train_loss)
    history['train goat'].append(train_acc)
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits, h_loss, _, _= model(batch_x.unsqueeze(-1))
            loss = criterion(logits, batch_y) + 0.1 * h_loss

            val_loss += loss.item() * batch_size
            val_correct += (logits.argmax(dim=1) == batch_y).sum().item()

    val_loss /= 10000
    val_acc = val_correct / 10000
    history['val Loss'].append(val_loss)
    history['val goat'].append(val_acc)
    print(f"Epoch [{epoch + 1}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    save_path = os.path.join('../../model/image', f'S4_fs4d_epoch_{epoch + 1}.pth')
    torch.save(model.state_dict(), save_path)

save_path = os.path.join('../../model/image', f'S4_fs4d_epoch_{epochs}.pth')
loss_path = os.path.join('../../model/image', f'S4_fs4d_loss.pth')
torch.save(model.state_dict(), save_path)
torch.save(history, loss_path)

