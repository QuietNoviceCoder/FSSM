import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import SSM_function as sf
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau
import random
import numpy as np
import fssm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import wandb

def set_seed(seed=44):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_ddp(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up distributed training"""
    dist.destroy_process_group()
#参数设置，以及修改内容
batch_size = 16
hidden_size = 64
step = 0.001
activation = 'tanh'
emb_dim = 256
text_len = 1024
total_samples = 50000
test_samples = 10000
epochs = 200
epoch_0 = 0
class Model(nn.Module):
    def __init__(self, batch_size, len, emb_dim, hidden_size, step, activation):
        super(Model, self).__init__()
        '''
        网络结构:
        输出层
        '''
        self.pre_fc = nn.Linear(1, emb_dim)
        self.emb_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(0.2)
        self.layer1 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.2, skip=True, norm='LN')
        self.layer2 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.2, skip=True, norm='LN')
        self.layer3 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.2, skip=True, norm='LN')
        self.layer4 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.2, skip=True, norm='LN')
        self.layer5 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.2, skip=True, norm='LN')
        self.layer6 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.2, skip=True, norm='LN')
        self.fc = nn.Linear(emb_dim,10)
    def forward(self,x):
        r = self.pre_fc(x)
        r = self.dropout(self.emb_norm(r))

        y2 = self.layer1(r)
        y3 = self.layer2(y2)

        y4 = self.layer3(y3)
        y5 = self.layer4(y4)
        y6 = self.layer5(y5)
        y7 = self.layer6(y6)

        # mean pooling
        sum_y = y7.sum(dim=1)
        mean_y = sum_y / (1024 + 1e-8)
        logits = self.fc(mean_y)
        return logits
def train(rank, world_size, args):
    """Main training function for each process"""
    n_ka = 0
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    # Load data
    # 加载数据
    data = torch.load('../../data/image/data.pt', weights_only=False)
    val_dataset = data['test']
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    model = Model(batch_size, text_len, emb_dim, hidden_size, step, activation)
    model.load_state_dict(torch.load('../../model/image/S4_s4d_epoch_67.pth'),strict=False)
    model = model.to(rank)
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    if rank == n_ka:
        history = {'train Loss': [], 'train goat': [],
                   'val Loss': [], 'val goat': []}
    model.eval()
    val_loss, val_correct = 0, 0
    val_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(rank, non_blocking=True)
            batch_y = batch_y.to(rank, non_blocking=True)
            logits = model(batch_x)
            batch_samples = batch_x.size(0)
            val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            val_samples += batch_samples
    val_correct_tensor = torch.tensor(val_correct, dtype=torch.float32, device=rank)
    val_samples_tensor = torch.tensor(val_samples, dtype=torch.float32, device=rank)
    val_loss_tensor = torch.tensor(val_loss, dtype=torch.float32, device=rank)

    dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_samples_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)

    val_acc = val_correct_tensor.item() / val_samples_tensor.item()
    val_loss = val_loss_tensor.item() / val_samples_tensor.item()
    if rank == n_ka:
        history['val Loss'].append(val_loss)
        history['val goat'].append(val_acc)

        print(
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | ")
    cleanup_ddp()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', type=int, default=4, help='number of GPUs to use')
    args = parser.parse_args()

    world_size = args.world_size

    # Launch distributed training
    torch.multiprocessing.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()


