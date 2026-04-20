import torch
from scipy.stats import cosine
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
from torch.amp import GradScaler, autocast
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
emb_dim = 128
text_len = 4000
text_samples = 25000
class Model(nn.Module):
    def __init__(self, batch_size, len, emb_dim, hidden_size, step, activation):
        super(Model, self).__init__()
        '''
        网络结构:
        输出层
        '''
        self.pad_id = 0
        self.embedding = nn.Embedding(256,embedding_dim=emb_dim)
        self.emb_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(0.2)
        self.layer1 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer2 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer3 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer4 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer5 = fssm.FS4Ddeq_model(hidden_size, activation, emb_dim, layers=2,
                                         final_act='gelu', skip=True, norm='LN', dropout=0.01,
                                         state_size=[batch_size, len, emb_dim],
                                         feed_size=[batch_size, len, emb_dim],
                                         feed_act='gelu',
                                         )
        self.fc = nn.Linear(emb_dim,2)
    def forward(self,x):
        mask = (x!=self.pad_id).float()
        r = self.embedding(x)
        r = self.dropout(self.emb_norm(r))

        y2 = self.layer1(r)
        y3 = self.layer2(y2)

        y4 = self.layer3(y3)
        y5 = self.layer4(y4)
        y1, loss, n_steps, zhanbi = self.layer5(y5)
        # y4 = self.layer3(y3)
        mask = mask.unsqueeze(-1)
        y5 = y1 * mask
        # mean pooling
        sum_y = y5.sum(dim=1)
        len_y = mask.sum(dim=1)  # [batch, 1]
        mean_y = sum_y / (len_y + 1e-8)
        logits = self.fc(mean_y)
        return logits,loss,n_steps,zhanbi
def train(rank, world_size, args):
    """Main training function for each process"""
    n_ka = 0
    # Setup DDP
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank)
    # Load data
    # 加载数据
    Idmb = torch.load("../../data/aclImdb/data.pt", weights_only=False)
    train_ds = Idmb["train"]
    test_ds = Idmb["test"]
    # Create DistributedSampler for training
    val_sampler = DistributedSampler(
        test_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    val_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    # checkpoint = torch.load("../../model/pathfinder/checkpoint35.pth")
    # Initialize model
    model = Model(batch_size, text_len, emb_dim, hidden_size, step, activation)
    model.load_state_dict(torch.load('../../model/text/S4_fs4d_epoch_45.pth'),strict=False)
    model = model.to(rank)
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    model.eval()
    val_loss, val_correct = 0, 0
    val_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(rank, non_blocking=True)
            batch_y = batch_y.to(rank, non_blocking=True)
            logits, jac_loss, n_steps, mean_zhanbi = model(batch_x)
            batch_samples = batch_x.size(0)
            val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            val_samples += batch_samples
    val_correct_tensor = torch.tensor(val_correct, dtype=torch.float32, device=rank)
    val_samples_tensor = torch.tensor(val_samples, dtype=torch.float32, device=rank)

    dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_samples_tensor, op=dist.ReduceOp.SUM)

    val_acc = val_correct_tensor.item() / val_samples_tensor.item()
    if rank == 0:
        print(f"Global Val Accuracy: {val_acc:.4f} "
              f"({int(val_correct_tensor.item())}/{int(val_samples_tensor.item())})")
    cleanup_ddp()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', type=int, default=2, help='number of GPUs to use')
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


