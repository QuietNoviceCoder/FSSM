import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import SSM_function as sf
import os
import random
import numpy as np
import fssm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
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
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed training"""
    dist.destroy_process_group()


class Model(nn.Module):
    def __init__(self, batch_size, len, emb_dim, hidden_size, step, activation):
        super(Model, self).__init__()
        self.pad_id = 0
        self.embeddind = nn.Linear(1, emb_dim)
        self.layer1 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.2, skip=True, norm='BN')
        self.layer2 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.2, skip=True, norm='BN')
        self.layer3 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.2, skip=True, norm='BN')
        self.layer4 = fssm.FS4Ddeq_model(hidden_size, activation, emb_dim, layers=2,
                                         final_act='relu', skip=True, norm='BN', dropout=0.0,
                                         state_size=[batch_size, len, emb_dim],
                                         feed_size=[batch_size, len, emb_dim],
                                         feed_act='relu',
                                         )
        self.layer5 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.2, skip=True, norm='BN')
        self.fc = nn.Linear(emb_dim, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        r = self.dropout(self.embeddind(x.unsqueeze(-1)))

        y1 = self.layer1(r)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer5(y3)
        y5, loss, n_steps, zhanbi = self.layer4(y4)
        # mean pooling
        mean_y = torch.mean(y5, dim=1)
        logits = self.fc(mean_y)
        return logits

def train(rank, world_size, args):
    """Main training function for each process"""
    # Setup DDP
    setup_ddp(rank, world_size)
    set_seed(40)  # Different seed per rank for data shuffling
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    # Load data
    image = torch.load('../../data/pathfinder/pathfinder32.pt',weights_only=False)
    # Hyperparameters
    batch_size = 64
    hidden_size = 64
    step = 0.001
    emb_dim = 256
    r_len = 1024
    activation = 'tanh'
    val_dataset = TensorDataset(image[160000:180000][0].float() / 255.0, image[160000:180000][1])
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
    # Initialize model
    model = Model(batch_size, r_len, emb_dim, hidden_size, step, activation)
    model.load_state_dict(torch.load('../../model/pathfinder/S4_fs4drel_epoch_100.pth'),strict=False)
    model = model.to(rank)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    criterion = nn.CrossEntropyLoss()
    val_loss, val_correct = 0, 0
    val_samples = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(rank, non_blocking=True)
            batch_y = batch_y.to(rank, non_blocking=True)
            logits  = model(batch_x)
            loss = criterion(logits, batch_y)

            batch_samples = batch_x.size(0)
            val_loss += loss.item() * batch_samples
            val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            val_samples += batch_samples

        val_loss /= val_samples
        val_acc = val_correct / val_samples
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | ")
    cleanup_ddp()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', type=int, default=1, help='number of GPUs to use')
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