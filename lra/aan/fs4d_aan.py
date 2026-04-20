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
    os.environ['MASTER_PORT'] = '12321'

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
text_len = 4000
total_samples = 5000
text_samples = 25000
epochs = 20
epoch_0 = 0
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
        self.dropout = nn.Dropout(0.1)
        self.layer1 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer2 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer3 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer4 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer5 = fssm.FS4Ddeq_model(hidden_size, activation, emb_dim, layers=2,
                                         final_act='relu', skip=True, norm='LN', dropout=0.0,
                                         state_size=[batch_size, len, emb_dim],
                                         feed_size=[batch_size, len, emb_dim],
                                         feed_act='relu',
                                         )
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, 2)
        )
    def encode(self,x):
        mask = (x!=self.pad_id).float()
        r = self.embedding(x)
        r = self.dropout(self.emb_norm(r))
        y1 = self.layer1(r)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        y5, loss, n_steps, zhanbi = self.layer5(y4)
        mask = mask.unsqueeze(-1)
        y5 = y5 * mask
        # mean pooling
        sum_y = y5.sum(dim=1)
        len_y = mask.sum(dim=1)  # [batch, 1]
        mean_y = sum_y / (len_y + 1e-8)
        return mean_y,loss,n_steps,zhanbi
    def forward(self, x1,x2):
        y1,jac_l1,n1,zhan1 = self.encode(x1)
        y2,jac_l2,n2,zhan2 = self.encode(x2)
        pair = torch.cat([y1, y2, torch.abs(y1 - y2), y1 * y2], dim=-1)
        logits = self.fc(pair)
        jac_l = (jac_l1 + jac_l2)/2
        n_steps = (n1 + n2)/2
        zhan = (zhan1 + zhan2)/2
        return logits,jac_l,n_steps,zhan
def train(rank, world_size, args):
    """Main training function for each process"""
    n_ka = 0
    # Setup DDP
    setup_ddp(rank, world_size)
    if rank == n_ka:
        run = wandb.init(
            project="retrieval",
            name = "4-s4d + 2-fs4d",
            config={
                "learning_rate": 1e-3,
                "ssm_learning_rate": 3e-4,
                "hidden_dim": hidden_size,
                "batch_size": 50,
                "epochs": epochs,
                "weight_decay": 1e-3,
                "architecture": "6-fs4d",
                "emb_dim": emb_dim,
            },
            resume = "allow",
        )
    set_seed(40)  # Different seed per rank for data shuffling
    torch.cuda.set_device(rank)
    # Load data
    # 加载数据
    aan_data = torch.load("../../data/ann/data.pt", weights_only=False)
    train_ds = aan_data["train"]
    eval_ds = aan_data["eval"]
    # Create DistributedSampler for training
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=40,
        drop_last=True
    )
    # Create DataLoaders with DistributedSampler
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_sampler = DistributedSampler(
        eval_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True
    )
    val_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    # Initialize model
    model = Model(batch_size, text_len, emb_dim, hidden_size, step, activation)
    # model.load_state_dict(torch.load('../../model/ann/S4_fs4d_epoch_11.pth'),strict=False)
    model = model.to(rank)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # GradScaler for mixed precision

    # Setup optimizer
    ssm_params = []
    feed_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'feed' in name:
            feed_params.append(param)
        elif 'ssm' in name:
            ssm_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW([
        {'params': ssm_params, 'lr': 1e-3,'weight_decay':5e-2},
        {'params': feed_params, 'lr': 3e-4,'weight_decay':5e-2},
        {'params': other_params, 'lr': 1e-3,'weight_decay':5e-2},
    ])

    criterion = nn.CrossEntropyLoss()

    # Learning rate schedulers
    # warm_scheduler = LinearLR(
    #     optimizer,
    #     start_factor=1e-5,
    #     end_factor=1,
    #     total_iters=10
    # )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs+5,
        eta_min=0,
    )
    # reduce_scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode='max',
    #     factor=0.5,
    #     patience=3,
    #     verbose=False,
    #     min_lr=1e-8,)
    # combined_scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warm_scheduler, cosine_scheduler],
    #     milestones=[10],
    # )
    # History tracking (only on rank 0)
    if rank == n_ka:
        history = {'train Loss': [], 'train goat': [],
                   'val Loss': [], 'val goat': []}

    # Training loop
    for epoch in range(epoch_0,epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # Important for proper shuffling

        total_loss, total_correct = 0.0, 0.0
        total_samples = 0.0

        # Only show progress bar on rank 0
        if rank == n_ka:
            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                ncols=120
            )
        else:
            pbar = enumerate(train_loader)
        for batch_idx, (batch_x1, batch_x2, batch_y) in pbar:
            batch_x1 = batch_x1.to(rank, non_blocking=True)
            batch_x2 = batch_x2.to(rank, non_blocking=True)
            batch_y = batch_y.to(rank, non_blocking=True)

            # Forward pass
            logits, jac_loss, n_steps, mean_zhanbi = model(batch_x1,batch_x2)
            loss = criterion(logits, batch_y) + 0.1 * jac_loss
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Accumulate metrics
            batch_samples = batch_x1.size(0)
            total_loss += loss.item() * batch_samples
            total_correct += (logits.argmax(dim=1) == batch_y).sum().float().item()
            total_samples += batch_samples
            train_acc = total_correct / total_samples
            train_loss = total_loss / total_samples
            if rank == n_ka:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "n_steps": f"{n_steps:.4f}"
                })
        if rank == n_ka:
            history['train Loss'].append(train_loss)
            history['train goat'].append(train_acc)
        # combined_scheduler.step()
        dist.barrier()

        model.eval()
        val_loss, val_correct = 0, 0
        val_samples = 0

        with torch.no_grad():
            for batch_x1, batch_x2, batch_y in val_loader:
                batch_x1 = batch_x1.to(rank, non_blocking=True)
                batch_x2 = batch_x2.to(rank, non_blocking=True)
                batch_y = batch_y.to(rank, non_blocking=True)
                logits, jac_loss, n_steps, mean_zhanbi = model(batch_x1,batch_x2)
                loss = criterion(logits, batch_y) + 0.1 * jac_loss
                batch_samples = batch_x1.size(0)
                val_loss += loss.item() * batch_samples
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
        cosine_scheduler.step()
        lr_list  =  cosine_scheduler.get_last_lr()
        current_lr = lr_list[0]
        if rank == n_ka:
            history['val Loss'].append(val_loss)
            history['val goat'].append(val_acc)

            print(f"Epoch [{epoch + 1}] "
                  f"Train Loss: {train_loss :.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"n_steps: {n_steps:.2f} | "
                  f"lr: {current_lr:.9f}")
            run.log({
                "acc": val_acc,
                "loss": val_loss,
                "n_steps": n_steps,
                "mean_zhanbi": mean_zhanbi,
            })
            save_path = os.path.join('../../model/ann', f'S4_fs4d_epoch_{epoch + 1}.pth')
            torch.save(model.module.state_dict(), save_path)
            loss_path = os.path.join('../../model/ann', f'S4_fs4d_loss.pth')
            torch.save(history, loss_path)

    if rank == n_ka:
        run.finish()
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


