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
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import wandb

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
    def __init__(self, batch_size, len, emb_dim, hidden_size, activation):
        super(Model, self).__init__()
        self.pad_id = 0
        self.embeddind = nn.Embedding(num_embeddings=17, embedding_dim=emb_dim)
        self.layer1 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer2 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer3 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer4 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer5 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer6 = sf.S4D_Block(hidden_size, activation, emb_dim, dropout=0.1, skip=True, norm='LN')
        self.layer7 = fssm.FS4Ddeq_model(hidden_size, activation, emb_dim, layers=2,
                                         final_act='gelu', skip=True, norm='LN', dropout=0.0,
                                         state_size=[batch_size, len, emb_dim],
                                         feed_size=[batch_size, len, emb_dim],
                                         feed_act='gelu',
                                         )
        self.fc = nn.Linear(emb_dim, 10)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        mask = (x != self.pad_id).float()
        r = self.dropout(self.embeddind(x))

        y1 = self.layer1(r)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        y5 = self.layer5(y4)
        y6 = self.layer6(y5)
        y7, loss1, n_steps1, zhanbi1 = self.layer7(y6)

        y = self.dropout(y7)

        mask = mask.unsqueeze(-1)
        y = y * mask
        # mean pooling
        sum_y = y.sum(dim=1)
        len_y = mask.sum(dim=1)  # [batch, 1]
        mean_y = sum_y / (len_y + 1e-8)
        logits = self.fc(mean_y)
        return logits, loss1, n_steps1, zhanbi1

def train(rank, world_size, args):
    """Main training function for each process"""

    # Setup DDP
    setup_ddp(rank, world_size)
    set_seed(42)  # Different seed per rank for data shuffling
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    # Load data
    data = torch.load('../../data/listops/data.pth')
    train_x, train_y, val_x, val_y, test_x, test_y = [
        data['train_x'], data['train_y'],
        data['val_x'], data['val_y'],
        data['test_x'], data['test_y']
    ]
    # Hyperparameters
    batch_size = 25
    hidden_size = 64
    emb_dim = 192
    r_len = 2000
    activation = 'gelu'
    epochs = 100
    epoch_0 = 0
    if rank == 0:
        run = wandb.init(
            project="listops",
            name = "6-s4d + 2-fs4d",
            config={
                "learning_rate": 1e-3,
                "SSM_learning_rate": 5e-4,
                "hidden_dim": hidden_size,
                "d_dim":emb_dim,
                "batch_size": 100,
                "epochs": epochs,
                "weight_decay": 1e-3,
                "architecture": "2-fs4d + 6-s4d",
            }
        )
    # Create datasets
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)

    # Create DistributedSampler for training
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=40,
        drop_last=True
    )

    # Create DataLoaders with DistributedSampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
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
    model = Model(batch_size, r_len, emb_dim, hidden_size, activation)
    # model.load_state_dict(torch.load('../../model/listops/S4_fs4d_epoch_45.pth'),strict=False)
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
        {'params': ssm_params, 'lr': 1e-3,'weight_decay':1e-2},
        {'params': feed_params, 'lr': 5e-4,'weight_decay':1e-2},
        {'params': other_params, 'lr': 1e-3,'weight_decay':1e-2},
    ])

    criterion = nn.CrossEntropyLoss()

    # Learning rate schedulers
    # warm_scheduler = LinearLR(
    #     optimizer,
    #     start_factor=1e-6,
    #     end_factor=1,
    #     total_iters=5
    # )
    cos_scheduler = CosineAnnealingLR(
        optimizer,
        T_max = epochs,
        eta_min = 1e-5
    )
    # scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode='max',
    #     factor=0.2,
    #     patience=3,
    #     verbose=False,
    #     min_lr=1e-8,)
    # combined_scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warm_scheduler, cosine_scheduler],
    #     milestones=[5],
    # )

    # History tracking (only on rank 0)
    if rank == 0:
        history = {'train Loss': [], 'train goat': [],
                   'val Loss': [], 'val goat': []}

    # Training loop
    for epoch in range(epoch_0,epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # Important for proper shuffling

        total_loss, total_correct = 0.0, 0.0
        total_samples = 0.0

        # Only show progress bar on rank 0
        if rank == 0:
            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                ncols=120
            )
        else:
            pbar = enumerate(train_loader)
        # print(optimizer.param_groups[0]['lr'])
        for batch_idx, (batch_x, batch_y) in pbar:
            batch_x = batch_x.to(rank, non_blocking=True)
            batch_y = batch_y.to(rank, non_blocking=True)

            # Forward pass
            logits, jac_loss, n_steps, mean_zhanbi = model(batch_x)
            loss = criterion(logits, batch_y) + 1 * jac_loss

            # if epoch < 10:model.module.layer4.deq_func.gamma.requires_grad = False
            # else:model.module.layer4.deq_func.gamma.requires_grad = False
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Accumulate metrics
            batch_samples = batch_x.size(0)
            total_loss += loss.item() * batch_samples
            total_correct += (logits.argmax(dim=1) == batch_y).sum().float().item()
            total_samples += batch_samples
            train_acc = total_correct / total_samples
            train_loss = total_loss / total_samples
            if rank == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "n_steps": f"{torch.mean(n_steps).item():.4f}"
                })
        if rank == 0:
            history['train Loss'].append(train_loss)
            history['train goat'].append(train_acc)

        # cosine_scheduler.step()
        dist.barrier()

        model.eval()
        val_loss, val_correct = 0, 0
        val_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(rank, non_blocking=True)
                batch_y = batch_y.to(rank, non_blocking=True)
                logits, jac_loss, n_steps, mean_zhanbi = model(batch_x)
                loss = criterion(logits, batch_y) + 1 * jac_loss

                batch_samples = batch_x.size(0)
                val_loss += loss.item() * batch_samples
                val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                val_samples += batch_samples

            val_loss /= val_samples
            val_acc = val_correct / val_samples

        cos_scheduler.step()
        lr_list  = cos_scheduler.get_last_lr()
        current_lr = lr_list[0]
        if rank == 0:
            history['val Loss'].append(val_loss)
            history['val goat'].append(val_acc)

            print(f"Epoch [{epoch + 1}] "
                  f"Train Loss: {train_loss :.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"lr: {current_lr:.9f}")
            run.log({
                "acc": val_acc,
                "loss": val_loss,
                "n_steps": n_steps,
                "mean_zhanbi": mean_zhanbi,
            })
            # Save model
            save_path = os.path.join('../../model/listops', f'S4_fs4d_epoch_{epoch + 1}.pth')
            torch.save(model.module.state_dict(), save_path)
            loss_path = os.path.join('../../model/listops', f'S4_fs4d_loss.pth')
            torch.save(history, loss_path)
    if rank == 0:
        run.finish()
    cleanup_ddp()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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