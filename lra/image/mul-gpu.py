import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import SSM_function as sf
import torch.optim as optim
from tqdm.auto import tqdm
import os
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import random
import numpy as np
import fssm
from torch.amp import GradScaler, autocast
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
        self.embeddind = nn.Linear(1, emb_dim)
        # self.layer1 = sf.S4D_Block(hidden_size, step, activation, emb_dim, dropout=0.2, skip=True, norm='LN')
        # self.layer2 = sf.S4D_Block(hidden_size, step, activation, emb_dim, dropout=0.2, skip=True, norm='LN')
        # self.layer3 = sf.S4D_Block(hidden_size, step, activation, emb_dim, dropout=0.2, skip=True, norm='LN')
        self.layer4 = fssm.FS4Ddeq_model(hidden_size, step, activation, emb_dim, layers=6,
                                         final_act='gelu', skip=False, norm='LN', dropout=0.0,
                                         input_size=[batch_size, len, emb_dim],
                                         feed_size=[batch_size, len, emb_dim],
                                         feed_act='tanh',
                                         )
        # self.layer4 = sf.S4D_Block(hidden_size, step, activation, emb_dim, dropout=0.2, skip=True, norm='LN')
        # self.layer5 = sf.S4D_Block(hidden_size, step, activation, emb_dim, dropout=0.2, skip=True, norm='LN')
        # self.layer6 = sf.S4D_Block(hidden_size, step, activation, emb_dim, dropout=0.2, skip=True, norm='LN')
        self.fc = nn.Linear(emb_dim, 10)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        r = self.dropout(self.embeddind(x))
        # y1 = self.layer1(r)
        # y2 = self.layer2(y1)
        # y3 = self.layer3(y2)
        y4, loss, h, n_steps = self.layer4(r)

        mean_y = torch.mean(y4, dim=1)
        logits = self.fc(mean_y)
        return logits, loss, h, n_steps

def train(rank, world_size, args):
    """Main training function for each process"""

    # Setup DDP
    setup_ddp(rank, world_size)
    set_seed(40)  # Different seed per rank for data shuffling
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    # Load data
    data = torch.load('../../data/image/data.pt',weights_only=False)
    train_data = data['train']
    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [45000, 5000])
    # Hyperparameters
    batch_size = 16
    hidden_size = 64
    step = 0.001
    emb_dim = 512
    r_len = 1024
    activation = 'tanh'
    epochs = 200
    epoch_0 = 0

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
    model = Model(batch_size, r_len, emb_dim, hidden_size, step, activation)
    # model.load_state_dict(torch.load('../../model/listops/S4_fs4d_epoch_50.pth'),strict=False)
    model = model.to(rank)

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Setup optimizer
    ssm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'ssm' in name:
            ssm_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW([
        {'params': ssm_params, 'lr': 1e-4},
        {'params': other_params, 'lr': 1e-4},
    ], weight_decay=1e-2)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.3)

    # Learning rate schedulers
    warm_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1,
        total_iters=20
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warm_scheduler, cosine_scheduler],
        milestones=[20]
    )

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
            logits, h_loss, h, feed_h = model(batch_x.unsqueeze(-1))
            loss = criterion(logits, batch_y) + 0.3 * h_loss

            # if epoch < 10:model.module.layer4.deq_func.gamma.requires_grad = False
            # else:model.module.layer4.deq_func.gamma.requires_grad = False
            # Backward pass
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Accumulate metrics
            batch_samples = batch_x.size(0)
            total_loss += loss.item() * batch_samples
            total_correct += (logits.argmax(dim=1) == batch_y).sum().float().item()
            total_samples += batch_samples
            train_acc = total_correct / total_samples
            train_loss = total_loss / batch_samples
            if rank == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "loss_h": f"{h_loss.item():.4f}",
                    "n_steps": f"{torch.mean(feed_h).item():.4f}"
                })
        if rank == 0:
            history['train Loss'].append(train_loss)
            history['train goat'].append(train_acc)
        # combined_scheduler_step(epoch)
        # cosine_scheduler.step()
        scheduler.step()
        dist.barrier()
        dist.barrier()
        # Gather metrics from all processes
        # total_loss_tensor = torch.tensor([total_loss], dtype=torch.float32, device=device)
        # total_correct_tensor = torch.tensor([total_correct], dtype=torch.float32, device=device)
        # total_samples_tensor = torch.tensor([total_samples], dtype=torch.float32, device=device)
        # dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        # dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        # dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

        model.eval()
        val_loss, val_correct = 0, 0
        val_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(rank, non_blocking=True)
                batch_y = batch_y.to(rank, non_blocking=True)
                logits, h_loss, _, _ = model(batch_x.unsqueeze(-1))
                loss = criterion(logits, batch_y) + 0.3 * h_loss

                batch_samples = batch_x.size(0)
                val_loss += loss.item() * batch_samples
                val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                val_samples += batch_samples

            val_loss /= val_samples
            val_acc = val_correct / val_samples

        # scheduler.step(val_acc)
        lr_list  = scheduler.get_last_lr()
        current_lr = lr_list[0]
        if rank == 0:
            history['val Loss'].append(val_loss)
            history['val goat'].append(val_acc)

            print(f"Epoch [{epoch + 1}] "
                  f"Train Loss: {train_loss / 704.0:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"lr: {current_lr:.9f}")
            # Save model
            save_path = os.path.join('../../model/image', f'S4_fs4d_epoch_{epoch + 1}.pth')
            torch.save(model.module.state_dict(), save_path)
            loss_path = os.path.join('../../model/image', f'S4_fs4d_loss.pth')
            torch.save(history, loss_path)

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