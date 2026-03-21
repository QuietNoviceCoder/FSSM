import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ReduceLROnPlateau
from tqdm.auto import tqdm
import yaml
import fssm


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(output_dir, level="INFO"):
    """Setup logging to both file and console."""
    # Create logs directory
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = logs_dir / "training.log"

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set up logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=[file_handler, console_handler]
    )

    return str(log_file)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_output_directory(config):
    """Create unique output directory for this experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config['experiment']['name']
    output_dir = Path("model/text") / f"{exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def save_config(config, output_dir, config_path):
    """Save the configuration file to output directory."""
    config_save_path = Path(output_dir) / "config.yaml"
    shutil.copy2(config_path, config_save_path)
    logging.info(f"Configuration saved to: {config_save_path}")


def get_device(device_config):
    """Get the appropriate device."""
    if device_config.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_config)
    else:
        return torch.device("cpu")


class TextClassificationModel(nn.Module):
    def __init__(self, config):
        super(TextClassificationModel, self).__init__()
        model_config = config['model']
        data_config = config['data']

        self.pad_id = 0
        self.embedding = nn.Embedding(
            model_config['embedding_vocab_size'],
            embedding_dim=model_config['emb_dim']
        )

        self.ssm1 = fssm.FSSM_model(
            model_config['hidden_size'],
            model_config['step'],
            model_config['activation'],
            data_config['text_len'],
            model_config['emb_dim'],
            mid_layers=model_config['fssm']['mid_layers'],
            skip=model_config['fssm']['skip'],
            dropout=model_config['fssm']['dropout'],
            norm=model_config['fssm']['norm'],
            h=model_config['fssm']['h'],
            final_act=model_config['fssm']['final_act'],
            input_size=[data_config['batch_size'], data_config['text_len'], model_config['emb_dim']],
            feed_size=[data_config['batch_size'], data_config['text_len'], model_config['emb_dim']],
            feed_act=model_config['fssm']['feed_act'],
            use_flash=model_config['fssm']['use_flash'],
        )

        self.fc = nn.Linear(model_config['emb_dim'], model_config['num_classes'])

    def forward(self, x):
        mask = (x != self.pad_id).float()
        r = self.embedding(x)
        y1, H1 = self.ssm1(r)

        mask = mask.unsqueeze(-1)
        y = y1 * mask
        # mean pooling
        sum_y = y.sum(dim=1)
        len_y = mask.sum(dim=1)  # [batch, 1]
        mean_y = sum_y / (len_y + 1e-8)
        logits = self.fc(mean_y)
        return logits, H1


def create_optimizer(model, config):
    """Create optimizer with different learning rates for different parameter groups."""
    optimizer_config = config['training']['optimizer']
    ssm_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'fssm' in name:
            ssm_params.append(param)
        else:
            other_params.append(param)

    if optimizer_config['type'] == 'AdamW':
        optimizer = optim.AdamW([
            {'params': ssm_params, 'lr': optimizer_config['ssm_lr']},
            {'params': other_params, 'lr': optimizer_config['other_lr']},
        ], weight_decay=optimizer_config['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")

    return optimizer


def create_scheduler(optimizer, config):
    """Create learning rate scheduler."""
    scheduler_config = config['training']['scheduler']
    epochs = config['training']['epochs']

    if scheduler_config['type'] == 'sequential':
        warm_scheduler = LinearLR(
            optimizer,
            start_factor=scheduler_config['warmup_start_factor'],
            end_factor=1.0,
            total_iters=scheduler_config['warmup_epochs']
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs-scheduler_config['warmup_start_factor'],
            eta_min=scheduler_config['cosine_eta_min'],
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warm_scheduler, cosine_scheduler],
            milestones=[scheduler_config['warmup_epochs']]
        )
    elif scheduler_config['type'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, epochs)
    elif scheduler_config['type'] == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config['plateau_mode'],
            factor=scheduler_config['plateau_factor'],
            threshold=scheduler_config['plateau_threshold'],
            patience=scheduler_config['plateau_patience'],
            verbose=True
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")

    return scheduler


def load_data(config):
    """Load and prepare data loaders."""
    data_config = config['data']

    # Load data
    Idmb = torch.load(data_config['data_path'], weights_only=False)

    # Set format
    Idmb['train'].set_format(type='torch', columns=['input_ids', 'label'])
    Idmb['test'].set_format(type='torch', columns=['input_ids', 'label'])

    # Create data loaders
    train_loader = DataLoader(
        Idmb['train'].select(range(data_config['total_samples'])),
        data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
    )
    test_loader = DataLoader(
        Idmb['test'].select(range(data_config['test_samples'])),
        data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device, config, epoch):
    """Train for one epoch."""
    model.train()
    total_loss, total_correct = 0, 0
    total_samples = config['data']['total_samples']
    batch_size = config['data']['batch_size']
    h_loss_weight = config['training']['loss']['h_loss_weight']

    pbar = tqdm(
        enumerate(train_loader),
        total=total_samples // batch_size,
        desc=f"Epoch {epoch + 1}/{config['training']['epochs']}",
        unit="batch"
    )

    for batch_idx, train in pbar:
        batch_x, batch_y = train['input_ids'].to(device), train['label'].to(device).long()

        # Forward pass
        logits, H1 = model(batch_x)
        loss1 = criterion(logits, batch_y)
        loss2 = fssm.loss_h(H1, h_loss_weight)
        loss = loss1 + loss2

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Batch": f"{batch_idx + 1}/{total_samples // batch_size}"
        })

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, test_loader, criterion, device, config):
    """Evaluate the model."""
    model.eval()
    test_loss, test_correct = 0, 0
    test_samples = config['data']['test_samples']
    h_loss_weight = config['training']['loss']['h_loss_weight']

    with torch.no_grad():
        for test in test_loader:
            batch_x, batch_y = test['input_ids'].to(device), test['label'].to(device)
            logits, H1 = model(batch_x)
            loss1 = criterion(logits, batch_y)
            loss2 = fssm.loss_h(H1, h_loss_weight)
            loss = loss1 + loss2
            test_loss += loss.item() * batch_x.size(0)
            test_correct += (logits.argmax(dim=1) == batch_y).sum().item()

    return test_loss / test_samples, test_correct / test_samples


def save_model(model, epoch, output_dir, is_final=False):
    """Save model checkpoint."""
    model_dir = Path(output_dir) / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)

    if is_final:
        save_path = model_dir / f"final_model.pth"
    else:
        save_path = model_dir / f"epoch_{epoch + 1}.pth"

    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved to: {save_path}")


def save_training_history(history, output_dir):
    """Save training history."""
    history_path = Path(output_dir) / "training_history.pth"
    torch.save(history, history_path)
    logging.info(f"Training history saved to: {history_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train FSSM text classification model")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create output directory
    output_dir = create_output_directory(config)

    # Setup logging
    setup_logging(output_dir, config.get('logging', {}).get('level', 'INFO'))

    # Save configuration
    save_config(config, output_dir, args.config)

    # Log experiment info
    logging.info(f"Starting experiment: {config['experiment']['name']}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Configuration loaded from: {args.config}")

    # Set seed for reproducibility
    set_seed(config['experiment']['seed'])
    logging.info(f"Random seed set to: {config['experiment']['seed']}")

    # Setup device
    device = get_device(config['training']['device'])
    logging.info(f"Using device: {device}")

    # Load data
    logging.info("Loading data...")
    train_loader, test_loader = load_data(config)
    logging.info(f"Training samples: {config['data']['total_samples']}")
    logging.info(f"Test samples: {config['data']['test_samples']}")

    # Create model
    logging.info("Creating model...")
    model = TextClassificationModel(config)
    model.to(device)

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }

    # Training loop
    logging.info("Starting training...")
    epochs = config['training']['epochs']
    save_interval = config['saving']['save_interval']

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, config, epoch
        )

        # Update scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(train_acc)
        else:
            scheduler.step()

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, config)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Log results
        logging.info(f"Epoch [{epoch + 1}/{epochs}] "
                     f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                     f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

        # Save model checkpoint
        if (epoch + 1) % save_interval == 0:
            save_model(model, epoch, output_dir)

    # Save final model
    save_model(model, epochs - 1, output_dir, is_final=True)

    # Save training history
    if config.get('logging', {}).get('save_history', True):
        save_training_history(history, output_dir)

    logging.info(f"Training completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()