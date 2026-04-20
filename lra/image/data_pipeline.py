import os
import torch
from torch.utils.data import TensorDataset
import torchvision
from torchvision import transforms


def build_lra_image_data_pt(
    save_path="../../data/image/data.pt",
    root="./data",
    normalize=False,
    save_val=True,
):
    """
    PyTorch version of LRA image preprocessing pipeline.

    LRA image pipeline behavior matched here:
    1. Load CIFAR-10
    2. Convert RGB -> grayscale
    3. Optionally normalize by /255
    4. Use train[:90%], train[90%:], test
    5. Save as TensorDataset into data.pt

    Args:
        save_path: output .pt path
        root: CIFAR-10 download root
        normalize: whether to divide grayscale values by 255
        save_val: whether to also save val split into data.pt
    """

    # Keep PIL -> Tensor only. This gives [C,H,W] in [0,1].
    transform = transforms.ToTensor()

    train_full = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    test_full = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )

    # Stack all images
    train_imgs = torch.stack([train_full[i][0] for i in range(len(train_full))], dim=0)  # [50000, 3, 32, 32]
    test_imgs = torch.stack([test_full[i][0] for i in range(len(test_full))], dim=0)    # [10000, 3, 32, 32]

    train_labels = torch.tensor(train_full.targets, dtype=torch.long)
    test_labels = torch.tensor(test_full.targets, dtype=torch.long)

    # Match tf.image.rgb_to_grayscale behavior in spirit:
    # grayscale = 0.2989 R + 0.5870 G + 0.1140 B
    train_gray = (
        0.2989 * train_imgs[:, 0:1, :, :]
        + 0.5870 * train_imgs[:, 1:2, :, :]
        + 0.1140 * train_imgs[:, 2:3, :, :]
    )  # [50000, 1, 32, 32]

    test_gray = (
        0.2989 * test_imgs[:, 0:1, :, :]
        + 0.5870 * test_imgs[:, 1:2, :, :]
        + 0.1140 * test_imgs[:, 2:3, :, :]
    )  # [10000, 1, 32, 32]

    # torchvision ToTensor already gives [0,1].
    # The TF pipeline casts to int32 first and only divides by 255 if normalize=True.
    # For practical PyTorch training, keeping float is fine.
    # To mimic the TF switch:
    if not normalize:
        # Put values back to 0~255-like scale to match "not normalized" behavior more closely
        train_gray = train_gray * 255.0
        test_gray = test_gray * 255.0

    # Flatten to sequence form [N, 1024, 1], convenient for your SSM code
    train_seq = train_gray.permute(0, 2, 3, 1).reshape(-1, 1024, 1).contiguous()
    test_seq = test_gray.permute(0, 2, 3, 1).reshape(-1, 1024, 1).contiguous()

    # Official LRA-style split from the file you sent:
    # train[:90%], train[90%:], test
    n_train = train_seq.size(0)  # 50000
    split = int(0.9 * n_train)   # 45000

    train_dataset = TensorDataset(train_seq[:split], train_labels[:split])
    val_dataset = TensorDataset(train_seq[split:], train_labels[split:])
    test_dataset = TensorDataset(test_seq, test_labels)

    save_dict = {
        "train": train_dataset,
        "test": test_dataset,
    }
    if save_val:
        save_dict["val"] = val_dataset

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)

    print(f"Saved to: {save_path}")
    print("train shape:", train_seq[:split].shape)
    if save_val:
        print("val shape:  ", train_seq[split:].shape)
    print("test shape: ", test_seq.shape)
    print("value range train:", train_seq.min().item(), train_seq.max().item())


if __name__ == "__main__":
    build_lra_image_data_pt(
        save_path="../../data/image/data.pt",
        root="./data",
        normalize=False,   # set True if you want /255
        save_val=True
    )