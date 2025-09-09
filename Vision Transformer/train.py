import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import VisionTransformer
from config import get_config, get_weights_file_path, latest_weights_file_path
from tqdm import tqdm


# Dataset Preparation
def get_ds(config):
    # Define data augmentation and preprocessing for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(config["img_size"], padding=4),  # Random cropping with padding
        transforms.RandomHorizontalFlip(),  # Random horizontal flipping
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),  # Normalize to [-1, 1]
    ])

    # Define preprocessing for validation/testing (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    # Load CIFAR-10 training and validation datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Wrap datasets in DataLoader for batching, shuffling, and parallel loading
    train_data = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_data = torch.utils.data.DataLoader(valset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    return train_data, val_data


# Model Initialization
def get_model(config):
    # Initialize Vision Transformer model with hyperparameters from config
    vit = VisionTransformer(
        config['num_channels'], config['embed_dim'], config['patch_size'], config['num_patches'],
        config['num_attn_heads'], config['mlp_hidden_dim'], config['num_blocks'], config['num_classes'],
        dropout=config["dropout"]
    )
    return vit


# Model Evaluation (Accuracy)
def check_accuracy(loader, model, device):
    # Function to evaluate model accuracy on a dataset (train/val)
    correct = 0
    total = 0
    model.eval()  # Set model to evaluation mode (disable dropout, BN updates)
    with torch.no_grad():  # No gradient calculation for evaluation
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)  # Get predicted class
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()  # Set model back to training mode
    return 100.0 * correct / total  # Return accuracy percentage


# Training Loop
def train(config):
    # Select device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create model checkpoint folder if it doesn’t exist
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # Load training and validation data
    train_loader, val_loader = get_ds(config)
    model = get_model(config).to(device)

    # Define optimizer, learning rate scheduler, and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    loss_fn = nn.CrossEntropyLoss()

    # TensorBoard writer for logging metrics
    writer = SummaryWriter(config["experiment_name"])


    # Resume from Checkpoint (if available)
    initial_epoch = 0
    global_step = 0
    model_filename = latest_weights_file_path(config)
    if config["preload"] == "latest" and model_filename:
        print(f"Resuming from checkpoint: {model_filename}")
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
    else:
        print("No checkpoint found, training from scratch.")


    # Epoch Loop
    for epoch in range(initial_epoch, config["num_epochs"]):
        running_loss = 0.0
        running_acc = 0.0

        # Progress bar for batches
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{config['num_epochs']}]")
        for images, targets in loop:
            images, targets = images.to(device), targets.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate batch accuracy
            _, preds = outputs.max(1)
            correct = (preds == targets).sum().item()
            acc = correct / targets.size(0)

            # Update metrics
            running_loss += loss.item()
            running_acc += acc

            # Update progress bar and TensorBoard logs
            loop.set_postfix(acc=acc, loss=loss.item())
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/accuracy", acc, global_step)
            global_step += 1

        # Step learning rate scheduler after each epoch
        scheduler.step()

        # Calculate average training loss and accuracy
        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = running_acc / len(train_loader)

        # Evaluate on validation set
        val_acc = check_accuracy(val_loader, model, device)

        # Print results
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {avg_train_acc * 100:.2f}%")
        print(f"Validation Accuracy: {val_acc:.2f}%\n")

        # Log validation accuracy
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.flush()

        # Save model checkpoint
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        }, model_filename)

    # Close TensorBoard writer after training finishes
    writer.close()


if __name__ == "__main__":
    # Load training configuration
    config = get_config()
    train(config)
