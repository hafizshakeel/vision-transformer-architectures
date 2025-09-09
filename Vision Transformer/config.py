import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Vision Transformer Training Configuration")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--experiment_name", type=str, default="runs/vit_cifar10", help="Experiment name")

    # Image/Model parameters
    parser.add_argument("--num_channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--img_size", type=int, default=32, help="Image size")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--transformer_blocks", type=int, default=4, help="Number of transformer blocks")
    parser.add_argument("--mlp_hidden_dim", type=int, default=512, help="MLP hidden dimension")
    parser.add_argument("--num_attn_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_blocks", type=int, default=6, help="Number of blocks")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Checkpointing
    parser.add_argument("--model_folder", type=str, default="weights", help="Folder to save weights")
    parser.add_argument("--model_basename", type=str, default="vit_model_", help="Model file base name")
    parser.add_argument("--preload", type=str, default="latest", help="Preload weights file: 'latest' or epoch number")

    return parser.parse_args()


def get_config():
    args = parse_args()

    num_patches = (args.img_size // args.patch_size) ** 2

    config = vars(args)  # Convert argparse Namespace to dict
    config["num_patches"] = num_patches

    return config


def get_weights_file_path(config, epoch: str):
    model_folder = Path(config['model_folder'])
    model_folder.mkdir(parents=True, exist_ok=True)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = Path(config['model_folder'])
    weights_files = list(model_folder.glob(f"{config['model_basename']}*.pt"))
    if not weights_files:
        return None
    weights_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
    return str(weights_files[-1])


# if __name__ == "__main__":
#     config = get_config()
#     print("Configuration:", config)
#     print("Latest weights file:", latest_weights_file_path(config))
#     print("Path for epoch 10:", get_weights_file_path(config, "10"))
