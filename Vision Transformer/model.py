"""
An Image is Worth 16x16 Words — Implementation

Paper: https://arxiv.org/abs/2010.11929

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com

"""

import torch
import torch.nn as nn
from torchinfo import summary

# Step 1: Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, num_channels, embed_dim, patch_size):
        super().__init__()
        # Split image into non-overlapping patches and project them to embeddings
        # Note: Kernel and stride must be equal to the patch size
        self.patch_embed = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.patch_embed(x)    # (B, embed_dim, H_patches, W_patches)
        x = x.flatten(2)           # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)      # (B, num_patches, embed_dim)
        return x

# Step 2: Transformer Encoder Block
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_attn_heads, mlp_hidden_dim, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)  # First LayerNorm before Attention
        # Multi-head Self Attention (batch_first=True keeps shape as (B, N, D))
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_attn_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)  # Second LayerNorm before MLP

        # Feedforward MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (B, num_patches, embed_dim)
        residual = x
        x = self.norm1(x)  # (B, num_patches, embed_dim)

        # Multi-head attention expects (B, N, D) when batch_first=True
        attn_out, _ = self.attn(x, x, x)  # Self-attention: Query=Key=Value=x
        # attn_out shape: (B, num_patches, embed_dim)

        x = residual + self.dropout(attn_out)  # (B, num_patches, embed_dim)
        residual = x

        x = self.norm2(x)  # (B, num_patches, embed_dim)
        x = self.mlp(x)  # (B, num_patches, embed_dim)

        x = residual + x  # (B, num_patches, embed_dim)
        return x


# Step 3: MLP Head for classification
class MLP_Head(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, embed_dim)
        x = self.norm(x)  # (batch_size, embed_dim)
        x = self.dropout(x)
        x = self.fc(x)    # (batch_size, num_classes)
        return x


# Step 4: Full Vision Transformer (ViT) model
class VisionTransformer(nn.Module):
    def __init__(self, num_channels, embed_dim, patch_size, num_patches, num_attn_heads, mlp_hidden_dim, num_blocks, num_classes, dropout):
        super().__init__()

        self.patch_embedding = PatchEmbedding(num_channels, embed_dim, patch_size)  # Patch Embedding
        # Learnable CLS token (1 token for classification per batch)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # (1, 1, embed_dim) or (B, 1 CLS token, embed_dim)
        # Learnable positional embeddings (CLS token + patches)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1 , embed_dim))  # (1, 1 + num_patches, embed_dim)

        self.dropout = nn.Dropout(dropout)
        # Transformer Encoder Blocks (stacked)
        self.transformer_blocks = nn.Sequential(
            *[TransformerEncoder(embed_dim, num_attn_heads, mlp_hidden_dim, dropout) for _ in range(num_blocks)]
        )

        # Step 3: MLP Head for classification
        self.mlp_heads = MLP_Head(embed_dim, num_classes, dropout)

    def forward(self, x):
        # Patchify the image and embed patches
        x = self.patch_embedding(x)  # (batch_size, num_patches, embed_dim)

        # Prepare CLS token for each batch
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1 CLS token/img, embed_dim) and
        # after expansion, this will create batch number of copies of the CLS token, one per image in the batch.

        # Concatenate CLS token with patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)  # (1, 1+N, D) or (batch_size, 1 + num_patches, embed_dim)

        # Add positional embeddings
        x = self.dropout(x + self.position_embedding)  # (batch_size, 1 + num_patches, embed_dim)

        # Apply Transformer blocks
        x = self.transformer_blocks(x)  # (batch_size, 1 + num_patches, embed_dim)

        # Extract CLS token output at position 0
        cls_token_final = x[:, 0]  # (batch_size, embed_dim)

        # Classification head
        logits = self.mlp_heads(cls_token_final)  # (batch_size, num_classes)

        return logits  # Unnormalized scores


# testing and debugging - CIFAR-10 dataset
# patches = PatchEmbedding(3, 128, 4)
# x = torch.randn(1, 3, 32, 32)  # img dim
# print(patches(x).shape)  # (1, 128, 8, 8)
# print(patches(x).flatten(2).shape)  # (1, 128, 64)
# print(patches(x).flatten(2).transpose(2, 1).shape)  # (1, 64, 128)



# Step 5: Test Model's Output
# def test_vit_model():
#     # Example hyperparameters
#     num_channels = 3       # RGB images
#     embed_dim = 128
#     patch_size = 4          # e.g., 32x32 image with 4x4 patches --> 8x8 = 64 patches
#     image_size = 32
#     num_patches = (image_size // patch_size) ** 2  # 8x8 = 64
#     num_attn_heads = 4
#     mlp_hidden_dim = 256
#     num_blocks = 2
#     num_classes = 10
#
#     # Instantiate the model
#     vit = VisionTransformer(num_channels, embed_dim, patch_size, num_patches,
#                              num_attn_heads, mlp_hidden_dim, num_blocks, num_classes)
#     dummy_images = torch.randn(8, 3, 32, 32)  # (batch_size, channels, H, W)
#     logits = vit(dummy_images)
#     print(f"Output logits shape: {logits.shape}")  # (batch, num_classes)
#
#     # summary(vit, input_size=(8, 3, 32, 32))  # Summary of the architecture
#
#
# # Run the test
# test_vit_model()

