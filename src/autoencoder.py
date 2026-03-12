"""
Autoencoder architectures for protoplanetary disk image reconstruction.

This module provides advanced autoencoder models with skip connections,
batch normalization, and accessible latent spaces for clustering analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class EncoderBlock(nn.Module):
    """Encoder block with convolution, batch norm, and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """Decoder block with transposed convolution, batch norm, and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        padding = kernel_size // 2
        output_padding = stride - 1
        
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ImprovedAutoencoder(nn.Module):
    """
    Improved autoencoder with U-Net style skip connections.
    
    Architecture:
        - Encoder: Progressive downsampling with increasing channels
        - Bottleneck: Latent representation
        - Decoder: Progressive upsampling with skip connections
        
    Input: [N, 1, 600, 600]
    Output: [N, 1, 600, 600]
    Latent: [N, latent_dim, H_latent, W_latent]
    """
    
    def __init__(self, latent_dim: int = 512, use_skip_connections: bool = True):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.use_skip_connections = use_skip_connections
        
        # Encoder
        self.enc1 = EncoderBlock(1, 32, kernel_size=3, stride=2)      # 600 -> 300
        self.enc2 = EncoderBlock(32, 64, kernel_size=3, stride=2)     # 300 -> 150
        self.enc3 = EncoderBlock(64, 128, kernel_size=3, stride=2)    # 150 -> 75
        self.enc4 = EncoderBlock(128, 256, kernel_size=3, stride=2)   # 75 -> 38 (rounded up)
        self.enc5 = EncoderBlock(256, 512, kernel_size=3, stride=2)   # 38 -> 19
        
        # Bottleneck (latent space)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec5 = DecoderBlock(latent_dim, 512, kernel_size=3, stride=2)  # 19 -> 38
        if use_skip_connections:
            self.dec4 = DecoderBlock(512 + 256, 256, kernel_size=3, stride=2)  # 38 -> 76
            self.dec3 = DecoderBlock(256 + 128, 128, kernel_size=3, stride=2)  # 76 -> 152
            self.dec2 = DecoderBlock(128 + 64, 64, kernel_size=3, stride=2)    # 152 -> 304
            self.dec1 = DecoderBlock(64 + 32, 32, kernel_size=3, stride=2)     # 304 -> 608
        else:
            self.dec4 = DecoderBlock(512, 256, kernel_size=3, stride=2)
            self.dec3 = DecoderBlock(256, 128, kernel_size=3, stride=2)
            self.dec2 = DecoderBlock(128, 64, kernel_size=3, stride=2)
            self.dec1 = DecoderBlock(64, 32, kernel_size=3, stride=2)
        
        # Final reconstruction layer
        self.final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output in range [-1, 1] for normalized images
        )
        
        # Adjustment layer to match exact input size (608 -> 600)
        self.size_adjust = nn.AdaptiveAvgPool2d((600, 600))
        
    def encode(self, x):
        """
        Encode input image to latent representation.
        
        Args:
            x: Input tensor [N, 1, 600, 600]
            
        Returns:
            Latent representation [N, latent_dim, H, W]
        """
        # Encoder with skip connections stored
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Bottleneck
        latent = self.bottleneck(e5)
        
        if self.use_skip_connections:
            return latent, (e1, e2, e3, e4)
        else:
            return latent, None
    
    def decode(self, latent, skip_connections=None):
        """
        Decode latent representation to reconstructed image.
        
        Args:
            latent: Latent tensor [N, latent_dim, H, W]
            skip_connections: Tuple of skip connection tensors (optional)
            
        Returns:
            Reconstructed image [N, 1, 600, 600]
        """
        d5 = self.dec5(latent)
        
        if self.use_skip_connections and skip_connections is not None:
            e1, e2, e3, e4 = skip_connections
            
            # Concatenate skip connections (resize skip connections to match decoder size)
            d4 = self.dec4(torch.cat([d5, self._match_size(e4, d5)], dim=1))
            d3 = self.dec3(torch.cat([d4, self._match_size(e3, d4)], dim=1))
            d2 = self.dec2(torch.cat([d3, self._match_size(e2, d3)], dim=1))
            d1 = self.dec1(torch.cat([d2, self._match_size(e1, d2)], dim=1))
        else:
            d4 = self.dec4(d5)
            d3 = self.dec3(d4)
            d2 = self.dec2(d3)
            d1 = self.dec1(d2)
        
        # Final reconstruction
        out = self.final(d1)
        
        # Adjust to exact input size
        out = self.size_adjust(out)
        
        return out
    
    def _match_size(self, skip, decoder):
        """
        Resize skip connection to match decoder feature map size.
        
        Args:
            skip: Skip connection tensor
            decoder: Decoder tensor to match size with
            
        Returns:
            Resized skip connection
        """
        if skip.shape[2:] != decoder.shape[2:]:
            skip = torch.nn.functional.interpolate(
                skip, size=decoder.shape[2:], mode='bilinear', align_corners=True
            )
        return skip
    
    def forward(self, x):
        """
        Full forward pass through autoencoder.
        
        Args:
            x: Input tensor [N, 1, 600, 600]
            
        Returns:
            Reconstructed image [N, 1, 600, 600]
        """
        latent, skip_connections = self.encode(x)
        reconstruction = self.decode(latent, skip_connections)
        return reconstruction
    
    def get_latent_features(self, x):
        """
        Extract flattened latent features for clustering.
        
        Args:
            x: Input tensor [N, 1, 600, 600]
            
        Returns:
            Flattened latent features [N, latent_dim * H * W]
        """
        latent, _ = self.encode(x)
        # Flatten spatial dimensions
        batch_size = latent.size(0)
        features = latent.view(batch_size, -1)
        return features


class SimpleAutoencoder(nn.Module):
    """
    Simple baseline autoencoder without skip connections.
    
    Useful for comparison with the improved version.
    """
    
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),     # 600 -> 300
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),    # 300 -> 150
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),    # 150 -> 75
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),   # 75 -> 38
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, 3, stride=2, padding=1),  # 38 -> 19
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 3, stride=2, padding=1, output_padding=1),  # 19 -> 38
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 38 -> 76
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 76 -> 152
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   # 152 -> 304
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),    # 304 -> 608
            nn.Tanh()
        )
        
        self.size_adjust = nn.AdaptiveAvgPool2d((600, 600))
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        reconstruction = self.size_adjust(reconstruction)
        return reconstruction
    
    def get_latent_features(self, x):
        latent = self.encoder(x)
        batch_size = latent.size(0)
        features = latent.view(batch_size, -1)
        return features


def train_autoencoder(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: str = '../models'
) -> Tuple[nn.Module, dict]:
    """
    Train the autoencoder model.
    
    Args:
        model: Autoencoder model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save the best model checkpoint
        
    Returns:
        Tuple of (trained model, history dict)
    """
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images,) in enumerate(train_loader):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, in val_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            import os
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, 'autoencoder_best.pth'))
    
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")
    
    return model, history


if __name__ == "__main__":
    # Example usage and model summary
    print("Autoencoder Module")
    print("\nImproved Autoencoder:")
    model = ImprovedAutoencoder(latent_dim=512)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nSimple Autoencoder:")
    model_simple = SimpleAutoencoder(latent_dim=256)
    print(f"  Parameters: {sum(p.numel() for p in model_simple.parameters()):,}")
