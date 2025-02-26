import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """
    Double convolution block for 3D UNet
    Modified to match Keras implementation: no batch norm, added bias
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock3D(nn.Module):
    """
    Downscaling block with maxpool, double conv
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock3D(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock3D(nn.Module):
    """
    Upscaling block with transposed conv, concatenation and double conv
    Simplified dimension handling to match Keras approach
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.1):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_channels, out_channels, dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Simplified dimension handling (Keras doesn't specifically handle this)
        # If dimensions don't match, we'll use center cropping or padding as needed
        if x2.size()[2:] != x1.size()[2:]:
            # Pad x1 if needed
            diff_z = x2.size()[2] - x1.size()[2]
            diff_y = x2.size()[3] - x1.size()[3]
            diff_x = x2.size()[4] - x1.size()[4]
            
            x1 = F.pad(x1, [
                max(0, diff_z // 2), max(0, diff_z - diff_z // 2),
                max(0, diff_y // 2), max(0, diff_y - diff_y // 2),
                max(0, diff_x // 2), max(0, diff_x - diff_x // 2)
            ])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, 
                 in_channels: int = 3, 
                 num_classes: int = 4, 
                 init_features: int = 16):
        """
        3D U-Net model aligned with Keras implementation
        Args:
            in_channels (int): number of input channels (default: 3 for flair, t1ce, t2)
            num_classes (int): number of output classes (default: 4 for brain tumor segmentation)
            init_features (int): number of features in first layer (default: 16)
        """
        super().__init__()
        
        # Save parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Initialize feature numbers for each level
        features = init_features
        
        # Encoder path with Keras-matched dropout rates
        self.enc1 = ConvBlock3D(in_channels, features, dropout_p=0.1)
        self.enc2 = DownBlock3D(features, features * 2, dropout_p=0.1)
        self.enc3 = DownBlock3D(features * 2, features * 4, dropout_p=0.2)
        self.enc4 = DownBlock3D(features * 4, features * 8, dropout_p=0.2)
        
        # Bridge
        self.bridge = DownBlock3D(features * 8, features * 16, dropout_p=0.3)
        
        # Decoder path with Keras-matched dropout rates
        self.up1 = UpBlock3D(features * 16, features * 8, dropout_p=0.2)
        self.up2 = UpBlock3D(features * 8, features * 4, dropout_p=0.2)
        self.up3 = UpBlock3D(features * 4, features * 2, dropout_p=0.1)
        self.up4 = UpBlock3D(features * 2, features, dropout_p=0.1)
        
        # Final convolution
        self.final_conv = nn.Conv3d(features, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # Bridge
        bridge = self.bridge(enc4)
        
        # Decoder
        dec4 = self.up1(bridge, enc4)
        dec3 = self.up2(dec4, enc3)
        dec2 = self.up3(dec3, enc2)
        dec1 = self.up4(dec2, enc1)
        
        # Final convolution and softmax
        logits = self.final_conv(dec1)
        return logits  # Note: softmax is applied in loss function, not here

    def initialize_weights(self):
        """Initialize model weights using He uniform initialization (like Keras)"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Example usage and testing
if __name__ == "__main__":
    # Create a sample input tensor
    batch_size = 2
    channels = 3
    depth = 128
    height = 128
    width = 128
    
    x = torch.randn(batch_size, channels, depth, height, width)
    
    # Initialize model
    model = UNet3D(in_channels=channels, num_classes=4)
    model.initialize_weights()
    
    # Forward pass
    output = model(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Calculate total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")