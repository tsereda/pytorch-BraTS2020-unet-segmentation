import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """
    Double convolution block for 3D UNet
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
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
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.1):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_channels, out_channels, dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handling cases where dimensions don't match exactly
        diff_x = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_z = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diff_z // 2, diff_z - diff_z // 2,
                       diff_y // 2, diff_y - diff_y // 2,
                       diff_x // 2, diff_x - diff_x // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, 
                 in_channels: int = 3, 
                 num_classes: int = 4, 
                 init_features: int = 16,
                 dropout_p: float = 0.1):
        """
        3D U-Net model
        Args:
            in_channels (int): number of input channels (default: 3 for flair, t1ce, t2)
            num_classes (int): number of output classes (default: 4 for brain tumor segmentation)
            init_features (int): number of features in first layer (default: 16)
            dropout_p (float): dropout probability (default: 0.1)
        """
        super().__init__()
        
        # Save parameters
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Initialize feature numbers for each level
        features = init_features
        
        # Encoder path
        self.enc1 = ConvBlock3D(in_channels, features, dropout_p)
        self.enc2 = DownBlock3D(features, features * 2, dropout_p)
        self.enc3 = DownBlock3D(features * 2, features * 4, dropout_p)
        self.enc4 = DownBlock3D(features * 4, features * 8, dropout_p)
        
        # Bridge
        self.bridge = DownBlock3D(features * 8, features * 16, dropout_p)
        
        # Decoder path
        self.up1 = UpBlock3D(features * 16, features * 8, dropout_p)
        self.up2 = UpBlock3D(features * 8, features * 4, dropout_p)
        self.up3 = UpBlock3D(features * 4, features * 2, dropout_p)
        self.up4 = UpBlock3D(features * 2, features, dropout_p)
        
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
        return F.softmax(logits, dim=1)

    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
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
    
    # Forward pass
    output = model(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Calculate total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")