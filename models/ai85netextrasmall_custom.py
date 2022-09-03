from torch import nn
import ai8x

class AI85NetExtraSmallCustom(nn.Module):
    """
    Minimal CNN for minimum energy per inference for MNIST
    """
    def __init__(self, num_classes=10, num_channels=1, dimensions=(28, 28),
                 fc_inputs=20, bias=False, **kwargs):
        super().__init__()

        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dReLU(in_channels=num_channels, out_channels=8, kernel_size=3,
                                          padding=1, bias=bias, **kwargs)
       
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(in_channels=8, out_channels=8, kernel_size=3,
                                                 pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= 2 

        self.conv3 = ai8x.FusedConv2dReLU(in_channels=8, out_channels=8, kernel_size=3,
                                                 padding=1,
                                                 bias=bias, **kwargs)

        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(in_channels=8, out_channels=fc_inputs, kernel_size=3,
                                                 pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= 2
      
        self.fc1 = ai8x.Linear(in_features=fc_inputs*dim*dim, out_features=num_classes, bias=True, wide=True, **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x): 
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
   
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)
     
        x = self.fc1(x)

        return x


def ai85netextrasmall_custom(pretrained=False, **kwargs):
    """
    Constructs a AI85NetExtraSmall_custom model.
    """
    assert not pretrained
    return AI85NetExtraSmallCustom(**kwargs)


models = [
    {
        'name': 'ai85netextrasmall_custom',
        'min_input': 1,
        'dim': 2,
    },
]
