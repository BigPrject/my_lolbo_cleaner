import torch
from collections import OrderedDict


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class _LinearBlock(torch.nn.Sequential):
    def __init__(self, input_dim, output_dim, swish,spectral_norm=False):
        layers = OrderedDict()
        
        # Fully connected layer
        fc = torch.nn.Linear(input_dim, output_dim)
        if spectral_norm:
            fc = torch.nn.utils.spectral_norm(fc)  # Apply spectral normalization
        layers["fc"] = fc
        
        if swish:
            layers["swish"] = Swish()
        else:
            layers["norm"] = torch.nn.BatchNorm1d(output_dim)
            layers["relu"] = torch.nn.ReLU(inplace=True)
        
        # init layers
        super().__init__(layers)


class DenseNetwork(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dims, swish=True):
        if isinstance(hidden_dims, str):
            model, output_dim = image_classifier(hidden_dims)
            layers = model
            self.output_dim = output_dim
        else:
            prev_dims = [input_dim] + list(hidden_dims[:-1])
            layers = OrderedDict([
                (f"hidden{i + 1}", _LinearBlock(prev_dim, current_dim, swish=swish,spectral_norm=True))
                for i, (prev_dim, current_dim) in enumerate(zip(prev_dims, hidden_dims))
            ])
            self.output_dim = hidden_dims[-1]

        super().__init__(layers)
