import torch
from collections import OrderedDict
import torch.nn as nn

class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class _LinearBlock(torch.nn.Sequential):
    def __init__(self, input_dim, output_dim, swish,spectral_norm=False):
        layers  = OrderedDict()
        
        fc = torch.nn.Linear(input_dim, output_dim)
        if spectral_norm:
            print("Spectral Normalization is on")
            fc = torch.nn.utils.spectral_norm(fc)
        else:
            print("No Spectral Normalization")
        layers["fc"] = fc
        
        if swish:
            layers["swish"] = Swish()
        else:
            layers['norm'] = torch.nn.BatchNorm1d(output_dim)
            layers['relu'] = torch.nn.RelU(inplace=True)
        super().__init__(layers)


class DenseNetwork(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dims, swish=True,spectral_norm=False):
        if isinstance(hidden_dims, str):
            model, output_dim = image_classifier(hidden_dims)
            layers = model
            self.output_dim = output_dim
        else:
            layers = OrderedDict()
            prev_dims = [input_dim] + list(hidden_dims[:-1])
            for i,(prev_dim,current_dim) in enumerate(zip(prev_dims,hidden_dims)):
                layers[f"hidden{i+1}"]  =  _LinearBlock(prev_dim, current_dim, swish=swish,spectral_norm=spectral_norm)
                
            self.output_dim = hidden_dims[-1]

        super().__init__(layers)
