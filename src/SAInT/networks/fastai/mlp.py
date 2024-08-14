from collections import OrderedDict
from fastai.tabular.all import torch, Module, nn


class MLP(Module):

    def __init__(self,
                 num_features: int,
                 output_size: int,
                 hidden_size: list,
                 p: float = 0.0,
                 bias: bool = True,
                 **kwargs):
        super().__init__()
        self.input_size = num_features
        self.output_size = output_size
        if len(hidden_size) < 1:
            raise RuntimeError("MLP Error: hidden_size must not be empty!")
        num_hidden = len(hidden_size) - 1
        layer_dict = OrderedDict()
        layer_dict['input_layer'] = nn.Linear(self.input_size,
                                              hidden_size[0],
                                              bias=bias)
        num_hidden = len(hidden_size) - 1
        for i in range(0, num_hidden):
            layer_dict['relu_' + str(i)] = nn.ReLU()
            layer_dict['dense_' + str(i)] = nn.Linear(hidden_size[i],
                                                      hidden_size[i + 1],
                                                      bias=bias)
        layer_dict['relu_' + str(num_hidden)] = nn.ReLU()
        if p > 0:
            layer_dict['dropout_' + str(num_hidden)] = nn.Dropout(p=p)
        layer_dict['output_layer'] = nn.Linear(hidden_size[num_hidden],
                                               output_size,
                                               bias=bias)
        self.layers = nn.Sequential(layer_dict)

    def forward(self, xcat: torch.Tensor, xcont: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((xcat, xcont), axis=1)
        inputs = inputs.view(-1, self.input_size)
        output = self.layers(inputs)
        return output
