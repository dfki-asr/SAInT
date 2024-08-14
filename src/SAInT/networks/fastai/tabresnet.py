from fastai.tabular.all import torch, Module, nn, SequentialEx, LinBnDrop


class TabResBlock(Module):

    def __init__(self,
                 n_in: int,
                 n_out: int,
                 bn: bool = False,
                 p: float = 0.0,
                 act: nn.Module = nn.ReLU(),
                 **kwargs):
        self.linpath = SequentialEx(
            LinBnDrop(n_in, n_out, act=act, bn=bn, p=p),
            LinBnDrop(n_out, n_out, act=act, bn=bn, p=p))
        idpath = []
        if n_in != n_out:
            idpath.append(nn.Linear(n_in, n_out))
        self.idpath = SequentialEx(*idpath)
        self.act = act

    def forward(self, xcont: torch.Tensor) -> torch.Tensor:
        output = self.act(self.linpath(xcont) + self.idpath(xcont))
        return output


class TabResNet(Module):

    def __init__(self,
                 num_features: int,
                 output_size: int,
                 bn: bool = False,
                 p: float = 0.0,
                 size: int = 256,
                 num_blocks: int = 5,
                 act: nn.Module = nn.ReLU(),
                 **kwargs):
        super().__init__()
        self.input_size = num_features
        self.output_size = output_size

        blocks = []
        for _ in range(num_blocks):
            blocks.append(TabResBlock(size, size, bn=bn, p=p, act=act))
        self.layers = SequentialEx(
            TabResBlock(self.input_size, size, bn=bn, p=p, act=act), *blocks,
            LinBnDrop(size, self.output_size))

    def forward(self, xcat: torch.Tensor, xcont: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((xcat, xcont), axis=1)
        inputs = inputs.view(-1, self.input_size)
        output = self.layers(inputs)
        return output
