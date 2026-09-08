import torch
from torch import nn


class InputConv(nn.Module):
    def __init__(self, in_chn, out_chn, dropout_rate=0.1, norm=True, **kwargs):
        super(InputConv, self).__init__(**kwargs)

        self.norm = norm
        self.lin = nn.Conv1d(in_chn, out_chn, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_chn, eps=0.001, momentum=0.6)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if self.norm:
            x = self.dropout(self.bn(self.act(self.lin(x))))
        else:
            x = self.act(self.lin(x))
        return x


class LinLayer(nn.Module):
    def __init__(self, in_chn, out_chn, dropout_rate=0.1, **kwargs):
        super(LinLayer, self).__init__(**kwargs)

        self.lin = nn.Linear(in_chn, out_chn)
        self.bn = nn.BatchNorm1d(out_chn, eps=0.001, momentum=0.6)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.bn(self.act(self.lin(x))))
        return x


class InputProcess(nn.Module):
    def __init__(self, 
                 cpf_conv,
                 npf_conv,
                 vtx_conv, 
                 **kwargs
        ):
        
        super(InputProcess, self).__init__(**kwargs)
        
        cpf_conv_full = cpf_conv.copy()
        npf_conv_full = npf_conv.copy()
        vtx_conv_full = vtx_conv.copy()

        self.cpf_bn = nn.BatchNorm1d(cpf_conv_full[0], eps=0.001, momentum=0.6)
        self.cpf_conv = nn.ModuleList([InputConv(cpf_conv_full[i], cpf_conv_full[i+1]) for i in range(len(cpf_conv_full) - 2)])
        self.cpf_conv.append(InputConv(cpf_conv_full[-2], cpf_conv_full[-1], norm=False))

        self.npf_bn = nn.BatchNorm1d(npf_conv_full[0], eps=0.001, momentum=0.6)
        self.npf_conv = nn.ModuleList([InputConv(npf_conv_full[i], npf_conv_full[i+1]) for i in range(len(npf_conv_full) - 2)])
        self.npf_conv.append(InputConv(npf_conv_full[-2], npf_conv_full[-1], norm=False))

        self.vtx_bn = nn.BatchNorm1d(vtx_conv_full[0], eps=0.001, momentum=0.6)
        self.vtx_conv = nn.ModuleList([InputConv(vtx_conv_full[i], vtx_conv_full[i+1]) for i in range(len(vtx_conv_full) - 2)])
        self.vtx_conv.append(InputConv(vtx_conv_full[-2], vtx_conv_full[-1], norm=False))

    def forward(self, cpf, npf, vtx):
        
        cpf = self.cpf_bn(torch.transpose(cpf, 1, 2))
        for conv in self.cpf_conv:
            cpf = conv(cpf)
        cpf = torch.transpose(cpf, 1, 2)

        npf = self.npf_bn(torch.transpose(npf, 1, 2))
        for conv in self.npf_conv:
            npf = conv(npf)
        npf = torch.transpose(npf, 1, 2)

        vtx = self.vtx_bn(torch.transpose(vtx, 1, 2))
        for conv in self.vtx_conv:
            vtx = conv(vtx)
        vtx = torch.transpose(vtx, 1, 2)

        return cpf, npf, vtx


class InputProcess_HLT(nn.Module):
    def __init__(self, 
                 cpf_conv,
                 vtx_conv, 
                 **kwargs
        ):
        
        super().__init__(**kwargs)
        
        cpf_conv_full = cpf_conv.copy()
        vtx_conv_full = vtx_conv.copy()

        self.cpf_bn = nn.BatchNorm1d(cpf_conv_full[0], eps=0.001, momentum=0.6)
        self.cpf_conv = nn.ModuleList([InputConv(cpf_conv_full[i], cpf_conv_full[i+1]) for i in range(len(cpf_conv_full) - 2)])
        self.cpf_conv.append(InputConv(cpf_conv_full[-2], cpf_conv_full[-1], norm=False))

        self.vtx_bn = nn.BatchNorm1d(vtx_conv_full[0], eps=0.001, momentum=0.6)
        self.vtx_conv = nn.ModuleList([InputConv(vtx_conv_full[i], vtx_conv_full[i+1]) for i in range(len(vtx_conv_full) - 2)])
        self.vtx_conv.append(InputConv(vtx_conv_full[-2], vtx_conv_full[-1], norm=False))

    def forward(self, cpf, vtx):
        
        cpf = self.cpf_bn(torch.transpose(cpf, 1, 2))
        for conv in self.cpf_conv:
            cpf = conv(cpf)
        cpf = torch.transpose(cpf, 1, 2)

        vtx = self.vtx_bn(torch.transpose(vtx, 1, 2))
        for conv in self.vtx_conv:
            vtx = conv(vtx)
        vtx = torch.transpose(vtx, 1, 2)

        return cpf, vtx
        

class MoDInputProcess(nn.Module):
    def __init__(self, **kwargs):
        super(MoDInputProcess, self).__init__(**kwargs)

        self.cpf_bn = nn.BatchNorm1d(cpf_conv_full[0], eps=0.001, momentum=0.6)
        self.cpf_conv = nn.ModuleList([InputConv(cpf_conv_full[i], cpf_conv_full[i+1]) for i in range(len(cpf_conv_full) - 2)])
        self.cpf_conv.append(InputConv(cpf_conv_full[-2], cpf_conv_full[-1], norm=False))

        self.vtx_bn = nn.BatchNorm1d(vtx_conv_full[0], eps=0.001, momentum=0.6)
        self.vtx_conv = nn.ModuleList([InputConv(vtx_conv_full[i], vtx_conv_full[i+1]) for i in range(len(vtx_conv_full) - 2)])
        self.vtx_conv.append(InputConv(vtx_conv_full[-2], vtx_conv_full[-1], norm=False))

    def forward(self, cpf, vtx):
        
        cpf = self.cpf_bn(torch.transpose(cpf, 1, 2))
        for conv in self.cpf_conv:
            cpf = conv(cpf)
        cpf = torch.transpose(cpf, 1, 2)

        vtx = self.vtx_bn(torch.transpose(vtx, 1, 2))
        for conv in self.vtx_conv:
            vtx = conv(vtx)
        vtx = torch.transpose(vtx, 1, 2)

        return cpf, vtx
        

class DenseClassifier(nn.Module):
    def __init__(self, dense_clas_dim_full, **kwargs):
        super(DenseClassifier, self).__init__(**kwargs)

        self.LinLayers = nn.ModuleList([
            LinLayer(dense_clas_dim_full[i], dense_clas_dim_full[i+1]) for i in range(len(dense_clas_dim_full) - 1)
        ])

    def forward(self, x):

        for layer in self.LinLayers:
            x = layer(x)
        
        return x


class MoDDenseClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(MoDDenseClassifier, self).__init__(**kwargs)

        self.LinLayer1 = LinLayer(268, 200)
        self.LinLayer2 = LinLayer(200, 100)
        self.LinLayer3 = LinLayer(100, 100)
        self.LinLayer4 = LinLayer(100, 100)
        self.LinLayer5 = LinLayer(100, 100)
        self.LinLayer6 = LinLayer(100, 100)
        self.LinLayer7 = LinLayer(100, 100)
        self.LinLayer8 = LinLayer(100, 100)

    def forward(self, x):
        x = self.LinLayer1(x)
        x = self.LinLayer2(x)
        x = self.LinLayer3(x)
        x = self.LinLayer4(x)
        x = self.LinLayer5(x)
        x = self.LinLayer6(x)
        x = self.LinLayer7(x)
        x = self.LinLayer8(x)

        return x
