import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import torch.utils.checkpoint as cp
from einops import rearrange
from torch import nn, einsum
from mmcv.cnn import build_conv_layer

def expand_dim(t, dim, k):
        t = t.unsqueeze(dim = dim)
        expand_shape = [-1] * len(t.shape)
        expand_shape[dim] = k
        return t.expand(*expand_shape)

def rel_to_abs(x):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim = 3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l-1):]
    return final_x

def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = expand_dim(logits, dim = 3, k = h)
    return logits

class LESA(nn.Module):
    def __init__(
        self, 
        lesa,
        in_planes, 
        out_planes, 
        kernel_size=56,
        stride=1, 
        bias=False,
        dcn=None,
        **kwargs,
    ):
        groups = lesa.groups
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super().__init__()
        
        assert lesa['type'] == 'LESA'
        self.lesa = lesa
        self.with_cp = lesa.with_cp_UB_terms_only
        self.fmap_size = kernel_size
        self.branch_planes = out_planes

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.qk_planes = out_planes // groups // 2
        self.v_planes = self.branch_planes // groups
        kernel_size = kernel_size ** 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        
        # Multi-head self attention
        self.qkv_transform = nn.Conv1d(
            in_planes, 
            (self.out_planes+self.branch_planes), 
            kernel_size=1, 
            stride=1,
            padding=0,
            groups=1,
            bias=False,
        )
        self.bn_qkv = nn.BatchNorm1d(self.out_planes+self.branch_planes)

        if lesa.pe_type == 'classification':
            self.bn_similarity = nn.BatchNorm2d(groups * 3)
            self.bn_output = nn.BatchNorm1d(self.branch_planes * 2)
        elif lesa.pe_type == 'detection_qr':
            self.bn_output = nn.BatchNorm1d(self.branch_planes)
            self.bn_similarity = nn.BatchNorm2d(groups * 2)
        else:
            raise NotImplementedError

        ReaPlanes = self.branch_planes

        if dcn is not None:
            x_layers = [
                build_conv_layer(
                    dcn,
                    in_planes,
                    self.branch_planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                    groups=lesa.groups,
                    bias=False,
                )
            ]
        else:
            x_layers = [
                nn.Conv2d(
                    in_planes, self.branch_planes, 
                    kernel_size=3,
                    padding=1,
                    groups=lesa.groups,
                    bias=False,
                )
            ]
        
        if lesa.groups != 1:
            x_layers += [
                nn.Conv2d(
                    self.branch_planes,
                    self.branch_planes,
                    kernel_size=1,
                    bias=False,
                )
            ]
        self.x_transform = nn.Sequential(*x_layers)
        self.bn_x = nn.BatchNorm2d(self.branch_planes)
        
        r_layers = []
        InChannels = self.branch_planes*2
        r_layers += [nn.ReLU(inplace=True)]
        for n_idx in range(len(lesa.df_channel_shrink)):
            r_layers += [
                nn.Conv2d(
                    InChannels,
                    int(InChannels/lesa.df_channel_shrink[n_idx]),
                    kernel_size=lesa.df_kernel_size[n_idx],
                    padding=(lesa.df_kernel_size[n_idx]-1)//2,
                    groups=lesa.df_group[n_idx],
                    bias=False,
                ),
                nn.BatchNorm2d(int(InChannels/lesa.df_channel_shrink[n_idx])),
                nn.ReLU(inplace=True),
            ]
            InChannels = int(InChannels/lesa.df_channel_shrink[n_idx])
        
        self.reasoning = nn.Sequential(*r_layers)
        TarPlanes = ReaPlanes
        proj_layers = []
        proj_layers.append(
            nn.Conv2d(
                InChannels, 
                TarPlanes,
                kernel_size=lesa.df_kernel_size[-1],
                groups=lesa.df_group[-1],
                bias=False,
            ),
        )
        proj_layers.append(nn.BatchNorm2d(TarPlanes))
        self.projection = nn.Sequential(*proj_layers)
        
        # Position embedding
        if lesa.pe_type == 'classification':
            
            self.pe_dim = self.qk_planes*2+self.v_planes

            self.relative = nn.Parameter(
                torch.randn(self.pe_dim, kernel_size * 2 - 1), 
                requires_grad=True,
            )

            query_index = torch.arange(kernel_size).unsqueeze(0)
            key_index = torch.arange(kernel_size).unsqueeze(1)
            relative_index = key_index - query_index + kernel_size - 1
            self.register_buffer('flatten_index', relative_index.view(-1))
        
        elif lesa.pe_type == 'detection_qr':

            self.pe_dim = self.qk_planes
            scale = self.pe_dim ** -0.5

            self.rel_height = nn.Parameter(
                torch.randn(self.fmap_size * 2 - 1, self.pe_dim) * scale
            )
            self.rel_width = nn.Parameter(
                torch.randn(self.fmap_size * 2 - 1, self.pe_dim) * scale
            )
        else:
            raise NotImplementedError
        
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def _rel_emb(self, q, rel_width, rel_height):
        h, w = self.fmap_size, self.fmap_size

        q = rearrange(q, 'b h d (x y) -> b h x y d', x = h, y = w)
        
        rel_logits_w = relative_logits_1d(q, rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        
        rel_logits_h = relative_logits_1d(q, rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h

    def _rel_emb_ve(self, q, rel_all):
        tmp = rearrange(rel_all, 'r d -> d r').unsqueeze(0)
        tmp = expand_dim(tmp, 2, self.kernel_size)
        tmp = rel_to_abs(tmp).squeeze(0)
        return einsum('bgij, cij -> bgci', q, tmp)

    def _binary_forward(self, x):

        N, C, HW = x.shape

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(
            qkv.reshape(N, self.groups, self.qk_planes*2+self.v_planes, HW), 
            [self.qk_planes, self.qk_planes, self.v_planes], 
            dim=2
        )
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        
        if self.lesa.pe_type is None:
            stacked_similarity = qk
            stacked_similarity = self.bn_similarity(stacked_similarity)
        elif self.lesa.pe_type == 'detection_qr':
            stacked_similarity = qk
            qr = self._rel_emb(q, self.rel_width, self.rel_height)
            stacked_similarity = self.bn_similarity(torch.cat([stacked_similarity, qr], dim=1)).view(N, 2, self.groups, HW, HW).sum(dim=1)
        elif self.lesa.pe_type == 'classification':
            all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(
                self.qk_planes*2+self.v_planes, 
                self.kernel_size, self.kernel_size,
            )
            q_embedding, k_embedding, v_embedding = torch.split(
                all_embeddings, 
                [self.qk_planes, self.qk_planes, self.v_planes], 
                dim=0,
            )
            qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
            kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
            stacked_similarity = torch.cat([qk, qr, kr], dim=1)
            stacked_similarity = self.bn_similarity(stacked_similarity).view(N, 3, self.groups, HW, HW).sum(dim=1)
        else:
            raise NotImplementedError

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        
        if self.lesa.pe_type == 'classification':
            sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
            stacked_binary = torch.cat([sv, sve], dim=-1).view(N, self.branch_planes * 2, HW)
            binary = self.bn_output(stacked_binary).view(N, self.branch_planes, 2, HW).sum(dim=-2)
        elif self.lesa.pe_type == 'detection_qr':
            stacked_binary = sv.reshape(N, self.branch_planes, HW)
            binary = self.bn_output(stacked_binary)
        elif self.lesa.pe_type == None:
            stacked_binary = sv.reshape(N, self.branch_planes, HW)
            binary = self.bn_output(stacked_binary)
        else:
            raise NotImplementedError
        return binary

    def _unary_forward(self, x):
        unary = self.bn_x(self.x_transform(x))
        return unary

    def forward(self, x):

        N,C,UnaryH,UnaryW = x.shape

        # unary
        if self.with_cp:
            unary = cp.checkpoint(self._unary_forward, x)
        else:
            unary = self._unary_forward(x)

        N,C,H,W = x.shape
        x = x.view(N,C,H*W)

        # binary
        if self.with_cp:
            binary = cp.checkpoint(self._binary_forward, x)
        else:
            binary = self._binary_forward(x)
            
        binary = binary.view(N, self.branch_planes, H, W)
        
        gate_in = torch.cat([unary, binary], dim=1)
        r = self.reasoning(gate_in)
        gate = self.projection(r)
        gate = torch.sigmoid(gate)

        binary = gate * binary
        output = binary + unary

        if self.stride > 1:
            output = self.pooling(output)
            
        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        if self.lesa.pe_type == 'classification':
            nn.init.normal_(self.relative, 0., math.sqrt(1. / self.v_planes*1))

def lesa3x3(**kwargs):
    return LESA(
        **kwargs,
    )
