import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils_imp import *
from weighted_ce_loss import *
import torch
from typing import Union, Tuple

# th.use_deterministic_algorithms(True)


def absolute(real, imag):
    return th.sqrt(real **2 + imag**2)

def complexVar(real: torch.Tensor,
                imag: torch.Tensor,
                dim: Union[int, Tuple[int, ...]] = None,
                keepdim: bool = False) -> torch.Tensor:
        """
        Compute the population variance of a complex-valued tensor given
        separate real and imaginary parts along specified dimension(s).

        Variances defined by:
            Var(Z) = E[|Z - E[Z]|^2]
                   = E[(a - μₐ)^2 + (b - μ_b)^2]

        Args:
            real     (torch.Tensor): Real-part tensor.
            imag     (torch.Tensor): Imag-part tensor (same shape as real).
            dim       (int or tuple): Dimension(s) along which to reduce.
            keepdim  (bool): Whether to retain reduced dimensions.

        Returns:
            torch.Tensor: Real-valued variance, reduced over `dim`.
        """
        # 1) Compute means for real and imaginary parts
        mean_real = real.mean(dim=dim, keepdim=True)
        mean_imag = imag.mean(dim=dim, keepdim=True)

        # 2) Compute squared deviations and sum
        sq_dev = (real - mean_real) ** 2 + (imag - mean_imag) ** 2

        # 3) Average over specified dimensions
        var = sq_dev.mean(dim=dim, keepdim=keepdim)
        return var


def Cmul(x, y):
    """
    Complex multiplication of two complex vectors.
    x: Tensor of shape [B, 2, C, H, W]
    y: Tensor of shape [B, 2, C, H, W]
    """
    a, b = x[:, 0], x[:, 1]
    c, d = y[:, 0], y[:, 1]

    real = (a*c - b*d)
    imag = (b*c + a*d)

    return th.stack([real, imag], dim=1)


def Cdiv(x, y, clamp=False):
    """
    Complex division of two complex vectors.
    x: Tensor of shape [B, 2, C, H, W]
    y: Tensor of shape [B, 2, C, H, W]
    clamp: Clamp the denominator to be non-zero, instead of adding a small value.
    """

    a, b = x[:, 0], x[:, 1]
    c, d = y[:, 0], y[:, 1]

    real = (a*c - b*d)
    imag = (b*c + a*d)

    if clamp:
        divisor = th.clamp(c**2 + d**2, 0.05)
    else:
        divisor = c**2 + d**2 + 1e-7

    real = (a*c + b*d)/divisor  # ac + bd
    imag = (b*c - a*d)/divisor  # (bc - ad)i

    return th.stack([real, imag], dim=1)


def Cconj(x):
    """
    Complex conjugate of a complex vector.
    x: Tensor of shape [B, 2, C, H, W]
    """
    a, b = x[:, 0], x[:, 1]
    return th.stack([a, -b], dim=1)


def abs_normalize(w):
    """
    Normalize the weights so that the sum of the absolute values is 1.
    """
    return w/(w.detach().abs().sum(dim=(1, 2, 3), keepdim=True)+1e-6)


def normalize(w):
    """
    Normalize the weights so that the sum is 1.
    """
    return w/(w.sum(dim=(1, 2, 3), keepdim=True)+1e-6)


def sq_normalize(w):
    """
    Normalize the weights so that the sum of the squares is 1.
    """
    w_sq = w**2
    return w_sq/(w_sq.sum(dim=(1, 2, 3), keepdim=True)+1e-6)


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(
        dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output


class reflect_pad(nn.Module):
    """
    Does 2D reflection padding (size 1), but deterministically.
    In some pytorch versions, the existing kernel is not deterministic.
    """

    def __init__(self) -> None:
        super(reflect_pad, self).__init__()

    def forward(self, x):
        shape = x.shape
        assert len(shape) == 4
        to_return = th.zeros(
            (shape[0], shape[1], shape[2]+2, shape[3]+2), device=x.device, dtype=x.dtype)
        to_return[..., 1:-1, 1:-1] = to_return[..., 1:-1, 1:-1] + x
        to_return[..., 0] = to_return[..., 0] + to_return[..., 1]
        to_return[..., -1] = to_return[..., -1] + to_return[..., -2]
        to_return[..., 0, :] = to_return[..., 0, :] + to_return[..., 1, :]
        to_return[..., -1, :] = to_return[..., -1, :] + to_return[..., -2, :]
        return to_return

class GMM(nn.Module):
    def __init__(self):
        super(GMM, self).__init__()



class MaxPoolMag(nn.Module):
    """
    Performs magnitude max pooling
    Pools the input with largest magnitude over neighbors
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        super(MaxPoolMag, self).__init__()
        self.mp = nn.MaxPool2d(kernel_size, stride,
                               padding, dilation, True, ceil_mode)

    def __repr__(self):
        return 'ComplexMagnitudePooling'

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        x_norm = th.norm(x, dim=1)
        _, indices = self.mp(x_norm)
        x_real = retrieve_elements_from_indices(x[:, 0, ...], indices)
        x_imag = retrieve_elements_from_indices(x[:, 1, ...], indices)
        return th.cat((x_real.unsqueeze(1), x_imag.unsqueeze(1)), dim=1)


class ComplexConv(nn.Module):
    # Our complex convolution implementation
    def __init__(self, in_channels, num_filters, kern_size, stride=(1, 1), padding=0, dilation=1, groups=1, reflect=False, bias=False, new_init=False, use_groups_init=False, fan_in=False, *args, **kwargs):
        super(ComplexConv, self).__init__()

        # Convolution parameters
        self.in_channels = in_channels
        self.kern_size = kern_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.reflect = reflect
        if reflect:
            self.pad_func = th.nn.ZeroPad2d(reflect)

        self.A = nn.Conv2d(in_channels, num_filters, kern_size,
                           stride=stride, padding=padding, groups=groups, bias=bias)
        self.B = nn.Conv2d(in_channels, num_filters, kern_size,
                           stride=stride, padding=padding, groups=groups, bias=bias)

        if new_init:
            if fan_in:
                c = in_channels
            else:
                c = num_filters
            if use_groups_init:
                c = c/groups

            gain = 1/np.sqrt(2)
            with th.no_grad():
                std = gain / np.sqrt(kern_size * kern_size * c)
                self.A.weight.normal_(0, std)
                self.B.weight.normal_(0, std)

    def __repr__(self):
        return 'ComplexConv'

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        if len(x.shape) == 5:
            N, CC, C, H, W = x.shape
            if self.reflect:
                x = self.pad_func(x.reshape(N*CC, C, H, W))
            x = x.reshape(N, CC, C, x.shape[-2], x.shape[-1])
            real = x[:, 0]
            imag = x[:, 1]
            out_real = self.A(real) - self.B(imag)
            out_imag = self.B(real) + self.A(imag)
            return th.stack([out_real, out_imag], dim=1)
        else:
            N, C, H, W = x.shape
            if self.reflect:
                x = self.pad_func(x)
            out_real = self.A(x)
            out_imag = self.B(x)
            return th.stack([out_real, out_imag], dim=1)


class ComplexConvFast(nn.Module):
    # Our complex convolution implementation
    def __init__(self, in_channels, num_filters, kern_size, stride=(1, 1), padding=0, dilation=1, groups=1, reflect=False, bias=False, new_init=False, use_groups_init=False, fan_in=False, *args, **kwargs):
        super(ComplexConvFast, self).__init__()

        # Convolution parameters
        self.in_channels = in_channels
        self.kern_size = kern_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.reflect = reflect
        if reflect:
            self.pad_func = th.nn.ZeroPad2d(reflect)

        self.A = nn.Conv2d(in_channels, num_filters, kern_size,
                           stride=stride, padding=padding, groups=groups, bias=bias)
        self.B = nn.Conv2d(in_channels, num_filters, kern_size,
                           stride=stride, padding=padding, groups=groups, bias=bias)

        if new_init:
            if fan_in:
                c = in_channels
            else:
                c = num_filters
            if use_groups_init:
                c = c/groups

            gain = 1/np.sqrt(2)
            with th.no_grad():
                std = gain / np.sqrt(kern_size * kern_size * c)
                self.A.weight.normal_(0, std)
                self.B.weight.normal_(0, std)

    def __repr__(self):
        return 'ComplexConv'

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        if len(x.shape) == 5:
            N, CC, C, H, W = x.shape
            if self.reflect:
                x = self.pad_func(x.reshape(N*CC, C, H, W))
            x = x.reshape(N, CC, C, x.shape[-2], x.shape[-1])
            real = x[:, 0]
            imag = x[:, 1]
            t1 = self.A(real)
            t2 = self.B(imag)

            t3 = F.conv2d(real+imag, weight=(self.A.weight + self.B.weight), stride=self.stride,
                          padding=self.padding, groups=self.groups)

            return th.stack([t1 - t2, t3 - t1 - t2], dim=1)
        else:
            N, C, H, W = x.shape
            if self.reflect:
                x = self.pad_func(x)
            out_real = self.A(x)
            out_imag = self.B(x)
            return th.stack([out_real, out_imag], dim=1)


class NaiveCBN(nn.Module):
    """
    Naive BatchNorm which concatenates real and imaginary channels
    """

    def __init__(self, channels):
        super(NaiveCBN, self).__init__()
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        x_shape = x.shape
        return self.bn(x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])).reshape(x_shape)


class VNCBN(nn.Module):
    """
    Equivariant Complex Batch Norm
    Computes magnitude of the complex input and applies batch norm on it
    """

    def __init__(self, channels):
        super(VNCBN, self).__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        mag = th.norm(x, dim=1)
        normalized = self.bn(mag)
        mag_factor = normalized/(mag+1e-6)
        return x*mag_factor[:, None, ...]


class DivLayer(nn.Module):
    """
    division layer
    """

    def __init__(self, in_channels, kern_size, stride=(1, 1), padding=0, dilation=1, groups=1, reflect=False, use_one_filter=True, new_init=False):
        super(DivLayer, self).__init__()

        self.kern_size = kern_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.reflect = reflect
        self.use_one_filter = use_one_filter

        if self.use_one_filter:
            self.conv = ComplexConv(
                in_channels, 1, kern_size, stride, padding, dilation, groups, reflect=reflect, new_init=new_init)
        else:
            self.conv = ComplexConv(in_channels, in_channels, kern_size,
                                    stride, padding, dilation, groups, reflect=reflect, new_init=new_init)

    def __repr__(self):
        return 'DivLayer'

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        N, CC, C, H, W = x.shape  # Batch, 2, channels, H, W

        y = x

        if self.use_one_filter:
            conv = self.conv(y)
            conv = conv.repeat(1, 1, C, 1, 1)
        else:
            conv = self.conv(y)

        # For center-cropping original input
        output_xdim = conv.shape[-2]
        output_ydim = conv.shape[-1]
        input_xdim = H
        input_ydim = W

        start_x = int((input_xdim-output_xdim)/2)
        start_y = int((input_ydim-output_ydim)/2)

        num = x[:, :, :, start_x:start_x +
                output_xdim, start_y:start_y+output_ydim]

        a, b = num[:, 0], num[:, 1]
        c, d = conv[:, 0], conv[:, 1]

        divisor = c**2 + d**2 + 1e-7

        real = (a*c + b*d)/divisor  # ac + bd
        imag = (b*c - a*d)/divisor  # (bc - ad)i

        return th.stack([real, imag], dim=1)


class ConjugateLayer(nn.Module):
    """
    conjugate layer
    """

    def __init__(self, in_channels, kern_size, stride=(1, 1), padding=0, dilation=1, groups=1, reflect=False, use_one_filter=False, new_init=False):
        super(ConjugateLayer, self).__init__()
        self.kern_size = kern_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.reflect = reflect
        self.use_one_filter = use_one_filter
        conv = ComplexConv

        if self.use_one_filter:
            self.conv = conv(
                in_channels, 1, kern_size, stride, padding, dilation, groups, reflect=reflect, new_init=new_init)
        else:
            self.conv = conv(in_channels, in_channels, kern_size,
                             stride, padding, dilation, groups, reflect=reflect, new_init=new_init)

    def __repr__(self):
        return 'Conjugate'

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        N, CC, C, H, W = x.shape  # Batch, 2, channels, H, W
        y = x

        if self.use_one_filter:
            conv = self.conv(y)
            conv = conv.repeat(1, 1, C, 1, 1)
        else:
            conv = self.conv(y)

        # For center-cropping original input
        output_xdim = conv.shape[-2]
        output_ydim = conv.shape[-1]
        input_xdim = H
        input_ydim = W

        start_x = int((input_xdim-output_xdim)/2)
        start_y = int((input_ydim-output_ydim)/2)

        num = x[:, :, :, start_x:start_x +
                output_xdim, start_y:start_y+output_ydim]

        a, b = num[:, 0], num[:, 1]
        c, d = conv[:, 0], conv[:, 1]
        real = (a*c + b*d)  # ac + bd
        imag = (b*c - a*d)  # (bc - ad)i

        x = th.stack([real, imag], dim=1)

        return x

def perpendicular_loss(real,imag, a, b):
    distances = th.zeros_like(real)
    for i in range(real.shape[2]):
    # for i in range(real.shape[2]):
        mag_input = absolute(real[..., i], imag[..., i])
        mag_preds = absolute(a, b)
        angle = th.atan2(imag[..., i], real[..., i]) - th.atan2(b, a)
        P = th.abs((real[..., i] * b) - (imag[..., i] * a)) / mag_input
        aligned_mask = (th.cos(angle) < 0).bool()
        final_term = th.zeros_like(P)
        final_term[aligned_mask] = mag_preds[aligned_mask] + (mag_preds[aligned_mask] - P[aligned_mask])
        final_term[~aligned_mask] = P[~aligned_mask]
        distances[..., i] = final_term + th.abs(mag_preds - mag_input)
    # distances = final_term + th.abs(mag_preds - mag_input)
    return distances.mean(dim=1)

class DistFeatures(nn.Module):
    """
    prototype distance layer, using Euclidean distance
    """

    def __init__(self, in_channels, num_prototypes=16):
        super(DistFeatures, self).__init__()

        # Convolution parameters
        self.in_channels = in_channels
        self.num_prototypes = num_prototypes

        # prototypes = th.rand(2, in_channels, num_prototypes)
        prototypes = th.rand(2, 10, in_channels, num_prototypes)
        self.prototypes = nn.Parameter(data=prototypes, requires_grad=True)
        self.temp = nn.Parameter(data=th.tensor(1.0), requires_grad=True)
        # self.log_sigma = nn.Parameter(data=th.zeros([1, num_prototypes]))
        self.log_sigma = nn.Parameter(data=th.zeros([1, 10, num_prototypes]))

    def __repr__(self):
        return 'DistFeats'

    def gmm(self, distance_sq, pro_real, pro_imag, pred_real, pred_imag):

        # distance2 = distance_sq.sum(dim=1)
        bc, ppc, dist, plen = distance_sq.size()
        distance2 = distance_sq.sum(dim=2)


        # Compute the unnormalized "likelihood" using a Gaussian kernel:
        sigma = th.exp(self.log_sigma)
        likelihoods = th.exp(-distance2.view(bc, ppc * plen) / (2 * sigma.view(sigma.shape[0], -1)**2 + 1e-8))  # (batch, num_prototypes)

        responsibilities = likelihoods / (likelihoods.sum(dim=1, keepdim=True) + 1e-8)  # shape: (batch, num_prototypes)
        # responsibilities = likelihoods / (likelihoods.sum(dim=2, keepdim=True) + 1e-8)  # shape: (batch, num_prototypes)

        # Now, compute the new prototype estimates as a weighted average of the sample features.
        # preds.complex has shape (batch, dist_features).
        # We compute weighted sums along the batch dimension.
        # Use Einstein summation: for each prototype k, new_proto[:, k] = sum_{i} responsibilities[i,k] * preds.complex[i,:] / sum_{i} responsibilities[i,k]
        # weighted_sum = th.einsum('bk,bd->kd', responsibilities+1j*th.zeros_like(responsibilities),
        #                          pred_real.squeeze() + 1j * pred_imag.squeeze())  # (num_prototypes, dist_features)
        weighted_sum = th.einsum('bk,bd->kd', responsibilities+1j*responsibilities,
                                 pred_real.squeeze() + 1j * pred_imag.squeeze())
        # weighted_sum = th.einsum('bck,bd->kcd', responsibilities+1j*responsibilities,
        #                          pred_real.squeeze() + 1j * pred_imag.squeeze())  # (num_prototypes, dist_features)
        sum_resp = responsibilities.sum(dim=0).unsqueeze(1)  # (num_prototypes, 1)
        # sum_resp = responsibilities.sum(dim=1).unsqueeze(2)  # (num_prototypes, 1)
        # new_prototypes = weighted_sum / (sum_resp.unsqueeze(1) + 1e-8)  # (num_prototypes, dist_features)
        new_prototypes = weighted_sum / (sum_resp + 1e-8)  # (num_prototypes, dist_features)
        # Transpose to match our prototype shape: (dist_features, num_prototypes)
        new_prototypes = new_prototypes.transpose(0, 1)
        new_prototypes = new_prototypes.reshape([ppc, dist, plen])
        # new_prototypes = new_prototypes.transpose(2, 3)
        prototypes_cv = pro_real + 1j * pro_imag

        # Define a prototype regularization loss that encourages current prototypes to be close to the updated values.
        loss_proto = th.mean(th.abs(new_prototypes - prototypes_cv)**2)

        return loss_proto



    def forward(self, x, y=None):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        N, CC, C = x.shape

        if y is not None:
            y = y[..., 0, 0]
            a, b = self.prototypes[None, 0], self.prototypes[None, 1]
            c, d = y[:, 0, ..., None], y[:, 1, ..., None]
            # real = a*c - b*d
            # imag = b*c + a*d
            real = a*c.unsqueeze(1) - b*d.unsqueeze(1)
            imag = b*c.unsqueeze(1) + a*d.unsqueeze(1)
        else:
            prototypes = self.prototypes
            real, imag = prototypes[None, 0, :, :], prototypes[None, 1, :, :]
        a, b = x[:, 0, :, None], x[:, 1, :, None]

        # dist = perpendicular_loss(real=real, imag=imag, a=a.squeeze(), b=b.squeeze())

        # dist_sq = (real-a)**2 + (imag-b)**2
        dist_sq = (real-a.unsqueeze(1))**2 + (imag-b.unsqueeze(1))**2
        # dist = th.sqrt(dist_sq.mean(dim=1))
        dist = th.sqrt(dist_sq.mean(dim=2))

        # add gmm layer
        l_proto = self.gmm(dist_sq, real, imag, a, b)

        # return -dist*self.temp, l_proto
        return -dist.mean(dim=1)*self.temp, l_proto


class infinite_mixture_prototype(nn.Module):
    def __init__(self):
        super(infinite_mixture_prototype, self).__init__()

        learn_sigma_l = 1.0
        #
        # self.in_channels = in_channels
        # self.num_prototypes = num_prototypes

        log_sigma_l = th.log(th.FloatTensor([learn_sigma_l]))

        if learn_sigma_l:
            self.log_sigma_l = nn.Parameter(log_sigma_l, requires_grad=True)
        else:
            self.log_sigma_l = Variable(log_sigma_l, requires_grad=True).cuda()

        hid_dim, z_dim = 64, 64

    def _add_cluster(self, nClusters, protos, radii, cluster_type='unlabeled', ex=None):
        """
        Args:
            nClusters: number of clusters
            protos: [B, nClusters, D] cluster protos
            radii: [B, nClusters] cluster radius,
            cluster_type: ['labeled','unlabeled'], the type of cluster we're adding
            ex: the example to add
        Returns:
            updated arguments
        """
        nClusters += 1
        bsize = protos.size()[0]
        dimension = protos.size()[2]

        zero_count = Variable(th.zeros(bsize, 1)).cuda()

        d_radii = Variable(th.ones(bsize, 1), requires_grad=False).cuda()

        if cluster_type == 'labeled':
            d_radii = d_radii * th.exp(self.log_sigma_l)
        else:
            d_radii = d_radii * th.exp(self.log_sigma_u)

        if ex is None:
            new_proto = self.base_distribution.data.cuda()
        else:
            # new_proto = ex.unsqueeze(0).unsqueeze(0).cuda()
            new_proto = ex.unsqueeze(0).unsqueeze(2).cuda()

        # protos = th.cat([protos, new_proto], dim=1)
        protos = th.cat([protos, new_proto], dim=2)
        radii = th.cat([radii, d_radii], dim=1)
        return nClusters, protos, radii

    def estimate_lambda(self, tensor_proto, semi_supervised):
        # estimate lambda by mean of shared sigmas
        # rho = tensor_proto[0].var(dim=0)
        rho = complexVar(tensor_proto[0, 0, ...], tensor_proto[0, 0, ...], dim=0)
        # rho = tensor_proto[0].var(dim=1)
        rho = rho.mean()

        if semi_supervised:
            sigma = (th.exp(self.log_sigma_l).data[0] + th.exp(self.log_sigma_u).data[0]) / 2.
        else:
            sigma = th.exp(self.log_sigma_l).data[0]

        alpha = 0.01
        lamda = -2 * sigma.cpu().numpy() * np.log(alpha) + sigma.cpu().numpy() * np.log(
            1 + rho.cpu().numpy() / sigma.cpu().numpy())

        return np.abs(lamda)

    def _compute_protos(self, h, probs):
        """Compute the prototypes
        Args:
            h: [B, N, D] encoded inputs
            probs: [B, N, nClusters] soft assignment
        Returns:
            cluster protos: [B, nClusters, D]
        """

        h = torch.unsqueeze(h, 2)       # [B, N, 1, D]
        probs = torch.unsqueeze(probs, 3)       # [B, N, nClusters, 1]
        prob_sum = torch.sum(probs, 1)  # [B, nClusters, 1]
        zero_indices = (prob_sum.view(-1) == 0).nonzero()
        if torch.numel(zero_indices) != 0:
            values = torch.masked_select(torch.ones_like(prob_sum), torch.eq(prob_sum, 0.0))
            prob_sum = prob_sum.put_(zero_indices, values)
        protos = torch.einsum('ijkmn, ijpq->ijmpn', h, probs)
        # protos = h*probs    # [B, N, nClusters, D]
        # protos = torch.sum(protos, 1)/prob_sum
        protos = torch.sum(protos, 1)/prob_sum.unsqueeze(1)
        return protos

    def _compute_distances_complex(self, protos, example):
        dist = torch.sum((example[0] - protos[:, 0, ...])**2 + (example[1] - protos[:, 1, ...])**2, dim=2)
        # dist = torch.sum((example - protos)**2, dim=2)
        return dist

    def delete_empty_clusters(self, tensor_proto, prob, radii, targets, eps=1e-3):
        column_sums = th.sum(prob[0], dim=0).data
        good_protos = column_sums > eps
        idxs = th.nonzero(good_protos).squeeze()
        return tensor_proto[:, :, idxs, :], radii[:, idxs], targets[idxs]

    def loss(self, logits, targets, labels):
        """Loss function to "or" across the prototypes in the class:
        take the loss for the closest prototype in the class and all negatives.
        inputs:
            logits [B, N, nClusters] of nll probs for each cluster
            targets [B, N] of target clusters
        outputs:
            weighted cross entropy such that we have an "or" function
            across prototypes in the class of each query
        """
        targets = targets.cuda()
        # determine index of closest in-class prototype for each query
        target_logits = th.ones_like(logits.data) * float('-Inf')
        target_logits[targets] = logits.data[targets]
        _, best_targets = th.max(target_logits, dim=1)
        # mask out everything...
        weights = th.zeros_like(logits.data)
        # ...then include the closest prototype in each class and unlabeled)
        unique_labels = np.unique(labels.cpu().numpy())
        for l in unique_labels:
            class_mask = labels == l
            class_logits = th.ones_like(logits.data) * float('-Inf')
            # batch_idx = torch.arange(class_logits.size(0))
            row, col = class_mask.repeat(logits.size(0), 1).nonzero(as_tuple=True)
            class_logits[row, col] = logits[row, col]
            # class_logits[class_mask.repeat(logits.size(0), 1)] = logits[class_mask].data.view(logits.size(0), -1).squeeze(1)
            _, best_in_class = th.max(class_logits, dim=1)
            weights[range(0, targets.size(0)), best_in_class] = 1.
        loss = weighted_loss(logits, Variable(best_targets), Variable(weights))
        return loss.mean()

    def forward(self, x, y, train_flag=True):

        y, _ = y.sort()

        # batch = self._process_batch(sample, super_classes=super_classes)
        if train_flag:
            self.nClusters = len(np.unique(y.data.cpu().numpy()))
            self.nInitialClusters = self.nClusters

            # run data through network
            # h_train = self._run_forward(x)
            # create probabilities for points
            _, idx = np.unique(y.squeeze().data.cpu().numpy(), return_inverse=True)
            prob_train = one_hot(y, self.nClusters).cuda()

            # make initial radii for labeled clusters
            bsize = x.size()[0]
            self.radii = Variable(th.ones(bsize, self.nClusters)).cuda() * th.exp(self.log_sigma_l)

            self.support_labels = th.arange(0, self.nClusters).cuda().long()

            # compute initial prototypes from labeled examples
            self.protos = self._compute_protos(x, prob_train)

            # estimate lamda
            lamda = self.estimate_lambda(self.protos.data, False)

            tensor_proto = self.protos.data
            # iterate over labeled examples to reassign first
            for i, ex in enumerate(x[0]):
                idxs = th.nonzero(y.data[0, i] == self.support_labels)[0]
                # distances = self._compute_distances(tensor_proto[:, idxs, :], ex.data)
                distances = self._compute_distances_complex(tensor_proto[:, :, idxs, :], ex.data)
                if (th.min(distances) > lamda):
                    self.nClusters, tensor_proto, self.radii = self._add_cluster(self.nClusters, tensor_proto, self.radii,
                                                                       cluster_type='labeled', ex=ex.data)
                    # self.support_labels = th.cat([self.support_labels, y[0, i].data], dim=0)
                    self.support_labels = th.cat([self.support_labels, y[0, i].reshape([1])], dim=0)

            # perform partial reassignment based on newly created labeled clusters
            if self.nClusters > self.nInitialClusters:
                support_targets = y.data[0, :, None] == self.support_labels
                prob_train = assign_cluster_radii_limited(Variable(tensor_proto), x, self.radii, support_targets)

            self.protos = Variable(tensor_proto).cuda()
            self.protos = self._compute_protos(x, Variable(prob_train.data, requires_grad=False).cuda())
            self.protos, self.radii, self.support_labels = self.delete_empty_clusters(self.protos, prob_train,
                                                                            self.radii, self.support_labels)

            logits = compute_logits_radii(self.protos, x, self.radii).squeeze()

            labels = y.data
            labels[labels >= self.nInitialClusters] = -1

            support_targets = labels[0, :, None] == self.support_labels
            loss = self.loss(logits, support_targets, self.support_labels)

            # map support predictions back into classes to check accuracy
            _, support_preds = th.max(logits.data, dim=1)
            y_pred = self.support_labels[support_preds]

            # acc_val = th.eq(y_pred, labels[0]).float().mean()
            acc_val = th.eq(y_pred, labels[0]).float().sum()

        else:

            # h_test = self._run_forward(x)

            logits = compute_logits_radii(self.protos, x, self.radii).squeeze()

            labels = y.data
            labels[labels >= self.nInitialClusters] = -1

            support_targets = labels[0, :, None] == self.support_labels
            loss = self.loss(logits, support_targets, self.support_labels)

            # map support predictions back into classes to check accuracy
            _, support_preds = th.max(logits.data, dim=1)
            y_pred = self.support_labels[support_preds]

            acc_val = th.eq(y_pred, labels[0]).float().sum()

        return loss, acc_val

            # # iterate over unlabeled examples
            # if batch.x_unlabel is not None:
            #     h_unlabel = self._run_forward(batch.x_unlabel)
            #     h_all = th.cat([h_train, h_unlabel], dim=1)
            #     unlabeled_flag = th.LongTensor([-1]).cuda()
            #
            #     for i, ex in enumerate(h_unlabel[0]):
            #         distances = self._compute_distances(tensor_proto, ex.data)
            #         if th.min(distances) > lamda:
            #             nClusters, tensor_proto, radii = self._add_cluster(nClusters, tensor_proto, radii,
            #                                                                cluster_type='unlabeled', ex=ex.data)
            #             support_labels = th.cat([support_labels, unlabeled_flag], dim=0)
            #
            #     # add new, unlabeled clusters to the total probability
            #     if nClusters > nTrainClusters:
            #         unlabeled_clusters = th.zeros(prob_train.size(0), prob_train.size(1), nClusters - nTrainClusters)
            #         prob_train = th.cat([prob_train, Variable(unlabeled_clusters).cuda()], dim=2)
            #
            #     prob_unlabel = assign_cluster_radii(Variable(tensor_proto).cuda(), h_unlabel, radii)
            #     prob_unlabel_nograd = Variable(prob_unlabel.data, requires_grad=False).cuda()
            #     prob_all = th.cat([Variable(prob_train.data, requires_grad=False), prob_unlabel_nograd], dim=1)
            #
            #     protos = self._compute_protos(h_all, prob_all)
            #     protos, radii, support_labels = self.delete_empty_clusters(protos, prob_all, radii, support_labels)
            # else:
        # logits = compute_logits_radii(protos, h_test, radii)

        # convert class targets into indicators for supports in each class

        # return loss, {
        #     'loss': loss.item(),
        #     'acc': acc_val,
        #     'logits': logits.data
        # }


class scaling_layer(nn.Module):
    """
    scaling layer for GTReLU
    """

    def __init__(self, channels, g_global=False):
        super(scaling_layer, self).__init__()
        if g_global:
            channels = 1
        self.a_bias = nn.Parameter(th.rand(channels,), requires_grad=True)
        self.b_bias = nn.Parameter(th.rand(channels,), requires_grad=True)

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        N, CC, C, H, W = x.shape  # Batch, 2, channels, H, W
        x_c = x[:, 0]
        x_d = x[:, 1]

        a_bias = self.a_bias[None, :, None, None]
        b_bias = self.b_bias[None, :, None, None]

        real_component = a_bias * x_c - b_bias * x_d
        imag_component = b_bias * x_c + a_bias * x_d

        return th.stack([real_component, imag_component], dim=1)


class Two_Channel_Nonlinearity(th.autograd.Function):
    """
    Non-linearity which thresholds phase
    """

    @staticmethod
    def forward(ctx, inputs):
        temp_phase = inputs

        phase_mask = temp_phase % (2*np.pi)
        phase_mask = (phase_mask <= np.pi).type(
            th.cuda.FloatTensor) * (phase_mask >= 0).type(th.cuda.FloatTensor)
        temp_phase = temp_phase * phase_mask

        ctx.save_for_backward(inputs, phase_mask)

        return temp_phase

    @staticmethod
    def backward(ctx, grad_output):
        inputs, phase_mask = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_input = grad_input*(1-phase_mask)

        return grad_input


class eqnl(nn.Module):
    """
    Equivariant version of the phase-only tangent ReLU
    """

    def __init__(self, channels, trelu_b=0.0, *args, **kwargs):
        # Applies tangent reLU to inputs.
        super(eqnl, self).__init__()
        self.phase_scale = nn.Parameter(
            th.ones(channels,), requires_grad=True)
        self.cn = Two_Channel_Nonlinearity.apply
        self.trelu_b = trelu_b

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        p1 = x
        p2 = th.mean(x, dim=2, keepdim=True)

        abs1 = th.norm(p1, dim=1, keepdim=True) + 1e-6
        abs2 = th.norm(p2, dim=1, keepdim=True) + 1e-6
        p2 = p2/abs2

        conjp2 = th.stack((p2[:, 0], -p2[:, 1]), 1)
        shifted = Cmul(p1, conjp2)
        phasediff = th.atan2(
            shifted[:, 1], shifted[:, 0] + (shifted[:, 0] == 0) * 1e-5)
        final_phase = self.cn(phasediff) * \
            th.relu(self.phase_scale[None, :, None, None])

        out = abs1 * \
            Cmul(th.stack([th.cos(final_phase), th.sin(final_phase)], 1), p2)

        return out


class GTReLU(nn.Module):
    """
    GTReLU layer
    """

    def __init__(self, channels, g_global=False, phase_scale=False):
        super(GTReLU, self).__init__()
        self.ps = phase_scale
        if g_global:
            channels = 1
        self.a_bias = nn.Parameter(th.rand(channels,), requires_grad=True)
        self.b_bias = nn.Parameter(th.rand(channels,), requires_grad=True)

        self.relu = nn.ReLU()
        self.cn = Two_Channel_Nonlinearity.apply
        if self.ps:
            self.phase_scale = nn.Parameter(
                th.ones(channels,)*phase_scale)

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        N, CC, C, H, W = x.shape  # Batch, 2, channels, H, W
        x_c = x[:, 0]
        x_d = x[:, 1]

        # Scaling
        a_bias = self.a_bias[None, :, None, None]
        b_bias = self.b_bias[None, :, None, None]

        real_component = a_bias * x_c - b_bias * x_d
        imag_component = b_bias * x_c + a_bias * x_d

        x = th.stack([real_component, imag_component], dim=1)

        # Thresholding
        temp_abs = th.norm(x, dim=1)
        temp_phase = th.atan2(
            x[:, 1, ...], x[:, 0, ...] + (x[:, 0, ...] == 0) * 1e-5)

        final_abs = temp_abs.unsqueeze(1)

        final_phase = self.cn(temp_phase)

        x = th.cat((final_abs * th.cos(final_phase).unsqueeze(1),
                   final_abs * th.sin(final_phase).unsqueeze(1)), 1)

        # Phase scaling [Optional]
        if self.ps:
            norm = th.norm(x, dim=1)
            angle = th.atan2(x[:, 1], x[:, 0] +
                             (x[:, 0] == 0) * 1e-5)
            angle = angle * \
                th.minimum(th.maximum(self.phase_scale[None, :, None, None],
                                      th.tensor(0.5)), th.tensor(2.0))

            x = th.stack([norm*th.cos(angle), norm*th.sin(angle)], dim=1)

        return x


class DCN_CBN(th.nn.Module):
    """
    DCN's CBN
    Mostly based on Pytorch th/nn/modules/batchnorm.py and DCN's Complex BatchNorm implementation
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, weight_init=None, bias_init=None):
        super(DCN_CBN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.Wrr = th.nn.Parameter(th.Tensor(num_features))
            self.Wri = th.nn.Parameter(th.Tensor(num_features))
            self.Wii = th.nn.Parameter(th.Tensor(num_features))
            self.Br = th.nn.Parameter(th.Tensor(num_features))
            self.Bi = th.nn.Parameter(th.Tensor(num_features))
        else:
            pass
        if self.track_running_stats:
            self.register_buffer('RMr',  th.zeros(num_features))
            self.register_buffer('RMi',  th.zeros(num_features))
            self.register_buffer('RVrr', th.ones(num_features))
            self.register_buffer('RVri', th.zeros(num_features))
            self.register_buffer('RVii', th.ones(num_features))
            self.register_buffer('num_batches_tracked',
                                 th.tensor(0, dtype=th.long))
        else:
            pass
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr .zero_()
            self.RMi .zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br .data.zero_()
            self.Bi .data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert(xr.shape == xi.shape)
        assert(xr.size(1) == self.num_features)

    def forward(self, xr, xi):
        self._check_input_dim(xr, xi)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1]*xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr = xr
            Mi = xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr-Mr, xi-Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr*xr
            Vri = xr*xi
            Vii = xi*xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr+self.eps
        Vri = Vri
        Vii = Vii+self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        tau = Vrr+Vii
        delta = th.addcmul(Vrr*Vii, -1, Vri, Vri)
        s = delta.sqrt()
        t = (tau + 2*s).sqrt()
        rst = (s*t).reciprocal()

        Urr = (s+Vii)*rst
        Uii = (s+Vrr)*rst
        Uri = (-Vri)*rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #

        if self.affine:
            Zrr = self.Wrr.view(Urr.shape)*Urr + self.Wri.view(Uri.shape)*Uri
            Zri = self.Wrr.view(Uri.shape)*Uri + self.Wri.view(Uii.shape)*Uii
            Zir = self.Wri.view(Urr.shape)*Urr + self.Wii.view(Uri.shape)*Uri
            Zii = self.Wri.view(Uri.shape)*Uri + self.Wii.view(Uii.shape)*Uii
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr, yi = Zrr*xr + Zri*xi, Zir*xr + Zii*xi

        if self.affine:
            yr = yr + self.Br[None, :, None, None]
            yi = yi + self.Bi[None, :, None, None]

        return yr, yi

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(
                   **self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        super(DCN_CBN, self)._load_from_state_dict(state_dict,
                                                   prefix,
                                                   local_metadata,
                                                   strict,
                                                   missing_keys,
                                                   unexpected_keys,
                                                   error_msgs)


class ComplexBN(th.nn.Module):
    """Wrapper around DCN_CBN"""

    def __init__(self, *args, **kwargs):
        super(ComplexBN, self).__init__()
        self.BN = DCN_CBN(*args, **kwargs)

    def forward(self, x):
        return th.stack(self.BN(x[:, 0, ...], x[:, 1, ...]), dim=1)
