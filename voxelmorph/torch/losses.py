import torch
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
    
class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true #[1, 1, 192, 160, 192]
        Ji = y_pred
        
        #print("y_true shape:",y_true.shape)
        #print("y_pred:", y_pred.shape)
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()

class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.reshape(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)

class CL:
    def __init__(self, loss_type='dv'):
        self.loss_type = loss_type
        #print("cl loss type: ",self.loss_type)

    def gradient(self,I):
        h = I.shape[2]
        w = I.shape[3]
        z = I.shape[4]
            
        I = F.pad(I,(1,1,1,1,1,1),'replicate')
        
        # Write central finite difference formula for Ix and Iy
        # Do not use any for loop
        Ix = 0.5*(I[:,:,1:h+1,2:w+2,1:z+1]-I[:,:,1:h+1,0:w,1:z+1])
        Iy = 0.5*(I[:,:,2:h+2,1:w+1,1:z+1]-I[:,:,0:h,1:w+1,1:z+1])
        Iz = 0.5*(I[:,:,1:h+1,1:w+1,2:z+2]-I[:,:,1:h+1,1:w+1,0:z])

        return Ix,Iy,Iz

    def ContrastiveLoss(self,I,J):
        tau = 2.0
        b,c,h,w,z = I.shape
        
        Ix,Iy,Iz = self.gradient(I)
        Jx,Jy,Jz = self.gradient(J)
        
        Imag = torch.sqrt(Ix**2 + Iy**2 +  Iz**2 + 1e-8)
        Jmag = torch.sqrt(Jx**2 + Jy**2 +  Iz**2 + 1e-8)
        
        Ix = Ix/Imag
        Iy = Iy/Imag
        Iz = Iz/Imag
        Jx = Jx/Jmag
        Jy = Jy/Jmag
        Jz = Jz/Jmag
        
        #numerator = torch.exp(-F.conv2d(torch.abs(Ix-Jx)+torch.abs(Iy-Jy),box1,padding='same')/tau)
        #numerator = F.conv2d(torch.abs(Ix-Jx)+torch.abs(Iy-Jy)+torch.abs(Iz-Jz),box1,padding='same')/tau
        #numerator = torch.exp(-torch.abs(Ix-Jx)/tau-torch.abs(Iy-Jy)/tau)

        numerator = (torch.abs(Ix-Jx)+torch.abs(Iy-Jy)+torch.abs(Iz-Jz))/tau
        
        ind_perm = torch.randperm(h*w*z).to("cuda")
        Jxp = Jx.view(b,c,h*w*z)[:,:,ind_perm].view(b,c,h,w,z).to("cuda")
        Jyp = Jy.view(b,c,h*w*z)[:,:,ind_perm].view(b,c,h,w,z).to("cuda")
        Jzp = Jz.view(b,c,h*w*z)[:,:,ind_perm].view(b,c,h,w,z).to("cuda")
            
        #denominator = torch.exp(-F.conv2d(torch.abs(Ix-Jxp)+torch.abs(Iy-Jyp)+torch.abs(Iz-Jzp),box1,padding='same')/tau)
        #denominator = torch.exp(-torch.abs(Ix-Jxp)/tau-torch.abs(Iy-Jyp)/tau)

        denominator = torch.exp(-(torch.abs(Ix-Jxp)+torch.abs(Iy-Jyp)+torch.abs(Iz-Jzp))/tau)

        #loss = torch.log(torch.mean(denominator))-torch.log(torch.mean(numerator))
        loss = torch.mean(denominator)+torch.mean(numerator)
        
        return loss

    def DVLoss(self,I,J):
        """tau = 2.0
        b,c,h,w,z = I.shape
        
        Ix,Iy,Iz = self.gradient(I)
        Jx,Jy,Jz = self.gradient(J)
        
        Imag = torch.sqrt(Ix**2 + Iy**2 +  Iz**2 + 1e-8)
        Jmag = torch.sqrt(Jx**2 + Jy**2 +  Iz**2 + 1e-8)
        
        Ix = Ix/Imag
        Iy = Iy/Imag
        Iz = Iz/Imag
        Jx = Jx/Jmag
        Jy = Jy/Jmag
        Jz = Jz/Jmag

        
        #numerator = F.conv2d(torch.abs(Ix-Jx)+torch.abs(Iy-Jy)+torch.abs(Iz-Jz),box1,padding='same')/tau
        numerator = (torch.abs(Ix-Jx)+torch.abs(Iy-Jy)+torch.abs(Iz-Jz))/tau
        
        ind_perm = torch.randperm(h*w*z).to("cuda")
        Jxp = Jx.view(b,c,h*w*z)[:,:,ind_perm].view(b,c,h,w,z)
        Jyp = Jy.view(b,c,h*w*z)[:,:,ind_perm].view(b,c,h,w,z)
        Jzp = Jz.view(b,c,h*w*z)[:,:,ind_perm].view(b,c,h,w,z)
            
        #denominator = torch.exp(-F.conv2d(torch.abs(Ix-Jxp)+torch.abs(Iy-Jyp)+torch.abs(Iz-Jzp),box1,padding='same')/tau)
        denominator = torch.exp(-(torch.abs(Ix-Jxp)+torch.abs(Iy-Jyp)+torch.abs(Iz-Jzp))/tau)

        loss = torch.mean(numerator) + torch.log(torch.mean(denominator))
        
        return loss"""

        b,c,h,w,z = I.shape
        
        numerator = -torch.mean(I*J,dim=1)
        
        ind_perm = torch.randperm(h*w*z).to("cuda")
        Jp = J.view(b,c,h*w*z)[:,:,ind_perm].view(b,c,h,w,z)
            
        denominator = torch.exp(torch.mean(I*Jp,dim=1))

        loss = torch.mean(numerator) + torch.log(torch.mean(denominator))
        
        return loss
    
    def loss(self, y_true, y_pred):

        if self.loss_type == 'dv':
            cl_loss = self.DVLoss(y_true, y_pred)
        elif self.loss_type == 'f-micl':
            cl_loss = self.ContrastiveLoss(y_true, y_pred)
        else:
            ValueError('Image loss should be "dv" or "f-micl", but found "%s"' % self.loss_type)
        
        return cl_loss
            


