import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

class ReflectedConvolution(nn.Module):
    
    def __init__(self, kernel_nums = 8, kernel_size = 3):
        super(ReflectedConvolution, self).__init__()
     
        self.kernel_nums = kernel_nums
        self.kernel_size = kernel_size  
        self.rg_bn = nn.BatchNorm2d(kernel_nums)
        self.gb_bn = nn.BatchNorm2d(kernel_nums)
        self.rb_bn = nn.BatchNorm2d(kernel_nums)
        self.filter = torch.nn.Parameter(torch.randn(self.kernel_nums, 1,  self.kernel_size,  self.kernel_size))
       # self.conv = nn.Conv2d(24, 24, 3, 1, 1, groups=1)
        
        self.init_weights()
  

      

    def init_weights(self):
        
        torch.nn.init.kaiming_normal_(self.filter)
        torch.nn.init.constant_(self.rg_bn.weight, 0.01)
        torch.nn.init.constant_(self.rg_bn.bias, 0)
        torch.nn.init.constant_(self.gb_bn.weight, 0.01)
        torch.nn.init.constant_(self.gb_bn.bias, 0)
        torch.nn.init.constant_(self.rb_bn.weight, 0.01)
        torch.nn.init.constant_(self.rb_bn.bias, 0)
        
    def mean_constraint(self, kernel):
        bs, cin, kw, kh =kernel.shape
        kernel_mean = torch.mean(kernel.view(bs, -1), dim=1, keepdim=True)
        kernel =  (kernel.view(bs, -1) - kernel_mean)
        return kernel.view(bs, cin, kw, kh)
    
    def forward(self, img):

        zeroMasks = torch.zeros_like(img)
        zeroMasks[img == 0] = 1
        log_img = torch.log(img + 1e-7)


        red_chan = log_img[:, 0, :, :].unsqueeze(1)
        green_chan = log_img[:, 1, :, :].unsqueeze(1)
        blue_chan = log_img[:, 2, :, :].unsqueeze(1)
        normalized_filter = self.mean_constraint(self.filter)

        # Red-Green
        filt_r1 = F.conv2d(red_chan, weight=normalized_filter, padding= self.kernel_size//2)
        filt_g1 = F.conv2d(green_chan, weight=-normalized_filter, padding= self.kernel_size//2)
        filt_rg = filt_r1 + filt_g1
        #filt_rg = torch.clamp(filt_rg, -1.0, 1.0)
        filt_rg = self.rg_bn(filt_rg)


        # Green-Blue
        filt_g2 = F.conv2d(green_chan, weight=normalized_filter, padding= self.kernel_size//2)
        filt_b1 = F.conv2d(blue_chan, weight=-normalized_filter, padding= self.kernel_size//2)
        filt_gb = filt_g2 + filt_b1
        #filt_gb = torch.clamp(filt_gb, -1.0, 1.0)
        filt_gb = self.gb_bn(filt_gb)


        # Red-Blue
        filt_r2 = F.conv2d(red_chan, weight=normalized_filter, padding= self.kernel_size//2)
        filt_b2 = F.conv2d(blue_chan, weight=-normalized_filter, padding= self.kernel_size//2)
        filt_rb = filt_r2 + filt_b2
       # filt_rb = torch.clamp(filt_rb, -1.0, 1.0)
        filt_rb = self.rb_bn(filt_rb)
  


        rg = filt_rg
        rg = torch.where(zeroMasks[:, 0:1, ...].expand(-1, self.kernel_nums, -1, -1)==1, 0, rg)
        gb = filt_gb
        gb = torch.where(zeroMasks[:, 1:2, ...].expand(-1, self.kernel_nums, -1, -1)==1, 0, gb)
        rb = filt_rb
        rb = torch.where(zeroMasks[:, 2:3, ...].expand(-1, self.kernel_nums, -1, -1)==1, 0, rb)
        
        
        out = torch.cat([rg, gb, rb], dim=1)
       
        return out


class IIBlock(nn.Module):
    #3, 24 , 32
    def __init__(self, kernel_nums=8, kernel_size=3, Gtheta = [0.6, 0.8]):
        self.Gtheta = Gtheta
        super(IIBlock, self).__init__()
        
        self.feat_projector = nn.Sequential(*[nn.Conv2d(3, 24, 3, 1, 1, groups=1), 
                                        nn.BatchNorm2d(24),
                                        nn.LeakyReLU(),
                                        ])
        self.fuse_net = nn.Sequential(*[nn.Conv2d(48, 32, 3, 1, 1, groups=2),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(32, 3, 3, 1, 1, groups=1)])
        self.iim = ReflectedConvolution(kernel_nums, kernel_size)
    
    def forward(self, x):
        
        """
        Q: why Gamma for orignal image?
        A: Illumination Invariant Loss, please refer to our appendix.
        """
        x_gma = torch.pow(x, np.random.uniform(self.Gtheta[0], self.Gtheta[1]))
        x_gma = torch.clamp(x_gma, min=0, max=1)
        

        feat_ii, feat_ii_gma = list(map(self.iim, [x, x_gma]))
        feats = self.feat_projector(x)

        feats_ = torch.concat((feats, feat_ii), dim=1)
        x_out = self.fuse_net(feats_)
        

        return x_out, (feat_ii, feat_ii_gma)




    
        


