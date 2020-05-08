import numpy as np
import copy
import torch 
import torch.nn as nn
import torch.nn.functional as F

from AC_modules.Layers import *

debug = False

class OheNet(nn.Module):
    def __init__(self, map_size, k_in=3, k_out=24, n_features=32, pixel_hidden_dim=128, 
                 pixel_n_residuals=4, feature_hidden_dim=64, feature_n_residuals=4, device=None):
        
        super(OheNet, self).__init__()
        
        self.n_features = n_features
        
        self.OHE_conv = Convolution(k_in=k_in, k_out=k_out)
        self.pos_enc = PositionalEncoding(n_kernels=k_out, n_features=n_features, device=device)

        pixel_res_layers = nn.ModuleList([ResidualLayer(map_size**2, pixel_hidden_dim) for _ in range(pixel_n_residuals)])
        self.pixel_res_block = nn.Sequential(*pixel_res_layers)

        self.maxpool = FeaturewiseMaxPool(pixel_axis=2)

        feature_res_layers = nn.ModuleList([ResidualLayer(n_features, feature_hidden_dim) for _ in range(feature_n_residuals)])
        self.feature_res_block = nn.Sequential(*feature_res_layers)
        
    def forward(self, x):
        """ Input shape (batch_dim, k_in, map_size+2, map_size+2) """
        
        x = self.OHE_conv(x)
        if debug: print("conv_state.shape: ", x.shape)
            
        x = self.pos_enc(x)
        if debug: print("After positional enc + projection: ", x.shape)
            
        x = x.permute(1,2,0)
        if debug: print("x.shape: ", x.shape)
            
        x = self.pixel_res_block(x) # Interaction between pixels feature-wise
        if debug: print("x.shape: ", x.shape)
            
        x = self.maxpool(x) # Feature-wise maxpooling
        if debug: print("x.shape: ", x.shape)
            
        x = self.feature_res_block(x) # Interaction between features -> final representation
        if debug: print("x.shape: ", x.shape)
        
        return x     
    
class SpatialNet(nn.Module):
    
    def __init__(self, n_features, state_size, out_channels):
        super(SpatialNet, self).__init__()
        
        self.size = state_size
        
        self.linear = nn.Linear(n_features, (state_size-6)*(state_size-6))
        
        self.conv_block = nn.Sequential(
                                        nn.ConvTranspose2d(in_channels=1, 
                                                           out_channels=out_channels, 
                                                            kernel_size=3),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=out_channels, 
                                                           out_channels=out_channels, 
                                                           kernel_size=3),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=out_channels, 
                                                              out_channels=out_channels, 
                                                              kernel_size=3)
                                        )
        
    def forward(self, state_rep):
        if debug: print("state_rep.shape: ", state_rep.shape)
            
        x = F.relu(self.linear(state_rep))
        if debug: print("x.shape (after linear): ", x.shape)
            
        x = x.reshape(x.shape[0], 1, self.size-6, self.size-6)
        if debug: print("x.shape (after reshape): ", x.shape)
            
        x = self.conv_block(x)
        if debug: print("x.shape (after conv block): ", x.shape)
            
        return x
