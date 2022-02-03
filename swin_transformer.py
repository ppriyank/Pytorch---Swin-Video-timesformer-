import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from utils import compute_hcf

from layers import Mlp , PatchEmbed, PatchMerging, BasicLayer

class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """
    def __init__(self, sec_len = 64, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        kwargs["sec_len"] = sec_len
        self.sec_len = sec_len

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution        
        
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # build layers
        self.layers = nn.ModuleList()

        sec_len = patches_resolution
        self.patches_resolution =[]
        self.real_size = []
        self.padding = nn.ModuleList()

        factor = None
        for i in range(self.num_layers):                
            if min(sec_len) > window_size:
                factor = window_size
            else:
                factor = compute_hcf(sec_len[0], sec_len[1])
            if factor % 2 != 0 :
                factor = factor * 2

            r1 , r2 = sec_len[0] % factor, sec_len[1] % factor
            self.real_size.append( (sec_len[0],sec_len[1]) )
            # self.padding.append( nn.ReplicationPad2d((0, 0, 0,4 - r)) )
            if r1 !=0 :
                sec_len[0] += factor - r1
                padding_windows = nn.ConstantPad2d((0, 0, 0,0,0,factor - r1) , 0 )
            else:
                padding_windows = None
            self.real_size.append( (sec_len[0],sec_len[1]) )
            self.padding.append(padding_windows)
            if r2 !=0 :
                sec_len[1] += factor - r2
                padding_windows = nn.ConstantPad2d((0, 0, 0,factor - r2) , 0 )
            else:
                padding_windows = None 
            self.padding.append(padding_windows)
            self.patches_resolution.append(sec_len.copy())
            sec_len[0] = sec_len[0]// 2        
            sec_len[1] = sec_len[1]// 2        

        print("Pads: ", self.patches_resolution)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            patch_size = self.patches_resolution[i_layer]
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patch_size[0],
                                                 patch_size[1]),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint, **kwargs)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        B, T, C, H, W = x.shape
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for i,layer in enumerate(self.layers):
            h_pad , w_pad = self.padding[2*i] , self.padding[2*i+1]
            if h_pad :
                x = x.view(x.shape[0], self.real_size[i*2][0], self.real_size[i*2][1], x.shape[-1])
                x = h_pad(x)
                x = x.view(x.shape[0], -1, x.shape[-1])
            if w_pad :
                x = x.view(x.shape[0], self.real_size[i*2+1][0], self.real_size[i*2+1][1], x.shape[-1])
                x = w_pad(x)
                x = x.view(x.shape[0], -1, x.shape[-1])
            x = layer(x)    
        
        
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1

        x = torch.flatten(x, 1)
        x = x.reshape(B, T, -1)
        x= x.mean(1)
        return x

    def forward(self, x):
        f = self.forward_features(x)
        x = self.head(f)
        return x, f


class Swin_timesformer(SwinTransformer):
# class Swin_timesformer(nn.Module):
    def __init__(self, sec_len = 64, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        print("Using Swin_timesformer")
        super().__init__(sec_len = sec_len, img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
            num_classes=num_classes, embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
            use_checkpoint=use_checkpoint, **kwargs)

        self.time_embed = nn.Parameter(torch.zeros(1, sec_len, 1, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)

    def forward_features(self, x):

        B, T, C, H, W = x.shape
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        n_patches = x.shape[1]
        ################ time emebedding ################
        x = x.view(B,T, -1 , self.embed_dim)        
        time_embed = self.time_embed.expand(1,T,n_patches, self.embed_dim)
        x += time_embed
        x = self.time_drop(x)
        ################ time emebedding ################
        x = x.view(B * T , -1, self.embed_dim)
        
        for i,layer in enumerate(self.layers):
            h_pad , w_pad = self.padding[2*i] , self.padding[2*i+1]
            if h_pad :
                x = x.view(x.shape[0], self.real_size[i*2][0], self.real_size[i*2][1], x.shape[-1])
                x = h_pad(x)
                x = x.view(x.shape[0], -1, x.shape[-1])
            if w_pad :
                x = x.view(x.shape[0], self.real_size[i*2+1][0], self.real_size[i*2+1][1], x.shape[-1])
                x = w_pad(x)
                x = x.view(x.shape[0], -1, x.shape[-1])
            x = layer(x)    
        
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = x.reshape(B, T, -1)
        x= x.mean(1)
        return x

    def forward(self, x):
        f = self.forward_features(x)
        x = self.head(f)
        return x, f