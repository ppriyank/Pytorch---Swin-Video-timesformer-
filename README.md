# Pytorch---Swin-Video-timesformer-

Swin is modified to acocunt for different resolution of window and Height and width. Original Swin code for a depth of "d" has constraints of Height and Width to be mutiple of 2^(d) and window size "w" 

### Space only is just temporal averaging
`mode=None`

### Joint Space-time   
`mode='joint_space_time'`


### Divided Space-time   
`mode='divided_space_time'`


### Debugging 
`mode='experimentation'`


```
import torch 
from swin_transformer import  Swin_timesformer

model = Swin_timesformer(sec_len=10, 
        img_size=(112, 112), 
        patch_size=10, 
        in_chans=3,
        num_classes=200,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[ 3, 6, 12, 24 ],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,    
        patch_norm=True,
        timesformer_mode='joint_space_time',
        )

model = model.cuda()
x =  torch.rand(2, 10, 3, 112, 112).cuda()
model(x)
```


