"""Keep source masks at native resolution; letterbox only at model input time."""
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF,InterpolationMode,ColorJitter

MEAN=(.485,.456,.406)
STD=(.229,.224,.225)


class SegmentationDataset(Dataset):
    def __init__(self,root,split,size=768,augment=False):
        self.root=Path(root);self.split=split;self.size=size;self.augment=augment
        self.ids=(self.root/'splits'/(split+'.txt')).read_text(encoding='utf-8').split()
        if not self.ids or len(self.ids)!=len(set(self.ids)):raise ValueError('Empty split or duplicate sample IDs')
        for name in self.ids:
            if Path(name).name!=name:raise ValueError('Invalid sample ID')
            if not (self.root/'images'/split/(name+'.png')).is_file() or not (self.root/'masks'/split/(name+'.png')).is_file():
                raise ValueError(f'Missing image/mask pair: {name}')
        self.jitter=ColorJitter(brightness=.2,contrast=.2,saturation=.12,hue=.02)

    def __len__(self):return len(self.ids)

    def __getitem__(self,index):
        name=self.ids[index]
        with Image.open(self.root/'images'/self.split/(name+'.png')) as file:image=file.convert('RGB')
        with Image.open(self.root/'masks'/self.split/(name+'.png')) as file:mask=file.convert('L')
        if image.size!=mask.size or not set(mask.tobytes())<={0,1,2}:raise ValueError(f'Invalid mask: {name}')
        w,h=image.size;scale=self.size/max(w,h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
        image=image.resize((nw,nh),Image.Resampling.BILINEAR)
        mask=mask.resize((nw,nh),Image.Resampling.NEAREST)
        padding=((self.size-nw)//2,(self.size-nh)//2,self.size-nw-(self.size-nw)//2,self.size-nh-(self.size-nh)//2)
        image=TF.pad(image,padding,fill=(124,116,104));mask=TF.pad(mask,padding,fill=255)
        if self.augment:
            angle=random.uniform(-12,12);translate=[round(random.uniform(-.025,.025)*self.size) for _ in range(2)]
            scale=random.uniform(.95,1.05)
            image=TF.affine(image,angle,translate,scale,[0.,0.],InterpolationMode.BILINEAR,fill=(124,116,104))
            mask=TF.affine(mask,angle,translate,scale,[0.,0.],InterpolationMode.NEAREST,fill=255)
            image=self.jitter(image)
        image=TF.normalize(TF.to_tensor(image),MEAN,STD)
        target=torch.from_numpy(np.array(mask,dtype=np.int64))
        return image,target
