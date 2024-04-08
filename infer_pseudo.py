#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
import sys
import unicodedata
import shutil
import json

from PIL import Image
from pathlib import Path, PurePath

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
from strhub.data.utils import CharsetAdapter
from strhub.data.dataset import LmdbDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')", default='/data/ocr/duyla4/Research/TEXT_RECOGNITION/WordArt/parseq/outputs/parseq_custom/2024-03-15_11-35-19/checkpoints/last.ckpt')
    parser.add_argument('--root_dir', default='/data/ocr/duyla4/DATASET/OCR/VTCC_DATA/Research_Dataset/Union14M/original/Union14M-U')
    parser.add_argument('--device', default='cuda')
    
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')
    
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    char_test = "0123456789abcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    
    total_paths = Path(args.root_dir).rglob('*/data.mdb')
    
    for path in total_paths:
        print(path.parent.as_posix())
        dset = LmdbDataset(path.parent.as_posix(), charset=char_test, max_label_len=None, unlabelled=True, transform=img_transform)
        dloader=DataLoader(
            dataset=dset,
            batch_size=32, 
            num_workers=4,
            pin_memory=True
        )

        count = 0
        imgs: torch.FloatTensor
        BUFFER=1000
        print('Start create pseudo label for {}'.format(path.as_posix()))
        for imgs, labels in tqdm(dloader, total=len(dloader)):
            imgs = imgs.to(args.device)
            probs = model(imgs).softmax(-1)
            preds, probs = model.tokenizer.decode(probs)
            for img, pred, prob in zip(imgs, preds, probs):
                if prob.mean().item() > 0.75 :
                    nuffer_save = str((count+1)//BUFFER)
                    #save img and pred (pseudo label) in appropriate directory
                    save_path = path.parent.parent.joinpath(path.parent.name + '_textlines').joinpath(nuffer_save)
                    
                    if not save_path.exists():
                        save_path.mkdir(parents=True)
                    
                    img = (img*0.5 + 0.5)*255.0
                    img = img.int()
                    img = img.cpu().permute(1,2,0).numpy()                    
                    img = img.astype('uint8')
                    img = Image.fromarray(img, mode='RGB')
                    
                    img.save(save_path.joinpath(str(count)).with_suffix('.png'))
                    with open(save_path.joinpath(str(count)).with_suffix('.txt'), 'w') as f:
                        f.write(pred)
                    count+=1
            
        print('Total number of pseudo label are {}'.format(count))
        
if __name__ == '__main__':
    main()
