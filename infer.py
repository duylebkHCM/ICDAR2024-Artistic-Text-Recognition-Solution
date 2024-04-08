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
import time
import numpy as np

from PIL import Image
from pathlib import Path, PurePath

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
from strhub.data.utils import CharsetAdapter
from strhub.data.dataset import LmdbDataset
import yaml
from tools.functional import *
from functools import partial


def preprocess_label(label, remove_whitespace=True, normalize_unicode=True, max_label_len=25, charset_adapter=None):
    # Normally, whitespace is removed from the labels.
    if remove_whitespace:
        label = ''.join(label.split())
    # Normalize unicode composites (if any) and convert to compatible ASCII characters
    if normalize_unicode:
        label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
    # Filter by length before removing unsupported characters. The original label might be too long.
    if len(label) > max_label_len:
        label = label[:max_label_len]
        
    label = charset_adapter(label)
    
    return label

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')", default='/data/ocr/duyla4/Research/TEXT_RECOGNITION/WordArt/parseq/outputs/parseq_viptr/2024-04-01_04-43-18/checkpoints/last.ckpt')
    parser.add_argument('--root_dir', default='/data/ocr/duyla4/DATASET/OCR/VTCC_DATA/Research_Dataset/WordArt/test/ICDAR24-WordArt_testB')
    parser.add_argument('--images', nargs='+', default=['test_image'], help='Images to read')
    parser.add_argument('--label_path', nargs='+', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save_wrong_img', default=False)
    parser.add_argument('--submit', default=True)
    parser.add_argument('--logging', default=False)
    parser.add_argument('--conf_logging', default=True)
    parser.add_argument('--textlines_logging', default=True)
    parser.add_argument('--not_skip_empty', default=True)
    
    #Postprocess config
    parser.add_argument('--decoding_scheme', default='AR')
    parser.add_argument('--tta', default=False)
    parser.add_argument('--ensemble', default=True)
    parser.add_argument('--beam_width', default=1)
    parser.add_argument('--conf_threshold', default=0.0)
    parser.add_argument('--refine_iter', default=None)
    
    parser.add_argument('--verification', default=True)
    
    
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')
    
    timestamp = int(round(time.time()*1000))
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime(timestamp/1000))
    
    if args.verification:
        save_path = Path(args.checkpoint).parent.joinpath('result')
    else:
        save_path = Path(args.checkpoint).parent.joinpath(timestamp)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    
    with open(save_path.joinpath('postprocess_conf.yaml'), 'w') as cfg:
        opt = vars(args)
        yaml.safe_dump(data=opt, stream=cfg)
    
    if args.save_wrong_img and not save_path.joinpath('wrong_imgs').exists():
        save_path.joinpath('wrong_imgs').mkdir(parents=True)
        
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    char_test = "0123456789abcdefghijklmnopqrstuvwxyz"
    char_adapter = CharsetAdapter(char_test)
    
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    label_dict = {}
    if args.label_path is not None:
        for label_path in args.label_path:
            labels = open(PurePath(args.root_dir, label_path), 'r').readlines()
            label_dict.update({Path(args.root_dir).joinpath(line.strip().split(maxsplit=1)[0]).as_posix():line.strip().split('\t', maxsplit=2)[-1] for line in labels})
    
    total_paths = []
    for item in args.images:           
        images = Path(args.root_dir).joinpath(item).rglob('*.*g')
        total_paths += images
    
    total_paths = sorted(total_paths, key=lambda x: x.name)
    
    if args.submit:
        submission = open(save_path.joinpath('answer.txt'), 'w')
    if args.logging:
        fout = open(save_path.joinpath('infer.txt'), 'w')
    if args.conf_logging:
        fconf = open(save_path.joinpath('conf.txt'), 'w')
    if args.textlines_logging:
        save_textline = save_path.joinpath('textlines')
        if not save_textline.exists():
            save_textline.mkdir(parents=True)
        
    wrong_info = {'data': []}
    fname: Path
    for fname in total_paths:
        print('Predict image', fname.as_posix())
        
        # Load image and prepare for input
        image = Image.open(fname.as_posix()).convert('RGB')
        image = img_transform(image).unsqueeze(0)

        if args.tta:
            img1 = torch_fliplr(image).squeeze(0)
            img2 = torch_flipud(image).squeeze(0)
            img3 = torch_flipud(torch_fliplr(image)).squeeze(0)
            image = image.squeeze(0)
            
            enhanced_images = [image, img1, img2, img3]
            enhanced_images = torch.stack(enhanced_images, dim=0)
            enhanced_logits = model(enhanced_images.to(args.device), decoding_scheme=args.decoding_scheme, conf_threshold=args.conf_threshold, refine_iter=args.refine_iter).softmax(-1) #p shape B,L,Class
            enhanced_logits = torch.unbind(enhanced_logits, dim=0)
            
            # import pdb
            # pdb.set_trace()
            enhanced_probs = []
            for logit in enhanced_logits:
                pred, p = model.tokenizer.decode(logit.unsqueeze(0))
                enhanced_probs.append((pred, p)) # list of batch

            average_scores = np.array([sum(prob[0].cpu())/max(1, len(prob[0].cpu())) for _, prob in enhanced_probs])
            max_idx = np.argmax(average_scores)
            pred, p = enhanced_probs[max_idx]
        else:
            logits = model(image.to(args.device), decoding_scheme=args.decoding_scheme, conf_threshold=args.conf_threshold, refine_iter=args.refine_iter).softmax(-1)    
            pred, p = model.tokenizer.decode(logits)
        
        if args.conf_logging:
            save_conf = fname.relative_to(fname.parents[1]).as_posix()
            fconf.write(save_conf + '\t')
            for prob in p[0]:
                prob = str(prob.item())
                fconf.write(prob + ' ')
            fconf.write('\n')
        
        gt = label_dict.get(fname.as_posix(), None)
        msg = f'{fname}| predict: {pred[0]}'
        if args.textlines_logging:
            with open(save_textline.joinpath(fname.name).with_suffix('.txt'), 'w') as tl:
                tl.write(pred[0])
        
        
        #Postprocessing
        
        
        if args.submit:
            nor_pred = char_adapter(pred[0])
            if len(nor_pred) > 0 or args.not_skip_empty:
                save_path = fname.relative_to(fname.parents[1]).as_posix()
                save_path = save_path.replace('/', '\\')
                if args.ensemble:
                    total_prob = p[0].mean().item()
                    submission.write(save_path + ' ' + nor_pred + ' ' + str(total_prob) + '\n')
                else:
                    submission.write(save_path + ' ' + nor_pred +'\n')
                    
        if gt is not None:
            gt = preprocess_label(gt, charset_adapter=char_adapter)
            msg += f'| groundtruth: {gt}'
            if args.save_wrong_img and gt != pred[0]:
                shutil.copy(fname, save_path.joinpath('wrong_imgs').joinpath(fname.name))
                wrong_info['data'].append({
                    'text': pred[0],
                    'image_path': save_path.joinpath('wrong_imgs').joinpath(fname.name).as_posix()
                })
                with open(save_path.joinpath('wrong_imgs').joinpath(fname.stem).with_suffix('.txt'), 'w') as f:
                    f.write(gt)
        
        if args.logging:
            io_ = [sys.stdout, fout]
        else:
            io_ = [sys.stdout]
        for out in io_:      
            print(msg, file=out)

    if len(wrong_info['data']) > 0:
        with open(save_path.joinpath('wrong_info.json'), 'w') as j:
            json.dump(wrong_info, j, indent=2, ensure_ascii=True)

    if args.submit:
        submission.close()
    if args.logging:
        fout.close()
    if args.conf_logging:
        fconf.close()
        
if __name__ == '__main__':
    main()
