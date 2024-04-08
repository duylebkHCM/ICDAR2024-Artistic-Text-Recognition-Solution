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
import glob
import io
import logging
import unicodedata
from pathlib import Path, PurePath
from typing import Callable, Optional, Union, List
import six
import lmdb
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
import torch
import cv2
from strhub.data.utils import CharsetAdapter, BaseTokenizer
from strhub.data.render_text_mask import  render_normal
log = logging.getLogger(__name__)


def build_tree_dataset(root: Union[PurePath, str, List[PurePath], List[str]], *args, **kwargs):
    try:
        kwargs.pop('root')  # prevent 'root' from being passed via kwargs
    except KeyError:
        pass
    
    if isinstance(root, list):
        datasets = []
        for r in root:
            dset = build_tree_dataset(r, *args, **kwargs)
            datasets += dset.datasets
        return ConcatDataset(datasets)
    else:
        root = Path(root).absolute()
        log.info(f'dataset root:\t{root}')
        datasets = []
        for mdb in glob.glob(str(root / '**/data.mdb'), recursive=True):
            mdb = Path(mdb)
            ds_name = str(mdb.parent.relative_to(root))
            ds_root = str(mdb.parent.absolute())
            dataset = LmdbDataset(ds_root, *args, **kwargs)
            log.info(f'\tlmdb:\t{ds_name}\tnum samples: {len(dataset)}')
            datasets.append(dataset)
        return ConcatDataset(datasets)


class LmdbDataset(Dataset):
    """Dataset interface to an LMDB database.

    It supports both labelled and unlabelled datasets. For unlabelled datasets, the image index itself is returned
    as the label. Unicode characters are normalized by default. Case-sensitivity is inferred from the charset.
    Labels are transformed according to the charset.
    """

    def __init__(self, root: str, charset: str, max_label_len: int, min_image_dim: int = 0,
                 remove_whitespace: bool = True, normalize_unicode: bool = True,
                 unlabelled: bool = False, transform: Optional[Callable] = None, font_path=None, use_class_binary_sup=False, font_size=0,font_strength=0, tokenizer: BaseTokenizer=None):
        self._env = None
        self.root = root
        self.unlabelled = unlabelled
        self.transform = transform
        self.labels = []
        self.filtered_index_list = []
        self.tokenizer=tokenizer
        self.num_samples = self._preprocess_labels(charset, remove_whitespace, normalize_unicode,
                                                   max_label_len, min_image_dim)
        # init for binary mask
        self.use_class_binary_sup = use_class_binary_sup
        if  font_path is not None:
            from pygame import freetype
            freetype.init()

            # init font
            self.font = freetype.Font(font_path)
            self.font.antialiased = True
            self.font.origin = True

            # choose font style
            self.font.size = font_size
            self.font.underline = False
            
            self.font.strong = True
            self.font.strength = font_strength
            self.font.oblique = False

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None    
    
    def _create_env(self):
        return lmdb.open(self.root, max_readers=1, readonly=True, create=False,
                         readahead=False, meminit=False, lock=False)

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env

    def _preprocess_labels(self, charset, remove_whitespace, normalize_unicode, max_label_len, min_image_dim):
        charset_adapter = CharsetAdapter(charset)
        
        with self._create_env() as env, env.begin() as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            
            if self.unlabelled:
                return num_samples
            
            for index in range(num_samples):
                index += 1  # lmdb starts with 1
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode()
                
                # Normally, whitespace is removed from the labels.
                if remove_whitespace:
                    label = ''.join(label.split())
                    
                # Normalize unicode composites (if any) and convert to compatible ASCII characters
                if normalize_unicode:
                    label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
                    
                # Filter by length before removing unsupported characters. The original label might be too long.
                if len(label) > max_label_len:
                    continue
                
                label = charset_adapter(label)
                
                # We filter out samples which don't contain any supported characters
                if not label:
                    continue
                
                # Filter images that are too small.
                if min_image_dim > 0:
                    img_key = f'image-{index:09d}'.encode()
                    buf = io.BytesIO(txn.get(img_key))
                    w, h = Image.open(buf).size
                    if w < self.min_image_dim or h < self.min_image_dim:
                        continue
                    
                self.labels.append(label)
                self.filtered_index_list.append(index)
                
        return len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.unlabelled:
            label = index
        else:
            label = self.labels[index]
            index = self.filtered_index_list[index]

        if self.use_class_binary_sup:
            binary_mask, bbs = render_normal(self.font, label)
            cate_aware_surf = np.zeros((binary_mask.shape[0], binary_mask.shape[1], len(self.tokenizer)-2)).astype(np.uint8)
            
            for char, bb in zip(label, bbs):
                char_id = self.tokenizer._stoi[char]
                cate_aware_surf[:, :, char_id][bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]] = binary_mask[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
                
            binary_mask = cate_aware_surf
            binary_mask = cv2.resize(binary_mask, (128//2, 32//2))
      
            if np.max(binary_mask)>0:
                binary_mask = binary_mask / np.max(binary_mask) # [0 ~ 1]
                binary_mask = torch.from_numpy(binary_mask).float()
                if binary_mask.dtype != torch.float32:
                    print(binary_mask.dtype)
      
        img_key = f'image-{index:09d}'.encode()
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
        buf = io.BytesIO(imgbuf)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.use_class_binary_sup:
            return img, label, binary_mask
        else:
            return img, label
