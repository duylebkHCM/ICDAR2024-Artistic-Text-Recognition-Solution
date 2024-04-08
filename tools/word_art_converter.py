#!/usr/bin/env python3

import sys
import re
import random

root = sys.argv[1]

with open(root + '/train_labels.txt', 'r') as f:
    d = f.readlines()

random.shuffle(d)

f_train = open(root + '/lmdb_train.txt', 'w')
f_val = open(root + '/lmdb_val.txt', 'w')

for line in d:
    line = line.replace('\\', '/')
    img_path = re.findall(r'^.*\s', string=line.rstrip())
    assert len(img_path) == 1
    img_path = img_path[0]
    label = line.rstrip().replace(img_path, '')
    if random.random() <= 0.8:
        f_train.write('\t'.join([img_path, label]) + '\n')
    else:
        f_val.write('\t'.join([img_path, label]) + '\n')


f_train.close()
f_val.close()