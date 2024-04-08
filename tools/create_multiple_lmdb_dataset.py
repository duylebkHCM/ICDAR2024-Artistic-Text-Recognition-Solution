import os
import lmdb
import cv2
import numpy as np
import yaml
import shutil
import time
from pathlib import Path

def load_yaml(yaml_path):
    with open(yaml_path, mode='r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    return data


def check_image_is_valid(image_bin):
    if image_bin is None:
        return False
    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
    img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def label_loader(label_path):
    with open(label_path, mode='r', encoding='utf-8') as f:
        label = f.read().rstrip()
    return label


def create_dataset(root_dir, output_dir, label_path, map_size=1048576):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        :param map_size: size for init env in lmdb
        :param label_path: list of label paths
        :param output_dir: LMDB output dir
        :param root_dir: root of raw dataset
    """
    annotation_path = os.path.join(root_dir, label_path)
    with open(annotation_path, mode='r', encoding='utf-8') as anno_file:
        lines = anno_file.readlines()
    lines = [line.strip() for line in lines]

    num_samples = len(lines)
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    env = lmdb.open(output_dir, map_size=map_size)
    cache = {}
    cnt = 1
    error = 0
    try:
        for idx in range(num_samples):
            image_path = lines[idx]
            image_path = os.path.join(root_dir, image_path)
            if (image_path.endswith('.jpg') or image_path.endswith('.png')) and os.path.exists(image_path):
                label_path = os.path.splitext(image_path)[0] + '.txt'
            else:
                error += 1
                print('{} does not exist!'.format(image_path))
                continue

            with open(image_path, 'rb') as f:
                image_bin = f.read()

            if not check_image_is_valid(image_bin):
                error += 1
                print('{} is not a valid image'.format(image_path))
                continue

            imageKey = ('image-%09d' % cnt).encode()
            labelKey = ('label-%09d' % cnt).encode()
            label = label_loader(label_path)

            cache[imageKey] = image_bin
            cache[labelKey] = label.encode()

            cnt += 1

            if cnt % 100 == 0:
                write_cache(env, cache)
                cache = {}

        num_samples = cnt - 1
        cache['num-samples'.encode()] = str(num_samples).encode()
        write_cache(env, cache)
        print('Created dataset with {} samples'.format(num_samples))
        print("Error = {} samples".format(error))
        return True
    except Exception as e:
        env.close()
        return False


def get_size(root_dir, annotation_path):
    annotation_path = os.path.join(root_dir, annotation_path)
    with open(annotation_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    total_size = 0
    for path_line in lines:
        fp = os.path.join(root_dir, path_line)
        lp = os.path.splitext(fp)[0] + '.txt'
        if not os.path.islink(fp):
            total_size += os.path.getsize(fp)
        if not os.path.islink(lp):
            total_size += os.path.getsize(lp)
    return total_size


def lmdb_building(config):
    root_dir = config['root_dir']
    output_dir = config['output_dir']
    label_paths = config['label_path']
    
    if not Path(label_paths).is_dir():
        label_paths = [Path(label_paths)]
    else:
        label_paths=list(Path(label_paths).rglob('*.txt'))
    for label_path in label_paths:
        sub_output_dir = Path(output_dir).joinpath(label_path.stem).as_posix()
        
        label_path = label_path.as_posix()
        total_size = get_size(root_dir, label_path)
        factor = config['base_factor']
        print("Size of Folder = {} bytes".format(total_size))
        success = False
        
        if os.path.exists(sub_output_dir):
            shutil.rmtree(sub_output_dir)

        while not success:
            print("factor = {}".format(factor))
            estimated_size = int(total_size * factor)
            success = create_dataset(root_dir, sub_output_dir, label_path, map_size=1099511627776)
            if success:
                print("Build LMDB dataset sucessully! The final factor is {}".format(factor))
            else:
                shutil.rmtree(sub_output_dir)
                factor = factor + config['increase_rate']


if __name__ == '__main__':
    begin = time.time()
    config_path = 'lmdb_config.yaml'
    cfg = load_yaml(config_path)
    lmdb_building(cfg)
    print("Processing time: {} s".format(time.time() - begin))
