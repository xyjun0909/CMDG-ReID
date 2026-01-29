# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import sys
import os
import os.path as osp
import random
from . import DATASET_REGISTRY
from .bases import ImageDataset
import json
import time
import errno
import numpy as np
import warnings
import PIL
import torch
from PIL import Image
import pdb
import glob
__all__ = ['VIPeR', ]


@DATASET_REGISTRY.register()
class VIPeR(ImageDataset):
    """VIPeR.
    Reference:
        Gray et al. Evaluating appearance models for recognition, reacquisition, and tracking. PETS 2007.
    URL: `<https://vision.soe.ucsc.edu/node/178>`_
    
    Dataset statistics:
        - identities: 632.
        - images: 632 x 2 = 1264.
        - cameras: 2.
    """
    dataset_dir = 'VIPeR'
    dataset_url = 'http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip'

    def __init__(self, root='', split_id=9, **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        required_files = [self.dataset_dir]
        self.check_before_run(required_files)

        train = self.process_dir_new(self.dataset_dir,'train')
        query = self.process_dir_new(self.dataset_dir,'query')
        gallery = self.process_dir_new(self.dataset_dir,'gallery')

        super().__init__(train, query, gallery, **kwargs)

    def process_dir_new(self, dataset_dir, split):
        josn_path = "./datasets/captions/viper.json"
        data = []
        type = split
        with open(josn_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        for item in json_data:
            if type == item.get('split'): 
                captions = item.get('captions')
                file_path = item.get('file_path')
                img_path = dataset_dir + '/' + file_path
                pid = item.get('id')
                camid = item.get('camid')
                data.append((img_path, pid, camid, 'viper',captions))
        return data

    def prepare_split(self):
        if not osp.exists(self.split_path):
            print('Creating 10 random splits of train ids and test ids')

            cam_a_imgs = sorted(glob.glob(osp.join(self.cam_a_dir, '*.bmp')))
            cam_b_imgs = sorted(glob.glob(osp.join(self.cam_b_dir, '*.bmp')))
            assert len(cam_a_imgs) == len(cam_b_imgs)
            num_pids = len(cam_a_imgs)
            print('Number of identities: {}'.format(num_pids))
            num_train_pids = num_pids // 2
            """
            In total, there will be 20 splits because each random split creates two
            sub-splits, one using cameraA as query and cameraB as gallery
            while the other using cameraB as query and cameraA as gallery.
            Therefore, results should be averaged over 20 splits (split_id=0~19).
            
            In practice, a model trained on split_id=0 can be applied to split_id=0&1
            as split_id=0&1 share the same training data (so on and so forth).
            """
            splits = []
            for _ in range(10):
                order = np.arange(num_pids)
                np.random.shuffle(order)
                train_idxs = order[:num_train_pids]
                test_idxs = order[num_train_pids:]
                assert not bool(set(train_idxs) & set(test_idxs)), \
                    'Error: train and test overlap'

                train = []
                for pid, idx in enumerate(train_idxs):
                    cam_a_img = cam_a_imgs[idx]
                    cam_b_img = cam_b_imgs[idx]
                    train.append((cam_a_img, pid, 0,'viper'))
                    train.append((cam_b_img, pid, 1,'viper'))

                test_a = []
                test_b = []
                for pid, idx in enumerate(test_idxs):
                    cam_a_img = cam_a_imgs[idx]
                    cam_b_img = cam_b_imgs[idx]
                    test_a.append((cam_a_img, pid, 0,'viper'))
                    test_b.append((cam_b_img, pid, 1,'viper'))

                # use cameraA as query and cameraB as gallery
                split = {
                    'train': train,
                    'query': test_a,
                    'gallery': test_b,
                    'num_train_pids': num_train_pids,
                    'num_query_pids': num_pids - num_train_pids,
                    'num_gallery_pids': num_pids - num_train_pids
                }
                splits.append(split)

                # use cameraB as query and cameraA as gallery
                split = {
                    'train': train,
                    'query': test_b,
                    'gallery': test_a,
                    'num_train_pids': num_train_pids,
                    'num_query_pids': num_pids - num_train_pids,
                    'num_gallery_pids': num_pids - num_train_pids
                }
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file saved to {}'.format(self.split_path))
            
def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))