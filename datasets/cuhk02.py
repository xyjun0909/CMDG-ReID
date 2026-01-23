import glob
import os.path as osp
import json
import re
import warnings

from .bases import ImageDataset
from . import DATASET_REGISTRY
import pdb
__all__ = ['CUHK02', ]

@DATASET_REGISTRY.register()
class CUHK02(ImageDataset):
    """CUHK02.
    Reference:
        Li and Wang. Locally Aligned Feature Transforms across Views. CVPR 2013.
    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html>`_
    
    Dataset statistics:
        - 5 camera view pairs each with two cameras
        - 971, 306, 107, 193 and 239 identities from P1 - P5
        - totally 1,816 identities
        - image format is png
    Protocol: Use P1 - P4 for training and P5 for evaluation.
    Note: CUHK01 and CUHK02 overlap.
    """
    dataset_dir = 'cuhk02'
    cam_pairs = ['P1', 'P2', 'P3', 'P4', 'P5']
    test_cam_pair = 'P5'
    dataset_name = "cuhk02"

    def __init__(self, root='', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.json_path = osp.join(self.dataset_dir,'Qwen2_cuhk02.json')

        required_files = [self.dataset_dir]
        self.check_before_run(required_files)
        self.train_pids=set()
        self.query_pids=set()
        self.gallery_pids=set()
        # train, query, gallery = self.get_data_list()
        train, query, gallery = self.get_data_from_json()
        super(CUHK02, self).__init__(train, query, gallery, **kwargs)
    
    def get_data_from_json(self):
        with open(self.json_path, 'r') as f:
            json_data = json.load(f)

        train, query, gallery = [], [], []
        num_train_pids = 0  # 关键：按摄像头对累计训练集PID数量（与原逻辑一致）
        camid = 0  # 关键：在摄像头对循环内递增，与原逻辑一致

        # 按摄像头对（P1-P5）依次处理，复刻原代码的循环逻辑
        for cam_pair in self.cam_pairs:  # 遍历P1→P2→P3→P4→P5
            # 1. 筛选当前摄像头对的所有JSON条目
            cam_pair_entries = [
                item for item in json_data
                if cam_pair in item['file_path']  # 条目属于当前摄像头对
            ]
            if not cam_pair_entries:
                continue

            # 2. 处理训练集（P1-P4）
            if cam_pair != self.test_cam_pair:  # P1-P4为训练集
                # 提取当前摄像头对的所有原始PID（如"877"）
                original_pids = set()
                for item in cam_pair_entries:
                    if item['split'] == 'train':  # 确保是训练集条目
                        original_pid = item['id'].split('_')[-1]
                        original_pids.add(original_pid)
                original_pids = sorted(original_pids)  # 排序确保映射稳定

                # 构建PID映射：当前摄像头对的PID从num_train_pids开始编号（与原pid2label一致）
                pid2label = {pid: num_train_pids + i for i, pid in enumerate(original_pids)}

                # 处理当前摄像头对的cam1和cam2
                for item in cam_pair_entries:
                    if item['split'] != 'train':
                        continue  # 只处理训练集条目

                    file_path = item['file_path']
                    captions = item['captions']
                    original_pid = item['id'].split('_')[-1]
                    cam = file_path.split('/')[1]  # "cam1"或"cam2"

                    # 映射PID并添加前缀
                    mapped_pid = pid2label[original_pid]
                    self.train_pids.add(mapped_pid)
                    final_pid = f"{self.dataset_name}_{mapped_pid}"

                    # 分配camid（cam1用当前camid，cam2用camid+1）
                    if cam == 'cam1':
                        final_camid = f"{self.dataset_name}_{camid}"
                    else:  # cam2
                        final_camid = f"{self.dataset_name}_{camid + 1}"

                    train.append((
                        osp.join(self.dataset_dir, file_path),
                        final_pid,
                        final_camid,
                        'cuhk02',
                        captions
                    ))

                # 更新camid（当前摄像头对用了2个camid：camid和camid+1）
                camid += 2
                # 更新num_train_pids（累计当前摄像头对的PID数量）
                num_train_pids += len(original_pids)

            # 3. 处理测试集（P5）
            else:  # cam_pair == 'P5'
                # 处理cam1（查询集）和cam2（画廊集）
                for item in cam_pair_entries:
                    file_path = item['file_path']
                    captions = item['captions']
                    original_pid = item['id'].split('_')[-1]
                    cam = file_path.split('/')[1]

                    # 测试集PID偏移：在训练集总PID数基础上递增（替代原固定+1577）
                    pid = int(original_pid) + num_train_pids
                    # 分配camid（cam1用当前camid，cam2用camid+1）
                    final_camid = camid if cam == 'cam1' else camid + 1

                    if cam == 'cam1':
                        self.query_pids.add(pid)
                        query.append((
                            osp.join(self.dataset_dir, file_path),
                            pid,
                            final_camid,
                            'cuhk02',
                            captions
                        ))
                    else:  # cam2
                        self.gallery_pids.add(pid)
                        gallery.append((
                            osp.join(self.dataset_dir, file_path),
                            pid,
                            final_camid,
                            'cuhk02',
                            captions
                        ))

                # 更新camid（P5用了2个camid）
                camid += 2
        return train, query, gallery

        
    def get_data_list(self):
        num_train_pids, camid = 0, 0
        train, query, gallery = [], [], []

        for cam_pair in self.cam_pairs:
            cam_pair_dir = osp.join(self.dataset_dir, cam_pair)

            cam1_dir = osp.join(cam_pair_dir, 'cam1')
            cam2_dir = osp.join(cam_pair_dir, 'cam2')

            impaths1 = glob.glob(osp.join(cam1_dir, '*.png'))
            impaths2 = glob.glob(osp.join(cam2_dir, '*.png'))

            if cam_pair == self.test_cam_pair:
                # add images to query
                for impath in impaths1:
                    pid = int(osp.basename(impath).split('_')[0])+1577
                    self.query_pids.add(pid)
                    # pid = self.dataset_name + "_" + str(pid)
                    query.append((impath, int(pid), int(camid),'cuhk02'))
                camid += 1

                # add images to gallery
                for impath in impaths2:
                    pid = int(osp.basename(impath).split('_')[0])+1577
                    pid = int(pid)
                    self.gallery_pids.add(pid)
                    # pid = self.dataset_name + "_" + str(pid)
                    gallery.append((impath, int(pid), int(camid),'cuhk02'))
                camid += 1

            else:
                pids1 = [
                    osp.basename(impath).split('_')[0] for impath in impaths1
                ]
                pids2 = [
                    osp.basename(impath).split('_')[0] for impath in impaths2
                ]
                pids = set(pids1 + pids2)
                pid2label = {
                    pid: label + num_train_pids
                    for label, pid in enumerate(pids)
                }

                # add images to train from cam1
                for impath in impaths1:
                    pid = osp.basename(impath).split('_')[0]
                    self.train_pids.add(pid2label[pid])
                    pid = self.dataset_name + "_" + str(pid2label[pid])
                    train.append((impath, pid, self.dataset_name + "_" + str(camid),'cuhk02'))
                camid += 1

                # add images to train from cam2
                for impath in impaths2:
                    pid = osp.basename(impath).split('_')[0]
                    self.train_pids.add(pid2label[pid])
                    pid = self.dataset_name + "_" + str(pid2label[pid])
                    train.append((impath, pid, self.dataset_name + "_" + str(camid),'cuhk02'))
                camid += 1
                num_train_pids += len(pids)

        return train, query, gallery