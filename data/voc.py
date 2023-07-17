# https://github.com/iCVTEAM/M3TR/blob/master/data/voc.py

import os
import random
import subprocess
from typing import Any, List, Optional, Tuple
import json
from PIL import Image
import torch
from torch.functional import Tensor
import shutil

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from loguru import logger

from .aug import Preprocess

DATA_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
DATA_URL_07 = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
)

LABEL2ID = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
}


def get_voc_labels() -> List[str]:
    keys = ["" for i in range(len(LABEL2ID))]
    for k in LABEL2ID:
        keys[LABEL2ID[k]] = k
    return keys


def prepare_voc12(root: str):
    work_dir = os.getcwd()

    if not os.path.exists(root):
        os.mkdir(root)

    tmp_dir = os.path.join(root, "tmp")

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    cached_file = os.path.join(tmp_dir, "VOCtrainval_11-May-2012.tar")
    if not os.path.exists(cached_file):
        logger.warning(f"downloading voc12 dataset")
        subprocess.call(f"wget -O {cached_file} {DATA_URL}", shell=True)
        logger.warning("done")

    extracted_dir = os.path.join(tmp_dir, "extracted")
    if not os.path.exists(extracted_dir):
        logger.warning(f"extracting dataset")
        os.mkdir(extracted_dir)
        subprocess.call(f"tar -xf {cached_file} -C {extracted_dir}", shell=True)
        logger.warning("done")

    vocdevkit = os.path.join(extracted_dir, "VOCdevkit", "VOC2012")
    assert os.path.exists(vocdevkit)

    img_dir = os.path.join(root, "img")
    if not os.path.exists(img_dir):
        logger.warning(f"refactor images dir structure")
        assert os.path.exists(os.path.join(vocdevkit, "JPEGImages"))
        subprocess.call(
            f"mv '{os.path.join(vocdevkit,'JPEGImages')}' '{img_dir}'", shell=True
        )
        logger.warning("done")

    cat_json = os.path.join(root, "categories.json")
    if not os.path.exists(cat_json):
        logger.warning(f"dumping categories mapping relations")
        json.dump(LABEL2ID, open(cat_json, "w"))
        logger.warning("done")

    anno_json = os.path.join(root, "annotations_train.json")
    if not os.path.exists(anno_json):
        logger.warning(f"generating annotations json")
        labels_dir = os.path.join(vocdevkit, "ImageSets", "Main")

        for phase in ["train", "val"]:
            img_annotations = {}
            img_ids = []
            for label in LABEL2ID:
                label_file = os.path.join(labels_dir, f"{label}_{phase}.txt")
                assert os.path.exists(label_file)
                with open(label_file, "r") as f:
                    for line in f:
                        spline = list(map(lambda x: x.strip(), line.strip().split(" ")))
                        image_id, positive = spline[0], spline[-1]
                        if positive == "1":
                            if image_id not in img_annotations:
                                img_annotations[image_id] = []
                                img_ids.append(image_id)
                            img_annotations[image_id].append(LABEL2ID[label])

            anno: List = []
            for i, image_id in enumerate(img_ids):
                t = {
                    "id": i,
                    "name": f"{image_id}.jpg",
                    "label": img_annotations[image_id],
                }
                anno.append(t)

            json_file = os.path.join(root, f"annotations_{phase}.json")
            json.dump(anno, open(json_file, "w"))

        logger.warning("done")


def prepare_voc07(root: str):
    work_dir = os.getcwd()

    if not os.path.exists(root):
        os.mkdir(root)

    tmp_dir = os.path.join(root, "tmp")

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    cached_file = os.path.join(tmp_dir, "VOCtrainval_06-Nov-2007.tar")
    if not os.path.exists(cached_file):
        logger.warning(f"downloading voc2007 dataset")
        subprocess.call(f"wget -O {cached_file} {DATA_URL_07}", shell=True)
        logger.warning("done")

    extracted_dir = os.path.join(tmp_dir, "extracted")
    if not os.path.exists(extracted_dir):
        logger.warning(f"extracting dataset")
        os.mkdir(extracted_dir)
        subprocess.call(f"tar -xf {cached_file} -C {extracted_dir}", shell=True)
        logger.warning("done")

    vocdevkit = os.path.join(extracted_dir, "VOCdevkit", "VOC2007")
    assert os.path.exists(vocdevkit)

    img_dir = os.path.join(root, "img")
    if not os.path.exists(img_dir):
        logger.warning(f"refactor images dir structure")
        assert os.path.exists(os.path.join(vocdevkit, "JPEGImages"))
        subprocess.call(
            f"mv '{os.path.join(vocdevkit,'JPEGImages')}' '{img_dir}'", shell=True
        )
        logger.warning("done")

    cat_json = os.path.join(root, "categories.json")
    if not os.path.exists(cat_json):
        logger.warning(f"dumping categories mapping relations")
        json.dump(LABEL2ID, open(cat_json, "w"))
        logger.warning("done")

    anno_json = os.path.join(root, "annotations_train.json")
    if not os.path.exists(anno_json):
        logger.warning(f"generating annotations json")
        labels_dir = os.path.join(vocdevkit, "ImageSets", "Main")

        for phase in ["train", "val"]:
            img_annotations = {}
            img_ids = []
            for label in LABEL2ID:
                label_file = os.path.join(labels_dir, f"{label}_{phase}.txt")
                assert os.path.exists(label_file)
                with open(label_file, "r") as f:
                    for line in f:
                        spline = list(map(lambda x: x.strip(), line.strip().split(" ")))
                        image_id, positive = spline[0], spline[-1]
                        if positive == "1":
                            if image_id not in img_annotations:
                                img_annotations[image_id] = []
                                img_ids.append(image_id)
                            img_annotations[image_id].append(LABEL2ID[label])

            anno: List = []
            for i, image_id in enumerate(img_ids):
                t = {
                    "id": i,
                    "name": f"{image_id}.jpg",
                    "label": img_annotations[image_id],
                }
                anno.append(t)

            json_file = os.path.join(root, f"annotations_{phase}.json")
            json.dump(anno, open(json_file, "w"))

        logger.warning("done")


class VOC12Dataset(Dataset):
    def __init__(self, root: str, preprocess: Preprocess, phase: str = "train"):
        assert phase == "train" or phase == "val"
        prepare_voc12(root)

        self.root = root
        self.preprocess = preprocess
        self.phase = phase

        with open(os.path.join(self.root, f"annotations_{self.phase}.json"), "r") as f:
            self.img_list = json.load(f)
        with open(os.path.join(self.root, f"categories.json"), "r") as f:
            self.cat2id = json.load(f)

        self.num_classes = len(self.cat2id)

        logger.warning(
            f"[VOC2012] load num of classes {len(self.cat2id)}, num images {len(self.img_list)}"
        )

    def __getitem__(self, index):
        item = self.img_list[index]

        _, filename, target = item["id"], item["name"], item["single_label"]
        image = self.preprocess.forward(
            Image.open(os.path.join(self.root, "img", filename)).convert("RGB")
        )

        target2 = torch.zeros(self.num_classes, dtype=torch.int)
        target2[target] = 1

        return {
            "index": index,
            "name": filename,
            "image": image,
            "target": target2,
        }

    def __len__(self):
        return len(self.img_list)

    def random(self, seed):
        random.seed(seed)
        random.shuffle(self.img_list)


    def drop_to_single(self, seed):
        random.seed(seed)
        # drop labels
        for img in self.img_list:
            # single_label is a list of label index
            # [0, 1, 9]
            img["single_label"] = [random.choice(img["label"])]


    def debug_retain_all_labels(self):
        for img in self.img_list:
            img["single_label"] = img["label"]

    def get_all_single_labels(self) -> Tensor:
        raw_targets = []
        for img in self.img_list:
            target = torch.zeros(self.num_classes, dtype=torch.int)
            target[img["single_label"]] = 1
            raw_targets.append(target.unsqueeze(0))

        return torch.cat(raw_targets)

    def get_real_labels(self):
        raw_targets = []
        for img in self.img_list:
            target = torch.zeros(self.num_classes, dtype=torch.int)
            target[img["label"]] = 1
            raw_targets.append(target.unsqueeze(0))

        return torch.cat(raw_targets)

    def get_real_labels_by_id(self, ids: Tensor, one_hot=True):
        labels = []
        for i in ids:
            if one_hot:
                label = torch.zeros(self.num_classes, dtype=torch.long)
                label[self.img_list[i.item()]["label"]] = 1
            else:
                label = self.data_lists[i.item()]["label"]
            labels.append(label.unsqueeze(0))
        return torch.cat(labels)


class VOC07Dataset(Dataset):
    def __init__(self, root: str, preprocess: Preprocess, phase: str = "train"):
        assert phase == "train" or phase == "val"
        prepare_voc07(root)

        self.root = root
        self.preprocess = preprocess
        self.phase = phase

        with open(os.path.join(self.root, f"annotations_{self.phase}.json"), "r") as f:
            self.img_list = json.load(f)
        with open(os.path.join(self.root, f"categories.json"), "r") as f:
            self.cat2id = json.load(f)

        self.num_classes = len(self.cat2id)

        logger.warning(
            f"[VOC2007] load num of classes {len(self.cat2id)}, num images {len(self.img_list)}"
        )

    def __getitem__(self, index):
        item = self.img_list[index]

        _, filename, target = item["id"], item["name"], item["single_label"]
        image = self.preprocess.forward(
            Image.open(os.path.join(self.root, "img", filename)).convert("RGB")
        )

        target2 = torch.zeros(self.num_classes, dtype=torch.int)
        target2[target] = 1

        return {
            "index": index,
            "name": filename,
            "image": image,
            "target": target2,
        }

    def __len__(self):
        return len(self.img_list)

    def random(self, seed):
        random.seed(seed)
        random.shuffle(self.img_list)


    def drop_to_single(self, seed):
        random.seed(seed)
        # drop labels
        for img in self.img_list:
            # single_label is a list of label index
            # [0, 1, 9]
            img["single_label"] = [random.choice(img["label"])]


    def debug_retain_all_labels(self):
        for img in self.img_list:
            img["single_label"] = img["label"]

    def get_all_single_labels(self) -> Tensor:
        raw_targets = []
        for img in self.img_list:
            target = torch.zeros(self.num_classes, dtype=torch.int)
            target[img["single_label"]] = 1
            raw_targets.append(target.unsqueeze(0))

        return torch.cat(raw_targets)

    def get_real_labels(self):
        raw_targets = []
        for img in self.img_list:
            target = torch.zeros(self.num_classes, dtype=torch.long)
            target[img["label"]] = 1
            raw_targets.append(target.unsqueeze(0))

        return torch.cat(raw_targets)

    def get_real_labels_by_id(self, ids: Tensor, one_hot=True):
        labels = []
        for i in ids:
            if one_hot:
                label = torch.zeros(self.num_classes, dtype=torch.long)
                label[self.img_list[i.item()]["label"]] = 1
            else:
                label = self.data_lists[i.item()]["label"]
            labels.append(label.unsqueeze(0))
        return torch.cat(labels)
