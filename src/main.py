import os
import torch
import shutil
import torchvision
import transforms as T
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import utils
from engine import train_one_epoch, evaluate
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from coco import coco_eval, coco_utils
from utils import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform(train):
    """
    Creates a set of image transforms.
    If training, includes a random horizontal flip.

    Args:
        train (bool): Whether or not the transform is for training.

    Returns:
        transforms (torchvision.transforms.Compose): A composition of transforms.
    """
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def read_xml(xml_path):
    """
    Reads XML annotation files and extracts bounding box information and labels.

    Args:
        xml_path (str): The path to the XML file.

    Returns:
        tuple: A tuple containing an array of bounding boxes and a list of labels.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    labels = []
    for element in root.findall('object'):
        label = element.find('name').text
        if label == 'steel':
            bndbox = element.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
    return np.array(boxes, dtype=np.float64), labels


# Creating PyTorch dataset for rebar
class RebarDataset(torch.utils.data.Dataset):
    """
    Custom dataset for rebar detection in images. Reads images and corresponding
    XML annotations and applies transforms.
    """

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))

        if ".ipynb_checkpoints" in self.imgs:
            self.imgs.remove(".ipynb_checkpoints")

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        box_path = os.path.join(self.root, "Annotations", self.imgs[idx].split(".")[0] + '.xml')
        img = Image.open(img_path).convert("RGB")
        boxes, _ = read_xml(box_path)

        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=DEVICE)
        labels = torch.ones((len(boxes),), dtype=torch.int64, device=DEVICE)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], device=DEVICE),
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# Load Faster R-CNN model pretrained with ResNet50 backbone
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True)
num_classes = 2  # 1​⬤