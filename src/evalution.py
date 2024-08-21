# load last epoch
import torch
if torch.__version__.count("1.4.0") == 0:
    print("This code uses pytorch 1.4.0!")
    assert False
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
from PIL import Image, ImageDraw
import numpy
import matplotlib.pyplot as plt
import cv2
import random

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2  # 1 rebar + background
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True,
    progress=True,
)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.roi_heads.detections_per_img=500 # max object detection
model.to(DEVICE)

trained_models = os.listdir('./model')
latest_epoch = -1
for model_name in trained_models:
    if not model_name.endswith('pth'):
        continue
    epoch = float(model_name.split('_')[1].split('.pth')[0])
    if epoch > latest_epoch:
        latest_epoch = epoch
        best_model_name = model_name
best_model_path = os.path.join('./model', best_model_name)
print('Loading model from', best_model_path)

model.load_state_dict(torch.load(best_model_path))
model.eval()

test_img_dir = r'./rebar_count_datasets/test_dataset'
files = os.listdir(test_img_dir)
random.shuffle(files)
if ".ipynb_checkpoints" in files:
    files.remove(".ipynb_checkpoints")
for i, file_name in enumerate(files[:15]):
    image_src = Image.open(os.path.join(test_img_dir, file_name)).convert("RGB")
    img_tensor = torchvision.transforms.ToTensor()(image_src)
    img_tensor
    with torch.no_grad():
        result_dict = model([img_tensor.to(DEVICE)])
    bbox = result_dict[0]["boxes"].cpu().numpy()
    scrs = result_dict[0]["scores"].cpu().numpy()

    image_draw = numpy.array(image_src.copy())

    rebar_count = 0
    for bbox,scr in zip(bbox,scrs):
        if len(bbox) > 0:
            if scr > 0.65:
                pt = bbox
                cv2.circle(image_draw, (int((pt[0] + pt[2]) * 0.5), int((pt[1] + pt[3]) * 0.5)), int((pt[2] - pt[0]) * 0.5 * 0.6), (255, 0, 0), -1)
                rebar_count += 1
    cv2.putText(image_draw, 'rebar_count: %d' % rebar_count, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    plt.figure(i, figsize=(15, 10))
    plt.imshow(image_draw)
    plt.show()