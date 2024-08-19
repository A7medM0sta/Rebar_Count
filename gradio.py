import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
from PIL import Image
import numpy as np
import cv2
import gradio as gr

# Load the model and set it to evaluation mode
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_classes = 2  # 1 rebar + background
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights="DEFAULT"  # Updated to use the latest weights parameter
)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.roi_heads.detections_per_img = 500  # max object detection
model.to(DEVICE)

# Load the latest trained model
trained_models = os.listdir("./model")
latest_epoch = -1
for model_name in trained_models:
    if not model_name.endswith("pth"):
        continue
    epoch = float(model_name.split("_")[1].split(".pth")[0])
    if epoch > latest_epoch:
        latest_epoch = epoch
        best_model_name = model_name

best_model_path = os.path.join("./model", best_model_name)
print("Loading model from", best_model_path)

model.load_state_dict(torch.load(best_model_path))
model.eval()


# Define the prediction function
def predict(image):
    image_src = Image.fromarray(image).convert("RGB")
    img_tensor = torchvision.transforms.ToTensor()(image_src).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        result_dict = model([img_tensor[0]])

    bbox = result_dict[0]["boxes"].cpu().numpy()
    scrs = result_dict[0]["scores"].cpu().numpy()

    image_draw = np.array(image_src.copy())

    rebar_count = 0
    for bbox, scr in zip(bbox, scrs):
        if scr > 0.65:
            pt = bbox
            cv2.circle(
                image_draw,
                (int((pt[0] + pt[2]) * 0.5), int((pt[1] + pt[3]) * 0.5)),
                int((pt[2] - pt[0]) * 0.5 * 0.6),
                (255, 0, 0),
                -1,
            )
            rebar_count += 1

    # Add the rebar count text on the image (if you still want it there)
    cv2.putText(
        image_draw,
        f"Rebar count: {rebar_count}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        3,
    )

    return image_draw, rebar_count


# Build the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[
        gr.Image(type="numpy", label="Processed Image with Predictions"),
        gr.Number(label="Number of Rebars Detected"),
    ],
)

# Launch the interface
interface.launch()
