import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
from PIL import Image
import numpy as np
import cv2
import gradio as gr
import unittest

class ModelLoader:
    """
    A class used to load the model for rebar detection.

    ...

    Attributes
    ----------
    device : torch.device
        The device to which the model is loaded.
    num_classes : int
        The number of classes for the model.
    detections_per_img : int
        The maximum number of object detections.
    model : torchvision.models.detection.faster_rcnn.FastRCNNPredictor
        The loaded model.

    Methods
    -------
    load_model():
        Loads the pre-trained model from torchvision.
    load_trained_model(model_dir="./model"):
        Loads the latest trained model from the specified directory.
    """

    def __init__(self, device, num_classes=2, detections_per_img=500):
        """
        Constructs all the necessary attributes for the ModelLoader object.

        Parameters
        ----------
            device : torch.device
                The device to which the model is loaded.
            num_classes : int, optional
                The number of classes for the model (default is 2).
            detections_per_img : int, optional
                The maximum number of object detections (default is 500).
        """
        self.device = device
        self.num_classes = num_classes
        self.detections_per_img = detections_per_img
        self.model = self.load_model()

    def load_model(self):
        """
        Loads the pre-trained model from torchvision.

        Returns
        -------
        model : torchvision.models.detection.faster_rcnn.FastRCNNPredictor
            The loaded model.
        """
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        model.roi_heads.detections_per_img = self.detections_per_img
        model.to(self.device)
        return model

    def load_trained_model(self, model_dir="./model"):
        """
        Loads the latest trained model from the specified directory.

        Parameters
        ----------
            model_dir : str, optional
                The directory where the trained models are stored (default is "./model").
        """
        trained_models = os.listdir(model_dir)
        latest_epoch = -1
        for model_name in trained_models:
            if not model_name.endswith("pth"):
                continue
            epoch = float(model_name.split("_")[1].split(".pth")[0])
            if epoch > latest_epoch:
                latest_epoch = epoch
                best_model_name = model_name

        best_model_path = os.path.join(model_dir, best_model_name)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()


class TestModelLoader(unittest.TestCase):
    """
    A class used to test the ModelLoader class.

    ...

    Methods
    -------
    test_model_loading():
        Tests that a model is loaded successfully.
    test_trained_model_loading():
        Tests that a trained model can be loaded.
    """

    def test_model_loading(self):
        """
        Tests that a model is loaded successfully.
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        loader = ModelLoader(device)
        self.assertIsNotNone(loader.model)

    def test_trained_model_loading(self):
        """
        Tests that a trained model can be loaded.
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        loader = ModelLoader(device)
        loader.load_trained_model()
        self.assertIsNotNone(loader.model)


if __name__ == "__main__":
    unittest.main()
