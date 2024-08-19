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

    def load_model(self):
        """
        Loads the pre-trained model from torchvision.

        Returns
        -------
        model : torchvision.models.detection.faster_rcnn.FastRCNNPredictor
            The loaded model.
        """

    def load_trained_model(self, model_dir="./model"):
        """
        Loads the latest trained model from the specified directory.

        Parameters
        ----------
            model_dir : str, optional
                The directory where the trained models are stored (default is "./model").
        """


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

    def test_trained_model_loading(self):
        """
        Tests that a trained model can be loaded.
        """
