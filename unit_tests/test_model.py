import unittest
import torch
from torchvision import transforms
from src.transforms import Compose, ToTensor, RandomHorizontalFlip

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'fasterrcnn_resnet50_fpn', pretrained=True)
        self.model.eval()
        self.image = torch.rand(1, 3, 256, 256)  # Dummy image

    def test_model_inference(self):
        with torch.no_grad():
            predictions = self.model(self.image)
        self.assertIn('boxes', predictions[0])
        self.assertIn('labels', predictions[0])
        self.assertIn('scores', predictions[0])

if __name__ == '__main__':
    unittest.main()