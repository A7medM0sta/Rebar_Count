import unittest
import torch
import numpy as np
from src.transforms import Compose, ToTensor, RandomHorizontalFlip

class TestTransforms(unittest.TestCase):
    def setUp(self):
        self.image = np.random.rand(256, 256, 3).astype(np.float32)
        self.target = {
            "boxes": torch.tensor([[50, 50, 100, 100]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64)
        }
        self.transforms = Compose([ToTensor(), RandomHorizontalFlip(prob=1.0)])

    def test_to_tensor(self):
        to_tensor = ToTensor()
        image, target = to_tensor(self.image, self.target)
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 256, 256))

    def test_random_horizontal_flip(self):
        flip = RandomHorizontalFlip(prob=1.0)
        image, target = flip(self.image, self.target)
        self.assertEqual(target["boxes"][0][0].item(), 156)  # Check if the box is flipped

    def test_compose(self):
        image, target = self.transforms(self.image, self.target)
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 256, 256))
        self.assertEqual(target["boxes"][0][0].item(), 156)  # Check if the box is flipped

if __name__ == '__main__':
    unittest.main()