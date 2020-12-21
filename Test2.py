from run import *
from dataset import VoxCelebDataset
import numpy as np
#x = np.arange(254*255*3).reshape((254, 255, 3)).transpose((2, 1, 0)).reshape((1,3,255,254))
#print(x)
t = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
        ])
