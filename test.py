from run import *
import numpy as np
import face_alignment
from skimage import io
from dataset import
from dataset import preprocess_dataset, VoxCelebDataset
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt

# generating face landmarks based on the source and driver images and bring them to required shape
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', flip_input=False)
source_img = io.imread('../talking-heads/dataset/faces/000002.jpg')
source_img = resize(source_img, (255, 255)) * 255
source_landmarks = fa.get_landmarks(source_img)
print(np.array(source_landmarks).shape)


driver_img = io.imread('../talking-heads/dataset/faces/000001.jpg')
driver_img = torch.from_numpy(resize(driver_img, (255, 255)) * 255)
driver_landmarks = fa.get_landmarks(driver_img)
driver_landmarks = plot_landmarks(driver_img, driver_landmarks)
# bringing images to shape required by network
t = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
        ])
source_img_net = torch.from_numpy(source_img.transpose((2, 1, 0)).reshape((1,3,255,255)))
source_landmarks_net = t(source_landmarks)

# creating required networks
G = network.Generator()
E = network.Embedder()

embeddings = E.forward(source_img_net, source_landmarks_net)
result = G.forward(driver_landmarks_net, embeddings)

io.imshow(result)
