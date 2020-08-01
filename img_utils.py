import numpy as np
import tqdm
import glob

from torch.utils.data import Dataset
from torchvision import transforms

from skimage.draw import random_shapes



def gen_one_image(h=64, w=64):

    shapes = ('rectangle', 'triangle', 'circle')
    n_images = 3 #np.random.randint(1, 9)
    matrix = np.zeros((h, w, n_images))
    image = np.zeros((h, w, 3))
    mask = np.full((h, w), -1)
    for i in range(n_images):
      n_shape = np.random.randint(len(shapes))
      tmp_image, _ = random_shapes((h, w), max_shapes=1, min_shapes=1, shape=shapes[n_shape],
                                #   multichannel=True, num_channels=3, 
                                  min_size=21, max_size=32, allow_overlap=True)
      matrix[tmp_image[:, :, 0] < 255, i] = 1
      mask[tmp_image[:, :, 0] < 255] = n_shape
      tmp_image[tmp_image == 255] = 0
      
      image[tmp_image[:, :, 0] > 0] = tmp_image[tmp_image[:, :, 0] > 0]
    image /= 255
    image[image == 0] = 1

    return image, mask, matrix


def gen_images(num_images: int):
    n_batches = int(np.ceil(num_images / 10**4))
    for i in range(n_batches):
        images, masks, matrices= [], [], []
        for j in tqdm.tqdm_notebook(range(10**4)):
            tmp_image, tmp_mask, tmp_matrix= gen_one_image()
            images.append(tmp_image)
            masks.append(tmp_mask)
            matrices.append(tmp_matrix)
        with open(f'image_batch_{i}.npy', 'wb') as f:
          np.save(f, images)
        with open(f'mask_batch_{i}.npy', 'wb') as f:
          np.save(f, masks)
        with open(f'matrix_batch_{i}.npy', 'wb') as f:
          np.save(f, matrices)


class ImageDataset(Dataset):
    def __init__(self, path):

        filelist = glob.glob(path + 'image_*.npy')
        tmp = []
        for el in filelist:
            tmp.append(np.load(el))
        self.images = np.concatenate(tmp)
        filelist = glob.glob(path + 'mask_*.npy')
        tmp = []
        for el in filelist:
            tmp.append(np.load(el))
        self.masks = np.concatenate(tmp)
        filelist = glob.glob(path + 'matrix_*.npy')
        tmp = []
        for el in filelist:
            tmp.append(np.load(el))
        self.matrices = np.concatenate(tmp)
        # filelist = glob.glob(path + 'objects_*.npy')
        # self.n_objects = []
        # for el in filelist:
        #     tmp.append(np.load(el))
        self.transform = transforms.ToTensor()
        
    def __len__(self):
       
        return len(self.images)


    def __getitem__(self, idx):
        
        x = self.transform(self.images[idx]).float()
        y = self.transform(self.masks[idx]).long()
        z = self.transform(self.matrices[idx]).float()
        # obj = torch.Tensor(self.objects[idx]).double()
        return x, y, z