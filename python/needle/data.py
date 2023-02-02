import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any

import gzip
import struct
import functools


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        # BEGIN YOUR SOLUTION
        if flip_img:
            return img[:, ::-1, :]
        return img
        # END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding+1, size=2)
        # BEGIN YOUR SOLUTION
        # https://github.com/j2kevin18/dlsyscourse_hw2_j2kevin18/blob/efbf321c2b33e4a3f4c669f158cef03977bb4007/python/needle/data.py#L39
        if shift_x <= 0:
            clip_x_up = shift_x + self.padding
            clip_x_down = clip_x_up + img.shape[0]
        else:
            clip_x_down = shift_x + self.padding + img.shape[0]
            clip_x_up = clip_x_down - img.shape[0]
        if shift_y <= 0:
            clip_y_up = shift_y + self.padding
            clip_y_down = clip_y_up + img.shape[1]
        else:
            clip_y_down = shift_y + self.padding + img.shape[1]
            clip_y_up = clip_y_down - img.shape[1]

        pad_img = np.pad(img, self.padding, 'constant')
        return pad_img[clip_x_up: clip_x_down, clip_y_up: clip_y_down, self.padding:self.padding+img.shape[2]]
        # def xCrop(img,shift):
        #     z=np.zeros_like(img)
        #     if shift>0:
        #         z[:,shift:,:]=img[:,:-shift,:]
        #     else :
        #         z[:,:shift,:]=img[:,-shift:,:]
        #     return z
        # def yCrop(img,shift):
        #     z=np.zeros_like(img)
        #     if shift>0:
        #         z[shift:,:,:]=img[:-shift,:,:]
        #     else :
        #         z[:shift,:,:]=img[-shift:,:,:]
        #     return z
        # return xCrop(yCrop(np.pad(img,self.padding,1),shift_y),shift_x)
        # END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)),
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        # BEGIN YOUR SOLUTION
        self.ordering_it = 0
        if self.shuffle:
            shuffle_range=np.arange(len(self.dataset))
            np.random.shuffle(shuffle_range)
            self.ordering = np.array_split(shuffle_range,range(self.batch_size, len(self.dataset), self.batch_size))
        # END YOUR SOLUTION
        return self

    def __next__(self):
        # BEGIN YOUR SOLUTION
        # if self.batch_size>1:
        #     print(self.batch_size)
        if self.ordering_it==len(self.ordering):
            raise StopIteration
        
        if type(self.dataset) is NDArrayDataset:
            if self.batch_size==1:
                index=self.ordering[self.ordering_it][0]
                result=Tensor(self.dataset[index])
                self.ordering_it=self.ordering_it+1
                return (result,)
            else:
                result=[]
                for index in self.ordering[self.ordering_it]:
                    # print(type(self.dataset[index]))
                    # print(len(self.dataset[index]))
                    result.append(self.dataset[index][0])
                self.ordering_it=self.ordering_it+1
                return (Tensor(np.array(result)),)

        if type(self.dataset) is MNISTDataset:
            result=[]
            for i in range(2):
                field_result=[]
                for index in self.ordering[self.ordering_it]:
                    # print(type(self.dataset[index]))
                    # print(len(self.dataset[index]))
                    field_result.append(self.dataset[index][i])
                result.append(Tensor(field_result))

            self.ordering_it=self.ordering_it+1
            return (result[0],result[1])

        # END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        # BEGIN YOUR SOLUTION
        self.data = parse_mnist(
            image_filesname=image_filename, label_filename=label_filename)
        self.transforms = transforms
        # END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        # BEGIN YOUR SOLUTION
        labels = self.data[1]
        img = self.data[0][index]
        newImg=None
        if self.transforms:
            newImg = np.reshape(img, (28, 28, 1))
            for func in self.transforms:
                newImg = func(newImg)

        return (img if newImg is None else newImg, labels[index])
        # END YOUR SOLUTION

    def __len__(self) -> int:
        # BEGIN YOUR SOLUTION
        return self.data[1].shape[0]
        # END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    # BEGIN YOUR SOLUTION
    def readLabels(filePath=label_filename):
        with gzip.open(filePath, 'rb') as f:
            return [struct.unpack('>II', f.read(8)), np.frombuffer(f.read(), dtype=np.uint8)]

    def readImages(filePath=image_filesname):
        with gzip.open(filePath, 'rb') as f:
            [magic, images, rows, cols] = struct.unpack('>IIII', f.read(16))
            return np.resize(np.frombuffer(f.read(), dtype=np.uint8), (images, rows*cols))/np.float32(255)
    return (readImages(), readLabels()[1])
    # END YOUR SOLUTION
