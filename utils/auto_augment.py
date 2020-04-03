# --*-- coding:utf-8 --*--
from __future__ import division

import random

import cv2
import numpy as np


class RandomSaturation(object):
    """随机的改变图片的饱和度, 会给图片中的绿色通道的值乘以一个 [lower, upper] 之间的随机值"""

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, image):
        if random.randint(0, 1):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image


class RandomHue(object):
    """随机改变图片的色调, 对蓝色通道值进行修改."""

    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(0, 1):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image


class RandomLightingNoise(object):
    """
    该类会随机给图片增加噪声, 随机调换 BGR 三通道的顺序.
    在该类的使用了 class SwapChannels(object) 类来辅助完成目的功能
    """

    def __init__(self):
        self.perms = (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0)
        )

    def __call__(self, image):
        if random.randint(0, 1):
            swap = random.randint(0, len(self.perms) - 1)
            shuffle = SwapChannels(self.perms[swap])
            image = shuffle(image)
        return image


class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args: image(Tensor):image tesnsor to be transformed
        Return: a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numoy()
        # else:
        #     image = np.array(image)
        print("swaps")
        image = image[:, :, self.swaps]
        return image


class ConvertColor(object):
    """颜色空间转换"""

    def __init__(self, current="BGR", transform="HSV"):
        self.current = current
        self.transform = transform

    def __call__(self, image):
        if self.current == "BGR" and self.transform == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == "HSV" and self.transform == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image


class Compose(object):
    """
    可以看到, Compose 的作用实际上就是按照顺序不断调用列表内的类函数来对数据进行 transformations,
     下面我们就根据默认调用的函数顺序对文件中的其他类进行解析.

     Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RandomContrast(object):
    """令图片中所有像素的值都乘以一个介于 [lower, upper] 之间的随机系数. 该操作会随机执行."""

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.upper
        assert self.lower >= 0

    def __call__(self, image):
        """随机执行"""
        if random.randint(0, 1):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image


class RandomBrightness(object):
    """随机调节图片的亮度, 给 BGR 三通道随机加上或减去一个值"""

    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(0, 1):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image


class PhotometricDistort(object):
    """该类会随机选择一些图片执行扭曲操作(distort)"""

    def __init__(self):
        # pd是一个列表，其中存放的多个transformations类
        # pd将会作为Compose的参数使用
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),  # opencv
            RandomContrast()
        ]
        # 随机调节亮度
        self.rand_brightness = RandomBrightness()
        # 随机增加噪声
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image):
        im = image.copy()
        im = self.rand_brightness(im)
        if random.randint(0, 1):  # 随机执行下面两者操作之一
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im = distort(im)

        return self.rand_light_noise(im)


#         return im


class ConvertFromInts(object):
    """将数据中的像素类型从整形变成浮点型"""

    # def __init__(self):
    #     pass

    def __call__(self, image):
        return image.astype(np.float32)


class ToAbsoluteCoords(object):
    """
    我们知道, 在进行目标检测时, 默认的生成的 boxes 的坐标值是按照图片的长宽比例来存储的,
    这里为了方便我们后续的 transformations 的执行,
    因此, 我们需要暂时将 boxes 的坐标切换成绝对值坐标. (事后会切换回来)
    """

    # def __init__(self):
    #     pass

    def __call__(self, image):
        width, height, channels = image.shape

        return image
