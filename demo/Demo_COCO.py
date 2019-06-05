import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from skimage.io import imread
from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo
import pylab



def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image


def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()
    #pylab.show()


def main():
    # this makes our figures bigger
    pylab.rcParams['figure.figsize'] = 20, 12

    config_file = '../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml'

    #'../configs/e2e_mask_rcnn_fbnet.yaml'
    #'../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml'
                  #'../configs/e2e_mask_rcnn_fbnet.yaml'

    #'../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml'


    #'../configs/e2e_mask_rcnn_fbnet.yaml'
    #'../configs/e2e_mask_rcnn_fbnet.yaml'
    #'../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml'

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.3, load_weight='../Trained Model/new/model_0027500.pth'
    )


    # from http://cocodataset.org/#explore?id=345434
    #image = load("https://www.abc.net.au/news/image/9857250-3x2-940x627.jpg")
    #imshow(image)

    # compute predictions

    #print(image)
    #image = Image.open('./0008.png').convert('RGB')
    #print(image.size)
    #'/mnt/DATA/Download/Dataset/t-less_v2/test_primesense/01/rgb/0004.png' /54
    image = imread('/mnt/DATA/Download/Dataset/t-less_v2/train_primesense/30/rgb/0001.png')
    #imread('./Test.jpg')

    image = np.array(image)[:, :, [2, 1, 0]]
    predictions = coco_demo.run_on_opencv_image(image)

    imshow(predictions)


if __name__ == '__main__':
    main()
