import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
import os
import yaml
from PIL.Image import open as open_img
from torch.utils.data import ConcatDataset, Dataset
import numpy
import random
from torchvision.transforms import functional as F
from torchvision.transforms import RandomResizedCrop, ColorJitter
#from torchvision import transforms as trans
import skimage.io as io
#from maskrcnn_benchmark.data.transforms.transforms import Compose as T_compose


class ObjectDataLoader(Dataset):
    """
    Pytorch dataset loader for T-Less
    it can loads each of the subfiles inside T-less Dataset
    """
#   def __init__(self, data_dir=None, train=None, cam=None, set_idx=0, transforms=None):

    def __init__(self, data_dir=None, set_idx=None, transforms=None):

        super(ObjectDataLoader, self)
        self.threshold = -0.8
        self.cont_flo = 1e-6

        self.set_idx = set_idx

        self.data_dir = (data_dir + '{:0>2d}/').format(set_idx)
        self.data_dir_img = self.data_dir + 'rgb/'
        self.data_dir_dp = self.data_dir + 'depth/'
        # These create the name list of rgb and depth image in data_dir_img and data_dir_dp
        self.img_name = [name for name in os.listdir(self.data_dir_img)
                         if os.path.isfile(os.path.join(self.data_dir_img, name))]
        self.depth_name = [name for name in os.listdir(self.data_dir_dp)
                           if os.path.isfile(os.path.join(self.data_dir_dp, name))]

        assert len(self.img_name) != len(self.data_dir_dp)

        self.gt_data_dir = os.path.join(self.data_dir, 'gt.yml')
        try:
            with open(self.gt_data_dir, 'r') as f:
                loaded = yaml.load(f, Loader=yaml.Loader)
                if 'train' in self.gt_data_dir:
                    self.gt = [Obj[0]['obj_bb'] for key, Obj in loaded.items()]
                    # print(gt)
                    self.cls_idx = [Obj[0]['obj_id'] for key, Obj in loaded.items()]
                else:
                    gt_dict = [Obj for key, Obj in loaded.items()]
                    self.gt = []
                    self.cls_idx = []
                    for i in range(len(gt_dict)):
                        self.gt.append([gt_dict[i][j]['obj_bb'] for j in range(len(gt_dict[i]))])
                        self.cls_idx.append([gt_dict[i][j]['obj_id'] for j in range(len(gt_dict[i]))])
        except FileNotFoundError:
            raise FileNotFoundError('YAML was not found')

        assert len(self.img_name) == len(self.gt)

        self._transform = transforms

    def __len__(self):
        #assert len(self.img_name) == len(self.data_dir_dp)
        return len(self.img_name)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir_img,
                                self.img_name[idx])

        image = open_img(img_name).convert('RGB')

        boxes = torch.as_tensor(self.gt[idx]).reshape(-1, 4)  # guard against no boxes

        target = BoxList(boxes, image.size, mode="xywh").convert("xyxy")

        classes = torch.as_tensor(self.cls_idx[idx])[None]
        target.add_field("labels", classes)

        masks = torch.as_tensor(1*self.gen_mask_from_img(image))
        masks = SegmentationMask(masks, image.size, mode='mask')
        #print(masks)
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self._transform:
            image, target = self._transform(image, target)
        return image, target, idx

    def __repr__(self):

        #   TODO

        pass

    def get_img_info(self, index):
        img_name = os.path.join(self.data_dir_img,
                                self.img_name[index])
        img = open_img(img_name)
        img_height, img_width = img.size
        return {"height": img_height, "width": img_width}

#   Mask reconstruction from the color image

    def gen_mask_from_img(self, img):
        return numpy.log10(numpy.sum(img/numpy.max(img, keepdims=2), axis=2) + self.cont_flo) >= self.threshold

#   The bbox from tless are not well constructed. We should reconstruct the bbox from the reconstructed mask.
#   Max and Min x,y non-zeros indices from the masks

    def cal_bbox_from_mask(self, mask):
        pass


class ObjectDataLoader_with_trans(ObjectDataLoader):
    """
    Pytorch dataset loader for T-Less
    it can loads each of the subfiles inside T-less Dataset
    """

    def __init__(self, data_dir=None, set_idx=None, transforms=None):

        #self.size = [round(720*0.8), round(540*0.8)]
        self.size = [380, 380]
        super(ObjectDataLoader_with_trans, self)
        super(ObjectDataLoader_with_trans, self).__init__(data_dir, set_idx, transforms)

    def __getitem__(self, idx):

        img_name = os.path.join(self.data_dir_img,
                                self.img_name[idx])

        image = open_img(img_name).convert('RGB')
        #print(self.size, image.size)
        i, j, h, w = self.get_params(image, self.size)

        #print("image size : {} ; {}, {}, {}, {}".format(image.size, i, j, h, w))
        image = F.crop(image, i, j, h, w)

        boxes = torch.as_tensor(self.gt[idx]).reshape(-1, 4)  # guard against no boxes
        #print(boxes, self.gt[idx])
        #print("Boxes :", boxes.shape)

        boxes = boxes - torch.as_tensor([i, j, 0, 0])
        #print(boxes)
        target = BoxList(boxes, image.size, mode="xywh").convert("xyxy")

        classes = torch.as_tensor(self.cls_idx[idx]).reshape(-1)
        #print("Classes :", classes.shape)
        target.add_field("labels", classes)

        masks = torch.as_tensor(1*self.gen_mask_from_img(image))
        masks = SegmentationMask(masks, image.size, mode='mask')

        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self._transform:
            image, target = self._transform(image, target)

        return image, target, idx

    @staticmethod
    def get_params(img, output_size):

        w, h = img.size
#   TODO : This is not correct for rectangular shape as the th and tw are switched. Need to check
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


class concat_tless(ConcatDataset):
    """
    data_dir (str) : the root directory of all T-Less dataset file
    train (str) : take either 'train' or 'val' or 'test'
    cam (str) : the camera used for the dataset
    set_idx (list) : [start_index, end_index]
    transform (callable) : transformation function of pytorch dataset.
    """

    def __init__(self, data_dir=None, train=None, cam=None, set_idx=[1, 20], transforms=None, other_dataset_list=None):

        self.data_dir = data_dir
        #self.train = train
        #self.cam = cam
        self.set_idx = set_idx
        self._transforms = transforms
        if not data_dir or not set_idx:
        #if not data_dir or not train or not cam or not set_idx:
            raise Exception('The dataset directory is not completed')
        #if transforms:
        #    print('Transforms is not defined')

        self.concate_dateset = self.contca(other_dataset_list)

        super(concat_tless, self)
        super(concat_tless, self).__init__(self.concate_dateset)

    def contca(self, other_dataset_list):

        tless = [ObjectDataLoader(data_dir=self.data_dir, set_idx=idx,
                                  transforms=self._transforms)
                 for idx in range(self.set_idx[0], self.set_idx[1] + 1, 1)]

        if other_dataset_list:
            return tless.extend(other_dataset_list)
        else:
            return tless

    def get_img_info(self, idx):
        # TODO: make it general
        #img, _, _ = self.__getitem__(idx)
        #print(img.shape)
        img_width, img_height = 400, 400 #img.shape[1:3]
        return {"height": img_height, "width": img_width}


class change_bg(object):
    """This is the background changing function for T-less
    It will load the whole background images in order to speed up the processing speed
    However, if there are large number of background, this is not desirable.
    Also, here assume the size of the bg is larger than the object
    Using resize maybe better than cropping
    """
    def __init__(self, data_dir):
        if 'train' in data_dir:
            self.bg_dir = data_dir[0:data_dir.find('train')] + 'background'
        else:
            self.bg_dir = data_dir[0:data_dir.find('test')] + 'background'

        self.bg_name = [name for name in os.listdir(self.bg_dir)
                        if os.path.isfile(os.path.join(self.bg_dir, name))]
        self.num_bg = len(self.bg_name)
        #print(self.num_bg)
        self.bg_object = [open_img(os.path.join(self.bg_dir, bg_name)).convert('RGB') for bg_name in self.bg_name]

        self.bg_scale = [0.1, 0.2, 0.4]

    def __call__(self, image, target):
        rand_num = random.randint(0, self.num_bg-1)

        bg = self.bg_object[rand_num]
        #print(rand_num)
        #print(bg.size, image.shape[1:3])
        i, j, h, w = self.get_params(bg, image.shape[1:3])
        bg = F.crop(bg, i, j, h, w)
        bg = F.to_tensor(bg)
        mask = target.get_field('masks').get_mask_tensor()
        mask = mask.float()#.type(torch.float32)
        #print(mask.dtype, image.dtype, bg.dtype)
        #print(mask.device, image.device, bg.device)
        img = image*mask + torch.eq(mask, 0).float()*bg
        #print(bg.shape, mask.shape, image.shape)

        return img, target

    @staticmethod
    def get_params(img, output_size):

        w, h = img.size
#   TODO : This is not correct for rectangular shape as the th and tw are switched. Need to check
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


# TODO: This will not work as transform take object 1 by 1. We cannot access the batch from here
# Either rewrite a loading function
# or we write a layer before the model.
class change_bg_rescale(object):
    """This is the background changing function for T-less
    It will load the whole background images in order to speed up the processing speed
    However, if there are large number of background, this is not desirable.
    Also, here assume the size of the bg is larger than the object
    Using resize maybe better than cropping
    """
    def __init__(self, data_dir, out_size=800):
        if 'train' in data_dir:
            self.bg_dir = data_dir[0:data_dir.find('train')] + 'background'
        else:
            self.bg_dir = data_dir[0:data_dir.find('test')] + 'background'

        self.bg_name = [name for name in os.listdir(self.bg_dir)
                        if os.path.isfile(os.path.join(self.bg_dir, name))]
        self.num_bg = len(self.bg_name)

        self.bg_object = [open_img(os.path.join(self.bg_dir, bg_name)).convert('RGB') for bg_name in self.bg_name]

        self.bg_scale = (0.5, 1.0)
        self.obj_scale = (0.2, 0.4, 0.6, 0.8, 1)
        #self.obj_size = obj_size
        self.bg_size = out_size

        self.rrc = RandomResizedCrop(size=out_size, scale=self.bg_scale)
        self.rrc_obj = self.class_resize(min_size=500, max_size=(600, 575, 550))
        self.color_jig = ColorJitter()

    def __call__(self, image, target):

        img, target = self.rrc_obj(image, target)
        #   target = target.convert('xywh')
        #   plt.figure()
        #   plt.subplot(1, 2, 1)
        #   print(target.bbox)
        #   plt.pcolor(torch.sum(F.to_tensor(image), dim=0))#.permute(1,2,0))
        #   show_bbox(target.bbox.reshape(-1))
        #   plt.subplot(1, 2, 2)
        #   plt.pcolor(target.get_field('masks').get_mask_tensor())
        #   plt.show()

#       Generate the Random Integer to pick the background from the backgound data
        rand_num = random.randint(0, self.num_bg-1)
        bg = self.bg_object[rand_num]
        bg = self.rrc(bg)
        bg = self.color_jig(bg)
        bg_size = bg.size
        r_mask = torch.zeros(bg_size)
        bg = F.to_tensor(bg)

        mask = target.get_field('masks').get_mask_tensor()
        classes = target.get_field('labels')
        nz_idx = torch.nonzero(mask)

        img_sx, img_sy = img.size
        img = F.to_tensor(img)

        rand_x, rand_y = random.randint(0, self.bg_size-img_sx-1), \
            random.randint(0, self.bg_size-img_sy-1)

        r_mask[nz_idx[:, 0] + rand_y, nz_idx[:, 1] + rand_x] = \
            mask[nz_idx[:, 0], nz_idx[:, 1]].float()

        bg[:, nz_idx[:, 0] + rand_y, nz_idx[:, 1] + rand_x] = \
            img[:, nz_idx[:, 0], nz_idx[:, 1]]

        #print('BBox :{}, Mode :{}'.format(target.bbox, target.mode))
        #print(rand_x, rand_y)

        boxes = target.bbox + torch.tensor([rand_x, rand_y, rand_x, rand_y], dtype=torch.float)
        target = BoxList(boxes, bg_size, mode="xyxy")

        target.add_field("labels", classes)
        r_mask = SegmentationMask(r_mask, bg_size, mode='mask')
        target.add_field("masks", r_mask)
        target.clip_to_image(remove_empty=True)
        return bg, target

    # To avoid mutual call, subclass is defined here.
    # Dependent call will raise infinite loop.
    class class_resize(object):
        def __init__(self, min_size, max_size):
            if not isinstance(min_size, (list, tuple)):
                min_size = (min_size,)
            self.min_size = min_size
            self.max_size = max_size

        # modified from torchvision to add support for max size
        def get_size(self, image_size):
            w, h = image_size
            size = random.choice(self.min_size)

            max_size = random.choice(self.max_size)
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

        def __call__(self, image, target):
            size = self.get_size(image.size)
            image = F.resize(image, size)
            target = target.resize(image.size)
            return image, target


# TODO
class mix_objects(object):
    def __init(self, trans_xy):
        if isinstance(trans_xy, (list, tuple)):
            self.trans_x = trans_xy[0]
            self.trans_y = trans_xy[1]
        else:
            self.trans_x, self.trans_y = trans_xy, trans_xy

    def __call__(self, image, target):
        size = image.size
        trans_x = torch.randint(0, self.trans_x, size)
        trans_y = torch.randint(0, self.trans_x, size)
        img = target.get_field('masks')
        bbox = target['bbox']
        return image, target


def main():

    import cv2
    from maskrcnn_benchmark.utils import cv2_util
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, image, target):
            for t in self.transforms:
                image, target = t(image, target)
            return image, target

        def __repr__(self):
            format_string = self.__class__.__name__ + "("
            for t in self.transforms:
                format_string += "\n"
                format_string += "    {0}".format(t)
            format_string += "\n)"
            return format_string

    def show_bbox(bbox):
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], facecolor='none', edgecolor='r', linewidth=3)
        plt.gca().add_patch(rect)

    # This is not working
    def compute_colors_for_labels(labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        colors = labels[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors
    # This is not working
    def overlay_boxes(image, predictions):

        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image
    # This is not working
    def overlay_mask(image, predictions):

        mask = predictions.get_field("masks").get_mask_tensor()[None].numpy()
        #print(masks.shape)
        labels = predictions.get_field("labels")

        color = compute_colors_for_labels(labels).tolist()

        #for mask, color in zip(masks, colors):
        #    print(mask.shape)
        thresh = mask[:, :, :]
        contours, hierarchy = cv2_util.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite

    print('Initializing the Dataset...')
    data_dir = '/mnt/DATA/Download/Dataset/t-less_v2/train_primesense/'
    #ran1 = random.randint(1, 30)
    ran1 = random.randint(1, 30)
    ran2 = random.randint(1, 1296) - 1
    Dataset = ObjectDataLoader(data_dir=data_dir, set_idx=ran1, transforms=Compose([change_bg_rescale(data_dir)]))
    #print(len(Dataset[ran2]))
    image, box, _ = Dataset[ran2]
    box = box.convert('xywh')
    bbox = box.bbox.reshape(-1)
    print(bbox)
    #image = Dataset[ran2][0].numpy()
    ##x = Dataset[ran2][1]
    ##result = torch.sum(torch.from_numpy(overlay_boxes(image, x)), dim=0)#.permute(1, 2, 0)
    result = torch.from_numpy(overlay_boxes(image.numpy(), box)).permute(1, 2, 0)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(result)#, origin='lower')
    show_bbox(bbox)
    plt.subplot(1, 2, 2)
    plt.imshow(box.get_field('masks').get_mask_tensor())#, origin='lower')
    plt.show()


    #Combined_set = concat_tless(data_dir='/mnt/2A9EAD839EAD47E1/Download/Dataset/T_Less_Dataset/',
    #                            train='train', cam='primesense', set_idx=[1, 2])
    #print(Combined_set.__class__)
    #dataloader = DataLoader(Combined_set, batch_size=400, num_workers=4)
    #print(type(Combined_set).__name))
    #print(ConcatDataset in type(Combined_set))
    #for img in range(len(Combined_set)):
    #    print('Processing Image : {}'.format(img))
    #    if img >= 5:
    #        break
    #    Data_dict = Combined_set[img * 500]
    # batch dataloader cannot work on this directly
    # img in this is PIL , but when we are running in normal program, the totensor function will make PIL to tensor
    #for i_batch, sample_batched in enumerate(dataloader):
    #    print(i_batch, sample_batched)
    #    if i_batch == 3:
    #        break
        #Data_dict = Combined_set[img*500]
        #x = Combined_set.target['mask']

    #dataloader = torch.utils.data.DataLoader(Combined_set, batch_size=400, num_workers=4)


if __name__ == '__main__':
    main()
