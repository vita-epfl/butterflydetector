import os
import copy
import logging
import numpy as np
import torch.utils.data
import torchvision
from PIL import Image
from .. import transforms, utils

class PIF_Category(object):
    def __init__(self, num_classes, catID_label):
        self.num_classes = num_classes
        self.catID_label = catID_label

    def __call__(self, anns):
        for ann in anns:
            temp_id = self.catID_label[ann['category_id']]
            x, y, w, h = ann['bbox']
            temp = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]*(temp_id)\
             + [x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2]\
             + [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]*(self.num_classes-(temp_id+1))
            ann['keypoints'] = copy.deepcopy(temp)
            ann['num_keypoints'] = 5
        return anns


# Swap function
def swapPositions(list, pos1, pos2):

    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

class CocoInstance(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Based on `torchvision.dataset.CocoDetection`.

    Caches preprocessing.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    train_image_dir = "data/mscoco/images/train2017/"
    val_image_dir = "data/mscoco/images/val2017/"
    train_annotations = "data/mscoco/annotations/instances_train2017.json"
    val_annotations = "data/mscoco/annotations/instances_val2017.json"
    test_path = {'val': "data/mscoco/annotations/instances_val2017.json"}
    def __init__(self, root, annFile, *, target_transforms=None, class_ids=None,
                 n_images=None, preprocess=None, all_images=False, all_persons=False):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)

        # Image ID
        if class_ids:
            self.ids = []
            for id in class_ids:
                self.ids.extend(list(self.coco.getImgIds(catIds=[id])))
            # Remove duplicates
            self.ids = list(set(self.ids))
        else:
            # All images
            self.ids = list(self.coco.imgs.keys())
        if n_images:
            self.ids = self.ids[:n_images]
        print('Images: {}'.format(len(self.ids)))

        # PifPaf
        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms
        self.log = logging.getLogger(self.__class__.__name__)

        # Cat ID (missing class)
        if class_ids:
            self.cat_ids = class_ids
            self.cat_ids = swapPositions(self.cat_ids, 0, self.cat_ids.index(19))
        else:
            self.cat_ids = self.coco.getCatIds()
        print("Number of classes: {}".format(len(self.cat_ids)))
        self.catID_label = {catid:label for label, catid in enumerate(self.cat_ids)}
        self.PIF_category = PIF_Category(num_classes=len(self.cat_ids), catID_label=self.catID_label)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)

        anns = copy.deepcopy(anns)

        image_info = self.coco.loadImgs(image_id)[0]

        self.log.debug(image_info)
        with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            image = Image.open(f).convert('RGB')

        meta_init = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
        }

        if 'flickr_url' in image_info:
            _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
            flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
            meta_init['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)

        # bbox center annotation
        anns = self.PIF_category(anns)

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update(meta_init)

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        # if there are not target transforms, done here
        self.log.debug(meta)
        # transform targets
        if self.target_transforms is not None:
            width_height = image.shape[2:0:-1]
            anns = [t(anns, width_height) for t in self.target_transforms]

        return image, anns, meta

    def __len__(self):
        return len(self.ids)

class ImageList(torch.utils.data.Dataset):
    def __init__(self, image_paths, preprocess=None):
        self.image_paths = image_paths
        self.preprocess = preprocess
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        anns = []
        image, anns, meta = self.preprocess(image, anns, None)
        meta.update({
            'dataset_index': index,
            'file_name': image_path,
        })

        return image, anns, meta

    def __len__(self):
        return len(self.image_paths)
