#import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pycocotools.coco import COCO
import numpy as np
import json as jsonmod
import time
import copy
import h5py
import torch.nn.functional as F
import re
from nlp_aug import SR, eda_4, RD, RS
import random
from random import choice

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def get_paths(path, name, use_restval=True):

    roots = {}
    ids = {}
    if 'coco' == name:
        #imgdir = os.path.join(path, 'images')
        imgdir = path
        capdir = os.path.join(path, 'annotations')
        roots['train'] = {
            'img': os.path.join(imgdir, 'train2014'),
            'cap': os.path.join(capdir, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(imgdir, 'val2014'),
            'cap': os.path.join(capdir, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }
        ids['train'] = np.load(os.path.join(capdir, 'coco_train_ids.npy'))
        ids['val'] = np.load(os.path.join(capdir, 'coco_dev_ids.npy'))[:5000]
        ids['test'] = np.load(os.path.join(capdir, 'coco_test_ids.npy'))
        ids['trainrestval'] = (
            ids['train'],
            np.load(os.path.join(capdir, 'coco_restval_ids.npy')))
        if use_restval:
            roots['train'] = roots['trainrestval']
            ids['train'] = ids['trainrestval']
    elif 'f8k' == name:
        imgdir = os.path.join(path, 'images')
        cap = os.path.join(path, 'dataset_flickr8k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}
    elif 'f30k' == name:
        imgdir = os.path.join(path, 'flickr30k-images')
        cap = os.path.join(path, 'dataset_flickr30k.json')
        roots['train'] = {'img': imgdir, 'cap': cap}
        roots['val'] = {'img': imgdir, 'cap': cap}
        roots['test'] = {'img': imgdir, 'cap': cap}
        ids = {'train': None, 'val': None, 'test': None}

    return roots, ids


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, vocab, region_det_file_prefix, transform=None, ids=None):
        """
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: v0ocabulary wrapper.
            transform: transformer for image.
        """
        self.root = root
        # when using `restval`, two json files are needed
        if isinstance(json, tuple):# True
            self.coco = (COCO(json[0]), COCO(json[1]))
        else:
            self.coco = (COCO(json),)
            self.root = (root,)
        # if ids provided by get_paths, use split-specific ids
        if ids is None: #False
            self.ids = list(self.coco.anns.keys())
        else:
            self.ids = ids

        # if `restval` data is to be used, record the break point for ids
        if isinstance(self.ids, tuple): #True
            self.bp = len(self.ids[0])
            self.ids = list(self.ids[0]) + list(self.ids[1])
        else:
            self.bp = len(self.ids)
        self.vocab = vocab
        print('load  data(sentence) {}'.format(len(self.ids)))
        self.transform = transform
        # self.region_bbox_file = region_bbox_file#'/remote-home/lyli/Workspace/burneddown/ECCV/joint-pretrain/COCO/region_feat_gvd_wo_bgd/coco_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5'
        self.region_det_file_prefix = region_det_file_prefix#'/remote-home/lyli/Workspace/burneddown/ECCV/joint-pretrain/COCO/region_feat_gvd_wo_bgd/feat_cls_1000/coco_detection_vg_100dets_gvd_checkpoint_trainval'
        self.max_words = 60 #30

        captions = []
        coco = self.coco[0]# train set
        self.image2anno = {}
        for ann_id in coco.anns:
            if coco.anns[ann_id]["image_id"] in self.image2anno:
                self.image2anno[coco.anns[ann_id]["image_id"]].append(coco.anns[ann_id]["caption"])
            else:
                self.image2anno[coco.anns[ann_id]["image_id"]] = [coco.anns[ann_id]["caption"]] 
        if len(self.coco)>1:
            coco = self.coco[1]# val set
            for ann_id in coco.anns:
                if coco.anns[ann_id]["image_id"] in self.image2anno:
                    self.image2anno[coco.anns[ann_id]["image_id"]].append(coco.anns[ann_id]["caption"])
                else:
                    self.image2anno[coco.anns[ann_id]["image_id"]] = [coco.anns[ann_id]["caption"]]
        print(len(self.image2anno)) #123287 images = 82783 train images + 40504 val images

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab
        root, caption, img_id, path, image, region_feat,captions= self.get_raw_item(index)

        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        """
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().encode('utf-8').decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        """
        caption = pre_caption(str(caption).encode('utf-8').decode('utf-8'), self.max_words)
        #增广一：拼接两个句子
        random_index1 = random.randint(0, len(captions) - 1)
        random_index2 = random.randint(0, len(captions) - 1)
        caption_aug1 = captions[random_index1] + ' ' + captions[random_index2] 

        #增广二：截断原句子
        caption_words_neg = caption.split(' ')
        caption_words_len = len(caption_words_neg)
        caption_aug2 = ' '.join(caption_words_neg[:caption_words_len // 2])

        #增广三：eda
        #text_aug = choice([SR,RD,RS])
        #caption_aug3 = text_aug(caption,0.2, 1)[0]

        caption_aug = choice([caption_aug1,caption_aug2])
        #caption_aug = caption_aug2

        return image,  region_feat, caption, index, img_id ,caption_aug

    def get_raw_item(self, index):
        if index < self.bp:
            coco = self.coco[0]
            root = self.root[0]
        else:
            coco = self.coco[1]
            root = self.root[1]
        ann_id = self.ids[index] #566435
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(root, path)).convert('RGB')
        region_feat = self.get_rcnn(path)
        
        captions = self.image2anno[img_id]
            

        return root, caption, img_id, path, image, region_feat ,captions

    def get_rcnn(self, path): #'COCO_train2014_000000050323.jpg'
        img_id = path.split('/')[-1].split('.')[0] #'COCO_train2014_000000050323'
        with h5py.File(self.region_det_file_prefix + '_feat' + img_id[-3:] + '.h5', 'r') as region_feat_f:
            # with h5py.File(self.region_det_file_prefix + '_cls' + img_id[-3:] + '.h5', 'r') as region_cls_f:
            img = torch.from_numpy(region_feat_f[img_id][:]).float()
                # cls = torch.from_numpy(region_cls_f[img_id][:]).float()
        # vis_pe = torch.randn(100,1601 + 6) # no position information
        img = img[0:36]
        # cls = F.layer_norm(cls, [1600])
        return img

    def __len__(self):
        return len(self.ids)


class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, json, split, vocab, feature_path, transform=None):
        self.root = root
        self.vocab = vocab
        self.split = split
        #         self.index_list = np.loadtxt('use_label_3.txt')
        self.transform = transform
        self.dataset = jsonmod.load(open(json, 'r'))['images']
        self.ids = []
        for i, d in enumerate(self.dataset):
            if d['split'] == split:
                self.ids += [(i, x) for x in range(len(d['sentences']))]
        print('load {} data(sentence) {}'.format(split,len(self.ids)))
        # print(self.ids)
        self.feature_path = feature_path

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        # print(index)
        vocab = self.vocab
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        caption = self.dataset[img_id]['sentences'][ann_id[1]]['raw']
        path = self.dataset[img_id]['filename']

        image = Image.open(os.path.join(root, path)).convert('RGB')
        # image.show()
        if self.transform is not None:
            image = self.transform(image)
        # print(path)
        path = path.replace('.jpg', '.npy')
        feature_path = self.feature_path

        region_feat = self.get_rcnn(os.path.join(feature_path, path))  # return img-feature 100 2048 & pos-feature

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().encode('utf-8').decode('utf-8'))
        # print(tokens)
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        # print(caption)
        target = torch.Tensor(caption)
        # print(target)
        return image,  region_feat, target, index, img_id

    def get_rcnn(self, img_path):
        if os.path.exists(img_path):
            # time1 = time.time()
            img = torch.from_numpy(np.load(img_path))
            img= img[0:36]
        else:
            #             print('no {}'.format(img_path.split('trainval')[-1]))
            img = torch.randn(36, 2048)
        # if os.path.exists(img_path.replace('.npy', '_cls_prob.npy')):
        #     # time1 = time.time()
        #     cls = torch.from_numpy(np.load(img_path.replace('.npy', '_cls_prob.npy')))
        #     cls = cls[0:36,1:]
        # 
        #     #             cls = cls[0:36,self.index_list]
        #     cls = torch.where(cls > 0.2, 1, 0)
        # else:
        #     #             print('no {} cls'.format(img_path.split('trainval')[-1]))
        #     cls = torch.randn(36, 1600)
        #     cls = torch.where(cls > 0.2, 1.0, 0.0)
        # cls = F.layer_norm(cls, [1600])
        return img


    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, region_feat, captions, ids, img_ids,captions_aug = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    region_feat = torch.stack(region_feat, 0)
    # region_cls = torch.stack(region_cls, 0)
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap.split()) for cap in captions]
    #targets = torch.zeros(len(captions), max(lengths)).long()
    #for i, cap in enumerate(captions):
    #    end = lengths[i]
    #    targets[i, :end] = cap[:end]

    return images,  region_feat,  captions, lengths, ids,captions_aug


def get_loader_single(data_name, split, root, json, vocab, transform, batch_size=100, shuffle=True,
                      num_workers=0, ids=None, collate_fn=collate_fn, feature_path=None):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if 'coco' in data_name: #True
        # COCO custom dataset
        dataset = CocoDataset(root=root,
                              json=json,
                              vocab=vocab,
                              region_det_file_prefix=feature_path,
                              transform=transform, ids=ids)

    elif 'f8k' in data_name or 'f30k' in data_name:
        dataset = FlickrDataset(root=root,
                                split=split,
                                json=json,
                                vocab=vocab,
                                feature_path=feature_path,
                                transform=transform)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


def get_transform(split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


def get_loaders(data_name, vocab, batch_size, workers, data_path, opt):
    dpath = os.path.join(data_path, data_name)

    roots, ids = get_paths(dpath, data_name) # data paths

    transform = get_transform( 'train', opt)
    train_loader = get_loader_single(opt.data_name, 'train',
                                     roots['train']['img'],
                                     roots['train']['cap'],
                                     vocab, transform, ids=ids['train'],
                                     batch_size=batch_size, shuffle=True,
                                     num_workers=workers,
                                     collate_fn=collate_fn,
                                     feature_path=opt.feature_path)

    transform = get_transform('val', opt)
    # print(1111111111111)
    val_loader = get_loader_single(opt.data_name, 'val',
                                   roots['val']['img'],
                                   roots['val']['cap'],
                                   vocab, transform, ids=ids['val'],
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=workers,
                                   collate_fn=collate_fn,
                                   feature_path=opt.feature_path)

    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, data_path, opt):
    dpath = os.path.join(data_path, data_name)

    roots, ids = get_paths(dpath, data_name)

    transform = get_transform( split_name, opt)
    test_loader = get_loader_single(opt.data_name, split_name,
                                    roots[split_name]['img'],
                                    roots[split_name]['cap'],
                                    vocab, transform, ids=ids[split_name],
                                    batch_size=batch_size, shuffle=False,
                                    num_workers=workers,
                                    collate_fn=collate_fn,
                                    feature_path=opt.feature_path)

    return test_loader
