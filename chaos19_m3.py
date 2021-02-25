# -*- coding: utf-8 -*-
import SimpleITK as sitk
from torch.utils.data import Dataset
import os
import numpy as np
import random
import cv2
import torch,pydicom
"""
mask[256，256]
Liver: 63 (55<<<70)
"""

class ChaosDataset_Syn_new(Dataset):
    def __init__(self, path="../datasets/chaos2019", split='train', modals=('t1','t2','ct'),transforms=None):
        super(ChaosDataset_Syn_new, self).__init__()
        for modal in modals:
            assert modal in {'t1','t2','ct'}
        fold = split + "/"
        path1 = os.path.join(path, fold+modals[0])
        path2 = os.path.join(path, fold + modals[1])
        path3 = os.path.join(path, fold + modals[2])

        list_path = sorted([os.path.join(path1, x) for x in os.listdir(path1)]) + sorted([os.path.join(path2, x) for x in os.listdir(path2)])
        raw_path = []
        label_path=[]
        for x in list_path:
            if "t1" in x:
                x += "/T1DUAL"
                c = np.array(0)
            elif "t2" in x:
                x += "/T2SPIR"
                c = np.array(1)
            for y in os.listdir(x):
                if "Ground" in y:
                    tmp = os.path.join(x, y)
                    raw_path.append([tmp.replace("Ground", "DICOM_anon"),c])
                    break
        #########
        self.raw_dataset = []
        self.label_dataset = []
        #######
        self.transfroms = transforms

        for i,c in raw_path:
            if c == 0:
                i += "/InPhase"
                for y in os.listdir(i):
                    tmp = os.path.join(i, y)
                    img = sitk.ReadImage(tmp)
                    img = sitk.GetArrayFromImage(img)[0]
                    self.raw_dataset.append([raw_preprocess(img, True),c])
                    a = tmp.replace("DICOM_anon/InPhase", "Ground")
                    img = sitk.ReadImage(a.replace(".dcm", ".png"))
                    img = sitk.GetArrayFromImage(img)
                    self.label_dataset.append(label_preprocess(img))
            elif c==1:
                for y in os.listdir(i):
                    tmp = os.path.join(i, y)
                    img = sitk.ReadImage(tmp)
                    img = sitk.GetArrayFromImage(img)[0]
                    self.raw_dataset.append([raw_preprocess(img, True),c])
                    a = tmp.replace("DICOM_anon", "Ground")
                    img = sitk.ReadImage(a.replace(".dcm", ".png"))
                    img = sitk.GetArrayFromImage(img)
                    self.label_dataset.append(label_preprocess(img))
        list_path = sorted([os.path.join(path3, x) for x in os.listdir(path3)])
        raw_path =[]
        assert len(raw_path)==0
        for x in list_path:
            c = np.array(2)
            for y in os.listdir(x):
                if "Ground" in y:
                    tmp = os.path.join(x, y)
                    label_path.append(tmp)
                    raw_path.append([tmp.replace("Ground", "DICOM_anon"), c])
                    break
        for i,c in raw_path:
            for y in sorted(os.listdir(i)):

                tmp = os.path.join(i, y)
                dcm = pydicom.dcmread(tmp)
                wc = dcm.WindowCenter[0]
                ww = dcm.WindowWidth[0]
                slope = dcm.RescaleSlope
                intersept = dcm.RescaleIntercept
                low = wc - ww // 2
                high = wc + ww // 2
                img = dcm.pixel_array * slope + intersept
                img[img < low] = low
                img[img > high] = high
                img = (img - low) / (high - low)
                shape= img.copy()
                shape[shape!=0]=1
                self.raw_dataset.append([[img, shape], c])

        for i in label_path:
            for y in sorted(os.listdir(i)):
                img = sitk.ReadImage(os.path.join(i, y))
                img = sitk.GetArrayFromImage(img)
                data = img.astype(dtype=int)
                new_seg = np.zeros(data.shape, data.dtype)
                new_seg[data != 0] = 1
                self.label_dataset.append(new_seg)
        self.split = split
        assert len(self.raw_dataset) == len(self.label_dataset)
        print("chaos train data load success!")
        print("modal:{},fold:{}, total size:{}".format(modals,fold,len(self.raw_dataset)))

    def __getitem__(self, item):
        img, shape_mask, class_label, seg_mask = self.raw_dataset[item][0][0], self.raw_dataset[item][0][1], \
                                                 self.raw_dataset[item][
                                                     1], self.label_dataset[item]
        if img.shape[0]!=256:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            seg_mask = cv2.resize(seg_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            shape_mask = cv2.resize(shape_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        t_img = img * seg_mask
        if self.split == 'train':
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                seg_mask = cv2.flip(seg_mask, 1)
                shape_mask = cv2.flip(shape_mask, 1)
                t_img = cv2.flip(t_img, 1)
        #  scale to [-1,1]
        img = (img - 0.5) / 0.5
        t_img = (t_img - 0.5) / 0.5
        return torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0), torch.from_numpy(t_img).type(
            torch.FloatTensor).unsqueeze(dim=0), \
               torch.from_numpy(shape_mask).type(
                   torch.LongTensor).unsqueeze(dim=0), torch.from_numpy(seg_mask).type(
            torch.LongTensor).unsqueeze(dim=0), torch.from_numpy(class_label).type(torch.FloatTensor)

    def __len__(self):
        return len(self.raw_dataset)


class ChaosDataset_Syn_Test(Dataset):

    def __init__(self, path="../datasets/chaos2019", split='test', modal='t1',gan=False, transforms=None):
        super(ChaosDataset_Syn_Test, self).__init__()
        assert modal in {'t1', 't2','ct'}
        fold = split + "/" + modal
        path = os.path.join(path, fold)
        list_path = sorted([os.path.join(path, x) for x in os.listdir(path)])
        raw_path = []
        label_path = []
        if gan is True:
            list_path = list_path[0:1]
        for x in list_path:
            if modal == "t1":
                x += "/T1DUAL"
            elif modal == "t2":
                x += "/T2SPIR"
            for y in os.listdir(x):
                if "Ground" in y:
                    tmp = os.path.join(x, y)
                    if "ct" in x:
                        label_path.append(tmp)
                    raw_path.append(tmp.replace("Ground", "DICOM_anon"))
                    break

        self.transfroms = transforms
        self.raw_dataset = []
        self.label_dataset = []
        self.index = []
        if modal == "t1":
            for i in raw_path:
                i += "/InPhase"
                n = 0
                for y in os.listdir(i):
                    tmp = os.path.join(i, y)
                    img = sitk.ReadImage(tmp)
                    img = sitk.GetArrayFromImage(img)[0]
                    self.raw_dataset.append(raw_preprocess(img))
                    a = tmp.replace("DICOM_anon/InPhase", "Ground")
                    img = sitk.ReadImage(a.replace(".dcm", ".png"))
                    img = sitk.GetArrayFromImage(img)
                    self.label_dataset.append(label_preprocess(img))
                    n += 1
                self.index.append(n)
        elif modal=='t2':
            for i in raw_path:
                n = 0
                for y in os.listdir(i):
                    tmp = os.path.join(i, y)
                    img = sitk.ReadImage(tmp)
                    img = sitk.GetArrayFromImage(img)[0]
                    self.raw_dataset.append(raw_preprocess(img))
                    a = tmp.replace("DICOM_anon", "Ground")
                    img = sitk.ReadImage(a.replace(".dcm", ".png"))
                    img = sitk.GetArrayFromImage(img)
                    self.label_dataset.append(label_preprocess(img))
                    n += 1
                self.index.append(n)
        else:
            for i in raw_path:
                n = 0
                for y in sorted(os.listdir(i)):
                    tmp = os.path.join(i, y)
                    dcm = pydicom.dcmread(tmp)
                    wc = dcm.WindowCenter[0]
                    ww = dcm.WindowWidth[0]
                    slope = dcm.RescaleSlope
                    intersept = dcm.RescaleIntercept
                    low = wc - ww // 2
                    high = wc + ww // 2
                    img = dcm.pixel_array * slope + intersept
                    img[img < low] = low
                    img[img > high] = high
                    img = (img - low) / (high - low)
                    self.raw_dataset.append(img)
                    n += 1
                self.index.append(n)
            for i in label_path:
                for y in sorted(os.listdir(i)):
                    img = sitk.ReadImage(os.path.join(i, y))
                    img = sitk.GetArrayFromImage(img)
                    data = img.astype(dtype=int)
                    new_seg = np.zeros(data.shape, data.dtype)
                    new_seg[data != 0] = 1
                    self.label_dataset.append(new_seg)
        self.split = split
        assert len(self.raw_dataset) == len(self.label_dataset)
        print("chaos test data load success!")
        print("modal:{},fold:{}, total size:{}".format(modal, fold, len(self.raw_dataset)))

    def __getitem__(self, item):
        img, mask = self.raw_dataset[item], self.label_dataset[item]
        if img.shape[0] != 256:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        img = (img - 0.5) / 0.5
        return torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0), torch.from_numpy(mask).type(torch.LongTensor)

    def __len__(self):
        return len(self.raw_dataset)

    def _getIndex(self):
        return self.index

def label_preprocess(data):
    data = data.astype(dtype=int)
    new_seg = np.zeros(data.shape, data.dtype)
    new_seg[(data > 55) & (data <= 70)] = 1
    return new_seg


def raw_preprocess(data, get_s=False):
    """
    :param data: [155,224,224]
    :return:
    """
    data = data.astype(dtype=float)
    data[data<50] = 0
    out = data.copy()
    out = (out - out.min()) / (out.max() - out.min())

    if get_s:
        share_mask = out.copy()
        share_mask[share_mask != 0] = 1
        return out, share_mask
    return out