import cv2
import numpy as np
import os
import torchvision
from torch.utils.data import DataLoader
from frcnn import FRCNN
from metadomain_aafg import Metadomian_AAFG
from meta_aafg_dataloader import MataAttack_YoloDataset,yolo_dataset_collate

from utils.utils import get_classes
from yolo import YOLO
from tqdm import tqdm
from centernet import CenterNet
from PIL import Image

if __name__ == "__main__":
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0
    #------------------------------------------------------------------#
    num_workers         = 20
    # ----------------------------------------------------#
    #   输入图像size
    # ----------------------------------------------------#
    input_shape = [512, 512]
    #---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    train_annotation_path   ='new_trainvallist.txt'
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()[9750::]
    # ---------------------------------------#
    #   构建数据集加载器。
    # ---------------------------------------#
    train_dataset = MataAttack_YoloDataset(train_lines, input_shape, num_classes, train=False)

    voc_yolo = YOLO(model_path="512logs/best_epoch_weights.pth")
    voc_frcnn = FRCNN(model_path='frcnn512logs/best_epoch_weights.pth')
    voc_centernet = CenterNet(model_path='cen512logs/best_epoch_weights.pth')
    coco_yolo = YOLO(model_path="512cocologs/best_epoch_weights.pth")
    coco_frcnn = FRCNN(model_path='frcnn512logs/coco.pth')
    coco_centernet = CenterNet(model_path='cen512logs/coco.pth')
    train_sampler = None
    shuffle = False
    batch_size = 1
    num = 0
    mulattack = Metadomian_AAFG(voc_yolo.net, voc_frcnn.net, voc_centernet.net, coco_yolo.net, coco_frcnn.net, coco_centernet.net)

    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)



    for iteration, batch in enumerate(gen):
        if iteration >= num and num!=0:
            break
        pbar = tqdm(desc=f'picture {iteration}/{len(gen)}', postfix=dict, mininterval=0.3)
        metas, filenames, images, targets, bboxes, labels, batch_hms, batch_whs, batch_regs, batch_reg_masks= batch
        adv_images = mulattack(images, metas, targets, bboxes, labels, batch_hms, batch_whs, batch_regs, batch_reg_masks)
        for n in range(len(adv_images)):
            y1, x1, y2, x2, size = metas[n]
            # w,h = size
            image_name = filenames[n].split('.')[0]
            img = adv_images[n,:, x1:x2, y1:y2]
            img = torchvision.transforms.ToPILImage()(img)
            # img = img.resize((w,h), Image.BICUBIC)
            img.save("5class_metadomain_images/"+image_name+".png")




