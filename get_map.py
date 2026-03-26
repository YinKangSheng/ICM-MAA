import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageChops
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from models.yolo import YOLO
import math
import matplotlib
matplotlib.use('Agg')  # 强制使用无界面模式 (Agg)，只保存图片不显示


if __name__ == "__main__":
    def PSNR(img1, img2):
        mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
        if mse < 1.0e-10:
            return 100
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    '''
    Recall和Precision不像AP是一个面积的概念，因此在门限值（Confidence）不同时，网络的Recall和Precision值是不同的。
    默认情况下，本代码计算的Recall和Precision代表的是当门限值（Confidence）为0.5时，所对应的Recall和Precision值。

    受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算不同门限条件下的Recall和Precision值
    因此，本代码获得的map_out/detection-results/里面的txt的框的数量一般会比直接predict多一些，目的是列出所有可能的预测框，
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容，默认为-1
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #--------------------------------------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    #--------------------------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #--------------------------------------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x，mAP0.x的意义是什么请同学们百度一下。
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #
    #   当某一预测框与真实框重合度大于MINOVERLAP时，该预测框被认为是正样本，否则为负样本。
    #   因此MINOVERLAP的值越大，预测框要预测的越准确才能被认为是正样本，此时算出来的mAP值越低，
    #--------------------------------------------------------------------------------------#
    MINOVERLAP      = 0.5
    #--------------------------------------------------------------------------------------#
    #   受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算mAP
    #   因此，confidence的值应当设置的尽量小进而获得全部可能的预测框。
    #   
    #   该值一般不调整。因为计算mAP需要获得近乎所有的预测框，此处的confidence不能随便更改。
    #   想要获得不同门限值下的Recall和Precision值，请修改下方的score_threhold。
    #--------------------------------------------------------------------------------------#
    confidence      = 0.001
    #--------------------------------------------------------------------------------------#
    #   预测时使用到的非极大抑制值的大小，越大表示非极大抑制越不严格。
    #   
    #   该值一般不调整。
    #--------------------------------------------------------------------------------------#
    nms_iou         = 0.5
    #---------------------------------------------------------------------------------------------------------------#
    #   Recall和Precision不像AP是一个面积的概念，因此在门限值不同时，网络的Recall和Precision值是不同的。
    #   
    #   默认情况下，本代码计算的Recall和Precision代表的是当门限值为0.5（此处定义为score_threhold）时所对应的Recall和Precision值。
    #   因为计算mAP需要获得近乎所有的预测框，上面定义的confidence不能随便更改。
    
    #   这里专门定义一个score_threhold用于代表门限值，进而在计算mAP时找到门限值对应的Recall和Precision值。
    #---------------------------------------------------------------------------------------------------------------#
    score_threhold  = 0.5
    
    #-------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    #-------------------------------------------------------#
    map_vis         = True
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    #-------------------------------------------------------#
    # map_out_path    = 'map_out_adv2nd'
    # map_out_path    = 'map_out_original'
    map_out_path    = 'map_out_AAFG_5classImage'
    #-------------------------------------------------------#
    #   攻击结果输出的文件夹，默认为attack_map_out
    #-------------------------------------------------------#
    attack_map_out_path    = 'map_out'

    txt = open(os.path.join("new_testlist.txt")).readlines()
    # image_ids = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    image_ids = [line.strip() for line in txt]

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)


    #-------------------------------------------------------#
    #  mode=-1,计算攻击后的map
    #-------------------------------------------------------#
    if map_mode == -1:
        if not os.path.exists(attack_map_out_path):
            os.makedirs(attack_map_out_path)
        if not os.path.exists(os.path.join(attack_map_out_path, 'ground-truth')):
            os.makedirs(os.path.join(attack_map_out_path, 'ground-truth'))
        if not os.path.exists(os.path.join(attack_map_out_path, 'detection-results')):
            os.makedirs(os.path.join(attack_map_out_path, 'detection-results'))
        if not os.path.exists(os.path.join(attack_map_out_path, 'images-optional')):
            os.makedirs(os.path.join(attack_map_out_path, 'images-optional'))
        p = 0
        print("Load model.")
        yolo = YOLO(confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            line = image_id.split()
            image_path = line[0]
            # print(image_path)
            img_name = image_path.split('/')[-1]
            # print(img_name)
            image_name = img_name.split('.')[0]
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_name + ".jpg")
            # if not os.path.exists(attack_image_path):
            #     continue
            attack_image_path = os.path.join("../Clip_codec/playground/experiments/objVOC_condi5_tune_1/" + image_name + ".png")
            attack_image       = Image.open(attack_image_path)
            image    = Image.open(image_path)
            p = p + PSNR(np.array(image), np.array(attack_image))

            yolo.get_map_txt(image_name, attack_image, class_names, attack_map_out_path, np.array(np.shape(image)[0:2]))

        psnr = p / len(image_ids)
        print(
            f"{psnr:.3f} psnr "
        )

    if  map_mode == -1:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            line = image_id.split()
            image_path = line[0]
            # print(image_path)
            img_name = image_path.split('/')[-1]
            # print(img_name)v
            image_name = img_name.split('.')[0]
            with open(os.path.join(attack_map_out_path, "ground-truth/"+image_name+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_name+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == -1:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold=score_threhold, path=attack_map_out_path)
        print("Get map done.")

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence = confidence, nms_iou = nms_iou,classes_path="model_data/voc_classes.txt",
                    model_path="logs/best_epoch_weights.pth", anchors_path="model_data/yolo_anchors.txt")
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            # 191~194是新加的
            line = image_id.split()
            image_path = line[0]
            img_name = image_path.split('/')[-1]
            image_trueId = img_name.split('.')[0]
            # image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_trueId+".jpg")
            image_path = os.path.join("5class_metaAttack_image", image_trueId + ".png")
            # image_path = os.path.join("VOCattack_images", image_trueId + ".png")
            image = Image.open(image_path)
            if map_vis:
                # 1. 调用 detect_image 进行预测并画框 (返回一张画好框的图)
                # 注意：这里使用的是当前 yolo 的 confidence (0.001)
                # 所以出来的图可能会有密密麻麻的红框，这是正常的 mAP 可视化
                r_image = yolo.detect_image(image.copy())
    
                # 2. 保存画好框的图
                r_image.save(os.path.join(map_out_path, "images-optional/" + image_trueId + ".jpg"))
            # yolo.get_map_txt(image_id, image, class_names, map_out_path)
            yolo.get_map_txt(image_trueId, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            line = image_id.split()
            image_path = line[0]
            img_name = image_path.split('/')[-1]
            image_trueId = img_name.split('.')[0]
            with open(os.path.join(map_out_path, "ground-truth/"+image_trueId+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_trueId+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
