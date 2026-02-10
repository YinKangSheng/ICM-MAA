import torchvision
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import numpy as np
from torchattacks.attack import Attack
from nets.centernet_training import focal_loss, reg_l1_loss
from nets.frcnn_training import FasterRCNNTrainer
from nets.yolo_training import YOLOLoss
from utils.utils import get_classes, get_anchors
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
class Meta_AAFG(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf
p
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model1,model2,model3, eps=8/255,
                 alpha=2/255, steps=8, T=10,random_start=False):
        super().__init__("PGD", model1)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.train_util = FasterRCNNTrainer(model2.eval(), optimizer=None)
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.T =T

    def forward(self, images, metas,targets, bboxes, labels, batch_hms, batch_whs, batch_regs, batch_reg_masks):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        targets = [ann.clone().detach().to(self.device) for ann in targets]
        batch_hms = batch_hms.clone().detach().to(self.device)
        batch_whs = batch_whs.clone().detach().to(self.device)
        batch_regs = batch_regs.clone().detach().to(self.device)
        batch_reg_masks = batch_reg_masks.clone().detach().to(self.device)

        y1, x1, y2, x2, size = metas[0]
        iw,ih=size
        T = torch.ones((3,ih,iw))
        pad_h = max(0, 512 - ih)
        pad_w = max(0, 512 - iw)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        T = F.pad(T, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        T=torch.squeeze(T).to(self.device)
        loss = self.yololoss()

        adv_images = images.clone().detach()
        adv_images_T = images.clone().detach()
        OR = images.clone().detach()
        for t in range(self.T):
            # 1
            for _ in range(self.steps):
                adv_images.requires_grad = True
                outputs = self.model1(adv_images)
                # ----------------------#
                #   计算损失
                # ----------------------#
                loss_value_all = 0
                for l in range(len(outputs)):
                    loss_item = loss(l, outputs[l], targets)
                    loss_value_all += loss_item

                hm, wh, offset = self.model3(adv_images)
                c_loss = focal_loss(hm, batch_hms)
                wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                centernet_loss = c_loss + wh_loss + off_loss

                cost = loss_value_all + centernet_loss
                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_images,
                                           retain_graph=False, create_graph=False)[0]

                adv_images = adv_images.detach() - self.alpha * grad.sign()
                delta = (adv_images - adv_images_T).detach()
                adv_images = (adv_images_T + delta).detach()
            adv_images.requires_grad = True
            frcnnlosslist = self.train_util(adv_images, bboxes, labels, 1)
            rcnnloss = frcnnlosslist[-1]

            cost = rcnnloss
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            new_adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = (new_adv_images - adv_images).detach()
            adv_images = (adv_images_T + delta).detach()
            delta2 = torch.clamp(adv_images - OR, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(OR + delta2, min=0, max=1).detach()
            adv_images_T = adv_images.clone().detach()
            # 2
            for _ in range(self.steps):
                adv_images.requires_grad = True
                outputs = self.model1(adv_images)
                # ----------------------#
                #   计算损失
                # ----------------------#
                loss_value_all = 0
                for l in range(len(outputs)):
                    loss_item = loss(l, outputs[l], targets)
                    loss_value_all += loss_item


                frcnnlosslist =self.train_util(adv_images, bboxes, labels, 1)
                rcnnloss = frcnnlosslist[-1]

                cost = loss_value_all + rcnnloss
                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_images,
                                           retain_graph=False, create_graph=False)[0]

                adv_images = adv_images.detach() - self.alpha * grad.sign()
                delta = (adv_images - adv_images_T).detach()
                adv_images = (adv_images_T + delta).detach()
            adv_images.requires_grad = True
            hm, wh, offset = self.model3(adv_images)
            c_loss = focal_loss(hm, batch_hms)
            wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
            off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)

            centernet_loss = c_loss + wh_loss + off_loss

            cost = centernet_loss
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            new_adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = (new_adv_images - adv_images).detach()
            adv_images = (adv_images_T + delta).detach()
            delta2 = torch.clamp(adv_images - OR, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(OR + delta2, min=0, max=1).detach()
            adv_images_T = adv_images.clone().detach()
            # 3
            for _ in range(self.steps):
                adv_images.requires_grad = True
                frcnnlosslist = self.train_util(adv_images, bboxes, labels, 1)
                rcnnloss = frcnnlosslist[-1]

                hm, wh, offset = self.model3(adv_images)
                c_loss = focal_loss(hm, batch_hms)
                wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)
                off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)
                centernet_loss = c_loss + wh_loss + off_loss

                cost = rcnnloss + centernet_loss
                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_images,
                                           retain_graph=False, create_graph=False)[0]

                adv_images = adv_images.detach() - self.alpha * grad.sign()
                delta = (adv_images - adv_images_T).detach()
                adv_images = (adv_images_T + delta).detach()
            adv_images.requires_grad = True
            outputs = self.model1(adv_images)
            loss_value_all = 0
            for l in range(len(outputs)):
                loss_item = loss(l, outputs[l], targets)
                loss_value_all += loss_item

            cost = loss_value_all
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            new_adv_images = adv_images.detach() - self.alpha * grad.sign()
            delta = (new_adv_images - adv_images).detach()
            adv_images = (adv_images_T + delta).detach()
            delta2 = torch.clamp(adv_images - OR, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(OR + delta2, min=0, max=1).detach()
            adv_images_T = adv_images.clone().detach()
        return adv_images

    def yololoss(self):
        anchors_path = 'model_data/yolo_anchors.txt'
        classes_path = 'model_data/voc_classes.txt'
        # ----------------------------------------------------#
        #   获取classes和anchor
        # ----------------------------------------------------#
        class_names, num_classes = get_classes(classes_path)
        anchors, num_anchors = get_anchors(anchors_path)
        # ----------------------------------------------------#
        #   输入图像size
        # ----------------------------------------------------#
        input_shape = [512, 512]
        # ----------------------------------------------------#
        #   Cuda    是否使用Cuda
        #           没有GPU可以设置成False
        # ---------------------------------#
        Cuda = True
        # ---------------------------------------------------------------------#
        #   anchors_mask    用于帮助代码找到对应的先验框，一般不修改。
        # ---------------------------------------------------------------------#
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # ----------------------#
        #   获得损失函数
        # ----------------------#
        yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)

        return yolo_loss

