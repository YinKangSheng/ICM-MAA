# Towards Generalized Image Coding for Machine Through Meta Adversarial Adaptation

This repo contains the official PyTorch implementation for the paper "Towards Generalized Image Coding for Machine Through Meta Adversarial Adaptation", published in **International Journal of Computer Vision (IJCV 2026)**.

## Updates

**Latest**
The code and pretrained models are released.

**2026/01/01**
*Towards Generalized Image Coding for Machine Through Meta Adversarial Adaptation* is accepted at **IJCV 2026**!

## Abstract

The existing Image Coding for Machine (ICM) paradigm aims at simultaneously fulfilling both machine analytics and human perception needs by incorporating the performance constraint of downstream machine vision models. However, the intrinsic semantic gap among different vision tasks and the reliance on the performance of specific models pose flexibility and generalization issues when handling unseen scenarios. To this challenge, this paper introduces a novel ICM paradigm that imposes an additional constraint on the reconstructed image from the **Meta-Adversarial-Adaptation (MAA)** perspective. 
Extensive experimental results have demonstrated the effectiveness of our design in achieving satisfactory perceptual quality, improved machine analytics performance, and powerful generalization capacity regarding unseen downstream models, image domains, and object-centric tasks.

<!-- 请将你的框架图命名为 framework.png 并放在 img 文件夹下 -->
<div align="center">
  <img src="img/table1.png" width="800"/>
</div>

<div align="center">
  <img src="img/table2.png" width="800"/>
</div>

## Environment

*   Python 3.8
*   PyTorch 1.10.2
*   CompressAI 1.2.4
*   CUDA 11.3

## Training:

### Stage One: Meta-AAFG Generation
Generate the adversarial augmented images using the Model Set (YOLOv3, Faster-RCNN, CenterNet).

```bash
cd ./MetaAAFG && python train_meta_aafg.py --dataset VOC --epsilon 8/255 --meta_steps 10 --save_path ./augmented_data

### Stage Two: Compression Fine-tuning
Fine-tune the LIC backbone using the Meta-AAFG augmented images as pseudo ground truth.

```bash
cd ./Compression && python train_compression.py --backbone bmshj2018 --quality 3 --lr 1e-4 --dataset ./augmented_data

## Testing:
Machine Analytics (Detection):
Evaluate the detection performance (mAP) on PASCAL VOC, COCO, or OID datasets.
