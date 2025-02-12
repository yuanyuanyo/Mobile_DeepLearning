# A Lightweight Deep Learning for Real-Time Road Distress Detection on Mobile Devices

## Abstract
MobiLiteNet is a lightweight deep learning framework designed for **real-time road distress detection** on mobile and mixed reality (MR) devices. By integrating **Efficient Channel Attention (ECA), structural refinement, sparse knowledge distillation, structured pruning, and quantization**, the model achieves **high detection accuracy with significantly reduced computational cost**. Field tests confirm its applicability in mobile road monitoring, while a **diverse dataset from Europe and Asia** ensures robustness. Experimental results show that MobiLiteNet outperforms baseline models, enabling **scalable and accurate road distress detection** for **intelligent transportation and infrastructure management**.


Key findings include:
- Proposed MobiLiteNet, a novel lightweight deep learning framework optimized for real-time road distress detection on mobile and MR devices.  
- Successful deployment on MR devices in road engineering projects in Aachen, Germany, demonstrating practical feasibility in complex environments.  
- Constructed a diverse dataset with road distress images from Europe and Asia, enhancing model robustness and generalization.  

## Dataset and Code
The dataset used in this research is available for download via the following link:

[Download Dataset for Classification](https://pan.baidu.com/s/1ZO0rKhjO_f2OE5SWqxbbjg)  
Extraction Code: `aej9`

[Download Dataset for Object Detection](https://pan.baidu.com/s/1evnkLWYLZ9VKDcnH5ueqHQ)  
Extraction Code: `qimr`

## Installation Guide

To set up the environment and run the code, follow these steps:

We recommend using `conda` to manage the environment for this project. First, ensure you have `conda` installed. Then, follow these steps to set up the environment:

1. Clone or download this repository and extract the contents.
2. Download the dataset and code from the provided link, and extract the zip file into the project directory.

3. Install the required environment by running the following command:

   ```bash
   conda env create -f environment.yml

## 🚀 Instruction for Use

###  1. Dataset Configuration
Organize your dataset in the following directory structure:

   ```plaintext
   dataset/
       train/
           class1/
               img1.png
               img2.png
           class2/
               img1.png
               img2.png
       val/
           class1/
               img1.png
           class2/
               img1.png

