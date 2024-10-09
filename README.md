# Advancing Infrastructure Monitoring: Lightweight Deep Learning for Real-Time Road Distress Detection on Mobile Devices

## Abstract
Cracks in road infrastructure critically affect service life and public safety, making timely detection essential. Currently, detections are primarily manual, relying on experienced engineers, which is time-consuming and labor-intensive. Recent advancements in Artificial Intelligence (AI) and intelligent mobile devices present promising solutions for efficient crack identification. 

This study proposes a lightweight model based on MobileNet V2, optimized for deployment on intelligent mobile devices such as Android smartphones and Mixed Reality (MR) devices, to facilitate rapid identification of road infrastructure defects during field inspections. 

Key findings include:
- The lightweight MobileNet V2 model reduces the number of parameters and the model size by 65% compared to the original version.
- The model maintains a high accuracy rate of 92% on the tested dataset when deployed on smartphones.
- The YOLOv8n and lightweight MobileNet V2 models are capable of real-time detection under varying conditions when deployed on MR devices.

Field inspections conducted in Aachen, Germany, using both smartphones and MR devices confirm the efficacy of the proposed method, demonstrating strong generalization potential in practical engineering applications.

## Dataset and Code
The dataset used in this research is available for download via the following link:

[Download Dataset and Code](https://pan.baidu.com/s/1zeS5c-QtbPu5x6yWR4f0cA)  
Extraction Code: `7h8r`

## Installation Guide

To set up the environment and run the code, follow these steps:

We recommend using `conda` to manage the environment for this project. First, ensure you have `conda` installed. Then, follow these steps to set up the environment:

1. Clone or download this repository and extract the contents.
2. Download the dataset and code from the provided link, and extract the zip file into the project directory.

3. Install the required environment by running the following command:

   ```bash
   conda env create -f environment.yml
