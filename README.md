# YOLOv11 with EfficientNet Backbone

This project is a modified version of [YOLOv11](https://github.com/ultralytics/ultralytics) with an EfficientNet backbone. By replacing the original backbone with EfficientNet, we aim to achieve enhanced efficiency and performance in object detection.

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/JYe9/YOLO11_EfficientNet.git
    cd YOLO11_EfficientNet
    ```

2. Install dependencies:

    ```bash
    pip install ultralytics
    ```

## Modifications

The main modifications are as follows:

1. **EfficientNet Backbone Integration**: EfficientNet has been added to YOLOv11 as the backbone to improve model efficiency.
2. **Updated Configurations**: A new configuration file `yolo11_EfficientNet.yaml` has been created to incorporate EfficientNet's architecture.

### Key Modification Steps

#### 1. Adding EfficientNet Code

In `/ultralytics/ultralytics/nn/modules/block.py`, the following classes were added to implement EfficientNet:
- `SeBlock`: Implements the Squeeze-and-Excitation block for adaptive average pooling.
- `drop_connect`: Applies drop connect regularization.
- `stem`, `MBConvBlock`: Builds EfficientNet's base structure.

#### 2. Adjusting `__init__.py`

In `modules/__init__.py`, import and register the newly added functions and classes.

#### 3. Creating EfficientNet YAML Configuration

The new file `yolo11_EfficientNet.yaml` in `/ultralytics/ultralytics/cfg/models/11` defines the EfficientNet backbone structure and model parameters.

## Usage

### Training

1. Adjust the dataset path in the configuration file as needed.
2. Run the training script:

    ```bash
    python train.py 
    ```
3. Run the detection script:

    ```bash
    python detect.py 
    ```
4. Run the validation script:

    ```bash
    python val.py 
    ```

This command will initiate training on the modified YOLOv11 model with the EfficientNet backbone.

## Acknowledgments

This project is based on the official [YOLOv11 repository](https://github.com/ultralytics/ultralytics). The backbone modification and integration of EfficientNet were inspired by the need to optimize model performance. 

Reference Paper [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946)

