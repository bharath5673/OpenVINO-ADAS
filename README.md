# OpenVINO-ADAS

## SIMPLE and EASY Real-time object detection and road segementaions 

### Description:

This repository provides a straightforward implementation of road segmentation and object detection using the OpenVINO toolkit, designed specifically for Advanced Driver Assistance Systems (ADAS). The code is kept clean and easy to understand, making it accessible even for those new to computer vision and OpenVINO.

![demo](output.gif)

## Key Features:

- Road Segmentation: Utilizes a pre-trained OpenVINO model to accurately segment road regions from input frames, enabling precise analysis of the driving environment.

- Object Detection: Employs a separate OpenVINO model to identify and classify objects within the scene, such as vehicles, pedestrians, and more.

- Efficient Inference: Leverages OpenVINO's optimized inference engine for fast and efficient processing on various hardware platforms.

- Clean Codebase: Well-structured and thoroughly commented code ensures readability and ease of modification for custom requirements.

### Prerequisites

- Python 3.x
- OpenCV
- PyTorch
- NumPy

### Installation

1. Clone this repository.
2. Install the required dependencies
3. conda env python>=3.9 recommended

```bash
pip3 install openvino-dev
```

### Usage

1. Download pre-trained YOLOv5 weights or train your own model.
2. Provide the path to the YOLOv5 weights in the code.
3. Run the script with the video file.
4. View the object detection results and Bird's Eye View visualization.

For more detailed usage instructions and options, refer to the project documentation.

### Run

```bash
python3 main.py
```

### Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.

### Acknowledgments

- OpenVINO: [INTEL-Openvino](https://docs.openvino.ai/2023.0/openvino_docs_install_guides_overview.html?ENVIRONMENT=DEV_TOOLS&OP_SYSTEM=WINDOWS&VERSION=v_2023_0_2&DISTRIBUTION=PIP)
- YOLOv5: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- OpenCV: [https://opencv.org/](https://opencv.org/)

