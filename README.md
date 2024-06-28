# Helmet Safety Detection

Fine-tuning pre-trained YOLOv10 model by using pre-prepared dataset. The dataset contains of images of construction or factory workers with coordinate of bounding boxes of their helmet.
Main goal is to have a specialized model that can check if workers wear a helmet or not.


## Getting Started

These instructions will help you run the fine-tuning process steps by steps on Google Colab.

### Prerequisites

Required libraries:
  - ultralytics
  - torch==2.0.1
  - torchvision==0.15.2
  - onnx==1.14.0
  - onnxruntime==1.15.1
  - pycocotools==2.0.7
  - PyYAML==6.0.1
  - scipy==1.13.0
  - onnxsim==0.4.36
  - onnxruntime-gpu==1.18.0
  - gradio==4.31.5
  - opencv-python==4.9.0.80
  - psutil==5.9.8
  - py-cpuinfo==9.0.0
  - huggingface-hub==0.23.2
  - safetensors==0.4.3

### Running the Code

0. **Open the Notebook in Google Colab**
  
    You can open the notebook directly in Google Colab by clicking on the link below: [Google Colab](https://colab.research.google.com/drive/1mxy5mqXwFHhMKrsIrd5_qRMb_1b8wFYx)


1. **Clone the Repository**

    First, clone the repository to your local machine:
    ```sh
    !git clone https://github.com/Khoa22213131/AIO_2024-HelmetSafetyDetection
    ```

2. **Download the Dataset**

    In the notebook, run the cell to download the dataset from Google Drive:
    ```python
    !gdown '1twdtZEfcw4ghSZIiPDypJurZnNXzMO7R'
    ```

    Unzip the downloaded file:
    ```python
    !mkdir safety_helmet_dataset
    !unzip -q '/content/Safety_Helmet_Dataset.zip' -d '/content/safety_helmet_dataset'
    ```

3. **Download Pre-trained Model and Install Required Libraries**

    Clone the YOLOv10 repository and install the required libraries:
    ```python
    !git clone https://github.com/THU-MIG/yolov10.git
    %cd yolov10
    !pip install -q -r requirements.txt
    !pip install -e .
    ```

4. **Initialize YOLOv10**

    Download the pre-trained weights:
    ```python
    !wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt
    ```

    Initialize the model:
    ```python
    !pip install ultralytics
    from ultralytics import YOLOv10

    MODEL_PATH = 'yolov10n.pt'
    model = YOLOv10(MODEL_PATH)
    ```

5. **Fine-tuning the Model**

    Prepare the training dataset and set the parameters for the fine-tuning process:
    ```python
    YAML_PATH = './safety_helmet_dataset/data.yaml'
    EPOCHS = 50
    IMG_SIZE = 640
    BATCH_SIZE = 256
    ```

    Train the model:
    ```python
    model.train(data=YAML_PATH, epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE)
    ```

6. **Evaluation**

    After training, evaluate the model:
    ```python
    TRAINED_MODEL_PATH = 'runs/detect/train/weights/best.pt'
    model = YOLOv10(TRAINED_MODEL_PATH)

    model.val(data=YAML_PATH, imgsz=IMG_SIZE, split='test')
    ```

By following these steps, you will be able to run the code in your Google Colab notebook successfully.
