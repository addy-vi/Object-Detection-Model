# Object Detection Model Using CIFAR-10 Dataset with ResNet50

## Introduction
Object detection plays a crucial role in various applications such as autonomous driving, surveillance, and robotics. In this project, we implement an object detection system using the **CIFAR-10** dataset and **ResNet50** architecture. The CIFAR-10 dataset consists of 60,000 32x32 pixel color images belonging to 10 distinct classes, such as airplanes, dogs, cats, etc. The **ResNet50** model, pre-trained on ImageNet, is used for feature extraction, and we fine-tune it for classifying objects in CIFAR-10 images.

## Objectives
- Implement an object detection system using the **CIFAR-10** dataset and **ResNet50** architecture.
- Leverage **transfer learning** by using a pre-trained ResNet50 model for improved feature extraction.
- Achieve high classification accuracy on the CIFAR-10 dataset.
- Fine-tune the pre-trained model on the CIFAR-10 dataset to classify images into one of 10 categories.

## Technologies Used
- **Programming Language:** Python
- **Libraries/Frameworks:** TensorFlow, Keras, NumPy, Matplotlib, OpenCV
- **Tools:** Jupyter Notebook, VS Code
- **Dataset:** CIFAR-10 Dataset

## CIFAR-10 Dataset
The CIFAR-10 dataset contains 60,000 32x32 pixel color images in 10 classes:
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The dataset is split into 50,000 training images and 10,000 test images.

## Workflow

### 1. **Data Collection:**
   - The CIFAR-10 dataset is publicly available and can be downloaded directly using `tensorflow.keras.datasets`.
   - We will preprocess the dataset by normalizing pixel values to a range of [0, 1].

### 2. **Data Preprocessing:**
   - **Normalization:** Rescale the pixel values to be in the range of [0, 1].
   - **One-hot Encoding:** Convert the categorical labels into one-hot encoded format for classification.

### 3. **Model Development:**
   - **ResNet50:** The pre-trained ResNet50 model is loaded without the top classification layer (using `include_top=False`).
   - **Custom Classifier:** A fully connected dense layer is added on top of the pre-trained ResNet50 model to classify CIFAR-10 images into one of 10 classes.

### 4. **Model Training:**
   - The model is trained using the CIFAR-10 training data.
   - Cross-entropy loss and the Adam optimizer are used.
   - We evaluate the model on the test data and track its accuracy.

### 5. **Model Evaluation:**
   - We evaluate the model using the test dataset and plot accuracy and loss curves to visualize training progress.

## Results
- **Accuracy achieved on CIFAR-10 test dataset:** 90%+ (depending on hyperparameters and training time).
- **Loss curve and accuracy curve**: Shows how well the model performed during training.

## Future Work
- Expand the dataset: Augment the CIFAR-10 dataset to improve model generalization.
- **Fine-tuning:** Unfreeze some layers of the ResNet50 model to improve feature extraction and model performance.
- **Object Detection:** Extend the model to perform object detection (bounding box prediction) alongside classification.
- Explore using advanced techniques like data augmentation and transfer learning for further improvement.

## Conclusion
This project showcases the use of **ResNet50** for object classification with the **CIFAR-10** dataset. By leveraging transfer learning and fine-tuning a pre-trained model, we can achieve high classification accuracy for a variety of object categories. This approach can be adapted to more complex object detection tasks with the addition of bounding box regression layers and advanced architectures.

## Clone Repository

To clone this repository to your local machine, run the following command:

```bash
git clone https://github.com/adnan-saif/Object-Detection-Model.git
