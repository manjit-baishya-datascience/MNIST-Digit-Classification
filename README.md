# 🔢 MNIST Digit Classification | TF CNN - ACC 0.99
![Uploading Asset 8.png…]()

## Overview
This project aims to classify handwritten digits using a Convolutional Neural Network (CNN) implemented in TensorFlow. The model achieves an impressive accuracy of 99% on the MNIST dataset, which consists of grayscale images of digits from 0 to 9. This project is beginner-friendly, providing a solid introduction to deep learning techniques and their applications in image recognition.


## Data
The project utilizes the MNIST dataset, a benchmark dataset for handwritten digit classification. It consists of 60,000 training images and 10,000 testing images, with each image being 28x28 pixels in grayscale.

## Data Preprocessing
### 1. Reshaping Image Dimensions
The dataset is reshaped from its original format to match the input shape required by CNNs, which is (number of samples, 28, 28, 1).

### 2. Normalizing Image Pixel Values
The pixel values of the images are normalized to a range of [0, 1] to ensure faster convergence during training.

### 3. Visualizing the Data
A random selection of images from the dataset is visualized to get an understanding of the input data.

## Modeling
### 1. Model Architecture
The CNN model consists of the following layers:
- **Conv2D Layers**: For feature extraction with ReLU activation functions.
- **MaxPooling2D Layers**: For downsampling the feature maps.
- **Flatten Layer**: To convert the 2D matrix into a vector.
- **Dense Layers**: Fully connected layers for classification.
- **Output Layer**: Uses softmax activation to classify the digits (0-9).

### 2. Model Training
The model is trained using the categorical cross-entropy loss function and the Adam optimizer. The training process includes monitoring the accuracy on the validation set to ensure the model generalizes well to unseen data.

## Evaluation
The model achieved an accuracy of 99% on the test set, demonstrating its effectiveness in digit classification tasks.

## Pipeline
1. **Data Loading**: Load the MNIST dataset.
2. **Data Preprocessing**: Reshape and normalize images.
3. **Model Definition**: Define the CNN architecture.
4. **Model Training**: Train the CNN model on the training set.
5. **Model Evaluation**: Evaluate the model on the test set.
6. **Visualization**: Visualize the model's predictions on random test samples.

## Testing the Model
After training, the model was tested on a separate set of images to evaluate its performance. The predictions matched the true labels in 99% of the cases.

## Conclusion
This project successfully demonstrates the application of CNNs in image classification tasks. The high accuracy achieved highlights the model's robustness and effectiveness, making it a strong foundation for more advanced deep learning projects.

## Requirements
The project requires the following Python libraries:
`tensorflow` `numpy` `matplotlib` `scikit-learn`
