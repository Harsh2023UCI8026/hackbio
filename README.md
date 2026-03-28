# hackbio

# Bone-Fracture-Detection-DeepLearning


Hello everyone.
In this project, I built a deep learning model that detects bone fractures from X-ray images.

The goal of this project is to automatically classify whether an X-ray contains a fracture or not using a Convolutional Neural Network (CNN).

This project demonstrates how deep learning can assist in medical image analysis, helping doctors detect fractures faster and more accurately.

Environment & Setup

This project was developed using Google Colab, which provides a cloud-based environment with optional GPU acceleration for training deep learning models.

The model is implemented using PyTorch, a popular deep learning framework used for building and training neural networks.

Libraries Used

The following libraries are used for data processing, training, and visualization:

PyTorch

Torchvision

NumPy

Matplotlib

Example imports:

Dataset

The dataset consists of X-ray images of bones divided into two classes:

Fractured

Non-Fractured

Dataset Structure
dataset/
    fractured/
        image1.jpg
        image2.jpg
    non-fractured/
        image3.jpg
        image4.jpg


Data Preprocessing

Before training the model, several preprocessing steps are applied to the images:

Resizing images to a fixed size (224 × 224)

Converting images to tensors

Preparing them for input into the neural network

Example transformation pipeline:

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
These transformations ensure that all images have the same format and dimensions, which is required for training neural networks.

Model Architecture

The model used in this project is a Convolutional Neural Network (CNN).

CNNs are highly effective for image analysis tasks because they can automatically learn important spatial patterns from images.

Key Components of the Model

1️⃣ Convolution Layers
Extract important features from X-ray images such as edges and fracture lines.

2️⃣ Activation Functions
Non-linear functions (such as ReLU) help the network learn complex patterns.

3️⃣ Pooling Layers
Reduce spatial dimensions and help the model focus on the most important features.

4️⃣ Fully Connected Layers
Combine all learned features to make the final classification decision.

These layers allow the model to learn patterns that indicate fractures in X-ray images.

Model Training

The model is trained on the dataset using the following process:

The model receives an input X-ray image.

It predicts whether the image contains a fracture.

A loss function calculates the difference between prediction and actual label.

The optimizer updates the model weights to reduce this error.

Loss Function

Binary Cross Entropy with Logits:
lossfun = torch.nn.BCEWithLogitsLoss()

Optimizer

Adam Optimizer:
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

Learning Rate Scheduler

A Cosine Annealing scheduler is used to gradually reduce the learning rate during training.
This helps the model converge more effectively and improves training stability.

📊 Model Evaluation

After training, the model is evaluated using the test dataset.

One important evaluation tool used is the Confusion Matrix.

The confusion matrix shows:

True Positives → correctly predicted fractures

True Negatives → correctly predicted non-fractures

False Positives → predicted fracture but actually normal

False Negatives → missed fracture

This helps analyze how well the model is performing.

Random Prediction Demo

The project also includes a feature that selects a random X-ray image from the dataset.

The model predicts whether the X-ray contains a fracture, and the predicted label is compared with the actual label.

This provides a simple way to visually verify the model’s predictions.

Grad-CAM Visualization (Explainable AI)

To make the model more interpretable, Grad-CAM visualization is used.

Grad-CAM highlights the regions of the X-ray image that the model focuses on while making predictions.

Red regions indicate areas that strongly influenced the model's decision.

These regions often correspond to fracture locations in the X-ray image.

This improves model explainability, which is especially important in medical applications.

Results

The trained model is able to:

Analyze X-ray images

Detect patterns indicating bone fractures

Classify images as fractured or non-fractured

With larger datasets and further optimization, this system could assist in automated medical diagnostics.


Conclusion

This project demonstrates how deep learning can be applied to medical image analysis.

By training a neural network on X-ray images, we can automatically detect fractures with promising results.

With larger datasets and more advanced architectures, such systems could potentially assist doctors in faster and more accurate diagnosis.
