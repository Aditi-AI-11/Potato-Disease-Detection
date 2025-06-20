# Potato-Disease-Detection
This project uses Convolutional Neural Networks (CNNs) to detect common potato leaf diseases such as Early Blight and Late Blight from image data. The model is trained on publicly available datasets and achieves high accuracy in classifying diseased vs. healthy leaves.

Potato Disease Detection Using Deep Learning
⚠️ IMPORTANT NOTE:
I have tried to include the error lines and warnings that might appear during training because many people may encounter these issues. This way, you can recognize these errors, troubleshoot them, and write the correct code. Please do not worry — the project is legit and fully functional. If you follow the notebook to the end, you will see the working code and successful results.

Project Overview -->

This project uses deep learning to classify and detect diseases in potato leaves from images. The model identifies multiple disease types such as Early Blight, Late Blight, and Healthy leaves, enabling farmers and agricultural experts to take timely actions to prevent crop loss.

The model is trained on a labeled dataset of potato leaf images and uses techniques like data augmentation, dropout, and early stopping to improve performance and reduce overfitting.

Key Features -->

High Accuracy: Achieved over 97% validation accuracy.

Multiple Classes: Classifies healthy leaves and common potato diseases.

Data Augmentation: Enhances model robustness by expanding the variety of training images.

Overfitting Prevention: Incorporates dropout layers and early stopping during training.

Prediction Functions: Includes functions to predict classes and confidence scores for new images.

Training Visualization: Provides plots for training and validation accuracy/loss to monitor model performance.

Dataset -->

The dataset consists of labeled images of potato leaves categorized into:

Healthy

Early Blight

Late Blight

-> Images are split into training, validation, and test sets.

Model Architecture & Training -->

Model compiled with Adam optimizer and softmax activation for multi-class classification.

Trained for up to 50 epochs with batch size of 32.

Early stopping monitors validation loss and stops training if no improvement for 5 epochs, restoring the best weights.

Dropout layers added to reduce overfitting.

Learning rate scheduling used to reduce fluctuations in accuracy.

Data augmentation applied to improve generalization.

Performance Metrics -->

Training Accuracy: ~99%

Validation Accuracy: ~98%

Test Accuracy: ~99%

Plots of training & validation accuracy and loss are generated for detailed analysis.

How to Use? -->

Load and preprocess the dataset with training, validation, and test splits.

Apply data augmentation and compile the model.

Train the model using the provided script with early stopping and dropout.

Evaluate the model on the test set.

Use the prediction function to classify new images and get confidence scores.

Visualize training progress and predictions using matplotlib plots.      

Prediction Example --> 

The model predicts the class and confidence score for input images.

Visualization plots show actual vs predicted labels along with confidence percentages.

Future Improvements -->

Explore more complex architectures or transfer learning.

Develop mobile or web apps for real-time disease detection.

Expand dataset to cover more disease classes and conditions.

Acknowledgements -->

Dataset contributors.

TensorFlow and Keras libraries.




