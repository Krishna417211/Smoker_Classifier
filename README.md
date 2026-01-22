Smoker Detector: Image Classification System1. Project GoalThe goal of this project is to create an automated deep learning system capable of analyzing image data (photographs of people) to determine the presence of smoking behavior. This technology can be utilized in health research, security monitoring, or automated content moderation.2. DatasetThe project utilizes the Smoking Dataset from Kaggle, containing labeled images categorized into two classes:SmokingNot SmokingData Distribution:Training Set: 716 filesTesting Set: 224 filesInput Dimensions: 256x256 pixels3. Technical ArchitectureThe model is built using TensorFlow and Keras using a Sequential Convolutional Neural Network (CNN) designed for binary classification.Key Model Components:Preprocessing: Rescaling(1./255) layer to normalize pixel values.Data Augmentation: Built-in RandomFlip("horizontal") and RandomRotation(0.1) layers to improve model robustness.Feature Extraction:3 x Conv2D layers (32, 64, and 128 filters) using ReLU activation.MaxPooling2D layers following each convolution to reduce dimensionality.Classification Head:Flatten layer followed by a Dense layer of 128 neurons.Dropout (0.5) to prevent overfitting during training.Sigmoid activation output for binary probability.4. Training ConfigurationOptimizer: Adam (Learning Rate: 0.0001).Loss Function: Binary Crossentropy.Epochs: 40.Batch Size: 32.5. Performance MetricsAfter 40 epochs, the model achieved a stable classification performance on the test set:Accuracy: ~75%.F1-Score: 0.75 for both Smoking and Non-Smoking classes.Classification Report:ClassPrecisionRecallF1-ScoreNot Smoking0.740.770.75Smoking0.760.730.756. How to UseTo predict a new image, use the provided predict_smoking function in the notebook:Pythonfrom tensorflow.keras.preprocessing import image
import numpy as np

def predict_smoking(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        print("Class: Smoking")
    else:
        print("Class: Not Smoking")
7. Installation & DependenciesEnsure you have Python 3.x installed along with the following libraries:tensorflownumpymatplotlibseabornscikit-learn
