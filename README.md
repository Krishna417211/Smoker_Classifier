# Smoker Detector: Image Classification System

## 1. Project Overview
This project is an automated deep learning system built to analyze images and detect smoking behavior. It is designed for applications in health research, safety monitoring, and automated content moderation.

## 2. Dataset Information
The model uses the **Smoking Dataset** from Kaggle.
* **Classes:** Smoking vs. Not Smoking
* **Total Training Images:** 716
* **Total Testing Images:** 224
* **Resolution:** 256x256 pixels

## 3. Model Architecture
Built using **TensorFlow/Keras**, the model follows a Sequential CNN structure:
* **Preprocessing:** Rescaling (1./255) and Data Augmentation (Flip/Rotation).
* **Convolutional Layers:** Three layers (32, 64, and 128 filters) with ReLU activation.
* **Pooling:** MaxPooling2D layers for spatial reduction.
* **Regularization:** Dropout (0.5) to prevent overfitting.
* **Output:** Dense layer with Sigmoid activation.

## 4. Training Results
The model was trained for **40 epochs** with the Adam optimizer.

### Final Performance Metrics:
| Metric | Value |
| :--- | :--- |
| **Accuracy** | ~75% |
| **F1-Score (Smoking)** | 0.75 |
| **F1-Score (Non-Smoking)** | 0.75 |

## 5. Usage
To test an image with the model, use the following snippet:

```python
# Predict function
def predict_smoking(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    return "Smoking" if prediction[0][0] > 0.5 else "Not Smoking"
