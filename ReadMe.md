# Hand Gesture Recognition using CNN

## Overview
This project implements a **Hand Gesture Recognition** system using a **Convolutional Neural Network (CNN)** to classify hand gestures captured using the **LeapGestRecog Dataset**. The model is trained on grayscale images and achieves high accuracy in recognizing various hand gestures.

## Dataset
The dataset used is the **LeapGestRecog** dataset available on Kaggle. It consists of:
- **10 different gestures** performed by **10 subjects**.
- Each gesture is stored in separate folders.
- Infrared grayscale images captured using the **Leap Motion Controller**.

Dataset Path in **Kaggle Notebook**:  
`/kaggle/input/leapgestrecog/leapGestRecog`

## Installation
### **1. Clone the Repository** (If running locally)
```sh
$ git clone https://github.com/TeefAlfadhli/PRODIGY_ML_04.git
$ cd PRODIGY_ML_04
```

### **2. Install Dependencies**
```sh
$ pip install -r requirements.txt
```

*(Ensure TensorFlow, OpenCV, NumPy, Matplotlib, and Scikit-learn are installed.)*

## Model Architecture
The CNN model consists of:
1. **Convolution Layers** (Feature Extraction)
2. **Max-Pooling Layers** (Dimensionality Reduction)
3. **Flattening Layer** (Vector Conversion)
4. **Fully Connected Dense Layers** (Classification)
5. **Softmax Activation** (Output Probability Distribution)

## Training the Model
```python
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
```

## Evaluation
- **Test Accuracy** is calculated using:
```python
test_accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Test Accuracy: {test_accuracy:.4f}")
```
- A **Classification Report** is generated to evaluate performance.

## Visualizing Predictions
A few random test images along with their predictions are displayed:
```python
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    idx = np.random.randint(len(X_test))
    img = X_test[idx].reshape(120, 120)
    true_label = GESTURES[y_true_classes[idx]]
    predicted_label = GESTURES[y_pred_classes[idx]]
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"True: {true_label}\nPred: {predicted_label}")
    axes[i].axis('off')
plt.show()
```

## Results
The model achieves a **high accuracy** in classifying hand gestures with an average accuracy above **99%**.

## Future Improvements
- Use **data augmentation** to improve generalization.
- Implement **real-time gesture recognition** using OpenCV and a webcam.
- Deploy the model as a **web application** using Flask.


## License
This project is licensed under the **MIT License**.

---
*Developed with ❤️ using TensorFlow and OpenCV.*


