# Fashion-MNIST Image Classification (CNN)

A complete deep-learning pipeline for classifying Fashion-MNIST images using a Convolutional Neural Network (CNN).  
Includes dataset loading, preprocessing, model training, cross-validation, result visualization, model saving, and real-image predictions.

---

## Features

- Load & preprocess Fashion-MNIST  
- Baseline CNN classifier  
- 5-fold cross-validation  
- Plot accuracy & loss curves  
- Save model: `final_model.h5`  
- Upload real images in Colab  
- Preprocess real images (crop, grayscale, resize, invert, normalize)  
- Predict with class labels  

---

## Dataset Classes

0 - T-shirt/top
1 - Trouser
2 - Pullover
3 - Dress
4 - Coat
5 - Sandal
6 - Shirt
7 - Sneaker
8 - Bag
9 - Ankle Boot


---

## CNN Architecture

Conv2D (32 filters, 3×3, ReLU)
MaxPooling2D (2×2)
Flatten
Dense (100, ReLU)
Dense (10, Softmax)
Optimizer: SGD (lr=0.01, momentum=0.9)
Loss: Categorical Crossentropy


---

## Training Results (5-Fold CV)

Mean Accuracy ≈ 91%
Std Dev ≈ 0.17
Epochs: 10
Batch Size: 32


---

## Example Predictions

### Shoes Example
Original Image → Model Input → Prediction

![Shoes](https://github.com/Kartikay77/fashion-mnist-image-classification/blob/main/Shoes.png)  
![Shoes Pred](https://github.com/Kartikay77/fashion-mnist-image-classification/blob/main/Shoes_predicted.png)

Predicted class: 8
Label: Bag


---

### T-Shirt Example

![Tshirt](https://github.com/Kartikay77/fashion-mnist-image-classification/blob/main/Tshirt.png)  
![Tshirt Pred](https://github.com/Kartikay77/fashion-mnist-image-classification/blob/main/Tshirt_Predicted.png)

Predicted class: 6
Label: Shirt
---

## Real Image Preprocessing Steps

1. Convert to grayscale
2. Center crop to remove background
3. Resize to 28×28
4. Apply Gaussian blur
5. Invert colors (white object on black background)
6. Normalize to [0,1]
7. Reshape to (1, 28, 28, 1)


---

## Run Prediction in Google Colab

```python
from google.colab import files
uploaded = files.upload()

run_example("your_image.png")
```
---
Predicted class ID: 6
Predicted label: Shirt

## Save the Model
model.save('final_model.h5')

## Future Improvements
- Add more convolution layers
- Use data augmentation
- Add dropout for regularization
- Train on real clothing datasets
- Fine-tune a pretrained CNN

