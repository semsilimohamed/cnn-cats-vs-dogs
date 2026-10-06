# CNN Image Classifier: Cats vs Dogs

A convolutional neural network built from scratch with TensorFlow/Keras to classify images as **cat** or **dog**. It's a deep-learning fundamentals project covering data augmentation, convolution and pooling, dense layers, binary classification, and reading the learning curves.

## Results

| Metric (epoch 25) | Train | Validation |
|---|---|---|
| Accuracy | 90.8 % | **80.1 %** |
| Loss | 0.223 | 0.522 |

![Training curves](images/training_curves.png)

**Interpretation:** validation accuracy plateaus around **79–81 % from epoch 9 onward**, while training accuracy keeps climbing to 91 %. Validation loss is lowest around epoch 14 (0.452) and rises after that, so the model starts **overfitting** in the second half of training. The *Next steps* section lists the fixes.

## Data

- 10,000 images from the Kaggle *Dogs vs. Cats* dataset: **8,000 for training** (4,000 per class) and **2,000 for testing**.
- The images aren't included in this repo. To run it, place them in this structure:
  ```
  dataset/
  ├── training_set/{cats,dogs}/
  └── test_set/{cats,dogs}/
  ```

## Method

| Step | Choice |
|---|---|
| Preprocessing | Resize to 64×64, rescale pixels to [0, 1] |
| Augmentation (train only) | Shear 0.2, zoom 0.2, horizontal flip |
| Architecture | Conv2D(32, 3×3, ReLU) → MaxPool(2×2) → Conv2D(32, 3×3, ReLU) → MaxPool(2×2) → Flatten → Dense(128, ReLU) → Dense(1, sigmoid) |
| Training | Adam, binary cross-entropy, batch 32, 25 epochs |
| Prediction | Probability ≥ 0.5 → dog, otherwise cat |

## Run it

Open `cnn_cats_vs_dogs.ipynb` in Google Colab (a GPU runtime is recommended), upload the dataset zip to Google Drive, update the path in the `unzip` cell, and run all cells. Training takes about 15 minutes on a Colab GPU.

## Next steps

- **EarlyStopping** on `val_loss` (patience ≈ 3) and keep the best weights. With these curves, training would stop around epoch 14.
- Add **Dropout** after the dense layer, and a third convolution block.
- Report a **confusion matrix**, precision and recall on the test set, not just accuracy.
- Try **transfer learning** (MobileNetV2 / EfficientNet), which typically reaches over 95 % on this task.
- Apply the same pipeline to a medical imaging task (e.g. pneumonia detection on chest X-rays).

## Stack

Python · TensorFlow / Keras · NumPy · Matplotlib · Google Colab

## Author

**Mohamed Semsili**, Data Scientist (health specialty), Casablanca
