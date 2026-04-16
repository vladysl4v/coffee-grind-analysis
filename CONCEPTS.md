# ML Concepts

A reference for domain knowledge relevant to this project.

---

## Loss vs Accuracy

**Loss** is what the model actually optimizes during training. It is a mathematical measure of how wrong the predictions are. We use **MSE (Mean Squared Error)** — the average of squared differences between predicted and actual fineness values. Lower is better.

**Accuracy** in the traditional sense applies to classification (cat vs dog — either right or wrong). Our task is **regression** — predicting a continuous fineness value — so there is no binary correct/incorrect. Instead we report:

- **MAE (Mean Absolute Error)** — on average, how many fineness units off is the prediction. This is the human-readable metric. If MAE = 3.5 it means the model is off by 3.5 fineness units on average.
- **MSE** — same idea but squares the errors, which penalizes large mistakes more heavily. Used as the training loss.

---

## Overfitting

Overfitting is when the model memorizes the training data instead of learning to generalize. You see it when training loss keeps going down but validation loss stays flat or goes up. This is what happened with our SimpleCNN baseline — the model was too small and the dataset too limited to train from scratch without overfitting.

Signs of overfitting:
- Large gap between train and val loss
- Val loss increases while train loss decreases
- Scatter plot shows tight predictions on train, scattered on val

---

## Pretrained Models & Fine-tuning

Training a model from scratch requires a lot of data. With 607 images we don't have enough. Instead we use a **pretrained model** — a network already trained on millions of images (ImageNet) that has learned to detect edges, textures, and shapes.

**Fine-tuning** means taking that pretrained model and adapting it to our specific task. We replace the final layer with a single regression output and train on our data.

Two strategies:
- **Frozen backbone** — only the final layer trains. Fast, less risk of overfitting. Good starting point.
- **Full fine-tune** — all layers train. Potentially better results but needs more data and careful tuning.

---

## Why ResNet18

ResNet18 is a well-established CNN architecture pretrained on ImageNet. It is lightweight enough to train quickly on limited hardware, has strong texture recognition from pretraining, and accepts variable input sizes above a minimum threshold. The ablation experiment (Rashid) confirmed it performs well on our dataset even with minimal augmentation.

---

## Augmentation

Augmentation artificially expands the dataset by generating modified versions of existing images. Flips, rotations, brightness shifts — variations the model would otherwise never see in 607 photos.

Key finding from our ablation: **minimal augmentation outperforms aggressive augmentation by 5× on this dataset**. Heavy transforms distort the grain texture the model needs to learn from. Keep augmentation conservative.

---

## Data Leakage

Our images were collected in groups of 4 — the same physical grain sample photographed 4 consecutive times. If you split randomly at the image level, photos of the same grain end up in both train and test. The model appears better than it is because it has effectively "seen" the test grain before.

We prevent this with a **group-aware split** — all 4 photos of a grain always go to the same split. This ensures val and test results are trustworthy.
