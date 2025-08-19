# 🖼️ Image Caption Generator with VGG16 + LSTM

This project demonstrates how to generate natural language captions for images using **deep learning**.
It combines a **CNN (VGG16)** for feature extraction with an **LSTM-based sequence model** for text generation.

The dataset used is **Flickr8k**, downloaded via [kagglehub](https://www.kaggle.com/datasets/adityajn105/flickr8k).

---

## 📌 Features

* Automatic download of the **Flickr8k dataset** using KaggleHub
* Image feature extraction with **VGG16** pretrained on ImageNet
* Text preprocessing with **Keras Tokenizer**
* Sequence generation using **LSTM**
* Custom training loop to combine **image embeddings + text sequences**
* Caption generation for unseen images
* Visualization with generated captions on test images

---

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/image-caption-generator.git
cd image-caption-generator

# Install dependencies
pip install kagglehub tensorflow numpy tqdm matplotlib pillow
```

---

## 📂 Dataset

The dataset is automatically downloaded via KaggleHub:

```python
import kagglehub

# Download Flickr8k dataset
path = kagglehub.dataset_download("adityajn105/flickr8k")
print("Path to dataset files:", path)
```

Dataset contains:

* **Images/** → Flickr8k image dataset
* **captions.txt** → Captions associated with each image

---

## 🧠 Model Architecture

1. **Feature Extractor**

   * Pretrained **VGG16** network (ImageNet)
   * Extracts a 4096-dim feature vector per image

2. **Sequence Model**

   * Tokenized captions with `<start>` and `<end>` tokens
   * Embedding layer + LSTM (256 units)
   * Dense layers for decoding combined features

3. **Final Model**

   * Input: Image features + Partial captions
   * Output: Next predicted word

---



## 📝 Caption Generation

Generate captions for new images:

```python
caption = generate_caption(model, tokenizer, photo, max_length)
print("Generated Caption:", caption)
```

Visualize the result:

```python
plt.imshow(image)
plt.axis('off')
plt.title("Caption: " + caption)
plt.show()
```

---



## 📊 Results

* **Vocabulary Size:** \~8,000 words
* **Max Caption Length:** \~35 words
* Captions generated are **basic but contextually aligned with the image**

---

## 🛠️ Tech Stack

* **Python** 🐍
* **TensorFlow / Keras** for Deep Learning
* **VGG16** (ImageNet pretrained)
* **LSTM** for sequence modeling
* **Matplotlib & PIL** for visualization
* **KaggleHub** for dataset download

---

## 📌 Future Improvements

* Use **Beam Search** instead of Greedy Search for better captions
* Train with **Flickr30k** or **MS-COCO** dataset for more robust captions
* Replace **VGG16** with **EfficientNet / ResNet** for stronger image features
* Use **Transformer-based Decoder (like GPT)** for more natural language fluency

---


Would you like me to also **add a folder structure + usage steps (run on Kaggle Notebook/Colab/Local)** so it’s beginner-friendly when someone lands on your repo?
