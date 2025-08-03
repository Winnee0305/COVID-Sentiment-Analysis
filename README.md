# Analyzing Public Sentiment on COVID-19 Vaccines
A Comparative Study of NLP and Machine Learning Techniques Across Social Media


## 📦 Libraries Used

<p align="left">
  <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"></a>
  <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/docs/transformers/index"><img src="https://img.shields.io/badge/Transformers-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black" alt="Transformers"></a>
  <a href="https://huggingface.co/docs/datasets/index"><img src="https://img.shields.io/badge/Datasets-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black" alt="Datasets"></a>
  <a href="https://github.com/cjhutto/vaderSentiment"><img src="https://img.shields.io/badge/VADER-4B0082?style=for-the-badge&logo=python&logoColor=white" alt="VADER"></a>
  <a href="https://textblob.readthedocs.io/"><img src="https://img.shields.io/badge/TextBlob-6A5ACD?style=for-the-badge&logo=python&logoColor=white" alt="TextBlob"></a>
  <a href="https://joblib.readthedocs.io/"><img src="https://img.shields.io/badge/joblib-696969?style=for-the-badge&logo=python&logoColor=white" alt="joblib"></a>
  <a href="#"><img src="https://img.shields.io/badge/pickle-003366?style=for-the-badge&logo=python&logoColor=white" alt="pickle"></a>
  <a href="https://matplotlib.org/"><img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib"></a>
  <a href="https://seaborn.pydata.org/"><img src="https://img.shields.io/badge/Seaborn-76B900?style=for-the-badge&logo=seaborn&logoColor=white" alt="Seaborn"></a>
  <a href="https://github.com/amueller/word_cloud"><img src="https://img.shields.io/badge/WordCloud-708090?style=for-the-badge&logo=python&logoColor=white" alt="WordCloud"></a>
  <a href="#"><img src="https://img.shields.io/badge/time-003366?style=for-the-badge&logo=python&logoColor=white" alt="time"></a>
  <a href="#"><img src="https://img.shields.io/badge/pathlib-003366?style=for-the-badge&logo=python&logoColor=white" alt="pathlib"></a>
</p>

## 📋 Project Overview

This project implements and compares multiple sentiment analysis models for analyzing COVID-19 related social media content. The analysis focuses on classifying text into three sentiment categories: **Positive**, **Negative**, and **Neutral**.

## 🎯 Objectives

- Implement and evaluate different sentiment analysis approaches
- Compare performance of rule-based, machine learning, and deep learning models
- Analyze COVID-19 related social media sentiment patterns
- Provide insights into model strengths and limitations

## 📊 Dataset

The project uses a COVID-19 related social media dataset collected from: [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FXJTBQM&utm_)

The characteristics of this dataset:

- **Total Samples:** 9,040 (training), 1,938 (validating), 1,938 (testing)
- **Features:** id, text, author, created_utc, No_of_comments, Subjectivity, Polarity, Analysis, Parent, Link
- **Labels:** Positive, Negative, Neutral
- **Class Distribution:**
  - Positive: 54.6%
  - Neutral: 23.2%
  - Negative: 22.2%

## 📁 Project Structure

```
COVID-Sentiment-Analysis/
├── data/
│   ├── Covid_without_dup.csv
│   ├── train_data.csv
│   ├── val_data.csv
│   └── test_data.csv
├── src/
│   ├── data_preprocessing.ipynb
│   ├── EDA.ipynb
│   ├── main.ipynb
│   ├── pretrained_model/
│   │   └── glove.6B.100d.txt
│   └── model/
│       ├── vader/
│       │   ├── vader.ipynb
│       │   ├── vader_sentiment_output.csv
│       │   ├── classification_report_vader.csv
│       │   ├── overall_accuracy_vader.csv
│       │   └── computation_time_vader.csv
│       ├── textblob/
│       │   ├── textblob.ipynb
│       │   ├── textblob_sentiment_output.csv
│       │   ├── classification_report_textblob.csv
│       │   ├── overall_accuracy_textblob.csv
│       │   └── computation_time_textblob.csv
│       ├── randomForest/
│       │   ├── randomForest_train.ipynb
│       │   ├── randomForest_test.ipynb
│       │   ├── randomForest.joblib
│       │   ├── classification_report_randomForest.csv
│       │   ├── overall_accuracy_randomForest.csv
│       │   └── computation_time_randomForest.csv
│       ├── svm/
│       │   ├── svm_train.ipynb
│       │   ├── svm_test.ipynb
│       │   ├── svm.joblib
│       │   ├── classification_report_svm.csv
│       │   ├── overall_accuracy_svm.csv
│       │   └── computation_time_svm.csv
│       ├── lstm/
│       │   ├── lstm_train.ipynb
│       │   ├── lstm_test.ipynb
│       │   ├── lstm.keras
│       │   ├── tokenizer.pkl
│       │   ├── classification_report_lstm.csv
│       │   ├── overall_accuracy_lstm.csv
│       │   ├── computation_time_lstm.csv
│       │   ├── accuracy_epochs_lstm.png
│       │   └── loss_epochs_lstm.png
│       └── DistilBERT/
│           ├── distilbert_train.ipynb
│           ├── distilbert_test.ipynb
│           ├── classification_metrics_per_class.csv
│           ├── overall_accuracy_distilbert.csv
│           ├── computation_time_distilbert.csv
│           ├── classification_barchart_distilbert.png
│           └── confusion_matrix_distilbert.png
└── README.md
```

## 🤖 Models Implemented

### 1. VADER (Valence Aware Dictionary and sEntiment Reasoner)

- **Type:** Rule-based lexicon approach
- **Method:** Dictionary-based sentiment scoring
- **Advantages:** Fast, interpretable, no training required
- **Performance:** 60.47% accuracy


### 2. TextBlob

- **Type:** Rule-based lexicon approach
- **Method:** Polarity-based sentiment scoring
- **Advantages:** Simple, fast, no training required
- **Performance:** 100.00% accuracy (Note: This appears to be an error in evaluation)

### 3. Random Forest

- **Type:** Machine learning ensemble
- **Features:** TF-IDF vectorization
- **Configuration:** 100 estimators, random_state=42
- **Performance:** 70.74% accuracy

### 4. SVM (Support Vector Machine)

- **Type:** Machine learning classifier
- **Features:** TF-IDF vectorization
- **Method:** Linear kernel SVM
- **Performance:** 75.59% accuracy

### 5. LSTM (Long Short-Term Memory)

- **Type:** Deep learning neural network
- **Architecture:**
  - Embedding layer (GloVe 100d)
  - LSTM layer (128 units)
  - Dropout (0.5)
  - Dense layers (64 → 3 classes)
- **Performance:** 76.83% accuracy

### 6. DistilBERT

- **Type:** Transformer-based deep learning
- **Architecture:** Distilled BERT model
- **Method:** Fine-tuned on sentiment classification
- **Performance:** 89.78% accuracy

## 📈 Results Summary

| Model             | Accuracy   | Training Time | Inference Time | Best Class              |
| ----------------- | ---------- | ------------- | -------------- | ----------------------- |
| **VADER**         | 60.47%     | N/A           | 0.65s          | Positive (67.93% F1)    |
| **TextBlob**      | 100.00%    | N/A           | ~0.5s          | All Classes (100% F1)   |
| **Random Forest** | 70.74%     | 15.45s        | 0.11s          | Positive (79.54% F1)    |
| **SVM**           | 75.59%     | 55.86s        | 1.98s          | Positive (81.65% F1)    |
| **LSTM**          | 76.83%     | 101.31s       | 1.12s          | Positive (83.19% F1)    |
| **DistilBERT**    | **89.78%** | 260.51s       | 5.24s          | **Neutral (94.01% F1)** |

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn vaderSentiment joblib textblob transformers torch
```

### Data Preparation

1. Download the GloVe embeddings from [Stanford NLP](https://nlp.stanford.edu/projects/glove/)
2. Place `glove.6B.100d.txt` in `src/pretrained_model/` directory
3. Ensure your dataset is in the `data/` directory

## 📊 Key Findings

1. **LSTM achieves the best overall performance** (76.83% accuracy)
2. **Class imbalance significantly affects model performance**
3. **Negative sentiment detection is challenging** for all models
4. **Neutral sentiment is most consistently classified**

## 🔧 Future Improvements

1. **Address Class Imbalance:**
   - Implement SMOTE or class weights
   - Data augmentation for minority classes
     
2. **Enhance Negative Detection:**
   - Feature engineering for sentiment-specific patterns
   - Context window optimization
   - Ensemble methods with lexicon-based approaches

3. **Optimize LSTM:**
   - Implement early stopping
   - Add regularization techniques
   - Use learning rate scheduling

## 📝 Usage Examples - Using Trained Models

#### TextBlob Analysis

```python
from textblob import TextBlob

text = "COVID-19 has been challenging for everyone"
blob = TextBlob(text)
sentiment = blob.sentiment.polarity
```

#### VADER Analysis

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
text = "COVID-19 has been challenging for everyone"
scores = analyzer.polarity_scores(text)
```

#### SVM Model

```python
import joblib

# Load model
model = joblib.load('src/model/svm/svm.joblib')

# Predict sentiment
text = "COVID-19 has been challenging for everyone"
prediction = model.predict([text])
```

#### Random Forest Model

```python
import joblib

# Load model
model = joblib.load('src/model/randomForest/randomForest.joblib')

# Predict sentiment
text = "COVID-19 has been challenging for everyone"
prediction = model.predict([text])
```

#### DistilBERT Model

```python
from transformers import pipeline

# Load model
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")

# Predict sentiment
text = "COVID-19 has been challenging for everyone"
result = classifier(text)
```

#### LSTM Model

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load model and tokenizer
model = load_model('src/model/lstm/lstm.keras')
with open('src/model/lstm/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Predict sentiment
text = "COVID-19 has been challenging for everyone"
sequences = tokenizer.texts_to_sequences([text])
padded = pad_sequences(sequences, maxlen=100)
prediction = model.predict(padded)
```

## Acknowledgments

- Stanford NLP for GloVe embeddings
- VADER sentiment analysis library
- TensorFlow and scikit-learn communities

---

**Note:** This project is for educational and research purposes. The models and results should be validated for specific use cases before deployment in production environments.
