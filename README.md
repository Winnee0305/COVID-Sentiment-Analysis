# COVID Sentiment Analysis Project

## 📋 Project Overview

This project implements and compares multiple sentiment analysis models for analyzing COVID-19 related social media content. The analysis focuses on classifying text into three sentiment categories: **Positive**, **Negative**, and **Neutral**.

## 🎯 Objectives

- Implement and evaluate different sentiment analysis approaches
- Compare performance of rule-based, machine learning, and deep learning models
- Analyze COVID-19 related social media sentiment patterns
- Provide insights into model strengths and limitations

## 📊 Dataset

The project uses a COVID-19 related social media dataset with the following characteristics:

- **Total Samples:** 9,040 (training), 1,938 (testing)
- **Features:** Text content, author, timestamp, comments count
- **Labels:** Positive, Negative, Neutral
- **Class Distribution:**
  - Positive: 54.6%
  - Neutral: 23.2%
  - Negative: 22.2%

## 🤖 Models Implemented

### 1. VADER (Valence Aware Dictionary and sEntiment Reasoner)

- **Type:** Rule-based lexicon approach
- **Method:** Dictionary-based sentiment scoring
- **Advantages:** Fast, interpretable, no training required
- **Performance:** 60.47% accuracy

### 2. Random Forest

- **Type:** Machine learning ensemble
- **Features:** TF-IDF vectorization
- **Configuration:** 100 estimators, random_state=42
- **Performance:** 70.74% accuracy

### 3. LSTM (Long Short-Term Memory)

- **Type:** Deep learning neural network
- **Architecture:**
  - Embedding layer (GloVe 100d)
  - LSTM layer (128 units)
  - Dropout (0.5)
  - Dense layers (64 → 3 classes)
- **Performance:** 76.83% accuracy

## 📈 Results Summary

| Model             | Accuracy   | Training Time | Inference Time | Best Class               |
| ----------------- | ---------- | ------------- | -------------- | ------------------------ |
| **VADER**         | 60.47%     | N/A           | 0.65s          | Positive (67.93% F1)     |
| **Random Forest** | 70.74%     | 15.45s        | 0.11s          | Positive (79.54% F1)     |
| **LSTM**          | **76.83%** | 101.31s       | 1.12s          | **Positive (83.19% F1)** |

## 🔍 Detailed Model Analysis

### VADER Performance

- **Strengths:** Fast inference, interpretable, good positive sentiment detection
- **Weaknesses:** Poor negative sentiment precision (41.59%), limited contextual understanding
- **Best For:** Quick prototyping, interpretable results

### Random Forest Performance

- **Strengths:** Fast training and inference, good overall accuracy
- **Weaknesses:** Severe class imbalance issues, poor negative sentiment recall (8.12%)
- **Best For:** Real-time applications, resource-constrained environments

### LSTM Performance

- **Strengths:** Highest accuracy, balanced performance across classes, contextual understanding
- **Weaknesses:** Long training time, computational intensive, moderate overfitting
- **Best For:** Production systems where accuracy is priority

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
│       ├── randomForest/
│       │   ├── randomForest_train.ipynb
│       │   ├── randomForest_test.ipynb
│       │   ├── randomForest.joblib
│       │   ├── classification_report_randomForest.csv
│       │   ├── overall_accuracy_randomForest.csv
│       │   └── computation_time_randomForest.csv
│       └── lstm/
│           ├── lstm_train.ipynb
│           ├── lstm_test.ipynb
│           ├── lstm.keras
│           ├── tokenizer.pkl
│           ├── classification_report_lstm.csv
│           ├── overall_accuracy_lstm.csv
│           ├── computation_time_lstm.csv
│           ├── accuracy_epochs_lstm.png
│           └── loss_epochs_lstm.png
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn vaderSentiment joblib
```

### Data Preparation

1. Download the GloVe embeddings from [Stanford NLP](https://nlp.stanford.edu/projects/glove/)
2. Place `glove.6B.100d.txt` in `src/pretrained_model/` directory
3. Ensure your dataset is in the `data/` directory

### Running the Models

#### VADER Analysis

```bash
cd src/model/vader/
jupyter notebook vader.ipynb
```

#### Random Forest Training & Testing

```bash
cd src/model/randomForest/
jupyter notebook randomForest_train.ipynb
jupyter notebook randomForest_test.ipynb
```

#### LSTM Training & Testing

```bash
cd src/model/lstm/
jupyter notebook lstm_train.ipynb
jupyter notebook lstm_test.ipynb
```

## 📊 Key Findings

### Model Performance Insights

1. **LSTM achieves the best overall performance** (76.83% accuracy)
2. **Class imbalance significantly affects model performance**
3. **Negative sentiment detection is challenging** for all models
4. **Neutral sentiment is most consistently classified**

### Training Insights

- **LSTM shows good convergence** over 10 epochs
- **Moderate overfitting observed** in LSTM (training vs validation gap)
- **Optimal training duration:** 6-7 epochs for LSTM
- **Random Forest suffers from class imbalance** affecting negative class detection

### Confusion Matrix Insights

- **Negative-Neutral confusion:** 25.1% of negative samples misclassified as neutral
- **Positive class bias:** Model tends to over-predict positive sentiment
- **Neutral class stability:** Most balanced precision/recall performance

## 🔧 Recommendations

### For Production Use

- **High Accuracy Required:** Use LSTM with early stopping at 6-7 epochs
- **Real-time Processing:** Use Random Forest with class balancing techniques
- **Interpretability Needed:** Use VADER for quick analysis

### Model Improvements

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

## 📝 Usage Examples

### Using Trained Models

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

#### Random Forest Model

```python
import joblib

# Load model
model = joblib.load('src/model/randomForest/randomForest.joblib')

# Predict sentiment
text = "COVID-19 has been challenging for everyone"
prediction = model.predict([text])
```

#### VADER Analysis

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
text = "COVID-19 has been challenging for everyone"
scores = analyzer.polarity_scores(text)
```

## Acknowledgments

- Stanford NLP for GloVe embeddings
- VADER sentiment analysis library
- TensorFlow and scikit-learn communities

---

**Note:** This project is for educational and research purposes. The models and results should be validated for specific use cases before deployment in production environments.
