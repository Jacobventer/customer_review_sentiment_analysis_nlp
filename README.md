# Customer Review Sentiment Analysis with NLP

This project implements a complete sentiment analysis pipeline for Amazon customer reviews using Natural Language Processing (NLP) and deep learning techniques.

The goal of the project is to automatically classify customer reviews into three sentiment categories:

- Negative
- Neutral
- Positive

The system compares a traditional machine learning approach with a modern transformer-based model to evaluate performance improvements.

# Project Overview

Customer reviews provide valuable insights for companies, but manually analyzing thousands of reviews is inefficient.  
Sentiment analysis allows automated classification of textual feedback to understand customer satisfaction and product perception.  

This project builds a sentiment analysis pipeline consisting of:  

1. Data inspection and preprocessing
2. Baseline machine learning model
3. Transformer-based deep learning model
4. Model evaluation and comparison
5. Iterative optimization


# Dataset

The project uses an Amazon customer reviews dataset containing **568,454 product reviews**.  

Each review includes:

- Review text
- Star rating (1–5)

The ratings are converted into three sentiment categories:

| Rating | Sentiment |
|------|------|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

To avoid class imbalance, the dataset was balanced to include **42,640 samples per class**.


# Project Pipeline

The following data processing pipeline was implemented:

```
Raw Amazon Dataset
        ↓
Initial Data Inspection
        ↓
Data Preparation
(Label Mapping & Cleaning)
        ↓
Balanced Dataset
        ↓
Train/Test Split
        ↓
Baseline Model
(TF-IDF + Logistic Regression)
        ↓
Transformer Model
(DistilBERT)
        ↓
Model Evaluation
```

---

# Models Implemented

## Baseline Model

TF-IDF vectorization with Logistic Regression.

Purpose:

- Establish a performance baseline
- Compare classical NLP methods with deep learning

Accuracy achieved:

**76%**
  

## Transformer Model

A fine-tuned **DistilBERT transformer model** was used for sentiment classification.  
  
Advantages:  

- Contextual word embeddings
- Better handling of negation and mixed sentiment
- Improved language understanding

Training configuration:  
  
- Batch size: 16
- Epochs: 2
- GPU acceleration (NVIDIA GTX 1650)

Accuracy achieved:  
 
**83%**  

# Model Comparison

| Model | Accuracy |
|------|------|
| TF-IDF + Logistic Regression | 0.76 |
| DistilBERT Transformer | 0.83 |

The transformer model significantly outperformed the classical NLP approach due to its ability to capture contextual relationships between words.
  
# Example Prediction

The trained transformer model can classify new customer reviews.

Example 1:

```
Input Review:
"This product is amazing and works perfectly."

Model Prediction:
Sentiment → Positive
Confidence → 0.93
```

Example 2:

```
Input Review:
"The product quality is not good and I would not recommend it."

Model Prediction:
Sentiment → Negative
Confidence → 0.975
```


Users can test their own reviews using the notebook:

```
notebooks/demo_sentiment_prediction.ipynb
```


# Iterative Optimization

An additional experiment was conducted to evaluate model performance with increased training iterations.

Optimization performed:

- Increased training epochs from **2 → 3**

This experiment demonstrates the iterative improvement phase of the project.
  

# Technical Challenge

During development, a GPU configuration issue occurred.

PyTorch was initially installed without CUDA support, resulting in the error:

```
Torch not compiled with CUDA enabled
```

The issue was resolved by reinstalling the CUDA-enabled PyTorch version (cu118) and verifying GPU availability using:

```
torch.cuda.is_available()
```

This enabled GPU acceleration for transformer training.
  

# Repository Structure

```
amazon-review-sentiment-analysis
│
├── data
├── models
├── notebooks
├── reports
└── images
```
  

# Libraries Used

Python libraries used in this project:

- pandas
- scikit-learn
- PyTorch
- HuggingFace Transformers
- Datasets
- Matplotlib
- Seaborn
  
# Author

Jaco Venter  
Artificial Intelligence Portfolio Project
