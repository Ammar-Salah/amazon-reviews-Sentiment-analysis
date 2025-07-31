
# Amazon Product Review Sentiment Analysis

## Overview

This project presents a complete machine learning pipeline for analyzing and classifying Amazon product reviews into three sentiment categories: **Positive**, **Negative**, and **Neutral**. By leveraging Natural Language Processing (NLP) and supervised learning techniques, we aim to help businesses and analysts automatically interpret customer feedback at scale.

The application includes data preprocessing, exploratory data analysis, model building using **Logistic Regression** and **Support Vector Machines (SVM)**, model evaluation, and a full deployment through a **Streamlit web application**.

---

## Dataset Overview

| Column Name                | Description                                                                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Id**                     | A unique identifier for each review (row index). Often just a running number.                                                                     |
| **ProductId**              | The unique ID of the product being reviewed.                                                                                                      |
| **UserId**                 | A unique identifier for the user who wrote the review.                                                                                            |
| **ProfileName**            | The display name of the reviewer.                                                                                                                 |
| **HelpfulnessNumerator**   | The number of users who found the review helpful (e.g., 3 out of 4).                                                                              |
| **HelpfulnessDenominator** | The total number of users who voted on the helpfulness of the review.                                                                             |
| **Score**                  | The actual rating given by the user (usually from 1 to 5 stars).                                                                                  |
| **Time**                   | The **Unix timestamp** of when the review was posted.                                                   |
| **Summary**                | The short title or headline of the review.                                                                                                        |
| **Text**                   | The full text/body of the customer’s review.                                                                                                      |
| **Review\_Length**         | This is likely **pre-computed** and represents the **length of the review text** (number of words or characters). |


---

## Project Objectives

- Clean and explore Amazon product review data.
- Convert raw text into numerical features suitable for ML models.
- Train and compare two classification models (Logistic Regression and SVM).
- Evaluate performance using classification metrics.
- Build and deploy an interactive web application for real-time sentiment prediction.

<img width="7000" height="200" alt="image" src="https://github.com/user-attachments/assets/b86ea973-df4e-4c26-87a4-4d95f043e782" />

---

## Tools & Technologies

| Category            | Tools & Libraries                             |
|---------------------|-----------------------------------------------|
| Language            | Python 3.8+                                    |
| Data Manipulation   | pandas, numpy                                  |
| Text Processing     | nltk, scikit-learn's TfidfVectorizer           |
| Modeling            | scikit-learn (Logistic Regression, SVM)       |
| Visualization       | matplotlib, seaborn                            |
| Deployment          | Streamlit                                      |
| Saving Models       | pickle                                         |

---

## Project Structure

```
amazon-sentiment-analysis/
├── models/
│   ├── svc_model.pkl
│   └── logistic_model.pkl
├── vectorizers/
│   └── vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## Exploratory Data Analysis (EDA)

We performed an in-depth analysis of the dataset to understand:
- Word frequencies in different sentiment classes.
- Review length distributions.
- Most common unigrams and bigrams.
- Sentiment class distribution.
- data balancing

---

## Preprocessing & Feature Engineering

- Lowercasing, punctuation & number removal
- Stopword removal using NLTK
- Lemmatization
- TF-IDF Vectorization for transforming text into numerical features

---

## Model Building

We implemented and compared two supervised learning models:

1. **Logistic Regression**
   - Simple and effective for linearly separable text data.
   - Fast and interpretable.

2. **Support Vector Machine (SVM)**
   - Robust to high-dimensional spaces.
   - Works well with text classification problems.

We evaluated both models using:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

---

## Results

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | 0.8939979310490426      |
| SVM                 | 0.902273734878713       |

---

## Deployment

We developed a user-friendly **Streamlit app** that allows users to:

- Input a review text manually
- Choose between **Logistic Regression** and **SVM** models
- Instantly get sentiment prediction: **Positive**, **Negative**, or **Neutral**

<img width="1400" height="800" alt="image" src="https://github.com/user-attachments/assets/ec1d3890-f540-4dea-a028-74a4bdb9de35" />

---

## Example Use Cases

| Review Text                                            | Prediction  |
|--------------------------------------------------------|-------------|
| "I love this product, it works perfectly!"             | Positive    |
| "It’s okay, not what I expected, but not bad either."  | Neutral     |
| "Worst purchase I’ve ever made, totally disappointed." | Negative    |

---

## Authors

This project was developed by:

- **Ammar Salah**  
- **Amira El Sayed**  
- **Ahmed Shaaban**

> *For academic, educational, and portfolio purposes.*

---

## Contact

Feel free to open an issue on GitHub or connect with the developers if you have suggestions or questions!
