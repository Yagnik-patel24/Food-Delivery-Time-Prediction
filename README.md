# 🍔 Food Delivery Time Prediction System

## 📌 Project Overview

The Food Delivery Time Prediction System is an end-to-end Machine Learning project designed to estimate the delivery time of food orders based on various factors such as distance, restaurant location, delivery partner details, weather conditions, traffic conditions, and order characteristics.

The project involves complete data preprocessing, feature engineering, model building, evaluation, and deployment. Multiple regression algorithms were tested, and **Polynomial Regression** was selected as the final model because it achieved a higher **R² Score** compared to Linear Regression.

The final solution was deployed as an interactive web application using **Streamlit**, allowing users to predict food delivery times in real-time.

---

## 🎯 Problem Statement

Food delivery platforms need accurate delivery time estimates to improve customer satisfaction and operational efficiency.

This project predicts the expected delivery time of an order based on available order and delivery-related information.

---

## 🚀 Features

- Predict food delivery time in real-time.
- Interactive Streamlit web application.
- Complete Machine Learning pipeline.
- Data cleaning and preprocessing.
- Feature engineering and encoding.
- Comparison of Linear Regression and Polynomial Regression.
- Higher prediction accuracy using Polynomial Regression.
- End-to-end deployment workflow.

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

### Machine Learning
- Linear Regression
- Polynomial Regression
- Feature Scaling
- Model Evaluation

### Deployment
- Streamlit
- GitHub

---

## 📂 Dataset Information

The dataset contains food delivery order information such as:

- Delivery Distance
- Restaurant Location
- Customer Location
- Weather Conditions
- Traffic Conditions
- Vehicle Type
- Delivery Partner Information
- Order Type
- Preparation Time
- Delivery Time (Target Variable)

---

## 🔄 Project Workflow

### 1. Data Collection

- Imported food delivery dataset.
- Explored dataset structure and quality.

### 2. Data Cleaning

- Removed duplicate records.
- Handled missing values.
- Corrected inconsistent data.
- Treated outliers where necessary.

### 3. Exploratory Data Analysis (EDA)

- Analyzed feature distributions.
- Identified important factors affecting delivery time.
- Visualized relationships between variables.

### 4. Feature Engineering

- Created meaningful input features.
- Encoded categorical variables.
- Prepared data for model training.

### 5. Data Preprocessing

- Feature Scaling
- Data Transformation
- Train-Test Split

### 6. Model Building

#### Linear Regression

Built a baseline Linear Regression model to understand linear relationships between features and delivery time.

#### Polynomial Regression

Applied Polynomial Features to capture non-linear relationships within the data and improve prediction performance.

### 7. Model Evaluation

Models were evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

Polynomial Regression achieved a higher R² Score and better overall prediction performance.

### 8. Deployment

- Developed an interactive Streamlit application.
- Integrated the trained model into the application.
- Deployed the complete project for end-user access.

---

## 🤖 Machine Learning Models Used

### Linear Regression

A statistical model that predicts delivery time using a linear relationship between input features and the target variable.

### Polynomial Regression

An advanced regression technique that captures non-linear relationships by transforming features into polynomial terms.

**Final Selected Model:** Polynomial Regression

Reason:
- Better fit on the dataset
- Higher R² Score
- Improved prediction accuracy

---

## 📊 Model Pipeline

```text
Food Delivery Dataset
          ↓
Data Cleaning
          ↓
EDA
          ↓
Feature Engineering
          ↓
Encoding & Scaling
          ↓
Train-Test Split
          ↓
Linear Regression
          ↓
Polynomial Regression
          ↓
Model Evaluation
          ↓
Best Model Selection
          ↓
Streamlit Deployment
```

---

## 📈 Evaluation Metrics

The model performance was evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

Polynomial Regression outperformed Linear Regression and was selected as the final model.

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/food-delivery-time-prediction.git](https://github.com/Yagnik-patel24/Food-Delivery-Time-Prediction)
```

### Navigate to Project Folder

```bash
cd Food-Delivery-Time-Prediction
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 💡 How to Use

1. Open the Streamlit application.
2. Enter delivery-related details.
3. Click the Predict button.
4. View the estimated delivery time instantly.

---

## 📷 Application Preview

Add screenshots of:

- Home Page
- User Input Form
- Prediction Results
- Model Performance Dashboard

---

## 🔮 Future Improvements

- Random Forest Regression
- XGBoost Regression
- Real-Time GPS Integration
- Live Traffic API Integration
- Weather API Integration
- Delivery Route Optimization
- Deep Learning Models

---

## 🎓 Learning Outcomes

Through this project, I gained hands-on experience in:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Regression Algorithms
- Linear Regression
- Polynomial Regression
- Model Evaluation Techniques
- Streamlit Development
- Machine Learning Deployment
- End-to-End Project Development

---

## 👨‍💻 Author

**Yagnik Patel**

Aspiring Data Scientist with expertise in:

- Python
- SQL
- Excel
- Power BI
- Machine Learning
- Data Analysis
- Data Visualization

---

⭐ If you found this project useful, please consider giving it a star on GitHub.
