# 🎬 Movie Recommendation System

## 📌 Project Overview

The Movie Recommendation System is an end-to-end Machine Learning project that recommends similar movies based on user-selected movies. The system analyzes movie information using Natural Language Processing (NLP) techniques and identifies movies with similar content.

The project was built using a dataset containing approximately **50,000 movie records**. After performing data cleaning and preprocessing, movie features were transformed using **TF-IDF Vectorization**, and similarities between movies were calculated using **Cosine Similarity**. The final model was deployed as an interactive web application using **Streamlit** and hosted through **GitHub**.

---

## 🎯 Problem Statement

With thousands of movies available across different platforms, users often struggle to discover movies that match their interests. This project helps users find relevant movie recommendations quickly by analyzing movie content and identifying similar movies.

---

## 🚀 Features

- Recommend movies based on a selected movie.
- NLP-based recommendation engine.
- Interactive and user-friendly Streamlit interface.
- Fast similarity search using Cosine Similarity.
- End-to-end Machine Learning workflow.
- Deployed and accessible through a web application.
- Handles a large dataset of approximately 50,000 movies.

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries & Frameworks
- Pandas
- NumPy
- Scikit-learn
- Streamlit

### NLP Techniques
- Text Preprocessing
- TF-IDF Vectorization

### Machine Learning Concepts
- Feature Engineering
- Similarity-Based Recommendation System
- Cosine Similarity

### Deployment
- Streamlit
- GitHub

---

## 📂 Dataset

- Dataset Size: Approximately **50,000 movie records**
- Data includes movie-related information such as:
  - Title
  - Genres
  - Overview/Description
  - Keywords
  - Cast
  - Crew
  - Other metadata

---

## 🔄 Project Workflow

### 1. Data Collection
- Loaded movie dataset containing around 50,000 records.

### 2. Data Cleaning
- Removed missing values.
- Handled duplicate records.
- Selected relevant columns.
- Standardized textual data.

### 3. Feature Engineering
- Combined important textual features.
- Created a single text representation for each movie.

### 4. Text Vectorization
- Applied **TF-IDF Vectorization** to convert movie text into numerical vectors.

### 5. Similarity Calculation
- Computed **Cosine Similarity** between movie vectors.
- Generated similarity scores for movie recommendations.

### 6. Recommendation Engine
- Selected the most similar movies based on similarity scores.
- Returned top recommended movies to users.

### 7. Deployment
- Built an interactive web application using Streamlit.
- Uploaded project to GitHub.
- Deployed the application for public access.

---

## 🧠 Machine Learning Approach

This project uses a **Content-Based Recommendation System**.

The recommendation process follows:

1. Convert movie content into TF-IDF vectors.
2. Measure similarity between movies using Cosine Similarity.
3. Recommend movies with the highest similarity scores.

This approach enables recommendations based on movie content rather than user ratings.

---

## 📊 Model Pipeline

```text
Movie Dataset
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Recommendation Engine
      ↓
Streamlit Web Application
```

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/movie-recommendation-system.git
```

### Move to Project Directory

```bash
cd movie-recommendation-system
```

### Install Dependencies

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
2. Select or enter a movie name.
3. Click on the recommendation button.
4. View the top recommended movies similar to your selected movie.

---

## 📈 Future Improvements

- Hybrid Recommendation System.
- Collaborative Filtering.
- Deep Learning-based Recommendations.
- User Authentication.
- Personalized Recommendations.
- Integration with Movie APIs for posters and ratings.
- Real-time recommendation updates.

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

- Data Cleaning and Preprocessing
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Cosine Similarity
- Recommendation Systems
- Machine Learning Workflow
- Streamlit Development
- GitHub Deployment
- End-to-End Project Building

---

## 👨‍💻 Author

**Yagnik Patel**

Aspiring Data Scientist skilled in:
- Python
- SQL
- Excel
- Power BI
- Machine Learning
- Natural Language Processing (NLP)

---

⭐ If you found this project useful, consider giving it a star on GitHub.
