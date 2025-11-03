# YouTube Sentiment Analysis API

This project is an end-to-end machine learning application that serves a sentiment analysis model through a REST API. The application is containerized with Docker for easy deployment and scalability.

## Features

- **ML Model:** Uses a Logistic Regression model and Naive bayes (not selected low accuracy) trained to classify YouTube comments as positive, negative, or neutral.
- model training on google colab.
- **API Server:** A high-performance REST API built with **FastAPI** for real-time predictions.
- **MLOps:** Uses **MLflow** for model tracking and **Git LFS** for versioning large model artifacts.
- **Containerization:** A **Dockerfile** is included to build a self-contained image of the application.
- 
### Prerequisites
- Docker Desktop installed
- Git and Git LFS installed

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/youtube-sentiment-analysis-api.git
cd youtube-sentiment-analysis-api

### Improvements

Training on a better model would skyrocket the prediction accuracy, currrently at 66%, fine-tuning on any llm would considerably increase the accuracy, but it requiers either a high end local GPU, or a
premium subscriptions to an online GPU, hosting on AWS services is another improvement, but since the accuracy of the model is not that high, I would rather not host the model there.
