## 📌 Hassaniya Sentiment Analysis System

This project is an **end-to-end sentiment analysis system for Hassaniya Arabic**, a low-resource dialect widely used in Mauritania. The system classifies text into positive, negative, or neutral sentiment using both classical Machine Learning and Deep Learning approaches.


## 🎯 Objective

The goal is to build a functional NLP system for Hassaniya text by:

* Creating a custom labeled dataset
* Training machine learning models
* Exploring deep learning approaches
* Deploying a working web application


## 🏷️ Data Annotation

Since no ready dataset exists for Hassaniya text, we:

* Collected raw text samples
* Performed **manual annotation**
* Assigned sentiment labels (positive / negative / neutral)

👉 This step is the core contribution of the project.


## 🤖 Models

We implemented and compared:

 **Naive Bayes** → strong baseline for small datasets
 
 **Deep Learning** → tested but affected by data scarcity

👉 Observation: Classical ML performs more stable than deep learning due to limited data.



## ⚙️ System Architecture

* **FastAPI** → backend API for prediction
* **Gradio** → user interface
* **Hugging Face Spaces** → deployment platform



## 📊 Key Insight

This project demonstrates that in low-resource languages like Hassaniya:

* Data quality is more important than model complexity
* Classical ML can outperform deep learning in small datasets
* Annotation is the most critical step in the pipeline


## 🚀 Deployment

The model is deployed using Hugging Face Spaces with a Gradio interface for real-time sentiment prediction.
