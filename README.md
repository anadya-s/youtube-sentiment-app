# YouTube Sentiment AI Microservice

A decoupled machine learning application that analyzes the sentiment of YouTube comment sections in real-time. 

### Architecture
This project uses a microservice architecture to separate the heavy ML inference from the lightweight UI:
* **Backend (Brain):** A FastAPI server hosting a Hugging Face RoBERTa Transformer (`cardiffnlp/twitter-roberta-base-sentiment-latest`).
* **Frontend (Face):** A Streamlit dashboard that handles user input and visualizes data via Matplotlib and Seaborn.
* **Scraper:** `youtube-comment-downloader` to bypass official API rate limits.

### How to Run Locally

You must start both the backend and frontend servers simultaneously.

**1. Start the API (Backend)**
```bash
uvicorn api:app --reload
```

**2. Start the UI (Frontend)**
Open a new terminal and run:
```bash
streamlit run app.py
```
