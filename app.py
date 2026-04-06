import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import requests

st.set_page_config(page_title="YouTube Sentiment UI", page_icon="🎬", layout="wide")

with st.sidebar:
    st.title("🤖 How it Works")
    st.markdown("""
    **1. Scraper:** Uses `youtube-comment-downloader` to fetch real comments without using the official API.
    
    **2. AI Brain:** Uses a **RoBERTa Transformer** model. It now runs on a lightning-fast decoupled FastAPI microservice!
    
    **3. Viz:** Built with `Matplotlib` & `Seaborn`.
    """)
    st.info("Built by You using Streamlit & FastAPI")

st.title("🎬 YouTube Sentiment Analyzer")
st.markdown("### Paste a video link to see what the internet *really* thinks.")

col1, col2 = st.columns([3, 1])
with col1:
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
with col2:
    analyze_btn = st.button("Analyze Sentiment 🚀", use_container_width=True)

if analyze_btn and url:
    with st.spinner('Sending request to AI Backend...'):
        try:
            
            api_url = "http://127.0.0.1:8000/analyze"
            payload = {"url": url}
            
            
            response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                json_data = response.json()
                comments_data = json_data["data"]
                
                df = pd.DataFrame(comments_data)
                st.success(f"Successfully analyzed {len(df)} comments!")
                
                
                sentiment_counts = df['label'].value_counts()
                col1, col2, col3 = st.columns(3)
                col1.metric("Positive", sentiment_counts.get('positive', 0), delta_color="normal")
                col2.metric("Neutral", sentiment_counts.get('neutral', 0), delta_color="off")
                col3.metric("Negative", sentiment_counts.get('negative', 0), delta_color="inverse")
                
                st.markdown("---")
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("Sentiment Distribution")
                    fig, ax = plt.subplots()
                    colors = {'positive': '#66c2a5', 'neutral': '#fc8d62', 'negative': '#e78ac3'}
                    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors, ax=ax)
                    st.pyplot(fig)
                
                with c2:
                    st.subheader("Word Cloud")
                    all_text = " ".join(df['text'])
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wordcloud, interpolation='bilinear')
                    ax_wc.axis("off")
                    st.pyplot(fig_wc)
                    
                with st.expander("📥 Download Data"):
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", csv, "sentiment_analysis.csv", "text/csv")
            
            
            else:
                st.error(f"Backend Error: {response.json()['detail']}")

        except Exception as e:
            st.error("Could not connect to the API. Make sure your FastAPI server is running in the other terminal!")