import streamlit as st
from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🎬 YouTube Sentiment Analyzer")
st.write("Enter a YouTube link to see what the audience is really thinking.")


@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

model = load_model()


url = st.text_input("Paste YouTube URL here:", "https://www.youtube.com/watch?v=Get7rqXYrwQ")


if url:
    if st.button("Analyze Comments"):
        with st.spinner('Scraping comments and analyzing... (this takes a moment)'):
            
            
            downloader = YoutubeCommentDownloader()
            comments = []
            try:
               
                generator = downloader.get_comments_from_url(url, sort_by=0) 
                for count, comment in enumerate(generator):
                    if count == 50: break
                    comments.append(comment['text'])
            except Exception as e:
                st.error(f"Error fetching comments: {e}")
            
            if comments:
                
                results = []
                for text in comments:
                    
                    truncated_text = text[:512]
                    results.append(model(truncated_text)[0])
                
                # Create DataFrame
                df = pd.DataFrame(results)
                df['text'] = comments
                
                
                st.success("Analysis Complete!")
                
                
                col1, col2, col3 = st.columns(3)
                sentiment_counts = df['label'].value_counts()
                
                
                pos = sentiment_counts.get('positive', 0)
                neu = sentiment_counts.get('neutral', 0)
                neg = sentiment_counts.get('negative', 0)
                
                col1.metric("Positive", pos)
                col2.metric("Neutral", neu)
                col3.metric("Negative", neg)
                
               
                st.subheader("Sentiment Distribution")
                fig, ax = plt.subplots()
                colors = {'positive': '#66c2a5', 'neutral': '#fc8d62', 'negative': '#e78ac3'}
                sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors, ax=ax)
                st.pyplot(fig)
                
                
                with st.expander("See Raw Comments"):
                    st.dataframe(df)
            else:
                st.warning("No comments found or URL is invalid.")
