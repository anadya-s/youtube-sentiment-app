import streamlit as st
from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


st.set_page_config(page_title="YouTube Sentiment AI", page_icon="🎬", layout="wide")


with st.sidebar:
    st.title("🤖 How it Works")
    st.markdown("""
    **1. Scraper:** Uses `youtube-comment-downloader` to fetch real comments without using the official API (no limits!).
    
    **2. AI Brain:** Uses a **RoBERTa Transformer** model (`cardiffnlp/twitter-roberta-base-sentiment`) fine-tuned on 58M tweets. It understands slang, emojis, and sarcasm better than standard BERT.
    
    **3. Viz:** Built with `Matplotlib` & `Seaborn`.
    """)
    st.info("Built by [Your Name] using Streamlit")


st.title("🎬 YouTube Sentiment Analyzer")
st.markdown("### Paste a video link to see what the internet *really* thinks.")


@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

model = load_model()


col1, col2 = st.columns([3, 1])
with col1:
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
with col2:
    analyze_btn = st.button("Analyze Sentiment 🚀", use_container_width=True)


if analyze_btn and url:
    with st.spinner('Accessing YouTube Mainframe... (Scraping & Inference)'):
        try:
           
            downloader = YoutubeCommentDownloader()
            comments = []
            generator = downloader.get_comments_from_url(url, sort_by=0) # Top comments
            for count, comment in enumerate(generator):
                if count == 100: break # Increased to 100 for better data
                comments.append(comment['text'])
            
            if not comments:
                st.error("No comments found! Check the URL.")
                st.stop()

            
            results = []
            for text in comments:
                truncated_text = text[:512]
                results.append(model(truncated_text)[0])
            
           
            df = pd.DataFrame(results)
            df['text'] = comments
            
            
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
                st.subheader("Word Cloud (What are they saying?)")
                
                all_text = " ".join(df['text'])
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                fig_wc, ax_wc = plt.subplots()
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis("off")
                st.pyplot(fig_wc)
                
            
            with st.expander("📥 Download Data"):
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "sentiment_analysis.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {e}")
