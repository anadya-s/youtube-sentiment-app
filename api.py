from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from youtube_comment_downloader import YoutubeCommentDownloader


app = FastAPI(title="YouTube Sentiment API")


print("Loading RoBERTa Model...")
model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")


class VideoRequest(BaseModel):
    url: str


@app.post("/analyze")
def analyze_video(request: VideoRequest):
    try:
        
        downloader = YoutubeCommentDownloader()
        comments = []
        generator = downloader.get_comments_from_url(request.url, sort_by=0)
        
        for count, comment in enumerate(generator):
            if count == 100: break
            comments.append(comment['text'])
            
        if not comments:
            raise HTTPException(status_code=404, detail="No comments found.")

        
        results = []
        for text in comments:
            truncated_text = text[:512] 
            prediction = model(truncated_text)[0]
            
            results.append({
                "text": text,
                "label": prediction['label'],
                "score": prediction['score']
            })
            
        
        return {"status": "success", "data": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))