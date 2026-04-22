import streamlit as st
from github import Auth, Github
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib
import plotly.express as px
from dotenv import load_dotenv
import os

nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()
model = joblib.load(".//Model//burnout_model_multi.pkl")
load_dotenv()

auth = Auth.Token(os.getenv("PAT"))
g = Github(auth=auth)

st.title("Developer Burnout & Code Quality Predictor")
st.write("Enter a public GitHub repo (e.g., numpy/numpy) to analyze commits for burnout risk.")

repo_name = st.text_input("GitHub Repo Name", "numpy/numpy")
num_commits = st.slider("Number of Commits to Analyze", 50, 500, 100)

if st.button("Analyze Repo"):
    with st.spinner("Fetching and analyzing commits..."):
        repo = g.get_repo(repo_name)
        commits = repo.get_commits()[:num_commits]
        
        data = []
        for commit in commits:
            message = commit.commit.message or "No message"
            date = commit.commit.author.date
            lines_added = commit.stats.additions if commit.stats else 0
            author = commit.commit.author.name
            data.append({
                "commit_message": message,
                "date": date,
                "lines_added": lines_added,
                "author": author
            })
        
        df = pd.DataFrame(data)
        if df.empty:
            st.error("No commits found. Try another repo.")
        else:
            df['date'] = pd.to_datetime(df['date'])
            df['week'] = df['date'].dt.isocalendar().week
            df['year'] = df['date'].dt.year
            df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str).str.zfill(2)
            weekly_commits = df.groupby(['author', 'year_week']).size().reset_index(name='commits_per_week')
            df = df.merge(weekly_commits, on=['author', 'year_week'], how='left')
            df['sentiment_score'] = df['commit_message'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
            
            features = ['commits_per_week', 'sentiment_score', 'lines_added']
            X = df[features].fillna(0)
            
            df['predicted_burnout'] = model.predict(X)
            
            author_summary = df.groupby('author').agg({
                'predicted_burnout': lambda x: x.value_counts().index[0],
                'commits_per_week': 'mean',
                'sentiment_score': 'mean',
                'lines_added': 'mean'
            }).reset_index()
            
            st.success("Analysis Complete!")
            st.write("### Author Burnout Predictions")
            st.dataframe(author_summary)
            
            fig1 = px.bar(author_summary, x='author', y='commits_per_week', color='predicted_burnout', title="Commits per Week by Burnout Risk")
            st.plotly_chart(fig1)
            fig2 = px.scatter(df, x='sentiment_score', y='lines_added', color='predicted_burnout', hover_data=['commit_message'], title="Sentiment vs Lines Added")
            st.plotly_chart(fig2)

            fig3 = px.line(df, x='year_week', y='sentiment_score', color='author', title="Sentiment Score Over Time")
            st.plotly_chart(fig3)

            df['quality'] = df['lines_added'].apply(lambda x: 'Low Quality' if x > 500 or x < 10 else 'High Quality')
            #st.dataframe(df[['author', 'commit_message', 'predicted_burnout', 'quality', 'sentiment_score']].head(10))

st.warning("This is an experimental model and not a real psychological measure")