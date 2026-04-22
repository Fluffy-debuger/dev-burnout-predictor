from github import Auth, Github
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()
auth = Auth.Token(os.getenv("PAT"))
g = Github(auth=auth)

repos = [
    "numpy/numpy",
    "tensorflow/tensorflow",
    "microsoft/vscode",
    "scikit-learn/scikit-learn",
    "python/cpython",
    "angular/angular",
    "pytorch/pytorch", 
    "django/django"     
]
total_commits = 200  

all_data = []
for repo_name in repos:
    repo = g.get_repo(repo_name)
    commits = repo.get_commits()[:total_commits]
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
            "author": author,
            "repo": repo_name
        })
    all_data.extend(data)

df = pd.DataFrame(all_data)
df['date'] = pd.to_datetime(df['date'])
df['week'] = df['date'].dt.isocalendar().week
df['year'] = df['date'].dt.year
df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str).str.zfill(2)
weekly_commits = df.groupby(['author', 'year_week']).size().reset_index(name='commits_per_week')
df = df.merge(weekly_commits, on=['author', 'year_week'], how='left').fillna(1)

df['sentiment_score'] = df['commit_message'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

def create_burnout_label(row):
    if (row['commits_per_week'] < 5 and row['sentiment_score'] < -0.1 and row['lines_added'] > 200):
        return 'High Burnout'
    elif row['commits_per_week'] < 3:
        return 'High Burnout'
    else:
        return 'Low Burnout'

df['burnout_label'] = df.apply(create_burnout_label, axis=1)

features = ['commits_per_week', 'sentiment_score', 'lines_added']
X = df[features].fillna(0)
y = df['burnout_label']

if len(y.unique()) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy: {accuracy:.2%} on {len(X_test)} test samples")
    print("label distribution:", y.value_counts().to_dict())
else:
    print("very few variety of label, add more diverse repos.")


joblib.dump(model, ".//Model//burnout_model_multi.pkl")
with open(".//Model//model_accuracy.txt", "w") as f:
    f.write(str(accuracy))
print("training complete and model saved")

df.to_csv(".//Datasets//github_commits_enhanced.csv", index=False)
print("data and analysis saved")