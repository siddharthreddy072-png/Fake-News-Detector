import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load the news dataset
data = pd.read_csv("data/news.csv")

# Remove missing values
data = data.dropna()

# Features and target
X = data["text"]
y = data["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a machine-learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "model/fake_news_model.pkl")

print("Fake News Detection model trained successfully!")
print("Model saved in model/fake_news_model.pkl")
