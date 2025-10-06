import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Bidirectional
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


# Dataset upload
df = pd.read_csv("/content/Twitter_Data.csv", encoding="latin-1")
df.head()
df.isnull().sum()
df=df.dropna()
df["category"].value_counts()

# Drop rows with missing sentiment values
df = df.dropna(subset=["category"])
df.columns = ["clean_text","category"]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df["category"])
y = to_categorical(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["category"], y, test_size=0.2, random_state=42, stratify=y
)

# Ensure text column has only strings
X_train = X_train.astype(str)
X_test = X_test.astype(str)

# Tokenization
max_words = 10000  # keep top 10k words
max_len = 100      # max length of sequences

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")


# Build LSTM model
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(3, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_test_pad, y_test),
    epochs=5,
    batch_size=1028,
    verbose=1
)

# Evaluate
loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

# Map sentiment labels to words
sentiment_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
df["sentiment_label"] = df["category"].map(sentiment_map)

# Function to generate word cloud
def generate_wordcloud(text, title):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        stopwords=stopwords
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.show()

# Loop through each sentiment category and plot word cloud
categories = df["sentiment_label"].unique()

for category in categories:
    texts = df[df["sentiment_label"] == category]["clean_text"].astype(str).values
    combined_text = " ".join(texts)
    generate_wordcloud(combined_text, f"Word Cloud for '{category}' Sentiment")



# Assume these are defined globally after training
# model, tokenizer, max_len

def predict_sentiment(text):
    """
    Predict sentiment for a single text using global model, tokenizer, and max_len.

    Args:
        text (str): Input text to predict.

    Returns:
        str: "Positive", "Negative", or "Neutral".
    """

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    pred_class = model.predict(padded).argmax(axis=1)[0]

    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map.get(pred_class, "Unknown")

text_to_predict = "I love this product, it works great!"
prediction = predict_sentiment(text_to_predict)

print(f"Predicted Sentiment: {prediction}")








def Extract(sentences,sentiment0):
    '''from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.asarray(X.mean(axis=0)).ravel().tolist()
    word_cloud = dict(zip(feature_names, tfidf_scores))
    sentiment = {"positive": sum(1 for s in sentiment0 if s > 0.1),
                 "negative": sum(1 for s in sentiment0 if s < -0.1),
                 "neutral": sum(1 for s in sentiment0 if -0.1 <= s <= 0.1)}
    return word_cloud, sentiment'''
    word_cloud = {"good":10000,"bad":1000,"risky":2000}
    sentiment = {"positive":100,
            "negative":20,
            "neutral":30}
    return word_cloud, sentiment
