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