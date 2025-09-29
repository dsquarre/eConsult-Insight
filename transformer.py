def Analysis(file):
    import pandas as pd
    import numpy as np
    #contents = file.file.read().decode("utf-8")
    sentences = ["good","bad","average","excellent","poor","satisfactory","unsatisfactory","happy","sad","joyful","angry"]
    sentiment_scores = [0.8,-0.6,0.1,0.9,-0.7,0.2,-0.8,0.7,-0.5,0.9,-0.6]
    arr = [3,6,7,8]
    best_sentence_index = {i:sentences[i] for i in arr}
    return sentences,sentiment_scores,best_sentence_index