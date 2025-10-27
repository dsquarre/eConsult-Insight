# eConsult-Insight
AI-assisted analysis of large-scale public feedback

How to run:
'''bash
python3 main.py
'''

Pipeline:
- React Input Page to load csv files containing all the comments.
- Internally calls transformer.py to preprocess, summarize and do sentiment analysis of each comment.
- Internally calls tf_idf.py to again do word count and sentiment analysis and double checking to prevent hallucination and returns a JSON file.
- Reads and displays JSON file as a dashboard to easily make sense of data.
