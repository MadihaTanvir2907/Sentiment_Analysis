#sentimental
from textblob import TextBlob
import pandas as pd
def get_sentiment(text):
    blob =TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    if sentiment_polarity > 0:
        sentiment_label = "Positive"
    elif sentiment_polarity < 0:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    result = {'polarity': sentiment_polarity, 'subjectivity': sentiment_subjectivity, 'sentiment': sentiment_label}
    return result
df = pd.read_csv('Russia_refined_retweets.csv')
df.head()
#polarity
ex1=df['0'].iloc[1]
get_sentiment(ex1)

#polarity subjectivity and sentiment of all the tweet

df['sentiment_results'] = df['0'].apply(get_sentiment)
df['sentiment_results']

#count of sentimental valuues
df['sentiment'].value_counts()

#plotting bar cart

df['sentiment'].value_counts().plot(kind='bar',title="Sentiment Analysis Bar chart")