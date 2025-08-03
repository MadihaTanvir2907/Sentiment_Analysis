import re
import tweepy
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

import numpy as np
import pandas as pd
import pickle as pk
from bs4 import BeautifulSoup
import unicodedata
import re

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer

from nltk.corpus import wordnet
from nltk import pos_tag
from nltk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.probability import FreqDist
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt

# # *********************************************
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from scipy.special  import softmax 


# *********************************************
# import seaborn as sns
# plt.style.use('ggplot')
# from wordcloud import WordCloud


# consumer_key ="7NHJSvlLvVJwMeWhSRaJQ05aA"
# consumer_secret = "dQNcWfSgNG1aFJ3zYsP0iKuOr9cL4Bw1IuUyXJZP2RXGzeFgxX"
# access_key ="170024948-vq0OlQTfKOds1BPbQxNWQ99si8MJtvx933teg2BB"
# access_secret ="OjbXv9NuXVX8h6Z8lbuyTVslQm8hEpCJ8D0q0O9nGWNZ8"

# # Twitter authentication
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_key, access_secret)

# # Creating an API object
# api = tweepy.API(auth)


# list = []
# date_tweets = tweepy.Cursor(api.search_tweets, q='"Data Science"' , lang = 'en', tweet_mode='extended').items(1000)

# for tweet in date_tweets:
#     text = tweet._json["full_text"]
#     print(text)
#     # preprocessing
#     text = text.lower()
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)

#     refined_tweet = {'text': text,
#                      'favorite_count': tweet.favorite_count
#                      ,
#                      'retweet_count': tweet.retweet_count,
#                      'created_at': tweet.created_at
#                      }

#     list.append(refined_tweet)



    

# import pandas as pd
# df = pd.DataFrame(list)
# df.to_csv('Data_Science.csv')


# # get your stopwords from nltk
# stop_words = set(stopwords.words('english'))

# # Read Data
# df = pd.read_csv('Data_Science.csv')
# df.head()

# ps = PorterStemmer()
# TfidfVector = TfidfVectorizer()
# new_list = []
# vextor = []
# for text in df['text']:
#     text = text.lower()

#     # Remove HTML Tag
#     print (BeautifulSoup(text,'html.parser').get_text())
#     text =  BeautifulSoup(text,'html.parser').get_text()

#     # Remove URLS
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)
#     print (text)

#     # Removing Accented Characters
#     text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
#     print(text)


#     # Removing Punctuation
#     text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
#     print(text)

#     # Removing irrelevant Characters (Numbers and Punctuation)

#     text = re.sub(r'[^a-zA-Z]', ' ', text)
#     print(text)

#     # Removing extra Whitespaces
#     text = re.sub(r'^\s*|\s\s*', ' ', text).strip()
#     print(text)
#     # tokenize
#     tokenized_sent = nltk.word_tokenize(text)

#     # remove stops
#     tokenized_sent_no_stops = [
#         tok for tok in tokenized_sent 
#         if tok not in stop_words
#     ]

#     # untokenize 
#     untokenized_sent = TreebankWordDetokenizer().detokenize(
#         tokenized_sent_no_stops
#     )
#      # to extract words from string 
#     res = untokenized_sent.split() 
        
#     # printing result 
#     stemmed_word = ''
#     for i in res:
#         stemmed_word += ' '+ ps.stem(i)
#     new_list.append(stemmed_word)

# df = pd.DataFrame(new_list)
# df.to_csv('refined_tweets_DS.csv')

# Read data from refined data for TfidfVectorizer

df = pd.read_csv('F:\Ecosystem\Assignment3\Python_Code\Russia_refined_retweets.csv')
df.head()
# *******************************************************************************
# Function for sentiment analysis
from tweepy import OAuthHandler
from textblob import TextBlob
def get_tweet_sentiment(tweet):



    # create TextBlob object of passed tweet text
    analysis = TextBlob(tweet)

    # set sentiment
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# *********************************************************************************
# Load Model and Tokenizer
# roberta = "cardiffnlp/twitter-roberta-base-sentiment"
# model = AutoModelForSequenceClassification.from_pretrained(roberta)
# tokenizer = AutoTokenizer.form_pretrained(roberta)
# labels = ["Negative","Neutral","Positive"]
# ***********************************************************************************
# Sentiment Analysis
sentiment_data = []
new_text = []
for text in df['0']:
    # text = [text]
    sentiment  = get_tweet_sentiment(text)
    sentiment_data.append(sentiment)
    new_text.append(text)
df['Sentiments'] = sentiment_data
df.to_csv('Sentiments_Rusisa.csv')
plot = df['Sentiments'].value_counts().plot(kind= 'bar', title = 'Sentiment Analysis on Russia Europe Relation')
# **************************************************************************************
# documents = df["0"].values.astype("U")
# Vectorizer = TfidfVectorizer(stop_words = "english")
# vector = Vectorizer.fit_transform(documents)
# # X, y = load_boston(return_X_y=True)
# K = range(1,12)
# wss = []
# for k in K:
#     model = KMeans(n_clusters = k, init = "k-means++",max_iter = 100, n_init = 1)
#     model.fit(vector)
#     wss_iter = KMeans.inertia_
#     wss.append(wss_iter)
# mycenters = pd.DataFrame({'Clusters' : K, 'WSS' : wss})

# df["cluster"]= model.labels_
# df.head()

# clusters = df.groupby('cluster')
# for cluster in clusters.groups:
#     f = open('cluster' + str(cluster) + '.csv','w')
#     data = clusters.get_group(cluster)['0']
#     f.write(data.to_csv(index_label = 'id'))
#     f.close()

# # *********************************************************************************
# # Vectorizer = TfidfVectorizer(stop_words = "english")
# # vextor = []
# # all_keyword = []
# # new_text = ''
# # for text in df['0']:
# #     # new_text +=text
# #     text = [text]
# # # new_text = [new_text]
# # # print(new_text)
# #     vector = Vectorizer.fit_transform(text)
# #     features_name = Vectorizer.get_feature_names_out()
# #     dense = vector.todense()
# #     dense_list = dense.tolist()
        
# #     for description in dense_list:
# #         x= 0
# #         key_words = []
# #         for word in description:
# #             if word > 0:
# #                 key_words.append(features_name[x])
# #             x = x+1
# #         all_keyword.append(key_words)

# # df = pd.DataFrame(all_keyword)
# # df.to_csv('all_keyword _DS.csv')

# # Model KMeans
# # true_k = 10
# # print(f"First Vector {vector}")

# # model = KMeans(n_clusters = true_k, init = "k-means++",max_iter = 100)
# # KMeans.fit(vector)
# # print(f"second Vector {vector}")
# # order_centroids = model.cluster_centers.argsort()[:,::-1]
# # terms = Vectorizer.get_feature_names()
# # df["Cluster"] = model.labels
# # df.head()