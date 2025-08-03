import tweepy
from tweepy import *
import pandas as pd
import csv
import re
import string
import preprocessor as p

consumer_key ="XA64LsdvcUMDtWNaRk0B4IPXp"
consumer_secret = "UoTAspAk3bIt48FZF0GZ8KlC8W0dgF64uiGuaL4yZDUKzVoesj"
access_key ="1473907385152548864-ZDntYYf6tfA0KbLp7DdUuGJwXfyBkQ"
access_secret ="6Ai9O371HGeo4E2u6R26Uu6fukYOMeKnPb1xhE3A9zjpq"

# Setup access to API
def connect_to_twitter_OAuth():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)

    api = tweepy.API(auth)
    return api


# Create API object
api = connect_to_twitter_OAuth()

# tweets from my stream
public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)


trump_tweets = api.user_timeline('realdonaldtrump')
for tweet in trump_tweets:
    print(tweet.text)