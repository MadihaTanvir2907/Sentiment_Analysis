import tweepy

consumer_key ="XA64LsdvcUMDtWNaRk0B4IPXp"
consumer_secret = "UoTAspAk3bIt48FZF0GZ8KlC8W0dgF64uiGuaL4yZDUKzVoesj"
access_key ="1473907385152548864-ZDntYYf6tfA0KbLp7DdUuGJwXfyBkQ"
access_secret ="6Ai9O371HGeo4E2u6R26Uu6fukYOMeKnPb1xhE3A9zjpq"

#creating the authentication object

auth=tweepy.OAuthHandler(consumer_key, consumer_secret)


#setting access to tokens and secrets
auth.set_access_token(access_key, access_secret)

#create the API object while passing in authentication information

api=tweepy.API(auth, wait_on_rate_limit=True)


#the twitter content I want to extract from

content='coronavirus'

#number of pulls (tweetcounts)
tweetCount=300

#calling the content_timeline function with the generated parameters
results=api.user_timeline(id=content, count=tweetCount)

#for each tweet pull the results
for tweet in results:
    #display tweets
    print(tweet.text)