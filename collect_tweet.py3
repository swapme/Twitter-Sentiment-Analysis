import sys
import jsonpickle
import os
import tweepy
import csv

consumer_key = ''
consumer_secret = ''

access_token =   ''
access_secret =  ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

searchQuery = '#bjp -filter:retweets'  # this is what we're searching for
maxTweets = 200 # Some arbitrary large number
tweetsPerQry = 100 # this is the max the API permits
fName = 'tweeter_bjp.csv' # We'll store the tweets in a csv file.


# If results from a specific ID onwards are reqd, set since_id to that ID.
# else default to no lower limit, go as far back as API allows
sinceId = None

# If results only below a specific ID are, set max_id to that ID.
# else default to no upper limit, start from the most recent tweet matching the search query.
max_id = -1L

tweetCount = 0
print("Downloading max {0} tweets".format(maxTweets))
with open(fName, 'w') as f:
    writer = csv.writer(f)
    while tweetCount < maxTweets:
        try:
            if (max_id <= 0):
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,  lang='en', show_user=True, tweet_mode="extended")
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            since_id=sinceId,  lang='en', show_user=True, tweet_mode="extended")
            else:
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1), lang='en', show_user=True, tweet_mode="extended")
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1),
                                            since_id=sinceId,  lang='en', show_user=True, tweet_mode="extended")
            if not new_tweets:
                print("No more tweets found")
                break
            for tweet in new_tweets:
                print(tweet)
                writer.writerow([tweet.author._json['name'].encode('utf-8'), tweet.full_text.encode('utf-8').replace('\n',' ')])

            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
            break

print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))
