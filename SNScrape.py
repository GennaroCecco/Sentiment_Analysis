import snscrape.modules.twitter as sntwitter
import pandas as pd

def get_tweets(query, num_tweets=100):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{query} since:2020-01-01 until:2023-01-01').get_items()):
        if i >= num_tweets:
            break
        tweets.append({
            'date': tweet.date,
            'content': tweet.content,
            'username': tweet.user.username
        })
    return tweets

# Esempio di utilizzo
query = 'Python'
num_tweets_to_extract = 10
tweets_data = get_tweets(query, num_tweets_to_extract)

# Creiamo un DataFrame per visualizzare i risultati
df = pd.DataFrame(tweets_data)
print(df)
