import os
import pickle
from TextPreprocessor import TextPreprocessor
from keras.src.saving.saving_api import load_model
from keras.src.utils import pad_sequences
import pandas as pd


class RNNGRU:
    def __init__(self, fileName):
        self.model = load_model('RNN-GRU/#2/modello.h5')
        with open('RNN-GRU/#2/tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.text_preprocessor = TextPreprocessor()
        self.new_tweets = self.load_tweets_from_text(fileName)

    def load_tweets_from_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        self.new_tweets = df['tweet_column_name'].tolist()

    def load_tweets_from_text(self, text_file):
        try:
            relative_path = os.path.join('uploadTxt', text_file)
            with open(relative_path, 'r', encoding='utf-8') as file:
                tweets = file.readlines()

            tweets = [tweet.strip() for tweet in tweets]
            return tweets
        except Exception as e:
            return f'Errore durante il caricamento dei tweet: {str(e)}'

    def analyze_sentiments(self):
        cleaned_tweets = [self.text_preprocessor.clean_text(tweet) for tweet in self.new_tweets]
        lemmatized_tweets = [self.text_preprocessor.lemmatize_text(tweet) for tweet in cleaned_tweets]

        sequences = self.tokenizer.texts_to_sequences(lemmatized_tweets)
        padded_sequences = pad_sequences(sequences, maxlen=43)
        self.predictions = self.model.predict(padded_sequences)

        sentiment_labels = ["Negativo" if sentiment < 0.1 else "Positivo" for sentiment in self.predictions]
        results = list(zip(self.new_tweets, sentiment_labels))

        return results

    def analyze_sentiments_Percentage(self):
        cleaned_tweets = [self.text_preprocessor.clean_text(tweet) for tweet in self.new_tweets]
        lemmatized_tweets = [self.text_preprocessor.lemmatize_text(tweet) for tweet in cleaned_tweets]

        sequences = self.tokenizer.texts_to_sequences(lemmatized_tweets)
        padded_sequences = pad_sequences(sequences, maxlen=44)
        self.predictions = self.model.predict(padded_sequences)

        sentiment_labels = ["Negativo" if sentiment < 0.1 else "Positivo" for sentiment in self.predictions]
        results = list(zip(self.new_tweets, sentiment_labels))

        # Calcola la percentuale di tweet positivi e negativi
        num_positives = sentiment_labels.count("Positivo")
        num_negatives = sentiment_labels.count("Negativo")
        total_tweets = len(sentiment_labels)

        percentage_positives = (num_positives / total_tweets) * 100
        percentage_negatives = (num_negatives / total_tweets) * 100

        return percentage_positives, percentage_negatives



