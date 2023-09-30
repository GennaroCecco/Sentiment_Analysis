import pickle

from sklearn.metrics import accuracy_score

from TextPreprocessor import TextPreprocessor
import numpy as np
from keras.src.saving.saving_api import load_model
from keras.src.utils import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split


class RNNGRU:
    def __init__(self):
        self.model = load_model('../RNN-GRU/#1/modello.h5')
        with open('../RNN-GRU/#1/tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.text_preprocessor = TextPreprocessor()
        #La lista è perfettamente bilanciata
        self.new_tweets = [
            "I just lost my wallet. What a terrible day! 😞",
            "But at least I found my favorite old book hidden in the bookshelf. What a pleasant surprise! 📚😊",
            "Traffic today is a nightmare. I missed my appointment. 😩🚗",
            "But I had a long chat with an old friend on the phone. It was great to catch up! ☎️😄",
            "Rainy days always make me feel gloomy. ☔😔",
            "I love how the rain makes everything feel fresh and clean. It's so peaceful. 🌧️😌",
            "Stuck in a long queue at the grocery store. This is so frustrating. 🛒😫",
            "But I just found a surprise sale, and I saved a lot of money on groceries! 💰💪",
            "I have so much work to do. I'll be up all night. 😓📚",
            "Finished my work early! Now I have the whole evening to relax. 🌅🎉",
            "My phone battery died when I needed it the most. This is just my luck. 📱🔋",
            "But I discovered a new coffee shop while waiting, and their coffee is amazing! ☕😃",
            "Got caught in the rain without an umbrella. I'm completely soaked. ☔😫",
            "But I met a kind stranger who shared their umbrella with me. There are good people out there! ☂️😇",
            "I burned dinner tonight. It's ruined. 🍳🔥",
            "But my partner surprised me with my favorite takeout. Dinner saved! 🍕🥳",
            "My computer crashed, and I lost all my work. This is a disaster. 💻😱",
            "But I had a backup, and it's a chance to improve my project. Silver lining! 💾🌟",
            "I missed my train and will be late for the meeting. Not a great start to the day. 🚆⏰",
            "But I used the extra time to grab my favorite coffee, and now I'm wide awake! ☕😄",
            "I can't believe I missed the bus again. This day is off to a bad start. 😞",
            "But I found some money on the street! Maybe my luck is turning around. 💰🍀",
            "This rain ruined my picnic plans. I'm so disappointed. 🌧️🧺",
            "At least I got to enjoy a cozy day indoors with a good book. 📖☕",
            "I had a fight with my friend today. It's been a tough one. 😢",
            "But we talked it out and made up. Friends forever! ❤️🤗",
            "I spilled coffee all over my favorite shirt. It's ruined. ☕😩",
            "But I got a compliment on my outfit at work. Silver lining! 👚💁",
            "I'm stuck in traffic again. This commute is the worst. 🚗🤬",
            "But I'm listening to my favorite music, so it's not that bad. 🎶🙂",
            "My phone battery died in the middle of an important call. Argh! 📱🔌",
            "But I found an old friend's number and reconnected. Great chat! ☎️👍",
            "I burnt dinner. Now I have to order takeout. 🍕🔥",
            "But I got my favorite pizza, so it's a win! 🍕😋",
            "I missed the last episode of my favorite show. Major bummer. 📺😩",
            "But I found out there's a new season coming soon! Can't wait! 🍿🥳",
            "I forgot my umbrella, and it started raining. I'm drenched. ☔😫",
            "But I ran into an old friend and had a heartwarming chat. ☂️❤️",
            "I lost my keys. This is a disaster. 🔑😱",
            "But I found my long-lost favorite childhood toy. Nostalgia! 🧸😊",
            "I had a flat tire on my way to an important meeting. Ugh! 🚗🔩",
            "But a kind stranger stopped to help me. There are good people out there! 🚗🤝",
            "I forgot my lunch at home. Now I'm hungry and grumpy. 😡🍔",
            "But a coworker shared their delicious sandwich with me. Lunchtime savior! 🥪🙏",
            "I got a parking ticket today. This is so frustrating. 🚗🎫",
            "But I received a surprise gift from a friend. Unexpected joy! 🎁😄",
            "I had a bad hair day. My confidence is low. 💇😞",
            "But a stranger complimented my smile. Made my day! 😄❤️",
            "I missed my morning coffee. I feel so tired. ☕😴",
            "But I got a free coffee coupon. Thank you, barista! ☕🙌",
            "I accidentally deleted an important email. Oops! 📧🙈",
            "But I found an even better solution to the problem. Crisis averted! 💡👍",
            "I got caught in the pouring rain without an umbrella. Soaked! ☔😫",
            "But I met someone new while seeking shelter. Interesting conversation! ☔🙂",
            "I overslept and was late for work. Not a good way to start the day. ⏰😩",
            "But I had a delicious breakfast, so I'm fueled up now. 🍳☕",
            "I spilled coffee on my laptop. It's a disaster. ☕💻",
            "But I learned how to fix it myself. Proud moment! 💪🤓",
            "I missed my flight. Vacation plans ruined. ✈️😢",
            "But I found a charming local spot to explore instead. Adventure time! 🌍🗺️",
            "I lost my wallet while shopping. Panic mode activated. 👛😱",
            "But a kind store employee found it and returned it to me. Faith in humanity restored! 👏🙏",
            "I locked myself out of the house. What a day. 🔑🚪",
            "But I had a lovely chat with a neighbor while waiting for help. Friendly community! 👋🏡",
            "I got a flat tire on my bike. Not again! 🚲🔩",
            "But I discovered a beautiful park I'd never been to before. Serendipity! 🌳😍",
            "I got caught in a sudden downpour. My clothes are drenched. ☔🌧️",
            "But I found a cozy cafe to dry off and enjoy a hot drink. Warm and toasty! ☕🙂",
            "I forgot my lunch at home. Hunger strikes. 🍱😫",
            "But a colleague shared their delicious homemade cookies. Office treats! 🍪🤤"
        ]
        self.ground_truth_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                                    0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                                    0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                                    0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                                    0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                                    0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                                    0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                                    ]

    def load_tweets_from_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        self.new_tweets = df['tweet_column_name'].tolist()

    def analyze_sentiments(self):
        cleaned_tweets = [self.text_preprocessor.clean_text(tweet) for tweet in self.new_tweets]
        lemmatized_tweets = [self.text_preprocessor.lemmatize_text(tweet) for tweet in cleaned_tweets]

        sequences = self.tokenizer.texts_to_sequences(lemmatized_tweets)
        padded_sequences = pad_sequences(sequences, maxlen=43)
        self.predictions = self.model.predict(padded_sequences)

        sentiment_labels = ["Negativo" if sentiment < 0.5 else "Positivo" for sentiment in self.predictions]
        results = list(zip(self.new_tweets, sentiment_labels))

        return results

    def calculate_accuracy(self):
        predicted_labels = [1 if sentiment >= 0.5 else 0 for sentiment in self.predictions]
        accuracy = accuracy_score(self.ground_truth_labels, predicted_labels) * 100
        return accuracy

if __name__ == '__main__':
    rnn = RNNGRU()
    results = rnn.analyze_sentiments()
    i = 1
    with open('../RNN-GRU/#0_TEST/risultato_test.txt', 'w', encoding='utf-8') as file:
        for tweet, sentiment in results:
            print(f"{i}) Tweet: {tweet}")
            print(f"Sentiment: {sentiment}")
            file.write(f"{i}) Tweet: {tweet}\n")
            file.write(f"Sentiment: {sentiment}\n")
            i += 1
        accuracy = rnn.calculate_accuracy()

        percent_positive = (sum(1 for sentiment in rnn.predictions if sentiment >= 0.5) / len(rnn.predictions)) * 100
        percent_negative = 100 - percent_positive

        print(f"Model Accuracy: {accuracy:.2f}%")
        file.write(f"Model Accuracy: {accuracy:.2f}%\n")
        print(f"Il modello ha classificato come tweet positivi il: {percent_positive:.2f}%")
        file.write(f"Il modello ha classificato come tweet positivi il: {percent_positive:.2f}%\n")
        print(f"Il modello ha classificato come tweet negativi il: {percent_negative:.2f}%")
        file.write(f"Il modello ha classificato come tweet negativi il: {percent_negative:.2f}%\n")
