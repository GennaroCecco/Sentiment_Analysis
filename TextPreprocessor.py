import re
import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, sentence):
        # Rimuovo gli URL, i tag HTML e le emoji usando una regex
        sentence = re.sub(r'http\S+|www\S+|https\S+|<.*?>', '', sentence, flags=re.MULTILINE)
        sentence = re.sub(r'[{}]'.format(re.escape(string.punctuation.replace('!', '').replace('?', ''))), '', sentence)
        sentence = re.sub("["
                          "\U0001F600-\U0001F64F"
                          "\U0001F300-\U0001F5FF"
                          "\U0001F680-\U0001F6FF"
                          "\U0001F700-\U0001F77F"
                          "\U0001F780-\U0001F7FF"
                          "\U0001F800-\U0001F8FF"
                          "\U0001F900-\U0001F9FF"
                          "\U0001FA00-\U0001FA6F"
                          "\U0001FA70-\U0001FAFF"
                          "\U00002702-\U000027B0"
                          "]+", '', sentence)

        # Rimuovo gli username
        sentence = re.sub(r'@\w+', '', sentence)

        # Porto tutto il testo in minuscolo
        sentence = sentence.lower()

        # Rimozione dei caratteri unici o ripetuti
        words = word_tokenize(sentence)
        cleaned_words = [re.sub(r'(.)\1+|\b(\w)\1+\b', r'\1', word) for word in words]

        # Rimozione di caratteri speciali aggiuntivi
        cleaned_words = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in cleaned_words]

        # Gestione delle negazioni
        cleaned_words = [re.sub(r'\b(?:not)\b', 'not_', word) for word in cleaned_words]

        # Rimozione delle stopwords
        cleaned_words = [word for word in cleaned_words if word.lower() not in self.stop_words]

        # Converto nuovamente in una stringa il testo ripulito
        cleaned_sentence = " ".join(cleaned_words)
        return cleaned_sentence

    def lemmatize_text(self, text):
        return " ".join([self.lemmatizer.lemmatize(w) for w in self.w_tokenizer.tokenize(text)])

