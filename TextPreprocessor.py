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
        # Rimuovo gli URL usando una regex
        sentence = re.sub(r'http\S+|www\S+|https\S+', '', sentence, flags=re.MULTILINE)

        # Rimuovo i tag HTML
        sentence = re.sub(r'<.*?>', '', sentence)

        #Rimuovo le emoji
        emoji_pattern = re.compile("["
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
                                   "]+", flags=re.UNICODE)
        sentence = emoji_pattern.sub(r'', sentence)

        # Rimuovo gli username
        sentence = re.sub(r'@\w+', '', sentence)

        sentence = re.sub(r'[{}]'.format(re.escape(string.punctuation.replace('!', '').replace('?', ''))), '', sentence)

        # Porto tutto il testo in minuscolo
        sentence = sentence.lower()

        # Tokenizzazione delle parole
        words = word_tokenize(sentence)

        # Rimozione dei caratteri unici o ripetuti
        cleaned_words = []
        for word in words:
            cleaned_word = re.sub(r'(.)\1+', r'\1', word)  # Rimuovi caratteri ripetuti
            cleaned_word = re.sub(r'\b(\w)\1+\b', r'\1', cleaned_word)  # Rimuovi caratteri unici multipli
            cleaned_words.append(cleaned_word)

        # Rimozione delle stopwords
        cleaned_words = [word for word in cleaned_words if word.lower() not in self.stop_words]

        # Converto nuovamente in una stringa il testo ripulito
        cleaned_sentence = " ".join(cleaned_words)
        return cleaned_sentence

    def lemmatize_text(self, text):
        st = ""
        for w in self.w_tokenizer.tokenize(text):
            st = st + self.lemmatizer.lemmatize(w) + " "
        return st

