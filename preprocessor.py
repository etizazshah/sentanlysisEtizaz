import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess_text(text):
    # Instantiate the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Handle contractions and expand abbreviations (you can add more)
    text = text.replace("can't", "can not")
    text = text.replace("don't", "do not")

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # Remove special characters and punctuation
    text = ''.join(char for char in text if char.isalnum() or char.isspace())

    # Lowercase the text
    text = text.lower()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = remove_stopwords(text)

    # Tokenization and lemmatization
    tokens = word_tokenize(text)
    text = ' '.join(lemmatizer.lemmatize(token) for token in tokens)

    return text
