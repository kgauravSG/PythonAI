import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize, sent_tokenize

# tokenizing sentences and words
# stemming and lemmatization
# stopwords removal
# POS tagging   
# stemming means reducing a word to its root form by removing suffixes; learerning -> learn
# lemmatization means reducing a word to its base form by considering the context and morphological analysis of the word
# stopwords are common words that do not add much meaning to a sentence and can be removed
# POS tagging means assigning a part of speech to each word in a sentence
# N-Grams are contiguous sequences of n items from a given sample of text or speech
# Example text

sentences = "Hello there, how are you doing today? The weather is great, and Python is awesome. Let's learn some Natural Language Processing with NLTK."
# Tokenize sentences
tokenized_sentences = sent_tokenize(sentences)
print(tokenized_sentences)

tokenised_words = word_tokenize(sentences)
print(tokenised_words)


