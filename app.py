# Importing necessary libraries
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the dataset
dataframe = pd.read_excel('/content/product_dataset/updated_product_dataset.xlsx')

# Display the first few rows of the dataframe
print(dataframe.head())

# Preprocess the text data
texts = dataframe['Text'].values

# Tokenization
max_words = 10000  # Max number of words in the vocabulary
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding
maxlen = 100  # Max length of each sequence
data = pad_sequences(sequences, maxlen=maxlen)

# Print the padded sequences for the first 3 samples
print(data[:3])
