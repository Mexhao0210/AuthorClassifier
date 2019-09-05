import pandas as pd
import nltk
import re
import numpy as np

nltk.download("stopwords")
from nltk.corpus import stopwords
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, Conv2D, MaxPooling2D, Reshape, concatenate, Flatten, Dropout, Dense
from keras import regularizers
from keras.optimizers import RMSprop
from keras.utils import to_categorical

train_df = pd.read_csv('train.csv')
Y = train_df['author']
X = train_df.drop(columns=['author'])

del train_df


def preprocess_text(sent):
    sent = str(sent)
    sent = sent.lower()

    sent = re.sub(r"what's", "what is ", sent)
    sent = re.sub(r"\'s", " ", sent)
    sent = re.sub(r"\'ve", " have ", sent)
    sent = re.sub(r"can't", "cannot ", sent)
    sent = re.sub(r"n't", " not ", sent)
    sent = re.sub(r"i'm", "i am ", sent)
    sent = re.sub(r"\'re", " are ", sent)
    sent = re.sub(r"\'d", " would ", sent)
    sent = re.sub(r"\'ll", " will ", sent)
    sent = re.sub(r",", " ", sent)
    sent = re.sub(r"\.", " ", sent)
    sent = re.sub(r"!", " ! ", sent)
    sent = re.sub(r"\/", " ", sent)
    sent = re.sub(r"\^", " ^ ", sent)
    sent = re.sub(r"\+", " + ", sent)
    sent = re.sub(r"\-", " - ", sent)
    sent = re.sub(r"\=", " = ", sent)
    sent = re.sub(r"'", " ", sent)

    sent = sent.split()
    return sent


vocab = dict()
inverse_vocab = ['<UNK>']
# embedding_file = "GoogleNews-vectors-negative300.bin.gz";
# glove_embedding = "glove.6B.50d.txt"
# embedding_file = "glove_50d.txt"

# glove2word2vec(glove_embedding, embedding_file)

stop_words = set(stopwords.words('english'))

# word2vec_embeddings = KeyedVectors.load_word2vec_format(embedding_file)

from gensim import models

word2vec_embeddings = models.KeyedVectors.load_word2vec_format(
    '../GoogleNews-vectors-negative300.bin', binary=True)

for index, row in X.iterrows():
    token_vector = []
    for word in preprocess_text(row['text']):
        if word in stop_words and word not in word2vec_embeddings.vocab:
            continue
        if word not in vocab:
            vocab[word] = len(inverse_vocab)
            token_vector.append(len(inverse_vocab))
            inverse_vocab.append(word)
        else:
            token_vector.append(vocab[word])
    X.set_value(index, 'text', token_vector)

embed_dim = 50
embeddings = 1 * np.random.randn(len(vocab) + 1, embed_dim)  # This will be the embedding matrix
embeddings[0] = 0

for word, index in vocab.items():
    if word in word2vec_embeddings.vocab:
        embeddings[index] = word2vec_embeddings.word_vec(word)


max_sent_length = X.text.map(lambda x: len(x)).max()
X_values = X.text
X_values = pad_sequences(X_values, maxlen=max_sent_length)

le = LabelEncoder()
Y = le.fit_transform(Y)
Y = to_categorical(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X_values, Y, test_size=0.2, random_state=2)
model = Sequential()
embedding_layer = Embedding(len(embeddings), embed_dim, weights=[embeddings], input_length=max_sent_length, trainable=True)
model.add(embedding_layer)
model.add(Reshape((max_sent_length, embed_dim, 1)))

model.add(Conv2D(filters=32, kernel_size=(10, embed_dim), activation="relu", padding="same", name="Conv1"))
model.add(MaxPooling2D(pool_size=(2, 2), name="Pool1"))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(3, activation="softmax", name="Dense1"))

rmsOpt = RMSprop()
model.compile(optimizer=rmsOpt, loss="categorical_crossentropy", metrics=["accuracy"])
epochs = 1
batch_size = 1000

model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=batch_size, validation_data=[X_test, Y_test], verbose=1)

