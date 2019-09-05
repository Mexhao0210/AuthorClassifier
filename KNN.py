import pandas as pd
import nltk
import re
import numpy as np

nltk.download("stopwords")
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

embedding_file = "GoogleNews-vectors-negative300.bin.gz"
word2vec_embeddings = KeyedVectors.load_word2vec_format(embedding_file, binary=True)

# Function to perform basic sentence cleaning
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
    sent = re.sub(r"'", " ", sent)
    
    sent = sent.split()
    return sent

# Function to convert word ids to word embeddings
def convert_to_embeddings(df):
    global word2vec_embeddings
    embeddings = 1 * np.random.randn(len(vocab) + 1, embed_dim)
    embeddings[0] = 0

    for word, index in vocab.items():
        if word in word2vec_embeddings.vocab:
            embeddings[index] = word2vec_embeddings.word_vec(word)

    embed_arr = []
    for sent in df:
        temp = [0 for x in range(300)]
        for word_id in sent:
            temp = np.add(temp, embeddings[word_id])
        embed_arr.append(temp)
    return np.asarray(embed_arr)

# Function that converts each sentence to a sequence of unique word ids
def convert_to_ids(X, vocab, inverse_vocab):
    global word2vec_embeddings
    stop_words = set(stopwords.words('english'))
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


if __name__=="__main__":
    embed_dim = 300
    vocab = dict()
    inverse_vocab = ['<UNK>']

    # Read data
    train_df = pd.read_csv('train.csv')

    # Seperate data into features and labels
    Y = train_df['author']
    Y = pd.DataFrame(Y, columns=['author'])
    X = train_df.drop(columns=['author'])

    # Convert sentences to word ids
    convert_to_ids(X, vocab, inverse_vocab)

    # Convert to embeddings
    X_values = convert_to_embeddings(X_values)

    # Convert author labels to categorical values
    le = LabelEncoder()
    Y['author'] = le.fit_transform(Y['author'])
    Y = Y.as_matrix()
    Y = Y.flatten()

    # Split into test and train sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_values, Y, test_size=0.2, random_state=2)

    # KNN classifier
    neigh_classifier = KNeighborsClassifier(n_neighbors=7)
    neigh_classifier.fit(X_train, Y_train)

    # Final accuracy on the validation set
    print(neigh_classifier.score(X_test, Y_test))