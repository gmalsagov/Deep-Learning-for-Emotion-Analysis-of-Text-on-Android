import csv
import random
import string

import os
import re
import matplotlib.pyplot as plt
import logging
import itertools
import numpy as np
import pandas as pd
from collections import Counter
import nltk
nltk.data.path.append('/Users/German/tensorflow/venv/lib/nltk_data')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation


"""
    File name: data_helper.py
    Author: German Malsagov
    Last modified: 04/09/2018
    Python Version: 2.7
"""


logging.getLogger().setLevel(logging.INFO)


def clean_text(text, remove_stopwords=False, stem_words=False):
    """This function does pre-processing of the input sentences"""


    # Remove punctuation
    text = text.translate(None, punctuation)

    # Convert sentences to lower case and split into individual words
    text = text.lower().split()

    # Remove stop words (Optional)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    # text = re.sub(r",", " ", text)
    # text = re.sub(r"\.", " ", text)
    # text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = strip_links(text)
    text = strip_all_entities(text)

    # Stemming (Optional)
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return text


# Removing URL's from tweets
def strip_links(text):
    """This function removes url links from input sentences"""

    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')
    return text

# Removing Tags and Hashtags
def strip_all_entities(text):
    """This function removes tags from input sentences"""

    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)


def load_embeddings_word2vec(vocabulary, filename):
    """This function loads word2vec embedding vectors from the file"""

    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initialise matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if True:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:

                    f.seek(binary_len, 1)
        f.close()
        return embedding_vectors


def load_embeddings_glove(vocabulary, filename, vector_size):
    """This function loads GloVe embedding vectors from the file"""

    # initialise matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors


def load_embeddings(vocabulary, dimension):
    """This function initialises embedding vectors randomly using vocabulary"""

    word_embeddings = {}
    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-0.25, 0.25, dimension)
    return word_embeddings


def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
    """This function pads sentences to one length for training or prediction"""

    if forced_sequence_length is None:  # Train
        sequence_length = max(len(x) for x in sentences)
    else:
        logging.critical('This is prediction, reading the trained sequence length')
        sequence_length = forced_sequence_length
    logging.critical('The maximum sentence length is {}'.format(sequence_length))

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)

        # Cut sentence if its longer than sequence length
        if num_padding < 0:
            logging.info('This sentence has to be cut off because it is longer than trained sequence length')
            padded_sentence = sentence[0:sequence_length]
        else:
            padded_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(padded_sentence)
    return padded_sentences


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """This function splitts data into mini-batches depending on number of epochs"""

    data = np.array(data)
    size = len(data)
    num_batches_per_epoch = int(size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, size)
            yield shuffled_data[start_index:end_index]


def count_words(filename):
    """This function provides word statistics about inputs"""

    sentences, labels, vocabulary, df = load(filename)

    num_words = []
    for line in sentences:
        counter = len(line.split())
        num_words.append(counter)

    num_files = len(num_words)

    return num_files, num_words


def load(filename):
    """This function loads csv files for training and testing"""

    df = pd.read_csv(filename)
    selected = ['label', 'text']
    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)
    df = df.dropna(axis=0, how='any', subset=selected)
    df = df.reindex(np.random.permutation(df.index))
    df = df[0:100000]

    # Map the actual labels to one hot labels
    labels = sorted(list(set(df[selected[0]].tolist())))
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    x_clean = df[selected[1]].apply(lambda x: clean_text(x, True, False)).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()

    # x_pad = pad_sentences(x_clean)
    vocabulary, vocabulary_inv = build_vocab(x_clean)

    # x = np.array([[vocabulary[word] for word in sentence] for sentence in x_pad])
    y = np.array(y_raw)

    return x_clean, y, vocabulary, labels


def load4(path):
    """This function was built for loading another dataset"""

    tweets = []
    affect = []

    os.chdir(path)

    for filename in os.listdir(path):

        f = open(filename, 'r')
        lines = f.readlines()[1:]
        for x in lines:
            tweets.append(x.split('\t')[1])
            affect.append(x.split('\t')[2])
        f.close()

    return tweets, affect


def split_train_test_data(filename):
    """This function randomly splitts dataset into train/test sets with equal number of samples per class"""

    classes = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]

    df = pd.read_csv(filename)
    selected = ['label', 'text']
    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)  # Drop non selected columns
    df = df.dropna(axis=0, how='any', subset=selected)  # Drop null rows
    df = df.reindex(np.random.permutation(df.index))  # Shuffle the dataframe
    df = df[0:100000]

    sentences = df[selected[1]].apply(lambda x: clean_text(x)).tolist()
    labels = df[selected[0]].tolist()

    joy = []
    fear = []
    anger = []
    sadness = []
    disgust = []
    shame = []
    guilt = []

    for i, e in enumerate(labels):
        if e == classes[0]:
            joy.append(sentences[i])
        elif e == classes[1]:
            fear.append(sentences[i])
        elif e == classes[2]:
            anger.append(sentences[i])
        elif e == classes[3]:
            sadness.append(sentences[i])
        elif e == classes[4]:
            disgust.append(sentences[i])
        elif e == classes[5]:
            shame.append(sentences[i])
        elif e == classes[6]:
            guilt.append(sentences[i])

    train_sentences = []
    train_labels = []
    test_labels = []

    count = 0

    joy_count = len(joy)
    fear_count = len(fear)
    anger_count = len(anger)
    sadness_count = len(sadness)
    disgust_count = len(disgust)
    shame_count = len(shame)
    guilt_count = len(guilt)

    while count < int(0.8 * joy_count):
        i = random.choice(range(len(joy)))
        train_sentences.append(joy[i])
        train_labels.append('joy')
        del joy[i]
        count = count + 1
    count = 0
    while count < int(0.8 * fear_count):
        i = random.choice(range(len(fear)))
        train_sentences.append(fear[i])
        train_labels.append('fear')
        del fear[i]
        count = count + 1
    count = 0
    while count < int(0.8 * anger_count):
        i = random.choice(range(len(anger)))
        train_sentences.append(anger[i])
        train_labels.append('anger')
        del anger[i]
        count = count + 1
    count = 0

    while count < int(0.8 * sadness_count):
        i = random.choice(range(len(sadness)))
        train_sentences.append(sadness[i])
        train_labels.append('sadness')
        del sadness[i]
        count = count + 1
    count = 0
    while count < int(0.8 * disgust_count):
        i = random.choice(range(len(disgust)))
        train_sentences.append(disgust[i])
        train_labels.append('disgust')
        del disgust[i]
        count = count + 1
    count = 0
    while count < int(0.8 * shame_count):
        i = random.choice(range(len(shame)))
        train_sentences.append(shame[i])
        train_labels.append('shame')
        del shame[i]
        count = count + 1
    count = 0
    while count < int(0.8 * guilt_count):
        i = random.choice(range(len(guilt)))
        train_sentences.append(guilt[i])
        train_labels.append('guilt')
        del guilt[i]
        count = count + 1

    test_sentences = joy + fear + anger + sadness + disgust + shame + guilt
    for x in joy:
        test_labels.append('joy')
    for x in fear:
        test_labels.append('fear')
    for x in anger:
        test_labels.append('anger')
    for x in sadness:
        test_labels.append('sadness')
    for x in disgust:
        test_labels.append('disgust')
    for x in shame:
        test_labels.append('shame')
    for x in guilt:
        test_labels.append('guilt')

    with open('./data/isear_train.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])
        writer.writerows(itertools.izip(train_labels, train_sentences))

    with open('./data/isear_test.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])
        writer.writerows(itertools.izip(test_labels, test_sentences))

    return


def plot_confusion_matrix(conf_mat, labels,
                          normalize=True,
                          title='',
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix built using sklearn.
    """
    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    else:
        pass


    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":

    split_train_test_data('./data/iseardataset.csv')
    # load_data('./data/isear_train.csv')
    # load4('./data/SemEval-2017/train/EI-reg-En-anger-train.txt')