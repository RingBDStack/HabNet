import pandas as pd
import nltk
import itertools
import pickle
import random
import numpy as np
from techniques import *
#from utils import read_vocab

# Hyper parameters
WORD_CUT_OFF = 1

# variables for preprocessing
stoplist = stopwords.words('english')
lemmatizer = WordNetLemmatizer() # set lemmatizer
stemmer = PorterStemmer() # set stemmer

def build_vocab(docs, save_path):
    print('Building vocab ...')

    sents = itertools.chain(*[nltk.sent_tokenize(text) for text in docs])
    #sents = nltk.sent_tokenize(docs[0])
    #tokenized_sents = [tokenize_word(sent) for sent in sents if len(tokenize_word(sent)) > 0]
    #tokenized_sents = [nltk.word_tokenize(sent) for sent in sents]
    tokenized_sents = []
    for index, sent in enumerate(sents):
        word_tokens_of_onesent = tokenize_word(sent)
        print("process sent: " + str(index))
        if len(word_tokens_of_onesent) > 0:
            tokenized_sents.append(word_tokens_of_onesent)

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sents))
    print("%d unique words found" % len(word_freq.items()))

    # Cut-off
    retained_words = [w for (w, f) in word_freq.items() if f > WORD_CUT_OFF]
    print("%d words retained" % len(retained_words))

    # Get the most common words and build index_to_word and word_to_index vectors
    # Word index starts from 1, 0 is reserved for padding
    word_to_index = {'PAD': 0}
    for i, w in enumerate(retained_words):
        word_to_index[w] = i + 1
    index_to_word = {i: w for (w, i) in word_to_index.items()}

    print("Vocabulary size = %d" % len(word_to_index))

    with open('{}-w2i.pkl'.format(save_path), 'wb') as f:
        pickle.dump(word_to_index, f)

    with open('{}-i2w.pkl'.format(save_path), 'wb') as f:
        pickle.dump(index_to_word, f)

    return word_to_index


def process_and_save(word_to_index, data, out_file):
    mapped_data = []
    for label, doc in zip(data.stars, data.text):
        #mapped_doc = [[word_to_index.get(word, 1) for word in tokenize_word(sent)] for sent in nltk.sent_tokenize(doc) if len(tokenize_word(sent)) > 0]
        mapped_doc = []
        sents = nltk.sent_tokenize(doc)
        for index, sent in enumerate(sents):
            word_tokens = tokenize_word(sent)
            print("sent: " + str(index))
            if len(word_tokens) > 0:
                word_index = [word_to_index.get(word, 1) for word in word_tokens]
                mapped_doc.append(word_index)
        mapped_data.append((label, mapped_doc))

    with open(out_file, 'wb') as f:
        pickle.dump(mapped_data, f)
    print(out_file + " saved successfully!")


def tokenize_word(text): # function to preprocess one sentence and tokenize this sentence
    onlyOneSentenceTokens = []  # tokens of one sentence each time
    # preprocessing
    text = removeUnicode(text)  # Technique 0
    # print(text) # print initial text
    wordCountBefore = len(re.findall(r'\w+', text))  # word count of one sentence before preprocess
    # print("Words before preprocess: ",wordCountBefore,"\n")

    text = replaceContraction(text)  # Technique 3: replaces contractions to their equivalents
    text = remove_single_letter_words(text)
    text = remove_blank_spaces(text)
    text = removeNumbers(text)  # Technique 4: remove integers from text
    text = removePunctuations(text)
    tokens = nltk.word_tokenize(text)  # it takes a text as an input and provides a list of every token in it
    for w in tokens:

        if w not in stoplist: # Technique 10: remove stopwords
            #final_word = addCapTag(w) # Technique 8: Finds a word with at least 3 characters capitalized and adds the tag ALL_CAPS_
            final_word = w
            final_word = final_word.lower() # Technique 9: lowercases all characters
            final_word = replaceElongated(final_word) # Technique 11: replaces an elongated word with its basic form, unless the word exists in the lexicon
            if len(final_word)>1:
                final_word = spellCorrection(final_word) # Technique 12: correction of spelling errors
            final_word = lemmatizer.lemmatize(final_word) # Technique 14: lemmatizes words
            final_word = stemmer.stem(final_word) # Technique 15: apply stemming to words
            onlyOneSentenceTokens.append(final_word)
    return onlyOneSentenceTokens



def read_data(data_file):
    data = pd.read_csv(data_file)
    print('{}, shape={}'.format(data_file, data.shape))
    shuffled_index = np.random.permutation(data.shape[0])
    #random.shuffle(data)
    train_data = data.iloc[shuffled_index[:7600],:]
    dev_data = data.iloc[shuffled_index[7600:8600],:]
    test_data = data.iloc[shuffled_index[8600:9600],:]

    return train_data,dev_data,test_data


if __name__ == '__main__':
    train_data,dev_data,test_data = read_data('data/ICLR_Review_all_processed.csv')
    word_to_index = build_vocab(train_data.text, 'data/ICLR_Review_all')
    process_and_save(word_to_index, train_data, 'data/ICLR_Review_all-train.pkl')

    process_and_save(word_to_index, dev_data, 'data/ICLR_Review_all-dev.pkl')

    process_and_save(word_to_index, test_data, 'data/ICLR_Review_all-test.pkl')
    print("data prepare finished!")
