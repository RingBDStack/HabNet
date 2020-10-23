import pandas as pd
import nltk
import itertools
import pickle
import random
import numpy as np
#from utils import read_vocab
from pandas import Series, DataFrame

# Hyper parameters
WORD_CUT_OFF = 1


def build_vocab(docs, save_path):
    print('Building vocab ...')

    #sents = itertools.chain(*[nltk.sent_tokenize(text) for text in docs])
    sents = []
    for ind, reviews in enumerate(docs):
        print(str(ind) + " reviews: ")
        for review in reviews:
            temp_sents = nltk.sent_tokenize(review)
            sents += temp_sents
    
    #tokenized_sents = [tokenize_word(sent) for sent in sents if len(tokenize_word(sent)) > 0]
    tokenized_sents = [nltk.word_tokenize(sent) for sent in sents]
    
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
    for label, reviews in zip(data.decision, data.reviews):
        #mapped_doc = [[word_to_index.get(word, 1) for word in tokenize_word(sent)] for sent in nltk.sent_tokenize(doc) if len(tokenize_word(sent)) > 0]
        mapped_reviews = []  # this is 3-d tensor representation of one sample (i.e. one paper)
        for review in reviews:
            mapped_review = []
            sents = nltk.sent_tokenize(review)
            for index, sent in enumerate(sents):
                word_tokens = nltk.word_tokenize(sent)
                #word_tokens = tokenize_word(sent)
                print("sent: " + str(index))
                if len(word_tokens) > 0:
                    word_index = [word_to_index.get(word, 1) for word in word_tokens]
                    mapped_review.append(word_index)
            mapped_reviews.append(mapped_review)
        mapped_data.append((label, mapped_reviews))

    with open(out_file, 'wb') as f:
        pickle.dump(mapped_data, f)
    print(out_file + " saved successfully!")


def read_data(data_file):
    data = pd.read_csv(data_file, sep=',', usecols=[0, 1, 3])
    #data = pd.read_csv(data_file)
    print('{}, shape={}'.format(data_file, data.shape))
    paperids = data.iloc[:,0]
    paperids = paperids.drop_duplicates()
    paperids_array = paperids.as_matrix()
    review_and_decision = []
    for index, paperid in enumerate(paperids_array):
        reviews_original = data[data['paperid']==paperid]
        reviews = []
        label = reviews_original.iloc[0,1]
        for row in reviews_original.iterrows():
            text = row[1]
            reviews.append(row[1]['text'])
        review_and_decision.append([label, reviews])

    #convert list to dataframe
    data_zipped = DataFrame(review_and_decision,columns=['decision','reviews'])

    shuffled_index = np.random.permutation(data_zipped.shape[0])
    #random.shuffle(data)
    train_data = data_zipped.iloc[shuffled_index[:2293],:]
    dev_data = data_zipped.iloc[shuffled_index[2293:2784],:]
    test_data = data_zipped.iloc[shuffled_index[2784:],:]
    
    return train_data,dev_data,test_data


if __name__ == '__main__':
    train_data,dev_data,test_data = read_data('data/ICLR_Review_all_with_decision_processed.csv')
    #train_data, dev_data, test_data = read_data('data/ICLR_Review_all_processed.csv')
    word_to_index = build_vocab(train_data.reviews, 'data/ICLR_Review_all_with_decision')
    process_and_save(word_to_index, train_data, 'data/ICLR_Review_all_with_decision-train.pkl')

    process_and_save(word_to_index, dev_data, 'data/ICLR_Review_all_with_decision-dev.pkl')

    process_and_save(word_to_index, test_data, 'data/ICLR_Review_all_with_decision-test.pkl')
    print("data prepare finished!")
