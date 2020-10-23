import tensorflow as tf
import numpy as np
import pickle
import os


def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims


def count_parameters(trained_vars):
  total_parameters = 0
  print('=' * 100)
  for variable in trained_vars:
    variable_parameters = 1
    for dim in variable.get_shape():
      variable_parameters *= dim.value
    print('{:70} {:20} params'.format(variable.name, variable_parameters))
    print('-' * 100)
    total_parameters += variable_parameters
  print('=' * 100)
  print("Total trainable parameters: %d" % total_parameters)
  print('=' * 100)


def read_vocab(vocab_file):
  print('Loading vocabulary ...')
  with open(vocab_file, 'rb') as f:
    word_to_index = pickle.load(f)
    print('Vocabulary size = %d' % len(word_to_index))
    return word_to_index


def batch_doc_normalize(docs):
  review_lengths = np.array([len(reviews) for reviews in docs], dtype=np.int32)
  max_review_length = review_lengths.max()
  #sent_lengths = np.array([len(doc) for doc in docs], dtype=np.int32)
  #max_sent_length = sent_lengths.max()
  sent_lengths = [[len(review) for review in reviews] for reviews in docs]
  max_sent_length = max(map(max, sent_lengths))
  #sent_lengths = np.array(sent_lengths, dtype=np.int32)

  word_lengths = [[[len(sent) for sent in review] for review in reviews] for reviews in docs]
  #max_word_length = max(map(max, word_lengths))
  max_word_length_temp = [max(map(max, each)) for each in word_lengths]
  max_word_length = max(max_word_length_temp)
  #word_lengths_array = np.array(word_lengths)
  #max_word_length = word_lengths_array.max()

  padded_docs = np.zeros(shape=[len(docs), max_review_length, max_sent_length, max_word_length], dtype=np.int32)  # PADDING 0
  sent_lengths = np.zeros(shape=[len(docs), max_review_length], dtype=np.int32)
  word_lengths = np.zeros(shape=[len(docs), max_review_length, max_sent_length], dtype=np.int32)
  for i, reviews in enumerate(docs):
    for j, review in enumerate(reviews):
      sent_lengths[i,j] = len(review)
      for k, sent in enumerate(review):
        word_lengths[i, j, k] = len(sent)
        for m, word in enumerate(sent):
          padded_docs[i, j, k, m] = word

  return padded_docs, review_lengths, max_review_length, sent_lengths, max_sent_length, word_lengths, max_word_length


def load_glove(glove_file, emb_size, vocab):
  print('Loading Glove pre-trained word embeddings ...')
  embedding_weights = {}
  f = open(glove_file, encoding='utf-8')
  for line in f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    embedding_weights[word] = vector
  f.close()
  print('Total {} word vectors in {}'.format(len(embedding_weights), glove_file))

  embedding_matrix = np.random.uniform(-0.5, 0.5, (len(vocab), emb_size)) / emb_size

  oov_count = 0
  for word, i in vocab.items():
    embedding_vector = embedding_weights.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
    else:
      oov_count += 1
  print('Number of OOV words = %d' % oov_count)

  return embedding_matrix

def create_folder_if_not_exists(directory):
    '''
    Create the folder if it doesn't exist already.
    '''
    if not os.path.exists(directory):
      os.makedirs(directory)
