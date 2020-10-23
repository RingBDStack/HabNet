import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from layers import bidirectional_rnn, attention
from utils import get_shape, batch_doc_normalize
from disan import disan


class Model:
    def __init__(self, cell_dim, att_dim, vocab_size, emb_size, num_classes, dropout_rate, pretrained_embs):
        self.cell_dim = cell_dim
        self.att_dim = att_dim
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.pretrained_embs = pretrained_embs

        self.docs = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='docs')
        self.sent_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sent_lengths')
        self.word_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths')
        self.max_word_length = tf.placeholder(dtype=tf.int32, name='max_word_length')
        self.max_sent_length = tf.placeholder(dtype=tf.int32, name='max_sent_length')
        self.labels = tf.placeholder(shape=(None), dtype=tf.int32, name='labels')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        self._init_embedding()
        self._init_sent_encoder()
        self._init_intra_review_encoder()
        self._init_classifier()

    def _init_embedding(self):
        with tf.variable_scope('embedding'):
            self.embedding_matrix = tf.get_variable(name='embedding_matrix',
                                                    shape=[self.vocab_size, self.emb_size],
                                                    initializer=tf.constant_initializer(self.pretrained_embs),
                                                    dtype=tf.float32)
            self.embedded_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.docs)

    def _init_sent_encoder(self): #sent-encoding
        with tf.variable_scope('sent-encoder') as scope:
            word_inputs = tf.reshape(self.embedded_inputs, [-1, self.max_word_length, self.emb_size])
            word_lengths = tf.reshape(self.word_lengths, [-1])

            word_inputs_before_mask = tf.reshape(self.docs, [-1,self.max_word_length])
            word_inputs_mask = tf.cast(word_inputs_before_mask, tf.bool)

            sent_encoding = disan(word_inputs, word_inputs_mask,'DiSAN',self.dropout_rate,self.is_training,0.,'elu', None,'sent-encoding')
            self.word_outputs = sent_encoding
            #self.word_outputs = tf.layers.dropout(word_outputs, self.dropout_rate, training=self.is_training)

    def _init_intra_review_encoder(self): #review encoding
        with tf.variable_scope('intra-review-encoder') as scope:
            sent_inputs = tf.reshape(self.word_outputs, [-1, self.max_sent_length, 2 * self.emb_size])
            sent_inputs_mask_temp = tf.cast(self.docs, tf.bool)
            sent_inputs_mask = tf.reduce_any(sent_inputs_mask_temp, reduction_indices=[2])
            print("sent_inputs shape: {0}".format(tf.shape(sent_inputs)[0]))
            print("mask shape: {0}".format(tf.shape(sent_inputs_mask)[0]))
            
            review_encoding = disan(sent_inputs, sent_inputs_mask,'DiSAN',self.dropout_rate,self.is_training,0.,'elu', None,'review-encoding')
            self.sent_outputs = review_encoding
            #self.sent_outputs = tf.layers.dropout(sent_outputs, self.dropout_rate, training=self.is_training)

    def _init_classifier(self):
        with tf.variable_scope('classifier'):
            self.logits = tf.layers.dense(inputs=self.sent_outputs, units=self.num_classes, name='logits')

    def get_feed_dict(self, docs, labels, training=False):
        padded_docs, sent_lengths, max_sent_length, word_lengths, max_word_length = batch_doc_normalize(docs)
        fd = {
            self.docs: padded_docs,
            self.sent_lengths: sent_lengths,
            self.word_lengths: word_lengths,
            self.max_sent_length: max_sent_length,
            self.max_word_length: max_word_length,
            self.labels: labels,
            self.is_training: training
        }
        return fd
