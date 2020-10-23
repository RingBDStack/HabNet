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

        self.docs = tf.placeholder(shape=(None, None, None, None), dtype=tf.int32, name='docs')
        self.review_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='review_lengths')
        self.sent_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='sent_lengths')
        self.word_lengths = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='word_lengths')
        self.max_word_length = tf.placeholder(dtype=tf.int32, name='max_word_length')
        self.max_sent_length = tf.placeholder(dtype=tf.int32, name='max_sent_length')
        self.max_review_length = tf.placeholder(dtype=tf.int32, name='max_review_length')
        self.labels = tf.placeholder(shape=(None), dtype=tf.int32, name='labels')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        self._init_embedding()
        self._init_sent_encoder()
        self._init_intra_review_encoder()
        self._init_inter_review_encoder()
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

    def _init_intra_review_encoder(self): #review(doc) encoding
        with tf.variable_scope('intra-review-encoder') as scope:
            sent_inputs = tf.reshape(self.word_outputs, [-1, self.max_sent_length, 2 * self.emb_size])
            sent_inputs_mask_temp = tf.cast(self.docs, tf.bool)
            sent_inputs_mask = tf.reduce_any(sent_inputs_mask_temp, reduction_indices=[3])
            sent_inputs_mask = tf.reshape(sent_inputs_mask, [-1, self.max_sent_length])

            
            review_encoding = disan(sent_inputs, sent_inputs_mask,'DiSAN',self.dropout_rate,self.is_training,0.,'elu', None,'review-encoding')
            self.sent_outputs = review_encoding
            #self.sent_outputs = tf.layers.dropout(sent_outputs, self.dropout_rate, training=self.is_training)

    def _init_inter_review_encoder(self):  # reviews encoding
        with tf.variable_scope('inter-review-encoder') as scope:
            review_inputs = tf.reshape(self.sent_outputs, [-1, self.max_review_length, 4 * self.emb_size])
            sent_inputs_mask_temp = tf.cast(self.docs, tf.bool)
            sent_inputs_mask = tf.reduce_any(sent_inputs_mask_temp, reduction_indices=[3])
            review_inputs_mask = tf.reduce_any(sent_inputs_mask, reduction_indices=[2])


            # reviews GRU encoder
            cell_fw = rnn.GRUCell(self.cell_dim, name='cell_fw')
            cell_bw = rnn.GRUCell(self.cell_dim, name='cell_bw')

            init_state_fw = tf.tile(tf.get_variable('init_state_fw',
                                                    shape=[1, self.cell_dim],
                                                    initializer=tf.constant_initializer(0)),
                                    multiples=[get_shape(review_inputs)[0], 1])
            init_state_bw = tf.tile(tf.get_variable('init_state_bw',
                                                    shape=[1, self.cell_dim],
                                                    initializer=tf.constant_initializer(0)),
                                    multiples=[get_shape(review_inputs)[0], 1])

            
            rnn_outputs, _ = bidirectional_rnn(cell_fw=cell_fw,
                                               cell_bw=cell_bw,
                                               inputs=review_inputs,
                                               input_lengths=self.review_lengths,
                                               initial_state_fw=init_state_fw,
                                               initial_state_bw=init_state_bw,
                                               scope=scope)

            
            reviews_encoding = disan(rnn_outputs, review_inputs_mask, 'DiSAN', self.dropout_rate, self.is_training, 0.,
                                    'elu', None, 'reviews-encoding')
            self.review_outputs = reviews_encoding
            # self.sent_outputs = tf.layers.dropout(sent_outputs, self.dropout_rate, training=self.is_training)

    def _init_classifier(self):
        with tf.variable_scope('classifier'):
            self.logits = tf.layers.dense(inputs=self.review_outputs, units=self.num_classes, name='logits')

    def get_feed_dict(self, docs, labels, training=False):
        padded_docs, review_lengths, max_review_length, sent_lengths, max_sent_length, word_lengths, max_word_length = batch_doc_normalize(docs)
        fd = {
            self.docs: padded_docs,
            self.review_lengths: review_lengths,
            self.sent_lengths: sent_lengths,
            self.word_lengths: word_lengths,
            self.max_review_length: max_review_length,
            self.max_sent_length: max_sent_length,
            self.max_word_length: max_word_length,
            self.labels: labels,
            self.is_training: training
        }
        return fd
