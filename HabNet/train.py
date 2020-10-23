import tensorflow as tf
from datetime import datetime
from data_reader import DataReader
from model import Model
from utils import read_vocab, count_parameters, load_glove, create_folder_if_not_exists
import sklearn.metrics
import utils_plots
import matplotlib
matplotlib.use('Agg')  # http://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
import matplotlib.pyplot as plt
import os
import numpy as np

# Parameters
# ==================================================

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_dir", 'checkpoints',
                       """Path to checkpoint folder""")
tf.flags.DEFINE_string("log_dir", 'logs',
                       """Path to log folder""")

tf.flags.DEFINE_integer("cell_dim", 100,
                        """Hidden dimensions of GRU cells (default: 50)""")
tf.flags.DEFINE_integer("att_dim", 100,
                        """Dimensionality of attention spaces (default: 100)""")
tf.flags.DEFINE_integer("emb_size", 50,
                        """Dimensionality of word embedding (default: 200)""")
tf.flags.DEFINE_integer("num_classes", 2,
                        """Number of classes (default: 5)""")

tf.flags.DEFINE_integer("num_checkpoints", 1,
                        """Number of checkpoints to store (default: 1)""")
tf.flags.DEFINE_integer("num_epochs", 100,
                        """Number of training epochs (default: 20)""")
tf.flags.DEFINE_integer("batch_size", 8,
                        """Batch size (default: 64)""")
tf.flags.DEFINE_integer("display_step", 20,
                        """Number of steps to display log into TensorBoard (default: 20)""")

tf.flags.DEFINE_float("learning_rate", 0.0005,
                      """Learning rate (default: 0.0005)""")
tf.flags.DEFINE_float("max_grad_norm", 5.0,
                      """Maximum value of the global norm of the gradients for clipping (default: 5.0)""")
tf.flags.DEFINE_float("dropout_rate", 0.5,
                      """Probability of dropping neurons (default: 0.5)""")

tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        """Allow device soft device placement""")

if not tf.gfile.Exists(FLAGS.checkpoint_dir):
  tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

if not tf.gfile.Exists(FLAGS.log_dir):
  tf.gfile.MakeDirs(FLAGS.log_dir)

train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train')
valid_writer = tf.summary.FileWriter(FLAGS.log_dir + '/valid')
test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

def loss_fn(labels, logits):
  onehot_labels = tf.one_hot(labels, depth=FLAGS.num_classes)
  cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                       logits=logits)
  tf.summary.scalar('loss', cross_entropy_loss)
  return cross_entropy_loss

def train_fn(loss):
  trained_vars = tf.trainable_variables()
  count_parameters(trained_vars)

  # Gradient clipping
  gradients = tf.gradients(loss, trained_vars)

  clipped_grads, global_norm = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)
  tf.summary.scalar('global_grad_norm', global_norm)

  # Define optimizer
  global_step = tf.train.get_or_create_global_step()
  optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
  train_op = optimizer.apply_gradients(zip(clipped_grads, trained_vars),
                                       name='train_op',
                                       global_step=global_step)
  return train_op, global_step


def eval_fn(labels, logits):
  predictions = tf.argmax(logits, axis=-1)
  correct_preds = tf.equal(predictions, tf.cast(labels, tf.int64))
  batch_acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
  tf.summary.scalar('accuracy', batch_acc)

  total_acc, acc_update = tf.metrics.accuracy(labels, predictions, name='metrics/acc')
  metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
  metrics_init = tf.variables_initializer(var_list=metrics_vars)

  return batch_acc, total_acc, acc_update, metrics_init, predictions

def save_optimized_presicion(all_y_true, all_y_pred, stats_graph_folder, name, epoch):
    #output_filepath = os.path.join(stats_graph_folder,'{1:03d}_{0}_optimized_precision.txt'.format(name, epoch))
    output_filepath = os.path.join(stats_graph_folder, '{0}_optimized_precision.txt'.format(name))
    classification_report = sklearn.metrics.classification_report(all_y_true, all_y_pred, digits=4)
    lines = classification_report.split('\n')
    acc = sklearn.metrics.accuracy_score(all_y_true, all_y_pred)
    classification_report = ['Accuracy: {:05.2f}%'.format(acc * 100)]
    recalls = sklearn.metrics.recall_score(all_y_true,all_y_pred,average=None)
    l=len(recalls)
    recalls_tile = np.tile(recalls,(len(recalls),1))
    recalls_vstack = np.array(10*[recalls[0]])
    for each in recalls[1:len(recalls)]:
        temp = np.array(10 * [each])
        recalls_vstack = np.vstack((recalls_vstack,temp))
    diff = recalls_vstack -recalls_tile
    diff1 = np.abs(diff)
    op = acc - diff1.sum()/(18*recalls.sum())
    print(name+" optimized precision: " + str(op))
    with open(output_filepath, 'a', encoding='utf-8') as fp:
        fp.write("{:.6f}\n".format(op))


def save_distance_measure(all_y_true, all_y_pred, stats_graph_folder, name, epoch):
    #output_filepath = os.path.join(stats_graph_folder,'{1:03d}_{0}_distance_measure.txt'.format(name, epoch))
    output_filepath = os.path.join(stats_graph_folder, '{0}_distance_measure.txt'.format(name))
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    dist = np.abs(y_pred-y_true)
    dist2 = np.mean(dist)
    dist_final = 1.0-dist2/9.0
    print(name+" distance measure: " + str(dist_final))
    with open(output_filepath, 'a', encoding='utf-8') as fp:
        fp.write("{:.6f}\n".format(dist_final))

def save_results(all_y_true, all_y_pred, stats_graph_folder, name, epoch):
    output_filepath = os.path.join(stats_graph_folder, 'classification_report_for_epoch_{0:04d}_in_{1}.txt'.format(epoch, name))
    plot_format = 'pdf'
    
    unique_labels = [0, 1]
    # classification_report = sklearn.metrics.classification_report(labels, predictions, digits=4,
    #                                                              labels=unique_labels)
    classification_report = sklearn.metrics.classification_report(all_y_true, all_y_pred, digits=4)
    acc = sklearn.metrics.accuracy_score(all_y_true, all_y_pred)
    lines = classification_report.split('\n')
    classification_report = ['Accuracy: {:05.2f}%'.format(sklearn.metrics.accuracy_score(all_y_true, all_y_pred) * 100)]
    for line in lines[2: (len(lines) - 1)]:
        new_line = []
        t = line.strip().replace(' avg', '-avg').split()
        if len(t) < 2: continue
        new_line.append(('        ' if t[0].isdigit() else '') + t[0])
        new_line += ['{:05.2f}'.format(float(x) * 100) for x in t[1: len(t) - 1]]
        new_line.append(t[-1])
        classification_report.append('\t'.join(new_line))
    classification_report = '\n'.join(classification_report)
    print('\n\n' + classification_report + '\n', flush=True)
    #with open(output_filepath + '_evaluation.txt', 'a', encoding='utf-8') as fp:
    with open(output_filepath, 'a', encoding='utf-8') as fp:
        fp.write(classification_report)

    output_filepath_acc = os.path.join(stats_graph_folder, '{0}_accuracy.txt'.format(name))
    with open(output_filepath_acc, 'a', encoding='utf-8') as f:
        f.write("{:.2f}\n".format(acc * 100))

    # save confusion matrix and generate plots
    confusion_matrix = sklearn.metrics.confusion_matrix(all_y_true, all_y_pred)
    #results['confusion_matrix'] = confusion_matrix.tolist()
    title = 'Confusion matrix for epoch {0} in {1}\n'.format(epoch, name)
    xlabel = 'Predicted'
    ylabel = 'True'
    xticklabels = yticklabels = unique_labels
    utils_plots.heatmap(confusion_matrix, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40,
                        figure_height=20, correct_orientation=True, fmt="%d", remove_diagonal=True)
    plt.savefig(os.path.join(stats_graph_folder,
                             'confusion_matrix_for_epoch_{0:04d}_in_{1}_{2}.{2}'.format(epoch, name,
                                                                                        plot_format)),
                dpi=300, format=plot_format, bbox_inches='tight')
    plt.close()


def main(_):
  vocab = read_vocab('data/ICLR_Review_all_with_decision-w2i.pkl')
  glove_embs = load_glove('glove.6B.{}d.txt'.format(FLAGS.emb_size), FLAGS.emb_size, vocab)
  data_reader = DataReader(train_file='data/ICLR_Review_all_with_decision-train.pkl',
                           dev_file='data/ICLR_Review_all_with_decision-dev.pkl',
                           test_file='data/ICLR_Review_all_with_decision-test.pkl')

  config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  with tf.Session(config=config) as sess:
    model = Model(cell_dim=FLAGS.cell_dim,
                  att_dim=FLAGS.att_dim,
                  vocab_size=len(vocab),
                  emb_size=FLAGS.emb_size,
                  num_classes=FLAGS.num_classes,
                  dropout_rate=FLAGS.dropout_rate,
                  pretrained_embs=glove_embs)

    loss = loss_fn(model.labels, model.logits)
    train_op, global_step = train_fn(loss)
    batch_acc, total_acc, acc_update, metrics_init, predictions = eval_fn(model.labels, model.logits)
    summary_op = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    train_writer.add_graph(sess.graph)
    saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)

    print('\n{}> Start training'.format(datetime.now()))
    result_save_folder = str(datetime.now())
    output_folder = os.path.join('.', 'output')
    create_folder_if_not_exists(output_folder)

    stats_graph_folder = os.path.join(output_folder, result_save_folder)  # Folder where to save graphs
    create_folder_if_not_exists(stats_graph_folder)


    epoch = 0
    valid_step = 0
    test_step = 0
    train_test_prop = len(data_reader.train_data) / len(data_reader.test_data)
    test_batch_size = int(FLAGS.batch_size / train_test_prop)
    best_acc = float('-inf')

    while epoch < FLAGS.num_epochs:
      epoch += 1
      print('\n{}> Epoch: {}'.format(datetime.now(), epoch))

      sess.run(metrics_init)
      all_labels = []
      all_y_pred = []
      for batch_docs, batch_labels in data_reader.read_train_set(FLAGS.batch_size, shuffle=True):
        _step, _, _loss, _acc, _, y_pred_batch = sess.run([global_step, train_op, loss, batch_acc, acc_update, predictions],
                                         feed_dict=model.get_feed_dict(batch_docs, batch_labels, training=True))
        all_labels += batch_labels
        #y_pred_batch_array = y_pred_batch.eval(session=sess)
        y_pred_batch_list = y_pred_batch.tolist()
        all_y_pred += y_pred_batch_list
        if _step % FLAGS.display_step == 0:
          _summary = sess.run(summary_op, feed_dict=model.get_feed_dict(batch_docs, batch_labels))
          train_writer.add_summary(_summary, global_step=_step)
      print('Training accuracy = {:.2f}'.format(sess.run(total_acc) * 100))
      save_results(all_labels, all_y_pred, stats_graph_folder, 'train', epoch)

      sess.run(metrics_init)
      all_valid_labels = []
      all_valid_y_pred = []
      for batch_docs, batch_labels in data_reader.read_valid_set(test_batch_size):
        _loss, _acc, _, valid_y_pred_batch = sess.run([loss, batch_acc, acc_update, predictions], feed_dict=model.get_feed_dict(batch_docs, batch_labels))
        all_valid_labels += batch_labels
        valid_y_pred_batch_list = valid_y_pred_batch.tolist()
        all_valid_y_pred += valid_y_pred_batch_list

        valid_step += 1
        if valid_step % FLAGS.display_step == 0:
          _summary = sess.run(summary_op, feed_dict=model.get_feed_dict(batch_docs, batch_labels))
          valid_writer.add_summary(_summary, global_step=valid_step)
      print('Validation accuracy = {:.2f}'.format(sess.run(total_acc) * 100))
      #save_optimized_presicion(all_valid_labels, all_valid_y_pred, stats_graph_folder, 'valid', epoch)
      #save_distance_measure(all_valid_labels, all_valid_y_pred, stats_graph_folder, 'valid', epoch)
      save_results(all_valid_labels, all_valid_y_pred, stats_graph_folder, 'valid', epoch)

      sess.run(metrics_init)
      all_test_labels = []
      all_test_y_pred = []
      for batch_docs, batch_labels in data_reader.read_test_set(test_batch_size):
        _loss, _acc, _, test_y_pred_batch  = sess.run([loss, batch_acc, acc_update, predictions], feed_dict=model.get_feed_dict(batch_docs, batch_labels))
        all_test_labels += batch_labels
        test_y_pred_batch_list = test_y_pred_batch.tolist()
        all_test_y_pred += test_y_pred_batch_list

        test_step += 1
        if test_step % FLAGS.display_step == 0:
          _summary = sess.run(summary_op, feed_dict=model.get_feed_dict(batch_docs, batch_labels))
          test_writer.add_summary(_summary, global_step=test_step)
      test_acc = sess.run(total_acc) * 100
      print('Testing accuracy = {:.2f}'.format(test_acc))
      #save_optimized_presicion(all_test_labels, all_test_y_pred, stats_graph_folder, 'test', epoch)
      #save_distance_measure(all_test_labels, all_test_y_pred, stats_graph_folder, 'test', epoch)
      save_results(all_test_labels, all_test_y_pred, stats_graph_folder, 'test', epoch)

      if test_acc > best_acc:
        best_acc = test_acc
        saver.save(sess, FLAGS.checkpoint_dir)
      print('Best testing accuracy = {:.2f}'.format(best_acc))

  print("{} Optimization Finished!".format(datetime.now()))
  print('Best testing accuracy = {:.2f}'.format(best_acc))


if __name__ == '__main__':
  tf.app.run()
