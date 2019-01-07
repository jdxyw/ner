import tensorflow as tf
import numpy as np
from functools import partial
import os


def parse_fn(line_words, line_tags):
    words = [w for w in line_words.strip().split()]
    tags = [t for t in line_tags.strip().split()]
    return (words, len(words)), tags


def generator(words, tags):
    with open(words) as f_words, open(tags) as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words, tags, params=None):
    shapes = (([
        None,
    ], ()), [
        None,
    ])
    types = ((tf.string, tf.int32), tf.string)
    defaults = (('<pad>', 0), 'O')

    dataset = tf.data.Dataset.from_generator(
        partial(generator, words, tags),
        output_shapes=shapes,
        output_types=types)
    dataset = (dataset.padded_batch(32, shapes, defaults).prefetch(1))
    return dataset


def model_fn(features, labels, mode, params):
    words, nword = features
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    with open(params['tags']) as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1

    words_id = vocab_words.lookup(words)
    glove = np.load(params['glove'])['embeddings']
    variable = np.vstack([glove, [[0.] * params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    embeddings_seq = tf.nn.embedding_lookup(variable, words_id)

    cells = []
    cell = tf.nn.rnn_cell.LSTMCell(num_units=128)
    cells.append(cell)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells)

    output, _ = tf.nn.dynamic_rnn(
        cell, embeddings_seq, sequence_length=nword, dtype=tf.float32)

    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nword)

    vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
    tags = vocab_tags.lookup(labels)
    log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
        logits, tags, nword, crf_params)
    loss = tf.reduce_mean(-log_likelihood)

    weights = tf.sequence_mask(nword)
    metrics = {
        'acc': tf.metrics.accuracy(tags, pred_ids, weights),
    }

    for metric_name, op in metrics.items():
        tf.summary.scalar(metric_name, op[1])

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer().minimize(
            loss, global_step=tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)


DATADIR = "*"

if __name__ == "__main__":
    params = {
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'words': os.path.join(DATADIR, 'vocab.words.txt'),
        'tags': os.path.join(DATADIR, 'vocab.tags.txt'),
        'glove': 'glove.npz'
    }

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(
        model_fn, config=cfg, params=params, model_dir="result/model")

    train_input_fn = partial(input_fn, os.path.join(DATADIR,
                                                    "train.words.txt"),
                             os.path.join(DATADIR, "train.tags.txt"))
    eval_input_fn = partial(input_fn, os.path.join(DATADIR, "test.words.txt"),
                            os.path.join(DATADIR, "test.tags.txt"))
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
