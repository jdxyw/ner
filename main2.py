import tensorflow as tf
import numpy as np
from functools import partial
import os
from tf_metrics import precision, recall, f1


def input_fn(file):
    def _parse_text_line(ln):
        pair = tf.string_split(
            tf.expand_dims(tf.string_strip(ln), axis=0), "\t").values

        words = tf.string_split([pair[0]]).values
        tags = tf.string_split([pair[1]]).values

        features = {
            "words": words,
            "rev_words": tf.reverse(words, axis=[0]),
            "seq_len": tf.cast(tf.shape(words)[0], dtype=tf.int32),
        }
        d = features, tags
        return d

    def _input_fn():

        dataset = tf.data.TextLineDataset(file)
        dataset = dataset.map(_parse_text_line)
        dataset = dataset.repeat(20)
        dataset = (dataset.padded_batch(
            batch_size=16,
            padded_shapes=({
                "words": (None, ),
                "rev_words": (None, ),
                "seq_len": (),
            }, (None, )),
            padding_values=({
                "words": "<pad>",
                "rev_words": "<pad>",
                "seq_len": 0,
            }, "O")))
        features, labels = dataset.make_one_shot_iterator().get_next()
        return features, labels

    return _input_fn


def model_fn(features, labels, mode, params):
    words = features["words"]
    rev_words = features["words"]
    nword = features["seq_len"]
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    with open(params['tags']) as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1

    words_id = vocab_words.lookup(words)
    rev_words_id = vocab_words.lookup(rev_words)
    glove = np.load(params['glove'])['embeddings']
    variable = np.vstack([glove, [[0.] * params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    embeddings_seq = tf.nn.embedding_lookup(variable, words_id)
    embeddings_seq = tf.layers.dropout(
        embeddings_seq, rate=0.5, training=training)
    rev_embeddings_seq = tf.nn.embedding_lookup(variable, rev_words_id)
    rev_embeddings_seq = tf.layers.dropout(
        rev_embeddings_seq, rate=0.5, training=training)

    cells = []
    cell = tf.nn.rnn_cell.LSTMCell(num_units=128, name="forward_lstm_cell")
    cells.append(cell)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells)

    rev_cells = []
    rev_cell = tf.nn.rnn_cell.LSTMCell(
        num_units=128, name="backward_lstm_cell")
    rev_cells.append(rev_cell)
    rev_cell = tf.nn.rnn_cell.MultiRNNCell(cells=rev_cells)

    output, _ = tf.nn.dynamic_rnn(
        cell, embeddings_seq, sequence_length=nword, dtype=tf.float32)
    rev_output, _ = tf.nn.dynamic_rnn(
        rev_cell, rev_embeddings_seq, sequence_length=nword, dtype=tf.float32)
    o = tf.concat([output, rev_output], axis=-1)
    o = tf.layers.dropout(o, rate=0.5, training=training)
    l1 = tf.layers.dense(o, 100)
    logits = tf.layers.dense(l1, num_tags)
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
        'precision': precision(tags, pred_ids, num_tags, indices, weights),
        'recall': recall(tags, pred_ids, num_tags, indices, weights),
        'f1': f1(tags, pred_ids, num_tags, indices, weights),
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

    train_input_fn = input_fn(os.path.join(DATADIR, "train.txt"))
    eval_input_fn = input_fn(os.path.join(DATADIR, "test.txt"))
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=50000)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)