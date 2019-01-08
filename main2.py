import tensorflow as tf
import numpy as np
from functools import partial
import os


def input_fn(file):
    def _parse_text_line(ln):
        #words_ln, tags_ln = ln.strip().split("\t")
        words_ln = tf.string_split(
            tf.expand_dims(tf.string_strip(ln), axis=0), "\t").values[0]
        tags_ln = tf.string_split(
            tf.expand_dims(tf.string_strip(ln), axis=0), "\t").values[1]

        # words_ln = pair[0]
        # tags_ln = pair[1]

        words = tf.string_split([words_ln], " ").values
        tags = tf.string_split([tags_ln], " ").values
        #words = tf.convert_to_tensor(words_ln.split(), dtype=tf.string)
        #tags = tf.convert_to_tensor(tags_ln.split(), dtype=tf.string)

        # sess = tf.Session()
        # print("***********************")
        # print(sess.run(words))
        # print(sess.run(tags))

        features = {
            "words": words,
            "rev_words": tf.reverse(words, axis=[0]),
            "seq_len": tf.cast(tf.shape(words)[0], dtype=tf.int32),
            #"pair": pair,
            "words_ln": words_ln
            #"origin_line": ln,
        }
        d = features, tags
        return d

    #def _input_fn():

    dataset = tf.data.TextLineDataset(file)
    dataset = dataset.map(_parse_text_line)
    dataset = (
        dataset.padded_batch(
            batch_size=1,
            padded_shapes=(
                {
                    "words": (None, ),
                    "rev_words": (None, ),
                    "seq_len": (),
                    #"pair": (None, ),
                    "words_ln": ()
                    #"origin_line": (),
                },
                (None, )),
            padding_values=(
                {
                    "words": "<pad>",
                    "rev_words": "<pad>",
                    "seq_len": 0,
                    #"pair": "",
                    "words_ln": "",
                    #"origin_line": "",
                },
                "0")))
    #features, labels = dataset.make_one_shot_iterator().get_next()
    #return features, labels
    interator = dataset.make_one_shot_iterator()
    features, label = interator.get_next()
    return {
        "words": features["words"],
        "seq_len": features["seq_len"],
        "label": label,
        #"pair": features["pair"],
        "words_ln": features["words_ln"]
    }
    #return dataset.prefetch(2).make_one_shot_iterator()

    #return _input_fn


def model_fn(features, labels, mode, params):
    #words, nword = features
    print(features)
    words = features["words"]
    nword = features["seq_len"]
    #labels = features["tags"]
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    with open(params['tags']) as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1
    print("**********************")
    print(num_tags)

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


DATADIR = "/Users/yongweixing/data/ner"

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

    # train_input_fn = partial(input_fn, os.path.join(DATADIR,
    #                                                 "train.words.txt"),
    #                          os.path.join(DATADIR, "train.tags.txt"))
    # eval_input_fn = partial(input_fn, os.path.join(DATADIR, "test.words.txt"),
    #                         os.path.join(DATADIR, "test.tags.txt"))
    inputs = input_fn(os.path.join(DATADIR, "train.txt"))
    sess = tf.Session()
    for i in range(3):
        #fea, label = train_input_fn.get_next()
        #inputs = train_input_fn()
        #print("original line")
        #print(sess.run(fea["origin_line"]))
        #print(sess.run(inputs["pair"]))
        print(sess.run(inputs["words_ln"]))
        print("the shape of the words")
        print(sess.run(tf.shape(inputs["words"])))
        print(sess.run(inputs["words"]))
        print("the shape of the tags")
        print(sess.run(tf.shape(inputs["label"])))
        print(sess.run(inputs["label"]))
        print("the value of the seq_len")
        print(sess.run(inputs["seq_len"]))
        #print(sess.run(label))
        # print(sess.run(fea))
        # print(sess.run(label))
    # eval_input_fn = input_fn(os.path.join(DATADIR, "test.txt"))
    # train_spec = tf.estimator.TrainSpec(
    #     input_fn=train_input_fn, max_steps=10000)
    # eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    # tf.logging.set_verbosity(tf.logging.INFO)
    # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)