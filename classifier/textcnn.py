import tensorflow as tf
import numpy as np
import random
import re
import time
import sys
sys.path.append('..')
from model import TextCNN
from options import *
from utils import constants
from utils.vocab import build_vocab_from_file, load_vocab_dict

os.environ["CUDA_VISIBLE_DEVICES"] = "2,1,0,3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_model(sess, args, vocab_size, mode=constants.TRAIN, load_pretrained_model=False, reuse=None):
    with tf.variable_scope(constants.CLS_VAR_SCOPE, reuse=reuse):
        model = TextCNN(mode, args.__dict__, vocab_size)

    if load_pretrained_model:
        try:
            model.saver.restore(sess, args.cls_model_save_dir)
            print("Loading model from", args.cls_model_save_dir)
        except Exception as e:
            model.saver.restore(sess, tf.train.latest_checkpoint(args.cls_model_save_dir))
            print("Loading model from", tf.train.latest_checkpoint(args.cls_model_save_dir))
    else:
        if reuse is None:
            print("Creating model with new parameters.")
            sess.run(tf.global_variables_initializer())
        else:
            print('Reuse parameters.')
    return model


def evaluate(sess, args, vocab, model, x, y, print_logs=True):
    probs = []
    batches = get_batches(x, y, word2id=vocab, batch_size=1)
    for batch in batches:
        p = sess.run(model.probs,
                     feed_dict={model.x: batch["x"],
                                model.dropout: 1})
        probs += p.tolist()
    y_hat = [p > 0.5 for p in probs]
    same = [p == q for p, q in zip(y, y_hat)]

    if print_logs:
        print("Saving classifier result at: %s" % args.log_path)
    with open(args.log_path, 'w') as f:
        for i in range(len(y)):
            f.write("%s\t%.3f\t%s\n" % (' '.join(x[i]), probs[i], same[i]))
    return (100.0 * sum(same)) / len(y), probs


def get_batches(x, y, word2id, batch_size, min_len=5):
    pad = word2id[constants.PADDING_TOKEN]
    unk = word2id[constants.UNKNOWN_TOKEN]

    batches = []
    s = 0
    sen_len = []
    while s < len(x):
        t = min(s + batch_size, len(x))

        _x = []
        max_len = max([len(sent) for sent in x[s:t]])
        max_len = max(max_len, min_len)  # sensitive to sentence-length
        sen_len.append(max_len)
        for sent in x[s:t]:
            sent_id = [word2id[w] if w in word2id else unk for w in sent]
            padding = [pad] * (max_len - len(sent))
            _x.append(padding + sent_id)

        batches.append({"x": _x,
                        "y": y[s:t]})
        s = t
    return batches


def prepare(paths, index_list=None, is_training=False):
    def load_sent(path, max_size=-1):
        data = []
        with open(path) as f:
            for line in f:
                if len(data) == max_size:
                    break
                words = line.split()
                if is_training:
                    if len(words) > 1:  # filter sentence only have one words
                        data.append(words)
                else:
                    data.append(words)
        return data

    if index_list is None:
        index_list = []
        for path in paths:
            i = int(re.findall('\d', path)[-1])
            if '.tsf' in path or 'reference' in path:
                i = 1-i
            index_list.append(i)

    data0 = load_sent(paths[0])
    if len(paths) >= 2:
        data1 = load_sent(paths[1])
        if is_training:
            min_c = min(len(data0), len(data1))
            np.random.shuffle(data0)
            np.random.shuffle(data1)
            # balance the bias caused by training data
            print("Dropped: %d, %s" % (len(data0) - min_c, paths[0]))
            data0 = data0[:min_c]
            print("Dropped: %d, %s" % (len(data1) - min_c, paths[1]))
            data1 = data1[:min_c]

        x = data0 + data1
        y = [index_list[0]] * len(data0) + [index_list[1]] * len(data1)
    else:
        x = data0
        y = [index_list[0]] * len(data0)

    z = sorted(zip(x, y), key=lambda i: len(i[0]))  # ranked by the length of sentences
    return zip(*z)


def evaluate_file(sess, args, vocab, eval_model, files, index_list, print_logs=True):
    x, y = prepare(files, index_list=index_list)
    acc, _ = evaluate(sess, args, vocab, eval_model, x, y, print_logs)
    return acc


if __name__ == "__main__":
    args = load_cls_arguments()

    if args.train_data and args.mode == constants.TRAIN:
        train_x, train_y = prepare(args.train_data, index_list=[0, 1], is_training=True)

        if not os.path.isfile(args.global_vocab_file):
            build_vocab_from_file(args.train_data, args.global_vocab_file)

    vocab, vocab_size = load_vocab_dict(args.global_vocab_file)
    print("Vocabulary size", vocab_size)

    if args.dev_data:
        dev_x, dev_y = prepare(args.dev_data)

    # if args.test_data and args.mode == constants.EVAL:
    if args.test_data:
        test_x, test_y = prepare(args.test_data)

    print("Prepare to Save model at: %s" % args.cls_model_save_dir)
    if not os.path.exists(args.cls_model_save_dir):
        os.makedirs(args.cls_model_save_dir)

    dump_args_to_yaml(args, args.cls_model_save_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    with tf.Session(config=config) as sess:
        with tf.device("/cpu:0"):   # gpu will face error
            if args.mode == constants.TRAIN:
                model = create_model(sess, args, vocab_size)
                if args.train_data:
                    batches = get_batches(train_x, train_y,
                                          vocab, args.batch_size)
                    random.shuffle(batches)

                    start_time = time.time()
                    step = 0
                    loss = 0.0
                    best_dev = float("-inf")
                    learning_rate = args.learning_rate

                    for epoch in range(1, 1 + args.n_epoch):
                        print("--------------------Epoch %d--------------------" % epoch)

                        for batch in batches:
                            step_loss, _ = sess.run([model.loss, model.optimizer],
                                                    feed_dict={model.x: batch["x"],
                                                               model.y: batch["y"],
                                                               model.dropout: args.keep_prob})

                            step += 1
                            loss += step_loss / args.steps_per_checkpoint

                            if step % args.steps_per_checkpoint == 0:
                                print("step %d, time %.0fs, loss %.2f" % (step, time.time() - start_time, loss))
                                loss = 0.0

                        if args.dev_data:
                            acc, _ = evaluate(sess, args, vocab, model, dev_x, dev_y)
                            print("Dev accuracy: %.2f" % acc)
                            if acc > best_dev:
                                best_dev = acc
                                print("Saving model to:  %s" % args.cls_model_save_dir)
                                model.saver.save(sess, args.cls_model_save_dir)
                        if args.test_data:
                            acc, _ = evaluate(sess, args, vocab, model, test_x, test_y)
                            print("Test accuracy: %.2f" % acc)
            else:
                if args.test_data:
                    eval_model = create_model(sess, args, vocab_size, load_pretrained_model=True)
                    acc, _ = evaluate(sess, args, vocab, eval_model, test_x, test_y)
                    print("Test accuracy: %.2f" % acc)
