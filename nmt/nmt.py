import tensorflow as tf
import numpy as np
import time
import sys
import re
sys.path.append("..")
from utils.data import load_dataset, load_paired_dataset
from utils.vocab import build_vocab_from_file, load_vocab
from model import NMT
from utils import constants
from options import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def create_model(sess, args, src_vocab_size, tgt_vocab_size, src_vocab_rev, tgt_vocab_rev, mode=constants.TRAIN,
                 reuse=None, load_pretrained_model=False, direction="", model_save_dir=None):
    sess.run(tf.tables_initializer())

    with tf.variable_scope(constants.NMT_VAR_SCOPE + direction, reuse=reuse):
        with tf.variable_scope("src"):
            src_emb = tf.get_variable("embedding", shape=[src_vocab_size, args.emb_dim])
        with tf.variable_scope("dst"):
            tgt_emb = tf.get_variable("embedding", shape=[tgt_vocab_size, args.emb_dim])

        model = NMT(mode, args.__dict__, src_vocab_size, tgt_vocab_size, src_emb, tgt_emb,
                    src_vocab_rev, tgt_vocab_rev, direction)

    if load_pretrained_model:
        if model_save_dir is None:
            model_save_dir = args.nmt_model_save_dir
            if direction not in model_save_dir:
                if direction[::-1] in model_save_dir:
                    model_save_dir = re.sub(direction[::-1], direction, model_save_dir)
                else:
                    model_save_dir = os.path.join(model_save_dir, direction)
        print(model_save_dir)
        try:
            print("Loading nmt model from", model_save_dir)
            model.saver.restore(sess, model_save_dir)
        except Exception as e:
            print("Error! Loading nmt model from", model_save_dir)
            print("Again! Loading nmt model from", tf.train.latest_checkpoint(model_save_dir))
            model.saver.restore(sess, tf.train.latest_checkpoint(model_save_dir))
    else:
        if reuse is None:
            print("Creating model with new parameters.")
            sess.run(tf.global_variables_initializer())
        else:
            print("Reuse parameters.")
    return model


def train(model):
    best = {
        "loss": 100.0,
        "step": 0,
    }
    global_step = 0
    for i in range(args.n_epoch):
        print("Epoch:%d" % i)
        sess.run(train_iterator.initializer)
        n_batch = 0
        t0 = time.time()
        while True:
            try:
                src = sess.run(train_next_op)
                feed_dict = {
                    model.input_ids: src["ids"],
                    model.input_length: src["length"],
                    model.target_ids_in: src["trans_ids_in"],
                    model.target_ids_out: src["trans_ids_out"],
                    model.target_length: src["trans_length"]
                }

                assert src["length"].shape == src["trans_length"].shape
                assert src["ids"].shape[0] == src["trans_ids"].shape[0]
                ops = [model.loss,
                       model.logits,
                       model.train_op, ]
                res = sess.run(ops, feed_dict=feed_dict)

                n_batch += 1
                global_step += 1
                if n_batch % 100 == 0:
                    eval_loss = eval(model)
                    if eval_loss < best["loss"]:
                        best["loss"] = eval_loss
                        best["step"] = global_step
                        print("Model save at: %s" % (args.nmt_model_save_dir+str(n_batch)))
                        model.saver.save(sess, args.nmt_model_save_dir, global_step=n_batch)

                    print("Epoch/n_batch:%d/%d\tTrain_loss:%.3f\tEval_loss:%.3f\tMin_eval_loss:%.3f\tBest_step:%d"
                          "\tTime:%d" % (i, n_batch, res[0], eval_loss, best["loss"], best["step"], time.time() - t0))

            except tf.errors.OutOfRangeError:  # next epoch
                print("Train---Total N batch:{}".format(n_batch))
                break


def eval(model):
    sess.run(dev_iterator.initializer)
    n_batch = 0
    t0 = time.time()
    loss = []
    while True:
        try:
            src = sess.run(dev_next_op)
            feed_dict = {
                model.input_ids: src["ids"],
                model.input_length: src["length"],
                model.target_ids_in: src["trans_ids_in"],
                model.target_ids_out: src["trans_ids_out"],
                model.target_length: src["trans_length"]
            }
            assert src["length"].shape == src["trans_length"].shape
            assert src["ids"].shape[0] == src["trans_ids"].shape[0]

            ops = [model.loss]
            res = sess.run(ops, feed_dict=feed_dict)
            loss.append(res[0])
            n_batch += 1
        except tf.errors.OutOfRangeError:  # next epoch
            print("Eval---Total N batch:{}\tLoss:{}\tCost time:{}".format(n_batch, np.mean(loss), time.time() - t0))
            break
    return np.mean(loss)


def inference(model, A, B, sess, args, src_test_iterator, src_test_next, src_vocab_rev, result_dir=None, step=None):
    sess.run([src_test_iterator.initializer])

    n_batch = 0
    t0 = time.time()
    probs = []
    if result_dir is not None:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        src_test_file = args.test_data[A].split("/")[-1]
        step = str(step) + "_" if step is not None else ''
        result_save_path1 = result_dir + "/" + step + src_test_file + '-' + str(B) + '.tsf'
        result_save_path2 = result_dir + "/" + step + src_test_file + '.tsf'
        print("Result save path:" + result_save_path1 + ", " + result_save_path2)
        dst_f1 = open(result_save_path1, "w")
        dst_f2 = open(result_save_path2, "w")

    while True:
        try:
            n_batch += 1
            src = sess.run(src_test_next)
            feed_dict = {
                model.input_ids: src["ids"],
                model.input_length: src["length"],
            }
            predictions = sess.run(model.predictions, feed_dict=feed_dict)
            probs.extend(predictions["log_probs"].reshape(-1))

            if result_dir is not None:
                def save_prediction():
                    batch_size = predictions["tokens"].shape[0]
                    decode_width = predictions["tokens"].shape[1]
                    src_tokens = src_vocab_rev.lookup(tf.cast(src["ids"], tf.int64))
                    src_tokens = sess.run(src_tokens)
                    for i in range(batch_size):
                        src_tokens_ = src_tokens[i][:src["length"][i]]
                        src_sent = " ".join(src_tokens_)
                        for j in range(decode_width):
                            log_probs = predictions["log_probs"][i][j]
                            pred_tokens = predictions["tokens"][i][j][:predictions["length"][i][j] - 1]  # ignore </s>
                            pred_sent = " ".join(pred_tokens)
                            dst_f1.write("%s\t%s\t%s\n" % (src_sent, pred_sent, log_probs))
                            if j == 0:
                                dst_f2.write("%s\n" % pred_sent)

                save_prediction()

        except tf.errors.OutOfRangeError:  # next epoch
            print("INFERENCE---Total N batch:{}\tGenerate probs:{}\tCost time:{}".format(
                n_batch, np.mean(probs), time.time() - t0))
            break
    if result_dir is not None:
        dst_f1.close()
        dst_f2.close()
    return np.mean(probs), result_save_path2


def get_nmt_direction(direction="0-1"):
    if direction == "0-1":
        A = 0
        B = 1
    else:
        A = 1
        B = 0
    return A, B


if __name__ == "__main__":
    args = load_nmt_arguments()

    # === Get translation direction
    A, B = get_nmt_direction(args.nmt_direction)
    print("A=%s, B=%s" % (A, B))

    # ===  Build vocab and load data
    if not os.path.isfile(args.global_vocab_file):
        build_vocab_from_file(args.train_data, args.global_vocab_file)
    vocab, vocab_size = load_vocab(args.global_vocab_file)
    src_vocab = tgt_vocab = vocab
    src_vocab_size = tgt_vocab_size = vocab_size
    print('Vocabulary size:src:%s' % vocab_size)

    vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
        args.global_vocab_file,  # target vocabulary file(each lines has a word)
        vocab_size=vocab_size - constants.NUM_OOV_BUCKETS,
        default_value=constants.UNKNOWN_TOKEN)
    src_vocab_rev = tgt_vocab_rev = vocab_rev

    with tf.device("/cpu:0"):  # Input pipeline should always be placed on the CPU.
        print("Use x'->y to update model f(x->y)")
        train_iterator = load_paired_dataset(args.tsf_train_data[B], args.train_data[B],
                                             src_vocab, tgt_vocab, batch_size=args.batch_size)
        dev_iterator = load_paired_dataset(args.tsf_dev_data[B], args.dev_data[B],
                                           src_vocab, tgt_vocab, batch_size=args.batch_size)

        src_test_iterator = load_dataset(args.test_data[A], src_vocab, mode=constants.INFER)

        train_next_op = train_iterator.get_next()
        dev_next_op = dev_iterator.get_next()
        src_test_next_op = src_test_iterator.get_next()

    # === Create session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=tf_config)

    # === Train
    if args.mode == "train":
        # Prepare for model saver
        print("Prepare for model saver")
        args.nmt_model_save_dir = "%s/%s-%s/" % (args.nmt_model_save_dir, A, B)
        print("Model save dir:", args.nmt_model_save_dir)
        if not os.path.exists(args.nmt_model_save_dir):
            os.makedirs(args.nmt_model_save_dir)

        dump_args_to_yaml(args, args.nmt_model_save_dir)

        # Initial and build model
        train_model = create_model(sess, args, src_vocab_size, tgt_vocab_size, src_vocab_rev, tgt_vocab_rev,
                                   mode=constants.TRAIN, direction=args.nmt_direction)
        infer_model = create_model(sess, args, src_vocab_size, tgt_vocab_size, src_vocab_rev, tgt_vocab_rev,
                                   mode=constants.INFER, direction=args.nmt_direction, reuse=True)
        train(train_model)

    # === Inference
    elif args.mode == "inference":
        print("Prepare for model saver")
        final_model_save_path = "%s%s-%s/" % (args.nmt_model_save_dir, A, B)
        print("Model save path:", final_model_save_path)
        eval_model = create_model(sess, args, src_vocab_size, tgt_vocab_size, src_vocab_rev, tgt_vocab_rev,
                                  mode=constants.EVAL, direction=args.nmt_direction, load_pretrained_model=True,
                                  model_save_dir=final_model_save_path)
        infer_model = create_model(sess, args, src_vocab_size, tgt_vocab_size, src_vocab_rev, tgt_vocab_rev,
                                   mode=constants.INFER, direction=args.nmt_direction, reuse=True)
        inference(infer_model, A=A, B=B, sess=sess, args=args, src_test_iterator=src_test_iterator,
                  src_test_next=src_test_iterator.get_next(),
                  result_dir=args.tsf_result_dir, src_vocab_rev=src_vocab_rev)

    # === Final inference (after dualRL training)
    elif args.mode == "final_inference":  # todo: check data loader
        print("Prepare for model saver")
        final_model_save_path = "%s%s-%s/" % (args.final_model_save_dir, A, B)

        args.decode_type = constants.RANDOM

        print("Model save path:", final_model_save_path)
        eval_model = create_model(sess, args, src_vocab_size, tgt_vocab_size, src_vocab_rev, tgt_vocab_rev,
                                  mode=constants.EVAL, direction=args.nmt_direction, load_pretrained_model=True,
                                  model_save_dir=final_model_save_path)
        infer_model = create_model(sess, args, src_vocab_size, tgt_vocab_size, src_vocab_rev, tgt_vocab_rev,
                                   mode=constants.INFER, direction=args.nmt_direction, reuse=True)
        print("INFERENCE TYPE:%s" % args.decode_type)
        inference(infer_model, A=A, B=B, sess=sess, args=args, src_test_iterator=src_test_iterator,
                  src_test_next=src_test_iterator.get_next(),
                  result_dir=args.final_tsf_result_dir, src_vocab_rev=src_vocab_rev)
