import tensorflow as tf
import numpy as np
import sys
import re
import time
sys.path.append('..')
from utils.data import load_dataset, load_paired_dataset
from utils.vocab import load_vocab
from nmt.nmt import create_model as nmt_create_model
from classifier.textcnn import create_model as cls_create_model
from common_options import *
from dual_options import load_dual_arguments
from utils import constants
from nmt.nmt import inference
from classifier.textcnn import evaluate_file as cls_evaluate_file
from utils.vocab import load_vocab_dict
from utils.evaluator import BLEUEvaluator
from utils.mid_data import process_mid_ids

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

safe_divide_constant = 1e-6
bleu_evaluator = BLEUEvaluator()

def main():
    # === Load arguments
    args = load_dual_arguments()
    dump_args_to_yaml(args, args.final_model_save_dir)

    cls_args = load_args_from_yaml(args.cls_model_save_dir)
    nmt_args = load_args_from_yaml(os.path.join(args.nmt_model_save_dir, '0-1'))
    nmt_args.learning_rate = args.learning_rate  # a smaller learning rate for RL
    min_seq_len = min(int(max(re.findall("\d", cls_args.filter_sizes))), args.min_seq_len)

    # === Load global vocab
    word2id, word2id_size = load_vocab_dict(args.global_vocab_file)
    global_vocab, global_vocab_size = load_vocab(args.global_vocab_file)
    print("Global_vocab_size: %s" % global_vocab_size)
    global_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
        args.global_vocab_file,
        vocab_size=global_vocab_size - constants.NUM_OOV_BUCKETS,
        default_value=constants.UNKNOWN_TOKEN)
    src_vocab = tgt_vocab = global_vocab
    src_vocab_size = tgt_vocab_size = global_vocab_size
    src_vocab_rev = tgt_vocab_rev = global_vocab_rev

    # === Create session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=tf_config)

    # === Initial and build model
    cls = cls_create_model(sess, cls_args, global_vocab_size, mode=constants.EVAL, load_pretrained_model=True)

    nmts_train = []
    nmts_random_infer = []
    nmts_greedy_infer = []
    train_data_next = []
    dev_data_next = []
    test_data_next = []
    train_iterators = []
    test_iterators = []
    paired_train_iterators = []
    paired_train_data_next = []
    final_model_save_paths = []

    # === Define nmt model
    for A, B in [(0, 1), (1, 0)]:
        with tf.device("/cpu:0"):  # Input pipeline should always be placed on the CPU.
            src_train_iterator = load_dataset(args.train_data[A], src_vocab, mode=constants.TRAIN,
                                              batch_size=args.batch_size, min_seq_len=min_seq_len)
            src_dev_iterator = load_dataset(args.dev_data[A], src_vocab, mode=constants.EVAL, batch_size=500)
            src_test_iterator = load_dataset(args.test_data[A], src_vocab, mode=constants.EVAL, batch_size=500)
            # Use (X', Y) to produce pseudo parallel data
            paired_src_train_iterator = load_paired_dataset(args.tsf_train_data[B], args.train_data[B],
                                                            src_vocab, tgt_vocab, batch_size=args.batch_size,
                                                            min_seq_len=min_seq_len)

            src_train_next_op = src_train_iterator.get_next()  # To avoid frequent calls of `Iterator.get_next()`
            src_dev_next_op = src_dev_iterator.get_next()
            src_test_next_op = src_test_iterator.get_next()
            src_paired_train_next_op = paired_src_train_iterator.get_next()

            train_data_next.append(src_train_next_op)
            dev_data_next.append(src_dev_next_op)
            test_data_next.append(src_test_next_op)
            paired_train_data_next.append(src_paired_train_next_op)

            train_iterators.append(src_train_iterator)
            test_iterators.append(src_test_iterator)
            paired_train_iterators.append(paired_src_train_iterator)

        direction = "%s-%s" % (A, B)
        nmt_args.sampling_probability = 0.5

        # == Define train model
        nmt_train = nmt_create_model(sess, nmt_args, src_vocab_size, tgt_vocab_size, src_vocab_rev, tgt_vocab_rev,
                                     mode=constants.TRAIN, direction=direction, load_pretrained_model=True)

        # == Define inference model
        decode_type_before = nmt_args.decode_type

        nmt_args.decode_type = constants.RANDOM
        nmt_random_infer = nmt_create_model(sess, nmt_args, src_vocab_size, tgt_vocab_size, src_vocab_rev,
                                            tgt_vocab_rev, mode=constants.INFER, direction=direction, reuse=True)

        nmt_args.decode_type = constants.GREEDY
        nmt_greedy_infer = nmt_create_model(sess, nmt_args, src_vocab_size, tgt_vocab_size, src_vocab_rev,
                                            tgt_vocab_rev, mode=constants.INFER, direction=direction, reuse=True)

        nmt_args.decode_type = decode_type_before  # restore to previous setting

        nmts_train.append(nmt_train)
        nmts_random_infer.append(nmt_random_infer)
        nmts_greedy_infer.append(nmt_greedy_infer)

        # == Prepare for model saver
        print("Prepare for model saver")
        final_model_save_path = "%s/%s-%s/" % (args.final_model_save_dir, A, B)
        if not os.path.exists(final_model_save_path):
            os.makedirs(final_model_save_path)
        print("Model save path:", final_model_save_path)
        final_model_save_paths.append(final_model_save_path)

    # === Start train
    n_batch = -1
    global_step = -1
    A = 1
    B = 0
    G_scores = []

    for i in range(args.n_epoch):
        print("Epoch:%s" % i)
        sess.run([train_iterators[A].initializer])
        sess.run([train_iterators[B].initializer])
        sess.run([paired_train_iterators[A].initializer])
        sess.run([paired_train_iterators[B].initializer])

        while True:
            n_batch += 1
            global_step += 1
            if n_batch % args.eval_step == 0:
                print('===== Start (N_batch: %s, Steps: %s): Evaluate on test datasets ===== ' % (n_batch, global_step))
                _, dst_f_A = inference(nmts_greedy_infer[A], sess=sess, args=nmt_args, A=A, B=B,
                                       src_test_iterator=test_iterators[A], src_test_next=test_data_next[A],
                                       src_vocab_rev=src_vocab_rev, result_dir=args.final_tsf_result_dir,
                                       step=global_step)
                _, dst_f_B = inference(nmts_greedy_infer[B], sess=sess, args=nmt_args, A=B, B=A,
                                       src_test_iterator=test_iterators[B], src_test_next=test_data_next[B],
                                       src_vocab_rev=src_vocab_rev, result_dir=args.final_tsf_result_dir,
                                       step=global_step)
                t0 = time.time()
                # calculate accuracy score
                senti_acc = cls_evaluate_file(sess, cls_args, word2id, cls, [dst_f_A, dst_f_B], index_list=[B, A])
                # calculate bleu score
                bleu_score_A = bleu_evaluator.score(args.reference[A], dst_f_A)
                bleu_score_B = bleu_evaluator.score(args.reference[B], dst_f_B)
                bleu_score = (bleu_score_A + bleu_score_B) / 2

                G_score = np.sqrt(senti_acc * bleu_score)
                H_score = 2/(1/senti_acc + 1/bleu_score)
                G_scores.append(G_score)
                print("%s-%s_Test(Batch:%d)\tSenti:%.3f\tBLEU(4ref):%.3f(A:%.3f+B:%.3f)"
                      "\tG-score:%.3f\tH-score:%.3f\tCost time:%.2f" %
                      (A, B, n_batch, senti_acc, bleu_score, bleu_score_A, bleu_score_B,
                       G_score, H_score, time.time() - t0))
                print('=====  End (N_batch: %s, Steps: %s): Evaluate on test datasets ====== ' % (n_batch, global_step))

            if n_batch % args.save_per_step == 0:
                print("=== Save model at dir:", final_model_save_paths[A], final_model_save_paths[B])
                nmts_train[A].saver.save(sess, final_model_save_paths[A], global_step=global_step)
                nmts_train[B].saver.save(sess, final_model_save_paths[B], global_step=global_step)

            if n_batch % args.change_per_step == 0:
                A, B = B, A
                print("============= Change to train model {}-{} at {} steps ==============".format(A, B, global_step))

            try:
                t0 = time.time()
                src = sess.run(train_data_next[A])  # get real data!!
                batch_size = np.shape(src["ids"])[0]
                decode_width = nmt_args.decode_width

                tile_src_ids = np.repeat(src["ids"], decode_width, axis=0)  # [batch_size*sample_size],
                tile_src_length = np.repeat(src['length'], decode_width, axis=0)
                tile_src_ids_in = np.repeat(src["ids_in"], decode_width, axis=0)
                tile_src_ids_out = np.repeat(src["ids_out"], decode_width, axis=0)
                tile_src_ids_in_out = np.repeat(src["ids_in_out"], decode_width, axis=0)

                random_predictions = sess.run(nmts_random_infer[A].predictions,
                                              feed_dict={nmts_random_infer[A].input_ids: src['ids'],
                                                         nmts_random_infer[A].input_length: src['length']})
                assert np.shape(random_predictions["ids"])[1] == nmt_args.decode_width
                mid_ids_log_prob = np.reshape(random_predictions["log_probs"], -1)
                mid_ids, mid_ids_in, mid_ids_out, mid_ids_in_out, mid_ids_length = \
                    process_mid_ids(random_predictions["ids"], random_predictions["length"],
                                   min_seq_len, global_vocab_size)

                greedy_predictions = sess.run(nmts_greedy_infer[A].predictions,
                                              feed_dict={nmts_greedy_infer[A].input_ids: src['ids'],
                                                         nmts_greedy_infer[A].input_length: src['length']})
                assert np.shape(greedy_predictions["ids"])[1] == 1
                mid_ids_bs, mid_ids_in_bs, mid_ids_out_bs, mid_ids_in_out_bs, mid_ids_length_bs = \
                    process_mid_ids(greedy_predictions["ids"], greedy_predictions["length"],
                                   min_seq_len, global_vocab_size)

                # Get style reward from classifier
                cls_probs = sess.run(cls.probs, feed_dict={cls.x: mid_ids, cls.dropout: 1})
                y_hat = [p > 0.5 for p in cls_probs]  # 1 or 0
                cls_acu = [p == B for p in y_hat]  # accuracy: count the number of style B
                style_reward = np.array(cls_acu, dtype=np.float32)

                # Get content reward from backward reconstruction
                feed_dict = {
                    nmts_train[B].input_ids: mid_ids,
                    nmts_train[B].input_length: mid_ids_length,
                    nmts_train[B].target_ids_in: tile_src_ids_in,
                    nmts_train[B].target_ids_out: tile_src_ids_out,
                    nmts_train[B].target_length: tile_src_length
                }
                nmtB_loss = sess.run(nmts_train[B].loss_per_sequence, feed_dict=feed_dict)  # nmtB_loss = -log(prob)
                nmtB_reward = nmtB_loss * (-1)  # reward = log(prob) ==> bigger is better

                # Get baseline reward from backward reconstruction
                feed_dict = {
                    nmts_train[B].input_ids: mid_ids_bs,
                    nmts_train[B].input_length: mid_ids_length_bs,
                    nmts_train[B].target_ids_in: src["ids_in"],
                    nmts_train[B].target_ids_out: src["ids_out"],
                    nmts_train[B].target_length: src["length"]
                }
                nmtB_loss_bs = sess.run(nmts_train[B].loss_per_sequence, feed_dict=feed_dict)
                nmtB_reward_bs = nmtB_loss_bs * (-1)  # nmt baseline reward

                def norm(x):
                    x = np.array(x)
                    x = (x - x.mean()) / (x.std() + safe_divide_constant)
                    # x = x - x.min()  # to make sure > 0
                    return x

                def sigmoid(x, x_trans=0.0, x_scale=1.0, max_y=1, do_norm=False):
                    value = max_y / (1 + np.exp(-(x - x_trans) * x_scale))
                    if do_norm:
                        value = norm(value)
                    return value

                def norm_nmt_reward(x, baseline=None, scale=False):
                    x = np.reshape(x, (batch_size, -1))  # x is in [-16, 0]
                    dim1 = np.shape(x)[1]

                    if baseline is not None:
                        x_baseline = baseline  # [batch_size]
                    else:
                        x_baseline = np.mean(x, axis=1)  # [batch_size]
                    x_baseline = np.repeat(x_baseline, dim1)  # [batch_size*dim1]
                    x_baseline = np.reshape(x_baseline, (batch_size, dim1))

                    x_norm = x - x_baseline

                    if scale:
                        # x_norm = sigmoid(x_norm, x_scale=0.5)  # x_norm: [-12, 12] => [0, 1]
                        x_norm = sigmoid(x_norm)  # Sharper normalization, x_norm: [-6, 6] => [0, 1]
                    return x_norm.reshape(-1)

                if args.use_baseline:
                    content_reward = norm_nmt_reward(nmtB_reward, baseline=nmtB_reward_bs, scale=True)
                else:
                    content_reward = norm_nmt_reward(nmtB_reward, scale=True)

                # Calculate reward
                style_reward += safe_divide_constant
                content_reward += safe_divide_constant
                reward = (1+0.25) * style_reward * content_reward / (style_reward + 0.25 * content_reward)
                if args.normalize_reward:
                    reward = norm(reward)

                # == Update nmtA via policy gradient training
                feed_dict = {
                    nmts_train[A].input_ids: tile_src_ids,
                    nmts_train[A].input_length: tile_src_length,
                    nmts_train[A].target_ids_in: mid_ids_in,
                    nmts_train[A].target_ids_out: mid_ids_out,
                    nmts_train[A].target_length: mid_ids_length,
                    nmts_train[A].reward: reward
                }
                ops = [nmts_train[A].lr_loss,
                       nmts_train[A].loss,
                       nmts_train[A].loss_per_sequence,
                       nmts_train[A].retrain_op]
                nmtA_loss_final, nmtA_loss_, loss_per_sequence_, _ = sess.run(ops, feed_dict=feed_dict)

                # == Update nmtA with pseudo data
                if args.MLE_anneal:
                    gap = min(args.anneal_max_gap, int(args.anneal_initial_gap * np.power(args.anneal_rate,
                                                                          global_step / args.anneal_steps)))
                else:
                    gap = args.anneal_initial_gap

                if n_batch % gap == 0:
                    # Update nmtA using original pseudo data (used as pre-training)
                    # This is not a ideal way since the quality of the pseudo-parallel data is not acceptable for
                    # the later iterations of training.
                    # We highly recommend you adopt back translation to generate the pseudo-parallel data on-the-fly
                    if "pseudo" in args.teacher_forcing:
                        data = sess.run(paired_train_data_next[A])  # get real data!!
                        feed_dict = {
                            nmts_train[A].input_ids: data["ids"],
                            nmts_train[A].input_length: data["length"],
                            nmts_train[A].target_ids_in: data["trans_ids_in"],
                            nmts_train[A].target_ids_out: data["trans_ids_out"],
                            nmts_train[A].target_length: data["trans_length"],
                        }
                        nmtA_pse_loss_, _ = sess.run([nmts_train[A].loss, nmts_train[A].train_op], feed_dict=feed_dict)

                    # Update nmtB using pseudo data generated via back_translation (on-the-fly)
                    if "back_trans" in args.teacher_forcing:
                        feed_dict = {
                            nmts_train[B].input_ids: mid_ids_bs,
                            nmts_train[B].input_length: mid_ids_length_bs,
                            nmts_train[B].target_ids_in: src["ids_in"],
                            nmts_train[B].target_ids_out: src["ids_out"],
                            nmts_train[B].target_length: src["length"],
                        }
                        nmtB_loss_, _ = sess.run([nmts_train[B].loss, nmts_train[B].train_op], feed_dict=feed_dict)

            except tf.errors.OutOfRangeError as e:  # next epoch
                print("===== DualTrain: Total N batch:{}\tCost time:{} =====".format(n_batch, time.time() - t0))
                n_batch = -1
                break


if __name__ == "__main__":
    main()
