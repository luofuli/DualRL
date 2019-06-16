import argparse
import os
from yaml import load, dump

base_path = os.getcwd()  # current working directory
base_path_ = base_path.split('/')
base_path = '/'.join(base_path_[:base_path_.index('DualRL') + 1])

dataset = "yelp"
# dataset = 'GYAFC'


def add_common_arguments(parser):
    # parser.add_argument("dataset", default="yelp")  # todo: add required=True

    # Original data path
    parser.add_argument("--train_data",
                        nargs=2,
                        default=["{}/data/{}/train.0".format(base_path, dataset),
                                 "{}/data/{}/train.1".format(base_path, dataset)],
                        help="Two train files (absolute path).")
    parser.add_argument("--dev_data",
                        nargs=2,
                        default=["{}/data/{}/dev.0".format(base_path, dataset),
                                 "{}/data/{}/dev.1".format(base_path, dataset)],
                        help="Two dev files (absolute path).")
    parser.add_argument("--test_data",
                        nargs="+",
                        default=["{}/data/{}/test.0".format(base_path, dataset),
                                 "{}/data/{}/test.1".format(base_path, dataset)],
                        help="Two test files (absolute path).")
    parser.add_argument("--reference",
                        nargs="+",
                        default=[[
                            "{}/references/{}/reference0.0".format(base_path, dataset),
                            "{}/references/{}/reference1.0".format(base_path, dataset),
                            "{}/references/{}/reference2.0".format(base_path, dataset),
                            "{}/references/{}/reference3.0".format(base_path, dataset)],
                            [
                                "{}/references/{}/reference0.1".format(base_path, dataset),
                                "{}/references/{}/reference1.1".format(base_path, dataset),
                                "{}/references/{}/reference2.1".format(base_path, dataset),
                                "{}/references/{}/reference3.1".format(base_path, dataset)]],
                        help="Two reference files (absolute path).")

    # Pseudo-data path for pre-training (as a warm start for RL)
    pseudo = 'template'
    parser.add_argument("--tsf_train_data",
                        nargs=2,
                        default=["{}/data/{}/tsf_{}/train.0.tsf".format(base_path, dataset, pseudo),
                                 "{}/data/{}/tsf_{}/train.1.tsf".format(base_path, dataset, pseudo)],
                        help="Two transfer-ed train files (absolute path).")
    parser.add_argument("--tsf_dev_data",
                        nargs=2,
                        default=["{}/data/{}/tsf_{}/dev.0.tsf".format(base_path, dataset, pseudo),
                                 "{}/data/{}/tsf_{}/dev.1.tsf".format(base_path, dataset, pseudo)],
                        help="Two transfer-ed dev files (absolute path).")
    parser.add_argument("--tsf_test_data",
                        nargs=2,
                        default=["{}/data/{}/tsf_{}/test.0.tsf".format(base_path, dataset, pseudo),
                                 "{}/data/{}/tsf_{}/test.1.tsf".format(base_path, dataset, pseudo)],
                        help="Two transfer-ed test files (absolute path).")

    # Hyperparameter for model
    parser.add_argument("--global_vocab_file",
                        default="{}/data/{}/vocab".format(base_path, dataset),
                        help="Total or global vocabulary file.")
    parser.add_argument("--min_seq_len",
                        default=5,
                        type=int,
                        help="Min sequence length.")
    parser.add_argument("--emb_dim",
                        default=300,
                        type=int,
                        help="The dimension of word embeddings.")

    # Model saved path
    parser.add_argument("--nmt_model_save_dir",
                        default="{}/tmp/model/{}/nmt_{}/".format(base_path, dataset, pseudo),
                        help="Model save dir.")
    parser.add_argument("--lm_model_save_dir",
                        default='{}/tmp/model/{}/lm/'.format(base_path, dataset),
                        help='Model save dir.')
    parser.add_argument("--cls_model_save_dir",
                        default='{}/tmp/model/{}/cls/'.format(base_path, dataset),
                        help='Model save dir.')
    parser.add_argument("--final_model_save_dir",
                        default="{}/tmp/model/{}/nmt_final/".format(base_path, dataset),
                        help="Final transfer model save dir")
    # Result saved path
    parser.add_argument("--tsf_result_dir",
                        default="{}/tmp/output/{}_{}".format(base_path, dataset, pseudo),
                        help="Transfer result dir (before dual training).")
    parser.add_argument("--final_tsf_result_dir",
                        default="{}/tmp/output/{}_final".format(base_path, dataset),
                        help="Final Transfer result dir (after dual training).")


def load_common_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser)
    return parser.parse_args()


def load_args_from_yaml(dir):
    args = load(open(os.path.join(dir, 'conf.yaml')))
    return args


def dump_args_to_yaml(args, dir):
    dump(args, open(os.path.join(dir, 'conf.yaml'), 'w'))
