"""Standalone script to generate word vocabularies from monolingual corpus."""

import argparse

from utils import constants
from opennmt import tokenizers
from opennmt import utils
import tensorflow as tf


def build_vocab_from_file(src_file, save_path, min_frequency=5, size=0, without_sequence_tokens=False):
    """
    Generate word vocabularies from monolingual corpus.
    :param src_file: Source text file.
    :param save_path: Output vocabulary file.
    :param min_frequency: Minimum word frequency.  # for yelp and amazon, min_frequency=5
    :param size: Maximum vocabulary size. If = 0, do not limit vocabulary.
    :param without_sequence_tokens: If set, do not add special sequence tokens (start, end) in the vocabulary.
    :return: No return.
    """

    special_tokens = [constants.PADDING_TOKEN]
    if not without_sequence_tokens:
        special_tokens.append(constants.START_OF_SENTENCE_TOKEN)
        special_tokens.append(constants.END_OF_SENTENCE_TOKEN)

    vocab = utils.Vocab(special_tokens=special_tokens)
    if isinstance(src_file, list):
        for data_file in src_file:
            vocab.add_from_text(data_file)
    else:
        vocab.add_from_text(src_file)
    vocab = vocab.prune(max_size=size, min_frequency=min_frequency)
    vocab.serialize(save_path)


def load_vocab(vocab_file):
    """Returns a lookup table and the vocabulary size."""

    def count_lines(filename):
        """Returns the number of lines of the file :obj:`filename`."""
        with open(filename, "rb") as f:
            i = 0
            for i, _ in enumerate(f):
                pass
            return i + 1

    vocab_size = count_lines(vocab_file) + 1  # Add UNK.
    vocab = tf.contrib.lookup.index_table_from_file(
        vocab_file,
        vocab_size=vocab_size - 1,
        num_oov_buckets=1)
    return vocab, vocab_size


def load_vocab_dict(vocab_file):
    """Returns a dictionary and the vocabulary size."""

    def count_lines(filename):
        """Returns the number of lines of the file :obj:`filename`."""
        with open(filename, "rb") as f:
            i = 0
            for i, _ in enumerate(f):
                pass
            return i + 1

    # vocab_size = count_lines(vocab_file) + 1  # Add UNK.

    vocab_dict = {}
    vocab_size = 0
    with open(vocab_file) as f:
        for line in f:
            word = line.strip()
            vocab_dict[word] = vocab_size
            vocab_size += 1
    vocab_dict[constants.UNKNOWN_TOKEN] = vocab_size
    vocab_size += 1
    return vocab_dict, vocab_size


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "data", nargs="+",
        help="Source text file.")
    parser.add_argument(
        "--save_vocab", required=True,
        help="Output vocabulary file.")
    parser.add_argument(
        "--min_frequency", type=int, default=1,
        help="Minimum word frequency.")
    parser.add_argument(
        "--size", type=int, default=0,
        help="Maximum vocabulary size. If = 0, do not limit vocabulary.")
    parser.add_argument(
        "--without_sequence_tokens", default=False, action="store_true",
        help="If set, do not add special sequence tokens (start, end) in the vocabulary.")
    tokenizers.add_command_line_arguments(parser)
    args = parser.parse_args()

    tokenizer = tokenizers.build_tokenizer(args)

    special_tokens = [constants.PADDING_TOKEN]
    if not args.without_sequence_tokens:
        special_tokens.append(constants.START_OF_SENTENCE_TOKEN)
        special_tokens.append(constants.END_OF_SENTENCE_TOKEN)

    vocab = utils.Vocab(special_tokens=special_tokens)
    for data_file in args.data:
        vocab.add_from_text(data_file, tokenizer=tokenizer)
    vocab = vocab.prune(max_size=args.size, min_frequency=args.min_frequency)
    vocab.serialize(args.save_vocab)


def test_vocab():
    import tensorflow as tf
    import numpy as np
    import os
    from common_options import load_common_arguments

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Load global vocab
    args = load_common_arguments()
    global_vocab, global_vocab_size = load_vocab(args.global_vocab_file)

    vocab, vocab_size = load_vocab_dict(args.global_vocab_file)

    assert global_vocab_size == vocab_size

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        i = 0
        ks = vocab.keys()
        vs = vocab.values()

        v1 = sess.run(global_vocab.lookup(tf.convert_to_tensor(ks)))
        for i in range(len(vs)):
            assert vs[i] == v1[i]


if __name__ == "__main__":
    main()
    test_vocab()
