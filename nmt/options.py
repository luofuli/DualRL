import argparse
from common_options import *


def add_optimizer_arguments(parser):
    parser.add_argument("--optimizer", default="AdamOptimizer", help="The name of the optimizer class.")
    parser.add_argument("--decay_type", default="exponential_decay", help="The type of decay.")
    parser.add_argument("--decay_rate", default=0.9, type=float, help="The decay rate.")
    parser.add_argument("--decay_steps", default=10000, type=int, help="Decay every this many steps.")
    parser.add_argument("--clip_gradients", default=1.0, type=float, help="Maximum gradients norm (default: 1.0).")
    parser.add_argument("--regularization", help="A dict of Weights regularization penalty type and scale.")


def add_nmt_arguments(parser):
    parser.add_argument("--mode", default="train", help="train, inference, final_inference")
    parser.add_argument("--nmt_direction", default="1-0", help="Translation direction(`0-1 or 1-0`.)")
    parser.add_argument("--n_epoch", default=5, type=int, help="Max n epoch during training.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size of training.")
    parser.add_argument("--maximum_iterations", default=100, type=int, help="Maximum decoding iterations (default: 100).")
    parser.add_argument("--initializer", default="uniform_unit_scaling", help="Initializer for model.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")

    parser.add_argument("--encoder_decoder_type", default="bilstm", help="Type of decode: `bilstm`, `transformer`")
    parser.add_argument("--encoder_units", default=256, type=int, help="encoder unit size.")
    parser.add_argument("--decoder_units", default=256, type=int, help="decoder unit size.")
    parser.add_argument("--n_layer", default=1, type=int, help="rnn layer.")
    parser.add_argument("--decode_type", default="greedy", help="Type of decode: `greedy`, `random`, `beam`.")
    parser.add_argument("--decode_width", default=10, type=int, help="Width of the random or beam search.")

    add_optimizer_arguments(parser)


def load_nmt_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser)
    add_nmt_arguments(parser)
    parser = parser.parse_args()
    return parser

