from nmt.options import *
from common_options import *
import ast


def add_dual_arguments(parser):
    # Arguments for training
    parser.add_argument("--n_epoch", default=10, type=int, help="Max n epoch during dual training.")
    parser.add_argument("--learning_rate", default=0.00001, type=float, help="Learning rate")
    parser.add_argument("--change_per_step", default=10, type=int, help="Change dual training direction per n steps")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size of dual training (You can enlarge the batch_size if you have enough memory).")

    # Arguments for evaluation and model saving
    parser.add_argument("--save_per_step", default=1000, type=int, help="Save model per n steps")
    parser.add_argument("--eval_step", default=100, type=int, help="Evaluate model.")

    # Arguments for calculating reward
    parser.add_argument("--use_baseline", type=ast.literal_eval, default=True,
                        help="Use baseline in reward calculation, input should be either 'True' or 'False'.")
    parser.add_argument("--normalize_reward", type=ast.literal_eval, default=False,   # important !!
                        help="normalize reward or not, input should be either 'True' or 'False'.")

    # Arguments for anneal teacher-forcing (MLE)
    parser.add_argument("--teacher_forcing", nargs='+', default=["back_trans"],
                        help="Corpus used in teacher forcing (MLE), must in [`pseudo`, `back_trans`")
    parser.add_argument("--MLE_anneal", action='store_true', help="Anneal the use of pseudo data via MLE")
    parser.add_argument("--anneal_rate", default=1.1, type=float, help="The decay rate.")
    parser.add_argument("--anneal_steps", default=1000, type=int, help="Increase gap")
    parser.add_argument("--anneal_initial_gap", default=1, type=int, help="Initial gap value.")
    parser.add_argument("--anneal_max_gap", default=100, type=int, help="Max gap value (Smaller can be more stable).")


def load_dual_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_dual_arguments(parser)
    add_common_arguments(parser)
    add_optimizer_arguments(parser)
    parser = parser.parse_args()
    return parser
