from common_options import *


def add_cls_arguments(parser):
    parser.add_argument("--mode", default="train", help="Mode: train or eval.")
    parser.add_argument("--n_epoch", default=10, help="Max n epoch during textcnn training.")
    parser.add_argument("--batch_size", default=16, help="Batch size of training")
    parser.add_argument("--keep_prob", default=0.5, type=float, help="Keep prob in dropout.")
    parser.add_argument("--steps_per_checkpoint", default=100, help="Print log gap.")
    parser.add_argument('--filter_sizes', type=str, default='1,2,3,4,5')
    parser.add_argument('--n_filters', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--load_model', action="store_true")
    parser.add_argument("--log_path", default='{}/tmp/cls_result_{}.txt'.format(base_path, dataset), help="Logs' path.")


def load_cls_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser)
    add_cls_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = load_cls_arguments()
    print("Save st: %s" % args.cls_model_save_dir)
    dump(args, open(os.path.join(args.cls_model_save_dir, 'conf.yaml'), 'w'))
    args = load(open(os.path.join(args.cls_model_save_dir, 'conf.yaml')))
    print(args.batch_size)
