import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--end_epoch", type=int, default=100)
parser.add_argument("--decay_epoch", type=int, default=50)
parser.add_argument("batch_size", type=int, default=1)
parser.add_argument("patch_size", type=int, default=192)
parser.add_argument("input_size", type=int, default=256)
parser.add_argument("lr", type=float, default=1e-4)
parser.add_argument("s_channel", type=float, default=1)
parser.add_argument("c_channel", type=float, default=3)

parser.add_argument("dataset_root", type=str, default='./dataset')
# parser.add_argument("", type=, default=)
parser.add_argument("dataset_root", type=str, default='./dataset')
parser.add_argument("model_dir", type=str, default='./models/')
args = parser.parse_args()
print(args)