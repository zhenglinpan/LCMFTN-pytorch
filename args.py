import argparse

parser = argparse.ArgumentParser()

# dataset setting
parser.add_argument("--video_dir", type=str, default='./dataset')
parser.add_argument("--video_root", type=str, default='./dataset/video')
parser.add_argument("--dataset_root", type=str, default='./dataset')
parser.add_argument("--models_root", type=str, default='./models')

# training setting
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--end_epoch", type=int, default=100)
parser.add_argument("--decay_epoch", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--input_height", type=int, default=256)
parser.add_argument("--input_width", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--s_channel", type=float, default=1)
parser.add_argument("--c_channel", type=float, default=3)
parser.add_argument("--n_cpu", type=float, default=4)
args = parser.parse_args()
print(args)