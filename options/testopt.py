import argparse

def _get_test_opt():
    parser = argparse.ArgumentParser(description = 'Evaluate performance of SARPN on NYU-D v2 test set')
    parser.add_argument('--backbone', default='SENet154', help='select a network as backbone')
    parser.add_argument('--testlist_path', required=True, help='the path of testlist')
    parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--root_path', required=True, help="the root path of dataset")
    parser.add_argument('--loadckpt', required=True, help="the path of the loaded model")
    parser.add_argument('--threshold', type=float, default=1.0, help="threshold of the pixels on edges")
    parser.add_argument('--pretrained_dir', type=str, required=True, help="the path of pretrained models")
    # parse arguments
    return parser.parse_args()
