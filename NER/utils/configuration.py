import os
import json
import argparse


def gen_config():
    parser = argparse.ArgumentParser()

    # parameters on data and models
    parser.add_argument('--START_TAG', type=str, default='<START>')
    parser.add_argument('--STOP_TAG', type=str, default='<STOP>')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--tag_to_ix', type=dict, default={'<START>': 0, '<STOP>': 1, 'O': 2,
                                                           'B_address': 3, 'I_address': 4,
                                                           'B_book': 5, 'I_book': 6,
                                                           'B_company': 7, 'I_company': 8,
                                                           'B_game': 9, 'I_game': 10,
                                                           'B_government': 11, 'I_government': 12,
                                                           'B_movie': 13, 'I_movie': 14,
                                                           'B_name': 15, 'I_name': 16,
                                                           'B_organization': 17, 'I_organization': 18,
                                                           'B_position': 19, 'I_position': 20,
                                                           'B_scene': 21, 'I_scene': 22})

    # parameters on training and validating
    parser.add_argument('--train_type', type=str, default='cross_validation')
    parser.add_argument('--k_fold', type=str, default=5)
    parser.add_argument('--ratio', type=list, default=[0.8, 0.2])
    parser.add_argument('--epochs_total', type=int, default=100)
    parser.add_argument('--epochs_current', type=int, default=0)
    parser.add_argument('--epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--log_iter_step', type=int, default=1)
    parser.add_argument('--log_epoch_step', type=int, default=1)

    # misc
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='BiLSTM_CRF')
    parser.add_argument('--model_save_path', type=str, default='models/model.pth')
    parser.add_argument('--checkpoint_save_path', type=str, default='checkpoints/model.tar')
    parser.add_argument('--train_data_path', type=str, default='data/train.json')
    parser.add_argument('--test_data_path', type=str, default='data/test.json')
    parser.add_argument('--train_log_filename', type=str, default='Log_train')
    parser.add_argument('--valid_log_filename', type=str, default='Log_valid')
    parser.add_argument('--test_result_save_path', type=str, default='checkpoint/result_test,txt')

    args = parser.parse_args()

    return args


def save_config(args, save_path):
    # args = parser.parse_args()
    with open(save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def load_config(load_path):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(load_path, 'r') as f:
        args.__dict__ = json.load(f)
    return args


if __name__ == '__main__':
    parser = gen_config()
    # save_config(parser, 'config.json')
    # config = load_config('config.json')
