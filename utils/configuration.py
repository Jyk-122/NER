import os
import json
import argparse


def gen_config():
    parser = argparse.ArgumentParser()

    # parameters on data and models
    parser.add_argument('--START_TAG', type=str, default='<SOS>')
    parser.add_argument('--STOP_TAG', type=str, default='<EOS>')
    parser.add_argument('--PAD_TAG', type=str, default='<PAD>')
    parser.add_argument('--if_padding', type=bool, default=True)
    parser.add_argument('--model_type', type=str, default='TENER')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--head_dims', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--attn_type', type=str, default='adatrans')
    parser.add_argument('--pos_embed', type=str, default=None)
    parser.add_argument('--after_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--fc_dropout', type=float, default=0.4)
    parser.add_argument('--tag_type', type=str, default='BIO')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--tag_to_ix', type=dict, default={ '<PAD>': 0,
                                                            '<SOS>': 1,
                                                            '<EOS>': 2,
                                                            'O': 3,
                                                            'B_address': 4,
                                                            'I_address': 5,
                                                            'B_book': 6,
                                                            'I_book': 7,
                                                            'B_company': 8,
                                                            'I_company': 9,
                                                            'B_game': 10,
                                                            'I_game': 11,
                                                            'B_government': 12,
                                                            'I_government': 13,
                                                            'B_movie': 14,
                                                            'I_movie': 15,
                                                            'B_name': 16,
                                                            'I_name': 17,
                                                            'B_organization': 18,
                                                            'I_organization': 19,
                                                            'B_position': 20,
                                                            'I_position': 21,
                                                            'B_scene': 22,
                                                            'I_scene': 23})

    # parameters on training and validating
    parser.add_argument('--train_type', type=str, default='hold_out')
    parser.add_argument('--k_fold', type=str, default=5)
    parser.add_argument('--ratio', type=list, default=[0.8, 0.2])
    parser.add_argument('--epochs_total', type=int, default=100)
    parser.add_argument('--epochs_current', type=int, default=0)
    parser.add_argument('--epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--log_iter_step', type=int, default=1)
    parser.add_argument('--log_epoch_step', type=int, default=1)

    # misc
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--model_save_path', type=str, default='models/model.pth')
    parser.add_argument('--checkpoint_save_path', type=str, default='checkpoints/model.tar')
    parser.add_argument('--train_data_path', type=str, default='data/train.json')
    parser.add_argument('--test_data_path', type=str, default='data/test.json')
    parser.add_argument('--train_log_filename', type=str, default='Log_train')
    parser.add_argument('--valid_log_filename', type=str, default='Log_valid')
    parser.add_argument('--test_result_save_path', type=str, default='checkpoint/result_test.txt')

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
