#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2018-12-04 10:46
# * Last modified : 2018-12-04 10:46
# * Filename      : main.py
# * Description   :
'''
'''
# **********************************************************
from train_log_deeplog import train_by_templates
from detect_log_deeplog import detect_by_templates
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', help='train_file.', type=str, default='train_log_seqence.txt')
    parser.add_argument('--seq_length', help='seq_length.', type=int, default=10)
    parser.add_argument('--template_index_map_path', help='template_index_map_path.', type=str, default='template_to_int.txt')
    parser.add_argument('--test_file', help='test_file.', type=str, default='test_log_seqence.txt')
    parser.add_argument('--n_candidates', help='n_candidates.', type=int, default=30)
    parser.add_argument('--windows_size', help='windows_size.', type=int, default=3)
    parser.add_argument('--step_size', help='step_size.', type=int, default=1)
    parser.add_argument('--model_filename', help='model_filename.', type=str, default='')
    parser.add_argument('--model_dir', help='model_dir.', type=str, default='weights/test/')
    parser.add_argument('--result_file', help='result_file.', type=str, default='precision_recall.txt')
    parser.add_argument('--label_file', help='label_file.', type=str, default='label.y')
    args = parser.parse_args()

    para_train = {
        'train_file': args.train_file,
        'seq_length':args.seq_length,
        'model_dir': args.model_dir,
        'template_index_map_path':args.template_index_map_path
        }

    para_test = {
        'test_file': args.test_file,
        'seq_length':args.seq_length,
        'n_candidates': args.n_candidates,
        'windows_size': args.windows_size,
        'step_size':args.step_size,
        'model_dir': args.model_dir,
        'model_filename': args.model_filename,
        'template_index_map_path':args.template_index_map_path,
        'result_file':args.result_file,
        'label_file':args.label_file
        }

    train_by_templates(para_train)
    detect_by_templates(para_test)




    

