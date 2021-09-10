#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2019-01-09 19:30
# * Last modified : 2019-08-07 14:46
# * Filename      : removeAnomaly.py
# * Description   :
'''
'''
# **********************************************************
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-input_seq', type=str, default='bgl.seq')
parser.add_argument('-input_label', type=str, default='bgl.label')
# parser.add_argument('-output_data', type=str, default='bgl2_log_20w_normal')
# parser.add_argument('-output_label', type=str, default='bgl2_label_20w_nromal')
args = parser.parse_args()

normal_seq_f = open(args.input_seq+'_normal', 'w')
#不需要normal的lable
#normal_label_f = open(args.input_label+'_normal', 'w')
save_index = 0
total_index = 0
remove_index = 0
with open(args.input_seq) as fp1:
    with open(args.input_label) as fp2:
        for data, label in zip(fp1,fp2):
            label = label.strip()
            total_index += 1
            if label == '0':
                save_index+=1
                normal_seq_f.writelines(data)
                #normal_label_f.writelines(label+'\n')
            if label == '1':
                remove_index+=1
print('finished')



