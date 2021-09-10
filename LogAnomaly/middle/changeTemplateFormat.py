#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2019-01-20 19:02
# * Last modified : 2019-01-20 19:12
# * Filename      : changeTemplateFormat.py
# * Description   :
'''
'''
# **********************************************************
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-input',help='input file name ',type = str, default ='bgl_log_20w.template')
arg = parser.parse_args()
input = arg.input
output = input+'_for_training'

f = open(output,'w')
with open(input) as IN:
    for line in IN:
        f.writelines(line.strip()+' ')
f.writelines('\n')

print('input:',input,'output:',output)
