#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2018-10-07 22:58
# * Last modified : 2019-01-19 13:24
# * Filename      : train_vector_deeplog.py
# * Description   :
'''
'''
# **********************************************************
# Larger LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import sklearn
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import argparse
from template2vec import Template2Vec
from sklearn import preprocessing
import time


def createDir(path):
    import os
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(path)

def Normalize(data):
    m = numpy.mean(data)
    mx = max(data)
    mn = min(data)
    return [(float(i) - m) / (mx - mn) for i in data]

def train_by_vector(para):
    t1 =time.time()
    filename = para['train_file']
    seq_length = para['seq_length']# l1,l2...l10 -> l_next
    model_dir = para['model_dir']
    template_num = para['template_num']
    template2Vec_file = para['template2Vec_file']
    tempalte_file = para['template_file']
    count_matrix_flag = para['count_matrix']
    temp2Vec = Template2Vec(template2Vec_file, tempalte_file)


    createDir(model_dir)
    template_index_map_path = para['template_index_map_path']#保存模板号与向量里数值的对应关系
    raw_text = []
    with open(filename) as IN:
        for line in IN:
            l=line.strip().split()
            if l[1] != '-1':
                raw_text.append(l[1]) #先将template_index添加进来

    if template_num == 0:
        # 如果template_num为0，则根据模板序列文件来生成映射create mapping of unique chars to integers
        chars = sorted(list(set(raw_text)))
        template_to_int = dict((c, i) for i, c in enumerate(chars))
        f = open(template_index_map_path,'w')
        for k in template_to_int:
            f.writelines(str(k)+' '+str(template_to_int[k])+'\n')
        f.close()
    else:
        # 如果template_num不为0，则根据其构造映射,int(用于lstm)从0开始，template（即模板号）从1开始
        template_to_int = dict((str(i+1), i) for i in range(template_num))


    # summarize the loaded data
    n_chars = len(raw_text)
    n_templates = len(template_to_int)
    print ("length of log sequence: ", n_chars)
    print ("# of templates: ", n_templates)

    # prepare the dataset of input to output pairs encoded as integers
    dataX = []
    dataY = []
    vectorX = []
    vectorY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        #dataX.append([template_to_int[char] for char in seq_in])
        #dataY.append(template_to_int[seq_out])
        temp_list = []
        for seq in seq_in:
            if count_matrix_flag == 0:
                #不拼接，直接用拼接template vector
                temp_list.append(list(temp2Vec.model[seq]))
            else:
                #拼接template vector和count vector
                cur_count_vector = [0 for i in range(n_templates)]
                for t in seq_in:
                    cur_index = template_to_int[t]
                    cur_count_vector[cur_index]+=1
                #extend 没有返回值，但会在已存在的列表中添加新的列表内容
                l =list(temp2Vec.model[seq])
                l.extend(cur_count_vector)
                temp_list.append(l)

        vectorX.append(numpy.array(temp_list))
        vectorY.append(temp2Vec.model[seq_out])

    n_patterns = len(vectorX)
    print ("# of patterns:", n_patterns)

    #normalize

    #print("len(vectorX[1])", type(vectorX[1]))
    # reshape X to be [samples, time steps, features]
    if count_matrix_flag == 0:
        X = numpy.reshape(vectorX, ( -1, seq_length, temp2Vec.dimension)) #
    else:
        X = numpy.reshape(vectorX, ( -1, seq_length, temp2Vec.dimension + n_templates))
    y = numpy.reshape(vectorY,(-1,temp2Vec.dimension))



    '''
    # normalize 我们需要将整数重新缩放到0到1的范围，以使默认情况下使用sigmoid激活函数的LSTM网络更容易学习模式
    X = X / float(n_templates) #X已经向量化了，不需要再缩放
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY, num_classes = n_templates, dtype='float16')#转成one hot,维度为总的tags数量
    '''

    # define the LSTM model
    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(temp2Vec.dimension))
    #model.add(Dense(temp2Vec.dimension, activation='softmax'))
    model.compile(loss="mse",optimizer="adam") #loss:mse

    # define the checkpoint
    if count_matrix_flag == 0:
        s = 'only_vector'
    else:
        s = 'contact_matrix'
    filepath = model_dir+"log_weights-"+ s +"-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model
    model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
    t2 = time.time()
    print('training time:',(t2-t1)/60,'mins')
    return n_templates



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', help='train_file.', type=str, default='../../middle/bgl_log_20w.seq')
    parser.add_argument('-seq_length', help='seq_length.', type=int, default=10)
    parser.add_argument('-model_dir', help='网络参数的输出文件夹', type=str, default='../weights/vector_deeplog/')
    parser.add_argument('-template_num', help='若为0，则根据输入文件统计，否则，根据输入确定。默认0', type=int, default=0)
    parser.add_argument('-template2Vec_file', help='template2Vec_file', type=str, default='../../model/bgl_log_20w.template_vector')
    parser.add_argument('-count_matrix', help='默认为0。1表示统计count_matrix，0不统计',type = int, default = 0)
    parser.add_argument('-template_file', help='template_file', type=str, default='../../middle/bgl_log_20w.template')
    args = parser.parse_args()

    para_train = {
        'train_file': args.train_file,
        'seq_length':args.seq_length,
        'model_dir': args.model_dir,
        'template_index_map_path':args.train_file+'_map',
        'template_num': args.template_num,
        'template2Vec_file': args.template2Vec_file,
        'template_file': args.template_file,
        'count_matrix': args.count_matrix
        }

    n_templates = train_by_vector(para_train)
    from keras import backend as K
    K.clear_session()
    print('training has finished')



