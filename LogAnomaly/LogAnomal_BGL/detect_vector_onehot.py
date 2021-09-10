#!/usr/bin/python
# -*- coding: UTF-8 -*-

# **********************************************************
# * Author        : Weibin Meng
# * Email         : m_weibin@163.com
# * Create time   : 2018-10-08 02:24
# * Last modified : 2019-01-24 15:20
# * Filename      : detect_vector_onehot.py
# * Description   :
'''

'''
# **********************************************************
# Load Larger LSTM network and generate text
import sys
from keras.layers import BatchNormalization
import math
from sklearn.metrics import precision_recall_fscore_support
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from getWindowsTime import getRawTime
import argparse
import os
from template2vec import Template2Vec

def findnewestfile(dir_path):
    filenames = os.listdir(dir_path)
    name_ = []
    time_ = []
    for filename in filenames:
        if 'DS' not in  filename and 'hdf5' in filename: #此处可以指定后缀名
            # print filename
            c_time = os.path.getctime(dir_path+filename)

            # print type(mtime)
            name_.append(dir_path+filename)
            time_.append(c_time)
            # print filename,mtime
    newest_file = name_[time_.index(max(time_))]
    #print(name_)
    #print(time_)
    # print('new file:',newest_file)
    return newest_file


def detect_by_vector(para):
    import time
    t1=time.time()

    filename = para['test_file']
    seq_length = para['seq_length'] # l1,l2...l10 -> l_next
    n_candidates = para['n_candidates']#top n probability of the next tag
    windows_size = para['windows_size']#hours
    step_size = para['step_size']#时间窗口的滑动步长，hours
    onehot = para['onehot'] #1表示统计使用onehot，0表示使用template2vec
    model_filename = para['model_filename']#训练好的参数
    model_dir = para['model_dir']  #模板数量，要与train的一致
    template_index_map_path = para['template_index_map_path']#保存模板号与向量里数值的对应关系
    result_file = para['result_file']
    template_num = para['template_num']
    label_file = para['label_file'] #label文件，本样例中从日志文件中抽取label
    template2Vec_file = para['template2Vec_file']
    tempalte_file = para['template_file']
    count_matrix_flag = para['count_matrix']
    temp2Vec = Template2Vec(template2Vec_file, tempalte_file)
    # prediction_file = 'detection_result' #保存top1的log key异常和时间窗口异常的结果

    #如果没有指定model_filename, 则从weight/文件夹中找出最新生成的文件
    if model_filename == '':
        model_filename = findnewestfile(model_dir)
        print('cur_model_filename',model_filename)


    template_to_int = {}
    int_to_template = {}
    if template_num == 0:
        # 如果template_num为0，则根据模板序列文件来生成映射create mapping of unique chars to integers
        with open(template_index_map_path) as IN:
            for line in IN:
                l = line.strip().split()
                c = l[0]
                i = int(l[1])
                template_to_int[c] = i
                int_to_template[i] = c
    else:
        # 如果template_num不为0，则根据其构造映射,int从0开始，char从1开始
        template_to_int = dict((str(i+1), i) for i in range(template_num))
        int_to_template = dict((i, str(i+1)) for i in range(template_num))

    raw_text = []
    raw_time_list = []
    raw_label_list = []
    with open(filename) as line_IN:
        with open(label_file) as label_IN:
            for line, label_line in zip(line_IN, label_IN):
                l=line.strip().split()
                if l[1] != '-1' and l[1] !='0' and l[1] in template_to_int:
                    raw_text.append(l[1])
                    raw_label_list.append(int(label_line.strip()))

                    #raw_text.append(l[1])
                    #raw_time_list.append(int(l[0]))

    # create mapping of unique chars to integers, and a reverse mapping
    chars = sorted(list(set(raw_text)))



    # summarize the loaded data
    n_chars = len(raw_text)
    n_templates = len(template_to_int)
    print ("length of log sequence: ", n_chars)
    print ("# of templates: ", n_templates)
    # prepare the dataset of input to output pairs encoded as integers
    #dataX = []
    #dataY = []
    #timeY = []
    charX = []
    label_list = []
    vectorX = []
    vectorY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        #time_out = raw_time_list[i + seq_length]
        label_out = raw_label_list[i + seq_length]
        #dataX.append([template_to_int[char] for char in seq_in])
        charX.append(seq_in)
        #dataY.append(template_to_int[seq_out])
        #timeY.append(time_out)
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
        vectorX.append(temp_list)
        vectorY.append(temp2Vec.model[seq_out])
        label_list.append(label_out)
    n_patterns = len(vectorX)
    print ("# of patterns: ", n_patterns)
    #split time into windows

    # reshape X to be [samples, time steps, features]
    if count_matrix_flag == 0:
        X = numpy.reshape(vectorX, ( -1, seq_length, temp2Vec.dimension)) #
    else:
        X = numpy.reshape(vectorX, ( -1, seq_length, temp2Vec.dimension + n_templates))
    y = numpy.reshape(vectorY,(-1,temp2Vec.dimension))

    ## normalize
    #X = X / float(n_templates)
    # one hot encode the output variable
    # y = np_utils.to_categorical(dataY)
    # define the LSTM model
    model = Sequential()
    model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    if onehot == 0:
        model.add(Dense(temp2Vec.dimension, activation='softmax'))
    else:
        model.add(Dense(n_templates, activation='softmax'))
    # load the network weights
    model.load_weights(model_filename)
    model.compile(loss='mse', optimizer='adam')
    if onehot ==1:
        model.compile(loss='categorical_crossentropy', optimizer='adam')

    # detect by tag
    total=0
    anomaly_count_dir = {}
    for i in range(n_candidates):
        anomaly_count_dir[i+1] = []
    test1_time = time.time()
    for x_char,x,aim_y_vector in zip(charX, X, y):
        total+=1
        if total%1000 ==0:
            test2_time = time.time()
            print(str(total)+'/'+str(len(X)),str( round(100*total/len(X),3) ),'% time:',(test2_time - test1_time)/60)
            test1_time = time.time()
        aim_y_char = temp2Vec.vector_to_most_similar(aim_y_vector, topn = 1)[0][0]
        if count_matrix_flag == 0:
            x = numpy.reshape(x, (1, seq_length, temp2Vec.dimension))
        else:
            x = numpy.reshape(x, (1, seq_length, temp2Vec.dimension + n_templates))
        prediction = model.predict(x, verbose=0)[0] #输出一个len(tags)的向量，数值越高的列对应概率最高的类别

        #获取与prediction最相似的topn
        if onehot == 1: #dense的y是onehot格式
            for i in range(n_candidates):
                i += 1
                top_n_index = prediction.argsort()[-i:]#[-i:]
                top_n_tag=[int_to_template[index] for index in top_n_index]
                ##将0/1的结果保存在anomaly_count_dir中
                if aim_y_char not in top_n_tag:
                    anomaly_count_dir[i].append(1)
                else:
                    anomaly_count_dir[i].append(0)

        else:#dense的y是temp2Vec格式
            top_n_tuple = temp2Vec.vector_to_most_similar(prediction, topn=n_candidates)
            for i in range(n_candidates):
                i += 1
                top_n =[t[0] for t in top_n_tuple[:i]] #[-i:]
                #print(top_n)
                #top_n_tag=[int_to_template[index] for index in top_n_index]
                ##将0/1的结果保存在anomaly_count_dir中
                if aim_y_char not in top_n:
                    anomaly_count_dir[i].append(1)
                else:
                    anomaly_count_dir[i].append(0)


    '''
    #count by windows
    window_count_dir = {}
    time_start=timeY[0]
    time_end=timeY[-1]
    windows_num = max(0, int(((time_end - time_start) - windows_size * 3600) / step_size / 3600)) +  1
    windows_start_time = [time_start+i*step_size*3600 for i in range(windows_num)]


    raw_windows_label_list = numpy.zeros(windows_num)
    for i in range(n_candidates):
        i += 1
        window_count_dir[i] = numpy.zeros(windows_num)
        for cur_time,cur_flag,label in zip(timeY,anomaly_count_dir[i],label_list):
            cur_index = int(max(0,cur_time - time_start - windows_size * 3600) / step_size / 3600)
            window_count_dir[i][cur_index] += cur_flag
            raw_windows_label_list[cur_index] += label
    windows_label_list = [ 1 if n >=1 else 0 for n in raw_windows_label_list]
    '''

    '''
    precision, recall, f1_score, _ = np.array(list(precision_recall_fscore_support(testing_labels, prediction)))[:, 1]
    print('=' * 20, 'RESULT', '=' * 20)
    print("Precision:  %.6f, Recall: %.6f, F1_score: %.6f" % (precision, recall, f1_score))
    '''


    f = open(result_file,'w')

    print('\nanomaly detection result:')
    for i in range(n_candidates):
        i += 1
        print('next tag  is not in top'+str(i)+' candidates:')
        # print('# of anomalous/total logs:',str(sum(anomaly_count_dir[i]))+'/'+str(len(anomaly_count_dir[i])))

        precision, recall, f1_score, _ = numpy.array(list(precision_recall_fscore_support(label_list, anomaly_count_dir[i])))[:, 1]
        print('=' * 20, 'RESULT', '=' * 20)

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for a,b, in zip(label_list, anomaly_count_dir[i]):
            if b == 1 and a == 1:
                tp += 1
            if a == 1 and b ==0:
                fn += 1
            if a ==0 and b == 0:
                tn += 1
            if a==0 and b == 1:
                fp += 1
        print("Precision:  %.6f, Recall: %.6f, F1_score: %.6f" % (precision, recall, f1_score))
        print('tp:',tp, 'fn:',fn,'tn:',tn,'fp:',fp,'total:',tp+tn+fp+fn)
        print('=' * 20, 'RESULT', '=' * 20)
        f.writelines(str(precision)+' '+str(recall)+'\n')
    '''
        windows_results = [ 1 if n >=1 else 0 for n in window_count_dir[i]]
        print('# of anomalous/total windows:',str(sum(windows_results))+'/'+str(len(windows_results)))
        precision, recall, f1_score, _ = numpy.array(list(precision_recall_fscore_support(windows_label_list, windows_results)))[:, 1]
        print("Precision:  %.6f, Recall: %.6f, F1_score: %.6f" % (precision, recall, f1_score))
        print('')
    '''
    f.close()
    t2 = time.time()
    print('testing time:',(t2-t1)/60,'mins')
    print ("\nDone.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-test_file', help='test_file.', type=str, default='../../middle/bgl_log_20w.seq')
    parser.add_argument('-seq_length', help='seq_length.', type=int, default=10)
    parser.add_argument('-n_candidates', help='n_candidates.', type=int, default=15)
    parser.add_argument('-windows_size', help='windows_size.', type=int, default=3)
    parser.add_argument('-step_size', help='step_size.', type=int, default=1)
    parser.add_argument('-model_filename', help='you can give a model file.', type=str, default='')
    parser.add_argument('-model_dir', help='model_dir.', type=str, default='../weights/vector_deeplog/')
    parser.add_argument('-template_index_map_path', help='template_index_map_path.', type=str, default='./bgl_log_20w_template_to_int.txt')
    parser.add_argument('-onehot', help='默认为1。1表示统计使用onehot，0表示使用template2vec',type = int, default = 1)
    parser.add_argument('-result_file', help='result_file.', type=str, default='../results/bgl_log_20w_log_pr.txt')
    parser.add_argument('-template_num', help='若为0，则根据输入文件统计，否则，根据输入确定。默认0', type=int, default=0)
    parser.add_argument('-label_file', help='label_file.', type=str, default='../../data/bgl2_label_20w')
    parser.add_argument('-count_matrix', help='默认为0。1表示统计count_matrix，0不统计',type = int, default = 0)
    parser.add_argument('-template2Vec_file', help='template2Vec_file', type=str, default='../../model/bgl_log_20w.template_vector')
    parser.add_argument('-template_file', help='template_file', type=str, default='../../middle/bgl_log_20w.template')

    args = parser.parse_args()

    para_detect = {
        'test_file': args.test_file,
        'seq_length':args.seq_length,
        'n_candidates': args.n_candidates,
        'windows_size': args.windows_size,
        'step_size':args.step_size,
        'model_dir': args.model_dir,
        'model_filename': args.model_filename,
        'template_index_map_path':args.template_index_map_path,
        'template_num' : args.template_num,
        'result_file':args.result_file,
        'label_file':args.label_file,
        'template2Vec_file': args.template2Vec_file,
        'template_file': args.template_file,
        'count_matrix': args.count_matrix,
        'onehot': args.onehot
        }

    detect_by_vector(para_detect)

    print('detection finish')

    from keras import backend as K
    K.clear_session()





