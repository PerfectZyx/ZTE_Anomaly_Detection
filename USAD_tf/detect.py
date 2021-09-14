# import numpy as np
# import pandas as pd
from eval_methods import pot_eval
import pickle, json
import numpy as np
import pandas as pd
from my import paint

dataset = '' # 数据名称

train_scores = pickle.load(open(f'./data/new/ZTE-{dataset}/train_score.pkl', 'rb')).T # 训练集分数，路径需要自己调整
test_scores = pickle.load(open(f'./data/new/ZTE-{dataset}/test_score.pkl', 'rb')).T # 测试集分数，路径需要自己调整

test_data = pickle.load(open(f'./data/new/ZTE-{dataset}_test.pkl', 'rb')).T # 测试集数据，路径需要自己调整

train_score = np.sum(train_scores, axis=0)
test_score = np.sum(test_scores, axis=0)

pot = pot_eval(train_score, test_score, level=0.995) # level为spot算法中的初始阈值，需要自行调整

thresholds = np.array(pot['thresholds'])
alarms = pot['alarms']

np.save(f'alarm_{dataset}.npy', np.array(alarms))


names = []
f = open(f'{dataset}_names.txt', 'r') # 打开指标名称文件，该文件在预处理时生成，名字为dataset_names.txt
names = f.readline()[1:-1]
names = names.split(',')

f = open(f'inter-{dataset}.csv', 'w')
f.write('alarm point,top 5 KPI\n')
for alarm in alarms:

    score = []
    for i in range(20):
        score.append((test_scores[i][alarm], i))
    sort = sorted(score, key=lambda x: x[0], reverse=True)
    KPI = ''
    KPI_value = ''

    time = ''

    for s in sort[:5]:
        KPI += names[str(s[1])] + f'({str(s[0])}), '
    f.write(f'{alarm},' + KPI + '\n')
f.close()

