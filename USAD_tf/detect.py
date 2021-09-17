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


pot = pot_evalnp.sum(train_scores, axis=0), np.sum(test_scores, axis=0), level=0.995) # level为spot算法中的初始阈值，需要自行调整

thresholds = np.array(pot['thresholds'])
alarms = pot['alarms']

np.save(f'alarm_{dataset}.npy', np.array(alarms))


names = []
f = open(f'{dataset}_names.txt', 'r') # 打开指标名称文件，该文件在预处理时生成，名字为dataset_names.txt
names = f.readline()[1:-1]
names = names.split(',')

f = open(f'inter-{dataset}.csv', 'w')
f.write('alarm point,top 5 KPI\n')

train_avg = np.mean(train_score, axis=0)
for alarm in alarms:
    scores = test_score[alarm] - train_avg
    score = []
    for i in range(train_avg.shape[0]):
        score.append((scores[i], i))
    sort = sorted(score, key=lambda x: x[0], reverse=True)
    KPI = ''
    KPI_value = ''

    time = ''

    for s in sort[:5]:
        KPI += names[str(s[1])] + f'({str(s[0])}), '
    f.write(f'{alarm},' + KPI + '\n')
f.close()

