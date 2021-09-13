import pandas as pd
import numpy as np
import os, time, math, pickle, json

from my import paint

def get_timestamp(s, type):
    if type == 1:
        return int(time.mktime(time.strptime(s, '%Y/%m/%d %H:%M')))
    elif type == 2:
        return int(time.mktime(time.strptime(s, '%Y-%m-%d %H:%M')))

def linear_interpolation(series, time_step=60):
    i = 0
    last_time = 0
    new_series = []

    while i < len(series):
        if i == 0:
            new_series.append(series[i])
            last_time = new_series[-1][0]
            i += 1
            continue

        if last_time + time_step != series[i][0]:
            # print(i, len(series))

            j = i
            while j < len(series) - 1 and (series[j][0] - last_time) % time_step != 0:
                j += 1
            i = j
            # print(last_time, series[i][0])

            gap_points = int((series[i][0] - last_time) / time_step)
            gap_value = float((series[i][1] - series[i-1][1]) / gap_points)

            # print(gap_points)
            for x in range(1, gap_points):
                new_series.append([series[i-1][0] + x * time_step, series[i - 1][1] + x * gap_value])
            new_series.append(series[i])
            last_time = new_series[-1][0]
            i += 1

        else:
            new_series.append(series[i])
            last_time = new_series[-1][0]
            i += 1

    return new_series

def process_1():
    new_data = []
    new_name = []
    temp_series = []
    begin = 0
    end = 0
    KPI_names = []

    for f in os.listdir('ZTE-5.0'):
        if f.startswith('x') or f.startswith('m'):
            names = list(pd.read_csv('ZTE-5.0/' + f, encoding='gb18030').keys())
            for i in range(len(names)):
                names[i] = f[:-4] + ' ' + names[i]
            KPI_names.append(names)

            data = pd.read_csv('ZTE-5.0/' + f, encoding='gb18030').values

            data = list(data)
            data = list(reversed(list(data)))
            series = []
            for d in data:
                temp = [get_timestamp(d[0], 1), get_timestamp(d[1], 1)]
                temp.extend(d[2:])
                temp[2] = f
                series.append(temp)
            series = linear_interpolation(series, 300)

            if series[0][0] > begin or begin == 0:
                begin = series[0][0]
            if series[-1][0] < end or end == 0:
                end = series[-1][0]

            temp_series.append(series)

            # print(begin, end)

        else:
            names = list(pd.read_csv('ZTE-5.0/' + f,encoding='gb18030').keys())
            for i in range(len(names)):
                names[i] = f[:-4] + ' ' + names[i]
            KPI_names.append(names)

            data = pd.read_csv('ZTE-5.0/' + f,encoding='gb18030').values

            data = list(data)
            data = list(reversed(list(data)))
            series = []
            for d in data:
                temp = [get_timestamp(d[0], 1)]
                temp.extend(d[1:])
                series.append(temp)
            series = linear_interpolation(series, 300)

            if series[0][0] > begin or begin == 0:
                begin = series[0][0]
            if series[-1][0] < end or end == 0:
                end = series[-1][0]

            temp_series.append(series)
            # print(begin, end)
        # break

    for i in range(len(temp_series)):
        # s = np.array(series)
        # breakpoint()
        grid = temp_series[i]
        series = [[row[i] for row in grid] for i in range(len(grid[0]))]
        print(len(series))

        if len(series) > 2:
            for j in range(3, len(series)):
                new_data.append(series[j])
                new_name.append(KPI_names[i][j])
        else:
            new_data.append(series[1])
            new_name.append(KPI_names[i][1])

    with open('names.txt', 'w') as f:
        f.write(str(new_name)[1:-1])

    np.save('new.npy', np.array(new_data).T)
    # print(np.array(new_data).shape)
    # print(len(new_name))


def process_2():

    raw_data = np.load('new.npy')

    train_data = raw_data[:int(9 * 24 * 60 / 5), :]
    test_data = raw_data[int(9 * 24 * 60 / 5):, :]

    with open('new_train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('new_test.pkl', 'wb') as f:
        pickle.dump(test_data, f)


if __name__ == '__main__':
    process_1()
    process_2()