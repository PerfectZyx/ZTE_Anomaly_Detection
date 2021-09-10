logdata = []
labeldata = []
with open("bgl.log","r") as fin:
    for i in range(100000):
        logdata.append(fin.readline())
with open("bgl10w.log","w") as fout:
    fout.writelines(logdata)

with open("bgl.label","r") as fin:
    for i in range(100000):
        labeldata.append(fin.readline())
with open("bgl10w.label","w") as fout:
    fout.writelines(labeldata)
