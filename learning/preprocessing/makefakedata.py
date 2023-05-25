# import numpy as np
#
# calibrateData = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
# np.savetxt("/home/gaofei/BiRing/Data/Calibration.txt",calibrateData)

# rawdata = np.loadtxt("/home/gaofei/BiRing/Data/processed__t1new.txt",dtype=np.float,encoding='utf-8')
# rawdata = np.array(rawdata)
# rawdatatest = float(rawdata)
# np.savetxt("/home/gaofei/BiRing/Data/processed__t1new.txt",rawdatatest)
# skiprows=1
# import pandas as pd
# df=pd.read_csv("/home/gaofei/BiRing/Data/processed__t1.csv") #filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。
# keys = df.keys()
# temarray = []
# for key in keys:
#     temarray.append(df[key].to_numpy())
#
# temarray = np.array(temarray).T
# print(temarray.shape)
#
# np.savetxt("/home/gaofei/BiRing/Data/processed__t1.txt",temarray[:,1:])
# print(df.keys().to_numpy())

# print(df.tail())
# print(df)

import pickle

# f = open('dict_word.pkl', 'rb')
# for line in f:
#     print(line)
word = pickle.load(open("/home/gaofei/BiRing/Data/calib.pkl", 'rb'), encoding='utf-8')
print(word)

