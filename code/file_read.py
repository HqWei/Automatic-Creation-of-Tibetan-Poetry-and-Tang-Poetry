#coding:utf-8
fp=open('/home/loghost/LSTM/data/data.txt', "r")
l=fp.readline()
print(l)
print fp.tell()
for line in fp:
        print(line)
        print fp.tell()

fp.close()