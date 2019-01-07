#!/usr/bin/python3
#-*- coding: UTF-8 -*-
import collections  
import numpy as np  
import tensorflow as tf  
'''
This one will produce a poetry 
author: log16
Date: 2018/5/4

'''  
#-------------------------------数据预处理---------------------------#  
   
poetry_file ='/home/loghost/LSTM/data/poetry.txt'
   
# 诗集  
poetrys = []  
with open(poetry_file, "r") as f:  
    for line in f:  
        try:  
            line = line.decode('UTF-8')
            line = line.strip(u'\n')
            title, content = line.strip(u' ').split(u':')  
            content = content.replace(u' ',u'')  
            if u'_' in content or u'(' in content or u'（' in content or u'《' in content or u'[' in content:  
                continue  
            if len(content) < 5 or len(content) > 79:  
                continue  
            content = u'[' + content + u']'  
            poetrys.append(content)  
        except Exception as e:   
            pass  
   
# 按诗的字数排序  
poetrys = sorted(poetrys,key=lambda line: len(line))  
print('唐诗总数: ', len(poetrys))  
   
# 统计每个字出现次数  
all_words = []  
for poetry in poetrys:  
    all_words += [word for word in poetry]  
counter = collections.Counter(all_words)  
count_pairs = sorted(counter.items(), key=lambda x: -x[1])  
words, _ = zip(*count_pairs)  
   
# 取前多少个常用字  
words = words[:len(words)] + (' ',)  
# 每个字映射为一个数字ID  
word_num_map = dict(zip(words, range(len(words))))  
# 把诗转换为向量形式，参考TensorFlow练习1  
to_num = lambda word: word_num_map.get(word, len(words))  
poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]  
#[[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],  
#[339, 3, 133, 31, 302, 653, 512, 0, 37, 148, 294, 25, 54, 833, 3, 1, 965, 1315, 377, 1700, 562, 21, 37, 0, 2, 1253, 21, 36, 264, 877, 809, 1]  
#....]  
   
# 每次取64首诗进行训练  
batch_size = 1
n_chunk = len(poetrys_vector) // batch_size  

class DataSet(object):
    def __init__(self,data_size):
        self._data_size = data_size
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._data_index = np.arange(data_size)

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        if start + batch_size > self._data_size:
            np.random.shuffle(self._data_index)
            self._epochs_completed = self._epochs_completed + 1
            self._index_in_epoch = batch_size
            full_batch_features ,full_batch_labels = self.data_batch(0,batch_size)
            return full_batch_features ,full_batch_labels 
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            full_batch_features ,full_batch_labels = self.data_batch(start,end)
            if self._index_in_epoch == self._data_size:
                self._index_in_epoch = 0
                self._epochs_completed = self._epochs_completed + 1
                np.random.shuffle(self._data_index)
            return full_batch_features,full_batch_labels

    def data_batch(self,start,end):
        batches = []
        for i in range(start,end):
            batches.append(poetrys_vector[self._data_index[i]])

        length = max(map(len,batches))

        xdata = np.full((end - start,length), word_num_map[' '], np.int32)  
        for row in range(end - start):  
            xdata[row,:len(batches[row])] = batches[row]  
        ydata = np.copy(xdata)  
        ydata[:,:-1] = xdata[:,1:]  
        return xdata,ydata

#---------------------------------------RNN--------------------------------------#  
   
input_data = tf.placeholder(tf.int32, [batch_size, None])  
output_targets = tf.placeholder(tf.int32, [batch_size, None])  
# 定义RNN  
def neural_network(model='lstm', rnn_size=128, num_layers=2):  
    if model == 'rnn':  
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':  
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':  
        cell_fun = tf.contrib.rnn.BasicLSTMCell
   
    cell = cell_fun(rnn_size, state_is_tuple=True)  
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
   
    initial_state = cell.zero_state(batch_size, tf.float32)  
   
    with tf.variable_scope('rnnlm'):  
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)])  
        softmax_b = tf.get_variable("softmax_b", [len(words)])  
        with tf.device("/cpu:0"):  
            embedding = tf.get_variable("embedding", [len(words), rnn_size])  
            inputs = tf.nn.embedding_lookup(embedding, input_data)  
   
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')  
    output = tf.reshape(outputs,[-1, rnn_size])  
   
    logits = tf.matmul(output, softmax_w) + softmax_b  
    probs = tf.nn.softmax(logits)  
    return logits, last_state, probs, cell, initial_state 


#-------------------------------生成古诗---------------------------------#  
# 使用训练完成的模型  
   
def gen_poetry():  
    def to_word(weights):  
        t = np.cumsum(weights)  
        s = np.sum(weights)  
        sample = int(np.searchsorted(t, np.random.rand(1)*s))  
        return words[sample]  
   
    _, last_state, probs, cell, initial_state = neural_network()
    Session_config = tf.ConfigProto(allow_soft_placement = True)
    Session_config.gpu_options.allow_growth=True 
   
    with tf.Session(config=Session_config) as sess: 
        with tf.device('/gpu:1'): 
            
            sess.run(tf.initialize_all_variables())  
       
            saver = tf.train.Saver(tf.all_variables())  
            saver.restore(sess, 'model/poetry.module-99')  
       
            state_ = sess.run(cell.zero_state(1, tf.float32))  
       
            x = np.array([list(map(word_num_map.get, '['))])  
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})  
            word = to_word(probs_)  
            #word = words[np.argmax(probs_)]  
            poem = ''  
            while word != ']':  
                poem += word  
                x = np.zeros((1,1))  
                x[0,0] = word_num_map[word]  
                [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})  
                word = to_word(probs_)  
                #word = words[np.argmax(probs_)]  
            return poem  
   
print(gen_poetry()) 
