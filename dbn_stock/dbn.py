"""
Deep Belief Network
author: Ye Hu
2016/12/20
"""
import timeit
import numpy as np
import tensorflow as tf
import input_data
from logisticRegression import LogisticRegression
from mlp import HiddenLayer
from rbm import RBM



def data_preprocess(data,pre_days):
    train_data=data
    target=data[:,3]#target是你想预测的属性 位于data的第几列，这里预测最高价，在data第4列
    for i in range(pre_days):
        target=np.delete(target,0,axis=0)
        train_data=np.delete(train_data,-1,axis=0)
        train_data=np.column_stack((train_data,target))
    tmp=target.shape[0]
    np.delete(train_data,[range(tmp,train_data.shape[0])],axis=0)
    return train_data
#f=open('qxdata_new_one_year.txt')  
#df=pd.read_table(f)
def get_test_data(time_step,test_begin):
    test_data=data_preprocess(data,pre_days)
    data_test=test_data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample 
    test_x,test_y=[],[]  
    test_x=normalized_test_data[:,:input_size]
    test_y=normalized_test_data[:,input_size:] 
    return mean,std,test_x,test_y    
##获取训练集
def get_data(train_data, batch_size,time_step,train_begin,train_end):
    dely_day=0
    batch_index=[]
    data_train=train_data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集 
    for i in range(len(normalized_train_data)-time_step-dely_day):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:input_size]
#       y=normalized_train_data[i+dely_day:i+dely_day+time_step,7:-1,np.newaxis]
       y=normalized_train_data[i+dely_day:i+dely_day+time_step,input_size:]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return np.mean(data_train,axis=0),np.std(data_train,axis=0),train_x,train_y


class DBN(object):
    """
    An implement of deep belief network
    The hidden layers are firstly pretrained by RBM, then DBN is treated as a normal
    MLP by adding a output layer.
    """
    def __init__(self, n_in=784, n_out=10, hidden_layers_sizes=[500, 500]):
        """
        :param n_in: int, the dimension of input
        :param n_out: int, the dimension of output
        :param hidden_layers_sizes: list or tuple, the hidden layer sizes
        """
        # Number of layers
        assert len(hidden_layers_sizes) > 0
        self.n_layers = len(hidden_layers_sizes)
        self.layers = []    # normal sigmoid layer
        self.rbm_layers = []   # RBM layer
        self.params = []       # keep track of params for training

        # Define the input and output
        self.x = tf.placeholder(tf.float32, shape=[None, n_in])
        self.y = tf.placeholder(tf.float32, shape=[None, n_out])

        # Contruct the layers of DBN
        with tf.name_scope('DBN_layer'):
            for i in range(self.n_layers):
                if i == 0:
                    layer_input = self.x
                    input_size = n_in
                else:
                    layer_input = self.layers[i-1].output
                    input_size = hidden_layers_sizes[i-1]
                # Sigmoid layer
                with tf.name_scope('internel_layer'):
                    sigmoid_layer = HiddenLayer(inpt=layer_input, n_in=input_size, n_out=hidden_layers_sizes[i],
                                                activation=tf.nn.sigmoid)
                self.layers.append(sigmoid_layer)
                # Add the parameters for finetuning
                self.params.extend(sigmoid_layer.params)
                # Create the RBM layer
                with tf.name_scope('rbm_layer'):
                    self.rbm_layers.append(RBM(inpt=layer_input, n_visiable=input_size, n_hidden=hidden_layers_sizes[i],
                                               W=sigmoid_layer.W, hbias=sigmoid_layer.b))
            # We use the LogisticRegression layer as the output layer
            with tf.name_scope('output_layer'):
                self.output_layer = LogisticRegression(inpt=self.layers[-1].output, n_in=hidden_layers_sizes[-1],
                                                    n_out=n_out)
        self.params.extend(self.output_layer.params)
        # The finetuning cost
        with tf.name_scope('output_loss'):
            self.cost = self.output_layer.cost(self.y)
        # The accuracy
        self.accuracy = self.output_layer.accuarcy(self.y)
    
    def pretrain(self, sess, train_x, batch_size=50, pretraining_epochs=10, lr=0.5, k=1, 
                    display_step=1):
        """
        Pretrain the layers (just train the RBM layers)
        :param sess: tf.Session
        :param X_train: the input of the train set (You might modidy this function if you do not use the desgined mnist)
        :param batch_size: int
        :param lr: float
        :param k: int, use CD-k
        :param pretraining_epoch: int
        :param display_step: int
        """
        print('Starting pretraining...\n')
        start_time = timeit.default_timer()
        # Pretrain layer by layer
        for i in range(self.n_layers):
            cost = self.rbm_layers[i].get_reconstruction_cost()
            train_ops = self.rbm_layers[i].get_train_ops(learning_rate=lr, k=k, persistent=None)
            batch_num = int(train_x.shape[0] / batch_size)

            for epoch in range(pretraining_epochs):
                avg_cost = 0.0
                for step in range(batch_num-1):
                    # 训练
                    x_batch = train_x[batch_num*batch_size:(batch_num+1)*batch_size]
                    sess.run(train_ops, feed_dict={self.x: x_batch})
                    # 计算cost
                    avg_cost += sess.run(cost, feed_dict={self.x: x_batch,})
                # 输出
                
                if epoch % display_step == 0:
                    print("\tPretraing layer {0} Epoch {1} cost: {2}".format(i, epoch, avg_cost))

        end_time = timeit.default_timer()
        print("\nThe pretraining process ran for {0} minutes".format((end_time - start_time) / 60))
    
    def finetuning(self, sess, train_x, train_y, test_x, test_y, training_epochs=10, batch_size=100, lr=0.5,
                   display_step=1):
        """
        Finetuing the network
        """
        print("\nStart finetuning...\n")
        start_time = timeit.default_timer()
        train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(self.cost)
        batch_num = int(train_x.shape[0] / batch_size)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs",sess.graph) 
        for epoch in range(training_epochs):
            avg_cost = 0.0
            for step in range(batch_num-1):
                x_batch = train_x[batch_num*batch_size:(batch_num+1)*batch_size]
                y_batch = train_y[batch_num*batch_size:(batch_num+1)*batch_size]
                # 训练
                sess.run(train_op, feed_dict={self.x: x_batch, self.y: y_batch})
                # 计算cost
                avg_cost += sess.run(self.cost, feed_dict={self.x: x_batch, self.y: y_batch}) / batch_num
                            # 输出
            if epoch % display_step == 0:
                val_acc = sess.run(self.accuracy, feed_dict={self.x: test_x,self.y: test_y})
                print("\tEpoch {0} cost: {1}, validation accuacy: {2}".format(epoch, avg_cost, val_acc))
           
            result = sess.run(merged,feed_dict={self.x: test_x,self.y: test_y})# 输出
            writer.add_summary(result,epoch)
        end_time = timeit.default_timer()
        print("\nThe finetuning process ran for {0} minutes".format((end_time - start_time) / 60))
import pandas as pd
if __name__ == "__main__":
    # mnist examples
    layer_num = 2 #lstm层数 
    keep_prob = 0.6#dropout的概率
    input_size = 9 #你输入的数据特征数量
    lr = 0.01
    pre_days = 3 #你想预测后面多少天
    time_step = 70 #你想用前多少天的数据
    batch_size  = 32
    train_begin = 0
    train_end = 1800
    train_ecpho = 10 #训练次数
    test_begin = train_end+1
    output_size = pre_days
    f=open('399300.csv',encoding='GBK') 
    df=pd.read_csv(f)    
    data=pd.DataFrame(df.iloc[:,2:].values)
    data = data.dropna()
    data = data.apply(lambda x: x.apply(lambda y: float(y))).values
    train_data=data_preprocess(data,pre_days)         
    m, s ,train_x,train_y=get_data(train_data, batch_size,time_step,train_begin,train_end)   
    mean, std, test_x, test_y=get_data(train_data, batch_size, time_step, test_begin, data.shape[0]) 
    x_train = pd.DataFrame()
    y_train = pd.DataFrame()   
    for item in train_x:
        item = pd.DataFrame(np.array(item).reshape([1,-1]))
        x_train = pd.concat([x_train, item])
    
    for item in train_y:
        item = pd.DataFrame(np.array(item[-1]).reshape([1,-1]))
        y_train = pd.concat([y_train, item])
    y_train_o = y_train.values
    x_train = x_train.values
    up =[0.0,1.0]
    down = [1.0,0.0]
    y_train = []
    for i in y_train_o:
        if i[1] - i[0] >0:
            y_train.append(up)
        else:
            y_train.append(down)    
    x_test = pd.DataFrame()
    y_test = pd.DataFrame()   
    for item in test_x:
        item = pd.DataFrame(np.array(item).reshape([1,-1]))
        x_test = pd.concat([x_test, item])
    
    for item in test_y:
        item = pd.DataFrame(np.array(item[-1]).reshape([1,-1]))
        y_test = pd.concat([y_test, item])
    y_test_o = y_test.values
    x_test = x_test.values
    up = [0.0,1.0]
    down = [1.0,0.0]
    y_train = []
    for i in y_train_o:
        if i[1] - i[0] >0:
            y_train.append(up)
        else:
            y_train.append(down)
    y_train = np.array(y_train)     
    y_test = []
    for i in y_test_o:
        if i[1] - i[0] >0:
            y_test.append(up)
        else:
            y_test.append(down)
    y_test = np.array(y_test)  
#    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    dbn = DBN(n_in=x_train.shape[1], n_out=2, hidden_layers_sizes=[900, 900, 500])
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.set_random_seed(seed=1111)
#    dbn.pretrain(sess, x_train, lr=lr)
    dbn.finetuning(sess, x_train, y_train, x_test, y_test, lr=lr,training_epochs=train_ecpho)
    