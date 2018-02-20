# Jan 26th 2018

__author__ = 'Garam Lee'
__email__ = 'goeastagent@gmail.com'

import tensorflow as tf

import numpy as np
import pandas as pd

from RNNComponent import *
from sklearn.linear_model import LogisticRegression

class MMRNN:
    # n_component : the number of LSTM components
    # output_dim : dimension of output
    def __init__(self):
        self.components = {}
        self.IDs = {}
        self.data = {}
        self.label = {}
        self.seqlen = {}
        
        self.training_overlapIDs = []
        self.test_overlapIDs = []
        
    def append_component(self, name, n_input, n_hidden, max_seq_len, cell=None, optimizer=None):
        x = tf.placeholder(tf.float32, [None, max_seq_len, n_input])
        y = tf.placeholder(tf.float32, [None])
        seqlen = tf.placeholder(tf.int32, [None])

        with tf.variable_scope(name):
            new_component = RNNComponent(n_hidden, max_seq_len, x, y, seqlen, cell, optimizer)
        self.components[name] = new_component

    def append_test_overlapIDs(self,IDs):
        self.test_overlapIDs = IDs

    def append_training_overlapIDs(self, IDs):
        self.training_overlapIDs = IDs
        
    def append_data(self, name, IDs, data, label, seqlen):
        self.data[name] = data
        self.IDs[name] = IDs
        self.label[name] = label
        self.seqlen[name] = seqlen
            
    def concatenation_based_integration(self,bucket):
        return tf.concat(bucket, axis=1)

    def kernel_based_integration(self,bucket):
        pass

    def pickup_by_ID(self,IDs, x, tIDs):
        new_x = []

        for ID in tIDs:
            new_x.append(x[IDs == ID][0])
        return np.array(new_x)
    
    def build_integrative_network(self,optimizer=None):
        bucket = []
        self.y = tf.placeholder(tf.float32, [None])
        
        for name, component in self.components.iteritems():
            bucket.append(component.outputs)
            
        self.predictor = tf.concat(bucket, axis=1)
        n_vector = int(self.predictor.shape[1])
        weights = tf.Variable(tf.random_normal([n_vector,1]))
        bias = tf.Variable(tf.random_normal([1]))
        
        self.pY = tf.matmul(predictor,weights) + bias
        # self.pY = tf.matmul(predictor,weights) + bias
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pY,labels=self.y))
        
        if optimizer == None:
            optimizer = tf.train.AdamOptimizer()
        self.train = optimizer.minimize(self.loss)
    
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
    def single_modal_training(self,name,batch_size):
        component = self.components[name]
        IDs = self.IDs[name]
        data = self.data[name]
        label = self.label[name]
        seqlen = self.seqlen[name]
        trainingIDs = self.IDs[name][~pd.Series(self.IDs[name]).isin(self.test_overlapIDs)]

        seqlen = self.pickup_by_ID(IDs, seqlen, trainingIDs)
        label = self.pickup_by_ID(IDs, label, trainingIDs)
        data = self.pickup_by_ID(IDs, data, trainingIDs)


        component.training(self.sess,data,label,seqlen, batch_size)
                    
    def evaluate(self):
        bucket = []
        for name in self.IDs:
            component = self.components[name]
            IDs = self.IDs[name]
            data = self.pickup_by_ID(IDs, self.data[name], self.test_overlapIDs)
            y = self.pickup_by_ID(IDs, self.label[name], self.test_overlapIDs)
            seqlen = self.pickup_by_ID(IDs, self.seqlen[name], self.test_overlapIDs)
            bucket.append(component.extract_output(self.sess, data, seqlen))

        estimated = self.clf.predict(np.concatenate(tuple(bucket), axis=1))
        print(sum(y==estimated)/float(len(estimated)))        
    
    def integrative_training(self):
        feed_dict = {}
        overlapIDs = self.training_overlapIDs
        bucket = []
        for name in self.IDs:
            component = self.components[name]
            IDs = self.IDs[name]
            x = self.pickup_by_ID(IDs, self.data[name], overlapIDs)
            y = self.pickup_by_ID(IDs, self.label[name], overlapIDs)
            seqlen = self.pickup_by_ID(IDs, self.seqlen[name], overlapIDs)
            bucket.append(component.extract_output(self.sess, x, seqlen))

        self.predictor = np.concatenate(tuple(bucket),axis=1)
        self.clf = LogisticRegression(penalty='l1', C=80, class_weight='balanced')
        self.clf.fit(self.predictor, y)
    
    def training(self,batch_size):
        for name in self.components:
            self.single_modal_training(name,batch_size)
        self.integrative_training()
