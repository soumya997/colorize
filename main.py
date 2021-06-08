
'''
SAR Colorization Using ALOS2 Data
by Qian Song on Jun. 13 2019
'''

import utils
import numpy as np
import time
import scipy.io as sio
import h5py
import tensorflow as tf
tf.compat.v1.Session()
flags = tf.app.flags
import random
import os
import librosa
import matplotlib.pyplot as plt
print(tf.__version__)
# from Speaker_Verification.configuration import get_config
# from Speaker_Verification.utils import similarity,loss_cal
random.seed(0)

flags.DEFINE_boolean('is_train', True, "Flag of opertion: True is for training")
flags.DEFINE_string('data', './data/data_SH.mat', "Path of training/test data")
flags.DEFINE_string('dir', './checkpoint', "Path of saving model")

FLAGS = flags.FLAGS

class Pol_SD(object):
    def __init__(self, sess,embeds=None):
        self.sess = sess
        self.embeds = embeds
        self.learning_rate = 0.0005
        self.output_size = 400
        self.feature_size = 1153
        self.model_build()

    def similarity_matrix(self,embeds):

        # Cosine similarity scaling (with fixed initial parameter values)
        similarity_weight = tf.Variable([10.])
        similarity_bias = tf.Variable([-5.])


        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        centroids_incl = tf.math.reduce_mean(embeds, axis=1,keepdims=True)
        # print("Centroid Shape",centroids_incl.shape) 
        mat_1 = tf.identity(centroids_incl)
        mat_2 = tf.norm(centroids_incl, axis=1,keepdims=True)
        # print("Mat Shapes",mat_1.shape,mat_2.shape)
        centroids_incl =  mat_1 / mat_2
        centroids_excl = (tf.math.reduce_sum(embeds, axis=1, keepdims=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1.0)
        centroids_excl =  tf.identity(centroids_excl) / tf.norm(centroids_excl, axis=2)
        sim_matrix = tf.zeros(speakers_per_batch, utterances_per_speaker,
                                      speakers_per_batch)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        
        
        for j in range(speakers_per_batch):
          mask = np.where(mask_matrix[j])[0]
          sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
          sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)

        sim_matrix = sim_matrix * similarity_weight + similarity_bias

        return sim_matrix

    def loss(self,embeds):
        # embeds = tf.reshape(embeds,(1,400,400,self.feature_size))
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                          speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = ground_truth
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        loss = loss_fn(sim_matrix, target)

        return loss

    def model_build(self):
        ##Build Model
        self.HHHH = tf.placeholder(tf.float32, [None, self.output_size, self.output_size, 1])
        self.H = tf.placeholder(tf.float32, [None, self.feature_size])
        self.X_true = tf.placeholder(tf.float32, [None, 32*9])

        # print(self.HHHH.shape)
        self.hypercolumn = utils.VGG16(self.HHHH)
        # self.X,self.X_ = utils.T_prediction(self.H)
        print("hypercolumn: ",self.hypercolumn.shape) # -> (160000, 1153)
        print("hypercolumn type: ",type(self.hypercolumn))
        #try two
        # hypcol = self.hypercolumn.eval(session=tf.compat.v1.Session())
        # try one
        # graph = tf.get_default_graph()
        # sess = tf.InteractiveSession(graph=graph)
        # with sess.as_default():
        #   name = hypcol.eval()
        # try three
        print("hypcol = tf.make_ndarray(self.hypercolumn.op.get_attr('value'))")
        hypcol = tf.make_ndarray(self.hypercolumn.op.get_attr('value'))

        print("hypcol type", type(hypcol))
        print("hypcol shape", hypcol.shape)
        # self.d_loss = - tf.reduce_mean(self.X_true*tf.log(self.X_+1e-7))
        # self.hypercolumn =
        self.d_loss = self.loss(self.hypercolumn)  # (160000, 1153)  --> .embed shape  torch.Size([64, 10, 256])
                                                                    # centroids_incl shapes  torch.Size([64, 1, 256])
        print("d_loss",self.d_loss.shape)
        # self.d_loss = loss_cal(self.similarity, type=config.loss)
        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-6) \
                                  .minimize(self.d_loss)
        self.saver = tf.train.Saver()

# no of speaker = no. of class, no of utterence = no. images/pixel


    def train(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

        data_X, data_V = self.load_data()
        temp_mean = np.zeros([self.training_size, self.feature_size])
        temp_var = np.zeros([self.training_size, self.feature_size])
        for i in range(self.training_size):
           batch_V = np.reshape(data_V[i, :, :], [1, self.output_size, self.output_size, 1])
           batch_H = self.sess.run(self.hypercolumn, feed_dict={self.HHHH: batch_V})
           temp_mean[i, :] = np.mean(batch_H, axis=0)
           temp_var[i, :] = np.std(batch_H, axis=0)
        temp_mean = np.mean(temp_mean, axis=0)
        temp_var = np.mean(temp_var, axis=0)
        temp_var[temp_var < 1.0] = 1.0
        sio.savemat('./data/mean_and_var.mat', {'mean': temp_mean, 'var': temp_var})

        start_time = time.time()

        # Training steps:=====================================
        counter = 0
        temp_list1 = np.linspace(0, self.output_size*self.output_size-1, self.output_size*self.output_size, dtype = 'int')
        temp_list2 = np.linspace(0, self.training_size-1, self.training_size, dtype = 'int')
        for epoch in range(10):
           batch_idxs = len(data_X)
           random.shuffle(temp_list2)
           random.shuffle(temp_list1)

           for idx in temp_list2:
               batch_V = np.reshape(data_V[idx, :, :], [1, self.output_size, self.output_size, 1])
               print("batch_V",batch_V.shape)
               temp_H = self.sess.run(self.hypercolumn, feed_dict={self.HHHH: batch_V})
               
               temp_H = (temp_H-temp_mean)/temp_var
               temp_X = utils.get_vectorised_T(data_X[idx, :, :, :])

               for index in range(10):
                   batch_H = temp_H[temp_list1[index*2000:(index+1)*2000], :]
                   batch_X = temp_X[temp_list1[index*2000:(index+1)*2000], :]
                   print("batch_H",batch_H.shape)
                   print("batch_X",batch_X.shape)
                   loss1, train_step, X = self.sess.run([self.d_loss, self.optim, self.X_], feed_dict={self.H: batch_H,
                                                                                                       self.X_true:
                                                                                                           batch_X})

                   counter += 1
                   if np.mod(counter, 10) == 9:
                       print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f" % (epoch, idx+1, batch_idxs, time.time()
                                                                                   - start_time, loss1))
           self.saver.save(self.sess, FLAGS.dir + "/Generate model")
           print("[*]Save Model...")

    def test(self):

        print("[*]Loading Model...")
        self.saver.restore(self.sess, FLAGS.dir + "/Generate model")
        print("[*]Load successfully!")

        matfn = './data/mean_and_var.mat'
        data1 = sio.loadmat(matfn)
        temp_mean = data1['mean']
        temp_var = data1['var']
# gradient tape
        test_V = self.load_data()

        Re_data = np.zeros([len(test_V), self.output_size, self.output_size, 9])
        for i in range(len(test_V)):
            batch_V = np.reshape(test_V[i, :, :], [1, self.output_size, self.output_size, 1])
            batch_H = self.sess.run(self.hypercolumn, feed_dict={self.HHHH: batch_V})
            batch_H = (batch_H-temp_mean)/temp_var
            val_X = self.sess.run(self.X_, feed_dict={self.H: batch_H})
            Re_data[i, :, :, :] = utils.inv_vetorization_T(val_X)

        f = h5py.File("./data/test_nj.mat", 'w')
        f.create_dataset('Re_data', data=Re_data)
        f.close()
        # sio.savemat('./data/test_sh2.mat', {'Re_data': Re_data})

    def load_data(self):

        if FLAGS.is_train == True:
            matfn = FLAGS.data
            data1 = h5py.File(matfn, 'r')
            data = data1['data']
            data = np.transpose(data, axes=[3, 2, 1, 0])
            data.shape = -1, self.output_size, self.output_size, 9
            self.training_size = len(data)

            data_H = data1['data_H']
            data_H = np.transpose(data_H, axes=[2, 1, 0])
            data_H = np.log10(data_H + 1e-10)    # Normalization of input image
            data_H[data_H > 13] = 13
            data_H[data_H < 9] = 9
            data_H = (data_H-9) / 4
            data_H.shape = self.training_size, self.output_size, self.output_size
            return data, data_H
        else:
            data1 = sio.loadmat(FLAGS.data)
            data_H = data1['data_H']
            self.test_size = len(data_H)
            data_H = np.log10(data_H + 1e-10)    # Normalization of input image
            data_H[data_H > 13] = 13
            data_H[data_H < 9] = 9
            data_H = (data_H - 9) / 4
            return data_H

def main(_):
    with tf.Session() as sess:
        print(FLAGS.data)
        sdgan = Pol_SD(sess)
        if FLAGS.is_train == True:
            sdgan.train()
        else:
            sdgan.test()

if __name__ == '__main__':
    tf.app.run() 
