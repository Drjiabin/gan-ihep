#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import shutil

from ops import *
#from utils_cgan import *
from utils_cgan2_minus_11 import *
class CGAN2_minus_11(object):
    def __init__(self, sess, epoch, batch_size, z_dim, y_dim, dataset_name, checkpoint_dir, result_dir, log_dir, test_dir, input_height,input_width,output_height,output_width,
                 lambd, disc_iters, learning_rate, beta1, epoch_num, input_num, structure, parameter_path, structure_path):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.test_dir = test_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.structure = structure
        self.model_name = "CGAN2_minus_11"  # name for checkpoint
        self.epoch_num = epoch_num
        self.input_num = input_num
        self.parameter_path = parameter_path
        self.structure_path = structure_path
        # mnist和fashion-mnist都是由28x28的灰色图像组成的
        if dataset_name == 'mnist' or dataset_name == 'comet':
            # parameters
            self.input_height = input_height
            self.input_width =  input_width
            self.output_height = output_height
            self.output_width = output_width

            self.z_dim = z_dim         # dimension of noise-vector
            self.y_dim = y_dim
            self.c_dim = 1

            # WGAN_GP parameter
            # WGAN_GP参数配置 lambd：数值越高，越稳定，但收敛速度越慢
            self.lambd = lambd       # The higher value, the more stable, but the slower convergence
            self.disc_iters = disc_iters     # The number of critic iterations for one-step of generator

            # train
            self.learning_rate = learning_rate
            self.beta1 = beta1

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            #self.data_X, self.data_y = load_mnist(self.dataset_name)
            self.data_X, self.data_Y= load_comet(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    # 判别器 (64,input_height,input_width,c_dim)-->(64,1)
    def discriminator(self, x, y, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            params_d = self.structure['discriminator']
            #y = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(x, y)
            net = lrelu(conv2d(x, params_d[0][0], params_d[0][1], params_d[0][2], params_d[0][3], params_d[0][4], name=params_d[0][5]))
            net = lrelu(conv2d(net, params_d[1][0], params_d[1][1], params_d[1][2], params_d[1][3], params_d[1][4], name=params_d[1][5]))
            net = lrelu(conv2d(net, params_d[2][0], params_d[2][1], params_d[2][2], params_d[2][3], params_d[2][4], name=params_d[2][5]))
            net = lrelu(conv2d(net, params_d[3][0], params_d[3][1], params_d[3][2], params_d[3][3], params_d[3][4], name=params_d[3][5]))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(linear(net, params_d[4][0], scope=params_d[4][1]))
            out_logit = linear(net, params_d[5][0], scope=params_d[5][1])
            out = tf.nn.sigmoid(out_logit)
            #改写
            #net = lrelu(conv1d(x, 64, 2, 1, name='d_conv1'))
            #net = lrelu(conv1d(x, 128, 2, 1, name='d_conv2'))
            #net = lrelu(conv1d(x, 256, 2, 1, name='d_conv3'))
            #net = tf.reshape(net, [self.batch_size, -1])
            #net = lrelu(bn(linear(net, 1024, scope='d_fc4' ), is_training=is_training, scope='d_bn4'))

            return out, out_logit, net

    # 生成器函数，对于不同的数据集判别器有不同的网络(64,z_dim)-->(64,output_height,output_width,c_dim)
    def generator(self, z, y, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            params_g = self.structure['generator']
            shape_y = y.get_shape()
            y = tf.reshape(y, [shape_y[0], shape_y[1],shape_y[3]])
            z = concat([z, y], 2)
            shape_z = z.get_shape()
            z = tf.reshape(z, [shape_z[0], shape_z[1]*shape_z[2]])
            net = tf.nn.relu(bn(linear(z, params_g[0][0], scope=params_g[0][1]), is_training=is_training, scope=params_g[0][2]))
            net = tf.nn.relu(bn(linear(net, params_g[1][0] * params_g[1][1] * params_g[1][2], scope=params_g[1][3]), is_training=is_training, scope=params_g[1][4]))
            net = tf.reshape(net, [self.batch_size, params_g[2][0], params_g[2][1], params_g[2][2]])
            net = tf.nn.relu(
                             bn(deconv2d(net, [self.batch_size, params_g[3][0][0], params_g[3][0][1], params_g[3][0][2]], params_g[3][1][0], params_g[3][1][1], params_g[3][1][2],
                             params_g[3][1][3], name=params_g[3][1][4]), is_training=is_training,
                             scope=params_g[3][2]))
            net = tf.nn.relu(
                             bn(deconv2d(net, [self.batch_size, params_g[4][0][0], params_g[4][0][1], params_g[4][0][2]], params_g[4][1][0], params_g[4][1][1],
                             params_g[4][1][2], params_g[4][1][3], name=params_g[4][1][4]), is_training=is_training,
                             scope=params_g[4][2]))
            net = tf.nn.relu(
                             bn(deconv2d(net, [self.batch_size, params_g[5][0][0], params_g[5][0][1], params_g[5][0][2]], params_g[5][1][0], params_g[5][1][1], 
                             params_g[5][1][2], params_g[5][1][3], name=params_g[5][1][4]), is_training=is_training,
                             scope=params_g[5][2]))
            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, params_g[6][0][0], params_g[6][0][1], params_g[6][0][2]], params_g[6][1][0], 
                                params_g[6][1][1], params_g[6][1][2], params_g[6][1][3], name=params_g[6][2]))

            return out

    # 最为重要的一个函数，控制着WGAN-GP模型的训练
    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        #labels
        self.y = tf.placeholder(tf.float32, [bs, self.input_height, 1, self.y_dim], name='y_d')
        #self.y_g = tf.placeholder(tf.float32, [bs, self.y_dim], name='y')
        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.input_height, self.z_dim], name='z')

        """ Loss Function """

        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.inputs, self.y, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.z, self.y, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G, self.y, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = - tf.reduce_mean(D_real_logits)
        d_loss_fake = tf.reduce_mean(D_fake_logits)
        self.d_loss = d_loss_real + d_loss_fake
        # get loss for generator
        self.g_loss = - d_loss_fake

        """ Gradient Penalty """#important 很重要的梯度惩罚实现
        # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
        alpha = tf.random_uniform(shape=self.inputs.get_shape(), minval=0., maxval=1.)
        differences = G - self.inputs # This is different from MAGAN
        interpolates = self.inputs + (alpha * differences)
        _, D_inter, _=self.discriminator(interpolates, self.y, is_training=True, reuse=True)
        gradients = tf.gradients(D_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        self.d_loss += self.lambd * gradient_penalty

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, self.y, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    # 训练模型WGAN-GP
    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.input_height, self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
            #if start_epoch == self.epoch:
            #    self.visualize_results_test(self.epoch)
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")
        if start_epoch != self.epoch:
            # summary writer
            self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # loop for epoch
        start_time = time.time()
        #simul_data = []
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_labels = self.data_Y[idx*self.batch_size:(idx + 1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.input_height, self.z_dim]).astype(np.float32)

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                               feed_dict={self.inputs: batch_images,
                                               self.z: batch_z,self.y:batch_labels})
                self.writer.add_summary(summary_str, counter)

                # update G network
                if (counter-1) % self.disc_iters == 0:
                    batch_z = np.random.uniform(-1, 1, [self.batch_size, self.input_height, self.z_dim]).astype(np.float32)
                    _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                           feed_dict={self.y:batch_labels, self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                counter += 1

                # display training status
                if np.mod(counter, 50) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

            # save training results for every epoch_num epochs
            #if np.mod(epoch, self.epoch_num) == 0:
            #    for i in range(self.input_num):
            #        samples = self.sess.run(self.fake_images,feed_dict={self.z: self.sample_z})
            #        samples = np.reshape(samples,(self.batch_size, self.input_height*self.input_width))
            #        simul_data.extend(samples)
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        # show temporal results
        #self.visualize_results(epoch)
        #save simu_data in csv file
        #simul_data = pd.DataFrame(simul_data)
        #path = self.create_time_file(self.result_dir, self.parameter_path, self.structure_path)
        #simul_data.to_csv(path+'/'+'1.csv', index=False)

        # save model for final step
        self.save(self.checkpoint_dir, counter)


    def create_time_file(self, result_dir,parameter_path, structure_path):
        data_now = time.strftime("%m%d-%H-%M-%S", time.localtime())
        path = result_dir + '/' + data_now
        if not os.path.exists(path):
            os.makedirs(path)
            print('Folder creation completed' + path)
        shutil.copy(parameter_path, path)
        shutil.copy(structure_path, path)
        return path
    #用训练好的生成器生成数据输出到root文件中
    def output_csv(self, output_batch, label_num, data_name):
        simul_data = []
        for i in range(output_batch):
            z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.input_height, self.z_dim))
            y_sample = np.zeros((self.batch_size, self.input_height, 1, 8))
            #11
            y_sample[:,:,:,label_num] = y_sample[:,:,:,label_num]+1
            samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample,self.y:y_sample})
            #print('samples shape is:')
            #print(samples.shape)
            #samples = (samples+1.)/2.
            samples = np.reshape(samples,(self.batch_size*self.input_height, self.input_width))
            #samples = samples[np.where(np.sqrt(samples[:, 3]**2+samples[:, 4]**2)<120)]
            simul_data.extend(samples)
        simul_data = pd.DataFrame(simul_data)
        #root
        simul_data = np.array(simul_data)
        print('simul_data shape is:', simul_data.shape)
        #print(simul_data.shape)
        with uproot.recreate(data_name) as f:
            f["ttree"] = uproot.newtree({"momentum": "float64",
                                         "phi": "float64",
                                         "theta": "float64",
                                         "x": "float64",
                                         "y": "float64",
                                         "time": "float64"})
            f["ttree"].extend({"momentum": np.array(simul_data)[:, 0]})
            f["ttree"].extend({"phi": np.array(simul_data)[:, 1]})
            f["ttree"].extend({"theta": np.array(simul_data)[:, 2]})
            f["ttree"].extend({"x": np.array(simul_data)[:, 3]})
            f["ttree"].extend({"y": np.array(simul_data)[:, 4]})
            f["ttree"].extend({"time": np.array(simul_data)[:, 5]})
        #simul_data = pd.DataFrame(simul_data)
        #simul_data.to_csv(data_name, index=False)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim,self.learning_rate,self.beta1,self.lambd,self.disc_iters)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def train_check(self):
        import re
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir, self.model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            start_epoch = (int)(counter / self.num_batches)
        if start_epoch == self.epoch:
            print(" [*] Training already finished! Begin to test your model")
