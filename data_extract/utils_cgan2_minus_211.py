from __future__ import division
import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import matplotlib.pyplot as plt
import os, gzip
import imageio
import tensorflow as tf
import tensorflow.contrib.slim as slim
import uproot
import h5py
import pandas as pd
import time

def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec


def load_comet(comet_data):
    time_extract = time.time()
    roo_data_sum = pd.DataFrame()
    #pdg_list = [11, 13, 22]
    pdg_list = [-211]
    #pdg_list = [-11,11,-13,13,22,-211,211,2212]
    #先不考虑gamma
    #pdg_list = [-11,11,-13,13,-211,211,2212]
    #-11, 11, -13
    #pdg_list = [-11,11,-13]
    for pdg_id in pdg_list:
        roo_data = pd.DataFrame()
        for file_id in range(1, 13):
            print(file_id)
            #path="./data/Geant4/circle_1"
            path="./data/Geant4/eight_particles"
            id_str = "%04d" % file_id
            file_name = os.path.join(path, "extracted_"+str(pdg_id)+"_rootracker_"+id_str+".txt.root")
            events = uproot.open(file_name)['T']
            df_all = events.pandas.df(['p','p_phi','p_theta','x','y','z','t'], flatten=True)
            #r<120,delete z
            df_all = df_all[(df_all['z']>4175.0027)&(df_all['z']<4175.003)&(np.sqrt(df_all['x']**2+df_all['y']**2)<120)]
            df_all = df_all.drop(['z'], axis=1)
            roo_data = roo_data.append(df_all)
            time_complete = time.time()-time_extract
            print("The time to extract rootracker data is:", time_complete)
        #p,time log transformation
        roo_data['t']=np.log(np.log(np.log(roo_data['t'])))
        roo_data['p']=np.log(roo_data['p'])
        #取每种粒子的十分之一用来训练
        data_tenpercent=roo_data.shape[0]//10
        roo_data = roo_data.iloc[0:data_tenpercent, :]
        #-11
        attribute_min_e = {'p':-3.08, 'p_phi':-3.142, 'p_theta':7.48e-05, 'x':-120, 'y':-120,'time':-0.032}
        attribute_max_e = {'p':6.752, 'p_phi':3.142, 'p_theta':1.71, 'x':120, 'y':120,'time':1.043}
        #11
        attribute_min_11 = {'p':-6.86, 'p_phi':-3.1415926, 'p_theta':5.206e-05, 'x':-120, 'y':-120,'time':-0.032}
        attribute_max_11 = {'p':7.05, 'p_phi':3.1415926, 'p_theta':3.14, 'x':120, 'y':120, 'time':1.05}
        #-13
        attribute_min_mu = {'p':-1.91, 'p_phi':-3.142, 'p_theta':9.95e-05, 'x':-120, 'y':-120,'time':-0.03}
        attribute_max_mu = {'p':7.1, 'p_phi':3.142, 'p_theta':2.35, 'x':120, 'y':120,'time':0.6}
        #13
        attribute_min_13 = {'p':-1.2, 'p_phi':-3.142, 'p_theta':9.4e-05, 'x':-120, 'y':-120, 'time':-0.03}
        attribute_max_13 = {'p':7.1, 'p_phi':3.2, 'p_theta':2, 'x':120, 'y':120, 'time':0.61}
        #22
        attribute_min_22 = {'p' :-9.3, 'p_phi':-3.142, 'p_theta':5e-06, 'x':-120, 'y':-120, 'time':-0.03}
        attribute_max_22 = {'p':7.72, 'p_phi':3.15, 'p_theta':1.58, 'x':120, 'y':120, 'time':1.05}
        #-211
        attribute_min_pi_minus = {'p':-2.1, 'p_phi':-3.142, 'p_theta':4.25e-05, 'x':-120, 'y':-120,'time':-2.89e-02}
        attribute_max_pi_minus = {'p':7.57, 'p_phi':3.142, 'p_theta':1.785, 'x':120, 'y':120,'time':0.53}
        #211
        attribute_min_pi = {'p':-1.34, 'p_phi':-3.142, 'p_theta':2e-05, 'x':-120, 'y':-120,'time':-0.03}
        attribute_max_pi = {'p':7.76, 'p_phi':3.142, 'p_theta':2.189, 'x':120, 'y':120,'time':0.53}
        #2212
        attribute_min_2212 = {'p':-1.088, 'p_phi':-3.142, 'p_theta':5.34e-05, 'x':-120, 'y':-120,'time':-0.02}
        attribute_max_2212 = {'p':8.19, 'p_phi':3.142, 'p_theta':2.75, 'x':120, 'y':120,'time':0.9}
        list_attribute = list(attribute_min_11.keys())
        roo_data = np.array(roo_data)
        if pdg_id == -11:
            for i,key in enumerate(list_attribute):
                roo_data[:, i] = (roo_data[:, i]-attribute_min_e[key])/(attribute_max_e[key]-attribute_min_e[key])
        elif pdg_id==11:
            for i,key in enumerate(list_attribute):
                roo_data[:, i] = (roo_data[:, i]-attribute_min_11[key])/(attribute_max_11[key]-attribute_min_11[key])
        elif pdg_id==-13:
            for i,key in enumerate(list_attribute):
                roo_data[:, i] = (roo_data[:, i]-attribute_min_mu[key])/(attribute_max_mu[key]-attribute_min_mu[key])
        elif pdg_id==13:
            for i,key in enumerate(list_attribute):
                roo_data[:, i] = (roo_data[:, i]-attribute_min_13[key])/(attribute_max_13[key]-attribute_min_13[key])
        elif pdg_id==22:
            for i,key in enumerate(list_attribute):
                roo_data[:, i] = (roo_data[:, i]-attribute_min_22[key])/(attribute_max_22[key]-attribute_min_22[key])
        elif pdg_id == -211:
            for i,key in enumerate(list_attribute):
                roo_data[:, i] = (roo_data[:, i]-attribute_min_pi_minus[key])/(attribute_max_pi_minus[key]-attribute_min_pi_minus[key])
        elif pdg_id == 211:
            for i,key in enumerate(list_attribute):
                roo_data[:, i] = (roo_data[:, i]-attribute_min_pi[key])/(attribute_max_pi[key]-attribute_min_pi[key])
        elif pdg_id == 2212:
            for i,key in enumerate(list_attribute):
                roo_data[:, i] = (roo_data[:, i]-attribute_min_2212[key])/(attribute_max_2212[key]-attribute_min_2212[key])
        roo_data = pd.DataFrame(roo_data)
        roo_data_sum = roo_data_sum.append(roo_data)

    input_height = 512
    input_width = 6
    roo_data_sum = np.array(roo_data_sum)
    roodata_block_num=roo_data_sum.shape[0]//input_height
    trX = roo_data_sum[0:roodata_block_num*input_height, :]
    trX = trX.reshape((roodata_block_num, input_height, input_width, 1))
    trY = np.load('/hpcfs/bes/mlgpu/cuijb/gan/data/Geant4/eight_particles/label_minus_211_onetenth.npy')
    #get -11,11,13
    #trY = trY[0:71276135, :]
    trY = trY[0:roodata_block_num*input_height, :]
    #trY_D是变形后用作与辨别器组合做输入的
    trY = trY.reshape((roodata_block_num, input_height, 1, trY.shape[1]))
    seed = 666
    np.random.seed(seed)
    np.random.shuffle(trX)
    np.random.seed(seed)
    np.random.shuffle(trY)
    print("trX shape is:")
    print(trX.shape)
    return trX,trY


def get_feature_from_rootracker_file(pdg,file_name):
    #root_path=os.path.join(file_name)
    events = uproot.open(file_name)['RooTrackerTree']
    df_all = events.pandas.df(["StdHepPdg","StdHepP4"],flatten=True)#,entrystart=-100
    data_by_pdg = df_all[df_all['StdHepPdg']==pdg]
    return data_by_pdg


def load_rootracker(dataset_name):
    path = "/hpcfs/bes/mlgpu/cuijb/rootracker"
    id_str = "%08d" % 1004
    file_name = os.path.join(path,"oa_xx_xxx_"+id_str+"_0000_user-SG4cat_001_cydet200.rootracker")
    data_by_pdg = get_feature_from_rootracker_file(11,file_name)
    return data_by_pdg




def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(path, image)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

""" Drawing Tools """
# borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)
    plt.close()

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
