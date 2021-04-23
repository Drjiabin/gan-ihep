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
    roo_data = pd.DataFrame()
    for file_id in range(1, 13):
        print(file_id)
        #path="./data/Geant4/circle_1"
        path="./data/Geant4/11_13_22"
        id_str = "%04d" % file_id
        file_name = os.path.join(path, "extracted_11_rootracker_"+id_str+".txt.root")
        events = uproot.open(file_name)['T']
        #df_all = events.pandas.df(['beam_px','beam_py','beam_pz','global_x','global_y', 'global_z', "beam_t"], flatten=True)
        df_all = events.pandas.df(['p','p_phi','p_theta','x','y','z','t'], flatten=True)
        #time<10000
        #df_all = df_all[df_all['beam_t']<10000]
        #r<120,delete z
        df_all = df_all[(df_all['z']>4175.0027)&(df_all['z']<4175.003)&(np.sqrt(df_all['x']**2+df_all['y']**2)<120)]
        df_all = df_all.drop(['z'], axis=1)
        roo_data = roo_data.append(df_all)
    time_complete = time.time()-time_extract
    print("The time to extract rootracker data is:", time_complete)
    #p,time log transformation
    roo_data['t']=np.log(np.log(np.log(roo_data['t'])))
    roo_data['p']=np.log(roo_data['p'])
    roo_data = np.array(roo_data)
    #normalization
    #all data beam_line
    #data shape(2.69352855e8, 7)
    #time 178s
    #attribute_min = {'px':-282.7, 'py':-247.7, 'pz':-9.0, 'phi':-3.14, 'r':0.0032, 'z':-531.6,'t':13.9}
    #attribute_max = {'px':228.06, 'py':225.54, 'pz':1196.42, 'phi':3.15, 'r':1710.1, 'z':10770.4,'t':5.183259994e7}
    #time<10000
    #data shape()
    #attribute_min = {'px':-282.7, 'py':-247.7, 'pz':-9.0, 'phi':-3.14, 'r':0.0032, 'z':-531.6,'t':13.9}
    #attribute_max = {'px':228.06, 'py':225.54, 'pz':1196.42, 'phi':3.15, 'r':1710.1, 'z':10770.4,'t':10000}
    #global x,y,z,beamliene  px py pz
    #circle,data shape(2.42238380e8,6)
    #time 211s
    #attribute_min = {'px':-224.73, 'py':-247.78, 'pz':9.63e-08, 'beam_phi':-120, 'beam_rho':-120,'time':13.94}
    #attribute_max = {'px':150.62, 'py':196.54, 'pz':1136.383, 'beam_phi':120, 'beam_rho':120, 'time':33739774}
    #circle_2
    #attribute_min = {'p':0.001, 'p_phi':-3.1415926, 'p_theta':3.14, 'x':-120, 'y':-120,'time':13.94}
    #attribute_max = {'p':1150.9, 'p_phi':3.1415926, 'p_theta':5.2e-05, 'x':120, 'y':120, 'time':33739774}
    #p,time log transformation
    #11
    attribute_min_11 = {'p':-6.86, 'p_phi':-3.1415926, 'p_theta':5.206e-05, 'x':-120, 'y':-120,'time':-0.032}
    attribute_max_11 = {'p':7.05, 'p_phi':3.1415926, 'p_theta':3.14, 'x':120, 'y':120, 'time':1.05}
    #13
    attribute_min_13 = {'p':0.4, 'p_phi':-3.2, 'p_theta':9.4e-05, 'x':-120, 'y':-120, 'time':14.1,'log(p)':-1.2,'log(log(log(time)))':-0.03}
    attribute_max_13 = {'p':1203.3, 'p_phi':3.2, 'p_theta':2, 'x':120, 'y':120, 'time':511.2,'log(p)':7.1,'log(log(log(time)))':0.61}
    #22
    attribute_min_22 = {'p' :0.0001, 'p_phi':-3.15, 'p_theta':5e-06, 'x':-120, 'y':-120, 'time':13}
    attribute_max_22 = {'p':2240, 'p_phi':3.15, 'p_theta':1.58, 'x':120, 'y':120, 'time':31753855}

    list_attribute = list(attribute_min_11.keys())
    for i,key in enumerate(list_attribute):
        roo_data[:, i] = (roo_data[:, i]-attribute_min_11[key])/(attribute_max_11[key]-attribute_min_11[key])
    input_height = 16
    input_width = 6
    roodata_block_num=roo_data.shape[0]//input_height
    trX = roo_data[0:roodata_block_num*input_height, :]
    trX = trX.reshape((roodata_block_num, input_height, input_width, 1))
    seed = 666
    np.random.seed(seed)
    np.random.shuffle(trX)
    return trX
def load_comet1(comet_data):
    scalar_data = np.zeros([60000, 7])
    scalar_data = scalar_data + 0.5
    input_height = 512
    input_width = 7
    scalar_block_num = scalar_data.shape[0]//input_height
    trX = scalar_data[0:scalar_block_num*input_height, :]
    trX = trX.reshape((scalar_block_num, input_height, input_width, 1))
    seed = 666
    np.random.seed(seed)
    np.random.shuffle(trX)
    return trX


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
