#global_x, global_y, global_z, px,py,pz,time
#df_all['global_z']>4175.0027)and(df_all['global_z']<4175.003)and(np.sqrt(df_all['global_x']**2+df_all['global_y']**2)<120
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use('Agg')
import yaml
import os
import sys
import time
import uproot

#extract fake data
def load_gan_beam_data(path,file_name):
    csv_path=os.path.join(path,file_name)
    return np.array(pd.read_csv(csv_path))

#def read_yml(yml_type, yml_file):
#    yml_path=os.path.join("/hpcfs/bes/mlgpu/cuijb/gan/yml/"+yml_type,yml_file)
#    with open(yml_path) as yml:
#        yml_data = yaml.load(yml)
#    return yml_data

#Anti-normalization
def Antinormalization(data,max,min):
    antinorm_data = data * (max-min) + min
    return antinorm_data

#Draw a histogram for comparing the two data
def my_plot_hist_compare(data,nbin,title,label,range_xy=None):
    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    #plt.figure(figsize=(5, 5))
    if range_xy != None:
        _ = plt.hist(data_x, bins=nbin, range=[range_xy[0], range_xy[1]],histtype='stepfilled', linewidth=2,
                     alpha=0.2,color=colors[0],
                     label=label1)
    else:
        _ = plt.hist(data_x, bins=nbin,histtype='stepfilled', linewidth=2,
                     alpha=0.2,color=colors[0],
                     label=label1)
    plt.tick_params(labelsize=15)
    loc = 'upper right'
    plt.legend(loc=loc, ncol=1, fontsize=13)
    plt.title(title, fontsize='x-large', fontweight='heavy')





def load_roodata(file_num_start, file_num_end):
    time_extract = time.time()
    roo_data = pd.DataFrame()
    for file_id in range(file_num_start, file_num_end):
        print(file_id)
        path="/hpcfs/bes/mlgpu/cuijb/gan/data/Geant4/11_13_22"
        id_str = "%04d" % file_id
        file_name = os.path.join(path, "extracted_22_rootracker_"+id_str+".txt.root")
        events = uproot.open(file_name)['T']
        df_all = events.pandas.df(['p','p_phi','p_theta','x','y','z','t'], flatten=True)
        print("before select shape: %d,%d" % df_all.shape)
        df_all = df_all[(df_all['z']>4175.0027)&(df_all['z']<4175.003)&(np.sqrt(df_all['x']**2+df_all['y']**2)<120)]
        print("after select shape: %d,%d" % df_all.shape)
        df_all = df_all.drop(['z'], axis=1)
        roo_data = roo_data.append(df_all)
    time_complete = time.time()-time_extract
    print("The time to extract rootracker data is:", time_complete)
    #p,time log transformation
    roo_data['t']=np.log(np.log(np.log(roo_data['t'])))
    roo_data['p']=np.log(roo_data['p'])

    roo_data = np.array(roo_data)
    return roo_data

def np2dict(data1, data2):
    dic_GAN = {'p':0, 'p_phi':0, 'p_theta':0,'x':0, 'y':0, 't':0}
    dic_Geant4 = {'p':0, 'p_phi':0, 'p_theta':0,'x':0, 'y':0, 't':0}
    list1 = list(dic_GAN.keys())
    for i, attribute in enumerate(list1):
        dic_GAN[attribute]=data1[:, i]
        dic_Geant4[attribute]=data2[:, i]
    return dic_GAN, dic_Geant4

def main(argv):
    #-------------------------------------------------------------------
    #extract Geant4 data,global_x,global_y,beam_z,px beam,py beam,pz beam,time
    file_num_start=1
    file_num_end = 4
    roo_data = load_roodata(file_num_start, file_num_end)
    data_min = [-6.86, -3.1415926,5.206e-05,-120,-120,-0.032]
    data_max = [7.05,3.1415926,3.14,120,120,1.05]


    with PdfPages('cgan2_11.pdf') as pdf:
        list_geant4 = ['p','p_phi','p_theta','x','y','t']
        list1 = [1,2,3,4,5,6]
        #gan vs geant4 1d
        fig1 = plt.figure(figsize=(20, 15))
        for index,g4_key in zip(list1,list_geant4):
            plt.subplot(3, 3, index)
            my_plot_hist_compare(roo_data[:, index-1],gan_data[:, index-1],100,g4_key,'G4_'+g4_key,'GAN_'+gan_key,list_range[index-1])
        pdf.savefig(fig1)


if __name__=='__main__':
    main(sys.argv)
