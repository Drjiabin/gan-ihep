#global_x, global_y, global_z, px,py,pz,time
#df_all['global_z']>4175.0027)and(df_all['global_z']<4175.003)and(np.sqrt(df_all['global_x']**2+df_all['global_y']**2)<120
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn import preprocessing
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
def my_plot_hist_compare(data_x,data_y,nbin,title,label1,label2,range_xy=None):
    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    #plt.figure(figsize=(5, 5))
    if range_xy != None:
        _ = plt.hist(data_x, bins=nbin, range=[range_xy[0], range_xy[1]],histtype='stepfilled', linewidth=2,
                     alpha=0.2,color=colors[0],#log='True',
                     label=label1)
        _ = plt.hist(data_y, bins=nbin,range=[range_xy[0], range_xy[1]],histtype='step', linewidth=2,
                     alpha=1, color=colors[0],#log='True',
                     label=label2)
    else:
        _ = plt.hist(data_x, bins=nbin,histtype='stepfilled', linewidth=2,
                     alpha=0.2,color=colors[0],#log='True',
                     label=label1)
        _ = plt.hist(data_y, bins=nbin,histtype='step', linewidth=2,
                     alpha=1, color=colors[0],#log='True',
                     label=label2)
    plt.tick_params(labelsize=15)
    loc = 'upper right'
    plt.legend(loc=loc, ncol=1, fontsize=13)
    plt.title(title, fontsize='x-large', fontweight='heavy')

def my_plot_2d_hist(data_x,data_y,nbin,title,range_xy=None):
    if range_xy!= None:
        plt.hist2d(data_x, data_y, bins=nbin, range=[[range_xy[0],range_xy[1]],[range_xy[2],range_xy[3]]],norm=colors.LogNorm())
    else:
        plt.hist2d(data_x, data_y, bins=nbin,norm=colors.LogNorm())
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=15)
    plt.tick_params(labelsize=15)
    plt.title(title, fontsize='x-large', fontweight='heavy')
    cb.set_label('Entries', size=13,weight='bold')



def load_roodata(file_num_start, file_num_end):
    time_extract = time.time()
    roo_data = pd.DataFrame()
    for file_id in range(file_num_start, file_num_end):
        print(file_id)
        path="/hpcfs/bes/mlgpu/cuijb/gan/data/Geant4/eight_particles"
        id_str = "%04d" % file_id
        file_name = os.path.join(path, "extracted_2212_rootracker_"+id_str+".txt.root")
        events = uproot.open(file_name)['T']
        df_all = events.arrays(['p','p_phi','p_theta','x','y','z','t'], library="pd")
        #data_roo = events.arrays(['p','p_phi','p_theta','x','y','z','t'])
        df_all = df_all[(df_all['z']>4175.0027)&(df_all['z']<4175.003)&(np.sqrt(df_all['x']**2+df_all['y']**2)<120)&(df_all['t']>300)&(df_all['t']<10000)]
        df_all = df_all.drop(['z'], axis=1)
        roo_data = roo_data.append(df_all)
    time_complete = time.time()-time_extract
    print("The time to extract rootracker data is:", time_complete)
    #p,time log transformation
    #roo_data['t']=np.log(np.log(np.log(roo_data['t'])))
    roo_data['p']=np.log(roo_data['p'])

    roo_data = np.array(roo_data)
    print('roodata shape is:===========', roo_data.shape)
    return roo_data



def load_gan(pdgid):
    path="/hpcfs/bes/mlgpu/cuijb/gan/root_gan"
    file_name = os.path.join(path, pdgid+".root")
    events = uproot.open(file_name)['ttree']
    data_arr = events.arrays(['momentum','phi','theta','x','y','time'], library="pd")
    data_arr = np.array(data_arr)
    print("gan data shape is :", data_arr.shape)
    #p,time log transformation
    return data_arr

def np2dict(data1, data2):
    dic_GAN = {'log(p)':0, 'p_phi':0, 'p_theta':0,'x':0, 'y':0, 't':0}
    dic_Geant4 = {'log(p)':0, 'p_phi':0, 'p_theta':0,'x':0, 'y':0, 't':0}
    list1 = list(dic_GAN.keys())
    for i, attribute in enumerate(list1):
        dic_GAN[attribute]=data1[:, i]
        dic_Geant4[attribute]=data2[:, i]
    return dic_GAN, dic_Geant4

def main(argv):
    #-------------------------------------------------------------------
    #extract Geant4 data,global_x,global_y,beam_z,px beam,py beam,pz beam,time
    file_num_start=1
    file_num_end = 13
    roo_data = load_roodata(file_num_start, file_num_end)
    print("roo_data shape is ", roo_data.shape)
    #time three log
    data_min = [-1.088, -3.142, 5.34e-05, -120, -120, 1.741]
    data_max = [8.19, 3.142, 2.75, 120, 120,2.22]
    #time no log
    #data_min = [-1.088, -3.142, 5.34e-05, -120, -120,14.368]
    #data_max = [8.19, 3.142, 2.75, 120, 120,120592]
    #
    with PdfPages('2212phi_qt9000.pdf') as pdf:
        #path='/hpcfs/bes/mlgpu/cuijb/gan/generate_data'
        #gan_data = load_gan_beam_data(path, '2212.csv')
        gan_data_original = load_gan('2212phi_qt9000')
        input_height=512
        input_width=6
        roo_data = roo_data[0:gan_data_original.shape[0],:]
        print("gandata shape is :", gan_data_original.shape)
        print("roodata shape is :", roo_data.shape)
        #Antinormalization
        gan_data = np.zeros((gan_data_original.shape[0],gan_data_original.shape[1]))
        for j in range(6):
            gan_data[:,j] = Antinormalization(gan_data_original[:,j],data_max[j],data_min[j])
        quantile_min = -5.2
        quantile_max = 5.2
        gan_data[:, 1] = gan_data_original[:, 1] * (quantile_max-quantile_min) + quantile_min
        #quantile transformation
        quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
        roo_data_load = quantile_transformer.fit_transform(roo_data)
        gandata_qt = quantile_transformer.inverse_transform(gan_data)
        gan_data[:, 1] = gandata_qt[:, 1]
        gan_data[:, 5] = np.exp(np.exp(gan_data[:, 5]))
        #dic_GAN['log(p)']=np.sqrt(np.square(dic_GAN['px'])+np.square(dic_GAN['py'])+np.square(dic_GAN['pz']))
        #list_gan = ['log(p)','p_phi','p_theta','x','y','log(log(log(t)))']
        #list_geant4 = ['log(p)','p_phi','p_theta','x','y','log(log(log(t)))']
        list_gan = ['log(p)','p_phi','p_theta','x','y','log(log(t))']
        list_geant4 = ['log(p)','p_phi','p_theta','x','y','log(log(t))']
        list1 = [1,2,3,4,5,6]
        #list_range=[[-1.1,8.19],[-3.15,3.15],[0, 2.75],[-120, 120],[-120, 120],[14.368, 120592]]
        list_range=[[-1.1,8.19],[-3.15,3.15],[0, 2.75],[-120, 120],[-120, 120],[5000, 10000]]
        #gan vs geant4 1d
        fig1 = plt.figure(figsize=(20, 15))
        #simul_data.dropna(axis=0, how='any')
        for index,g4_key, gan_key in zip(list1,list_geant4, list_gan):
            plt.subplot(3, 3, index)
            my_plot_hist_compare(roo_data[:, index-1],gan_data[:, index-1],100,g4_key,'G4_'+g4_key,'GAN_'+gan_key,list_range[index-1])
        #plt.subplot(3, 3, 7)
        #time_arr_G4 = np.array(dic_Geant4['log(log(log(t)))'])
        #time_arr_GAN = np.array(dic_GAN['log(log(log(t)))'])
        #my_plot_hist_compare(time_arr_G4[np.where((time_arr_G4>0)&(time_arr_G4<200))],
        #                     time_arr_GAN[np.where((time_arr_GAN>0)&(time_arr_GAN<200))],100,'0<time<200','G4_time','GAN_time')
        #plt.subplot(3, 3, 8)
        #my_plot_hist_compare(time_arr_G4[np.where((time_arr_G4>200)&(time_arr_G4<1e7))],
        #                     time_arr_GAN[np.where((time_arr_GAN>200)&(time_arr_GAN<1e7))],100,'200<time<1e7','G4_time','GAN_time')

        pdf.savefig(fig1)
        #gan vs geant4 2d
        dic_GAN, dic_Geant4 = np2dict(gan_data, roo_data)
        list1 = [[1,2,3],[4,5,6]]
        #list2 = [['log(p)','log(p)','p_phi'],['x'],['log(p)','log(p)'],['p_phi','p_phi'],['p_theta','p_theta'],['log(log(log(t)))','log(log(log(t)))','log(log(log(t)))'],['log(log(log(t)))','log(log(log(t)))']]
        #list3 = [['p_phi','p_theta','p_theta'],['y'],['x','y'],['x','y'],
        #             ['x','y'],['log(p)','p_phi','p_theta'],['x','y']]
        list2 = [['log(p)','log(p)','p_phi'],['x'],['log(p)','log(p)'],['p_phi','p_phi'],['p_theta','p_theta'],['t','t','t'],['t','t']]
        list3 = [['p_phi','p_theta','p_theta'],['y'],['x','y'],['x','y'],
                      ['x','y'],['log(p)','p_phi','p_theta'],['x','y']]

        list4 = [0,1,2]
        range_2d=[[[-1.1,8.2,-3.15,3.15],[-1.1,8.2,0,2.76],[-3.15,3.15,0,2.76]],
                      [[-120,120,-120,120]],
                      [[-1.1,8.2,-120,120],[-1.1,8.2,-120,120]],
                      [[-3.15,3.15,-120,120],[-3.15,3.15,-120,120]],
                      [[0,2.76,-120,120],[0,2.76,-120,120]],
                      #[[-0.02,0.9,-1.1,8.2],[-0.02,0.9,-3.15,3.15],[-0.02,0.9,0,2.76]],
                      #[[-0.02, 0.9, -120, 120],[-0.02, 0.9, -120, 120]]
                      #[[14.368,120592,-1.1,8.2],[14.368,120592,-3.15,3.15],[14.368,120592,0,2.76]],
                      [[300,10000,-1.1,8.2],[300,10000,-3.15,3.15],[300,10000,0,2.76]],
                      [[300, 10000, -120, 120],[300, 10000, -120, 120]]]
        for i in range(len(list2)):
            fig2 = plt.figure(figsize=(27, 15))
            for j,k,l,m in zip(list1[0], list2[i], list3[i],list4):
                plt.subplot(2,3,j)
                range_xy = range_2d[i][m]
                my_plot_2d_hist(dic_Geant4[k], dic_Geant4[l], 30,'G4 '+k+' vs '+l,range_xy)
            for j,k,l,m in zip(list1[1], list2[i],list3[i],list4):
                plt.subplot(2,3,j)
                range_xy = range_2d[i][m]
                my_plot_2d_hist(dic_GAN[k], dic_GAN[l], 30,'GAN '+k+' vs '+l, range_xy)
            pdf.savefig(fig2)

if __name__=='__main__':
    main(sys.argv)
