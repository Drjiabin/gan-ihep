#global_x, global_y, global_z, px,py,pz,time
#df_all['global_z']>4175.0027)and(df_all['global_z']<4175.003)and(np.sqrt(df_all['global_x']**2+df_all['global_y']**2)<120
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
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
                     alpha=0.2,color=colors[0],
                     label=label1)
        _ = plt.hist(data_y, bins=nbin,range=[range_xy[0], range_xy[1]],histtype='step', linewidth=2,
                     alpha=1, color=colors[0],
                     label=label2)
    else:
        _ = plt.hist(data_x, bins=nbin,histtype='stepfilled', linewidth=2,
                     alpha=0.2,color=colors[0],
                     label=label1)
        _ = plt.hist(data_y, bins=nbin,histtype='step', linewidth=2,
                     alpha=1, color=colors[0],
                     label=label2)
    plt.tick_params(labelsize=15)
    loc = 'upper right'
    plt.legend(loc=loc, ncol=1, fontsize=13)
    plt.title(title, fontsize='x-large', fontweight='heavy')


def load_roodata(file_num_start, file_num_end):
    time_extract = time.time()
    roo_data = pd.DataFrame()
    for file_id in range(file_num_start, file_num_end):
        print(file_id)
        path="./data/Geant4/circle_2"
        id_str = "%04d" % file_id
        file_name = os.path.join(path, "extracted_11_rootracker_"+id_str+".txt.root")
        events = uproot.open(file_name)['T']
        df_all = events.pandas.df(['p','p_phi','p_theta','x','y','z','t'], flatten=True)
        df_all = df_all[(df_all['z']>4175.0027)&(df_all['z']<4175.003)&(np.sqrt(df_all['x']**2+df_all['y']**2)<120)]
        df_all = df_all.drop(['z'], axis=1)
        roo_data = roo_data.append(df_all)
    time_complete = time.time()-time_extract
    print("The time to extract rootracker data is:", time_complete)
    #p,time log transformation
    roo_data['t']=np.log(np.log(np.log(roo_data['t'])))
    roo_data['p']=np.log(roo_data['p'])

    roo_data = np.array(roo_data)
    return roo_data

def main(argv):
    #-------------------------------------------------------------------
    #extract Geant4 data,global_x,global_y,beam_z,px beam,py beam,pz beam,time
    file_num_start=1
    file_num_end = 2
    roo_data = load_roodata(file_num_start, file_num_end)
    data_min = [-6.86, -3.1415926,5.206e-05,-120,-120,-0.032]
    data_max = [7.05,3.1415926,3.14,120,120,1.05]
    file_list=['test1.csv','test2.csv','test3.csv','test4.csv','test5.csv','test6.csv','test7.csv']
    batchsize=[16, 32, 64, 128, 256, 512, 1024]
    with PdfPages('train_results_visualise_batchsize.pdf') as pdf:
        for file_name,batch_size in zip(file_list,batchsize):
            path='/hpcfs/bes/mlgpu/cuijb/gan'
            gan_data = load_gan_beam_data(path, file_name)
            input_height=512
            input_width=6
            gan_data = gan_data.reshape((gan_data.shape[0]*input_height, input_width))
            roo_data = roo_data[0:gan_data.shape[0],:]
            #Antinormalization
            for j in range(6):
                gan_data[:,j] = Antinormalization(gan_data[:,j],data_max[j],data_min[j])
            #dic_GAN['p']=np.sqrt(np.square(dic_GAN['px'])+np.square(dic_GAN['py'])+np.square(dic_GAN['pz']))
            list_gan = ['p','p_phi','p_theta','x','y','t']
            list_geant4 = ['p','p_phi','p_theta','x','y','t']
            list1 = [1,2,3,4,5,6]
            list_range=[[-7,7.05],[-3.1415926,3.1415926],[5.206e-05,3.14],[-120, 120],[-120, 120],[-0.032, 1.05]]
            #gan vs geant4
            fig1 = plt.figure(figsize=(20, 15))
            for index,g4_key, gan_key in zip(list1,list_geant4, list_gan):
                plt.subplot(3, 3, index)
                my_plot_hist_compare(roo_data[:, index-1],gan_data[:, index-1],100,g4_key,'G4_'+g4_key,'GAN_'+gan_key,list_range[index-1])
            #plt.subplot(3, 3, 7)
            #time_arr_G4 = np.array(dic_Geant4['t'])
            #time_arr_GAN = np.array(dic_GAN['t'])
            #my_plot_hist_compare(time_arr_G4[np.where((time_arr_G4>0)&(time_arr_G4<200))],
            #                     time_arr_GAN[np.where((time_arr_GAN>0)&(time_arr_GAN<200))],100,'0<time<200','G4_time','GAN_time')
            #plt.subplot(3, 3, 8)
            #my_plot_hist_compare(time_arr_G4[np.where((time_arr_G4>200)&(time_arr_G4<1e7))],
            #                     time_arr_GAN[np.where((time_arr_GAN>200)&(time_arr_GAN<1e7))],100,'200<time<1e7','G4_time','GAN_time')
            fig1.suptitle("batch size: %d" % batch_size, fontsize=20)
            pdf.savefig(fig1)
if __name__=='__main__':
    main(sys.argv)
