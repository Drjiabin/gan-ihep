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


#Draw a histogram for comparison of individual data
def my_plot_hist_individual(data_x,nbin,title,label):
    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    plt.figure(figsize=(5, 5))
    _ = plt.hist(data_x, bins=100, histtype='stepfilled', linewidth=2,
                 alpha=0.2,color=colors[0],
                 label=label)
    plt.tick_params(labelsize=20)
    loc = 'upper right'
    plt.legend(loc=loc, ncol=1, fontsize=20)#, mode='expand', fontsize=20)
    plt.title(title, fontsize='x-large', fontweight='heavy')

#2dDraw two-dimensional histograms of two variables
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

#extract Extract data under n epochs
def extract_n_epoch(start_Epoch, end_Epoch, step,data1, data2):
    dic_GAN = {'px':0, 'py':0, 'pz':0,'beam_phi':0, 'beam_rho':0, 'beam_z':0, 'time':0}
    dic_Geant4 = {'px':0, 'py':0, 'pz':0,'beam_phi':0, 'beam_rho':0, 'beam_z':0, 'time':0}
    list1 = list(dic_GAN.keys())
    list2 = list(dic_Geant4.keys())
    for i, gan_attribute in enumerate(list1):
        dic_GAN[gan_attribute]=data1[start_Epoch:end_Epoch, i]
    for j, g4_attribute in enumerate(list2):
        dic_Geant4[g4_attribute]=data2[start_Epoch:end_Epoch, j]
    geant4_p = np.sqrt(np.square(dic_Geant4['px'])+np.square(dic_Geant4['py'])+np.square(dic_Geant4['pz']))
    dic_Geant4['p'] = geant4_p
    return dic_GAN, dic_Geant4

def load_roodata(file_num_start, file_num_end):
    time_extract = time.time()
    roo_data = pd.DataFrame()
    for file_id in range(file_num_start, file_num_end):
        print(file_id)
        path="./data/Geant4"
        id_str = "%04d" % file_id
        file_name = os.path.join(path, "extracted_11_rootracker_"+id_str+".txt.root")
        events = uproot.open(file_name)['T']
        df_all = events.pandas.df(['beam_px','beam_py','beam_pz','beam_phi','beam_r', 'beam_z', "beam_t"], flatten=True)
        roo_data = roo_data.append(df_all)
    time_complete = time.time()-time_extract
    print("The time to extract rootracker data is:", time_complete)
    roo_data = np.array(roo_data)
    return roo_data

def main(argv):
    #-------------------------------------------------------------------
    #extract Geant4 data,beam_phi,beam_rho,beam_z,px beam,py beam,pz beam,time
    file_num_start=1
    file_num_end = 13
    roo_data = load_roodata(file_num_start, file_num_end)
    #-------------------------------------------------------------------
    #
    time1=time.time()
    GAN_data_path = "./results/0810-21-14"
    with PdfPages(GAN_data_path+'/'+'train_results_visualise.pdf') as pdf:
        dic_Geant4 = extract_n_epoch(iEpoch, iEpoch+step, step, gan_data, roo_data)
        list_geant4 = ['beam_phi','beam_rho','beam_z','px','py','pz','p']
        list1 = [1,2,3,4,5,6,7]
            #gan vs geant4
            fig1 = plt.figure(figsize=(20, 15))
            for index,g4_key, gan_key in zip(list1,list_geant4, list_gan):
                plt.subplot(3, 3, index)
                my_plot_hist_compare(dic_Geant4[g4_key],dic_GAN[gan_key],1000,g4_key,'G4_'+g4_key,'GAN_'+gan_key)
            plt.subplot(3, 3, 8)
            time_arr_G4 = np.array(dic_Geant4['time'])
            time_arr_GAN = np.array(dic_GAN['time'])
            my_plot_hist_compare(time_arr_G4[np.where((time_arr_G4>0)&(time_arr_G4<5000))],
                                 time_arr_GAN[np.where((time_arr_GAN>0)&(time_arr_GAN<5000))],1000,'0<time<5000','G4_time','GAN_time')
            plt.subplot(3, 3, 9)
            my_plot_hist_compare(time_arr_G4[np.where((time_arr_G4>5000)&(time_arr_G4<1e7))],
                                 time_arr_GAN[np.where((time_arr_GAN>5000)&(time_arr_GAN<1e7))],1000,'5000<time<1e7','G4_time','GAN_time')
            pdf.savefig(fig1)
            #gan vs geant4 log
            fig2 = plt.figure(figsize=(20, 15))
            for index,g4_key, gan_key in zip(list1,list_geant4, list_gan):
                plt.subplot(3, 3, index)
                my_plot_hist_log_compare(dic_Geant4[g4_key],dic_GAN[gan_key],1000,g4_key,'G4_'+g4_key,'GAN_'+gan_key)
            plt.subplot(3, 3, 8)
            my_plot_hist_log_compare(time_arr_G4[np.where((time_arr_G4>0)&(time_arr_G4<1e4))],
                                 time_arr_GAN[np.where((time_arr_GAN>0)&(time_arr_GAN<1e4))],1000,'0<time<1e4','G4_time','GAN_time')
            plt.subplot(3, 3, 9)
            my_plot_hist_log_compare(time_arr_G4[np.where((time_arr_G4>1e4)&(time_arr_G4<1e7))],
                                 time_arr_GAN[np.where((time_arr_GAN>1e4)&(time_arr_GAN<1e7))],1000,'1e4<time<1e7','G4_time','GAN_time')
            pdf.savefig(fig2)
            #
            #2d
            #All attributes are compared in pairs, a total of 21 combinations, seven pictures
            #list1 stores the position of each picture in a fig
            list1 = [[1,2,3],[4,5,6]]
            #list2 list3 stores the attributes to be looped each time
            list2 = [['px','px','py'],['beam_phi','beam_phi','beam_rho'],['p','p','p'],['p','p','p'],['px','px','px'],['py','py','py'],['pz','pz','pz'],['time','time','time']]
            list3 = [['py','pz','pz'],['beam_rho','beam_z','beam_z'],['beam_phi','beam_rho','beam_z'],['px','py','pz'],['beam_phi','beam_rho','beam_z'],['beam_phi','beam_rho','beam_z'],
                     ['beam_phi','beam_rho','beam_z'],['beam_rho', 'beam_z', 'p']]
            for i in range(len(list2)):
                fig = plt.figure(figsize=(27, 15))
                for j,k,l in zip(list1[0], list2[i], list3[i]):
                    plt.subplot(2,3,j)
                    my_plot_2d_hist(dic_Geant4[k], dic_Geant4[l], 30,'G4 '+k+' vs '+l)
                for j,k,l in zip(list1[1], list2[i],list3[i]):
                    plt.subplot(2,3,j)
                    my_plot_2d_hist(dic_GAN[k], dic_GAN[l], 30,'GAN '+k+' vs '+l)
                pdf.savefig(fig)
            time_now=time.time()-time1
            print('time is:', time_now)
if __name__=='__main__':
    main(sys.argv)
