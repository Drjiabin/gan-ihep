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
    if title == '$p^,$':
        loc = 'upper left'
    elif title == '$\phi$':
        loc = 'upper center'
    else:
        loc = 'upper right'
    plt.legend(loc=loc, ncol=1, fontsize=13)
    plt.title(title, fontsize='x-large', fontweight='heavy')

def my_plot_hist_log_compare(data_x,data_y,nbin,title,label1,label2,range_xy=None):
    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3)) 
    #plt.figure(figsize=(5, 5))
    if range_xy != None:
        _ = plt.hist(data_x, bins=nbin,range=[range_xy[0], range_xy[1]],histtype='stepfilled', linewidth=2,
                     alpha=0.2,color=colors[0],log='True',
                     label=label1)
        _ = plt.hist(data_y, bins=nbin,range=[range_xy[0], range_xy[1]],histtype='step', linewidth=2,
                     alpha=1, color=colors[0],log='True',
                     label=label2)
    else:
        _ = plt.hist(data_x, bins=nbin,histtype='stepfilled', linewidth=2,
                     alpha=0.2,color=colors[0],log='True',
                     label=label1)
        _ = plt.hist(data_y, bins=nbin,histtype='step', linewidth=2,
                     alpha=1, color=colors[0],log='True',
                     label=label2)
    plt.tick_params(labelsize=30)
    loc = 'upper right'
    plt.legend(loc=loc, ncol=1, fontsize=30)
    plt.title(title, fontsize=30, fontweight='heavy')

def my_plot_2d_hist(data_x,data_y,nbin,title,x_label,y_label,range_xy=None):
    if range_xy!= None:
        plt.hist2d(data_x, data_y, bins=nbin, range=[[range_xy[0],range_xy[1]],[range_xy[2],range_xy[3]]],norm=colors.LogNorm())
    else:
        plt.hist2d(data_x, data_y, bins=nbin,norm=colors.LogNorm())
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=30)
    plt.tick_params(labelsize=30)
    plt.xlabel(x_label,fontsize=50,fontweight='heavy')
    plt.ylabel(y_label,fontsize=50,fontweight='heavy')
    plt.title(title, fontsize=30, fontweight='heavy')
    cb.set_label('Entries', size=30,weight='bold')



def load_roodata(file_num_start, file_num_end):
    time_extract = time.time()
    roo_data = pd.DataFrame()
    for file_id in range(file_num_start, file_num_end):
        print(file_id)
        path="/hpcfs/bes/mlgpu/cuijb/gan/data/Geant4/eight_particles"
        id_str = "%04d" % file_id
        file_name = os.path.join(path, "extracted_11_rootracker_"+id_str+".txt.root")
        events = uproot.open(file_name)['T']
        df_all = events.arrays(['p','p_phi','p_theta','x','y','z','t'], library="pd")
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


def load_gan(pdgid):
    path="/hpcfs/bes/mlgpu/cuijb/gan/root_gan"
    file_name = os.path.join(path, pdgid+".root")
    events = uproot.open(file_name)['ttree']
    df_all = events.arrays(['momentum','phi','theta','x','y','time'], library="pd")
    df_all = np.array(df_all)
    return df_all

def np2dict(data1, data2):
    dic_GAN = {'$p^,$':0, '$\phi$':0, r'$\theta$':0,'x':0, 'y':0, '$t^,$':0}
    dic_Geant4 = {'$p^,$':0, '$\phi$':0, r'$\theta$':0,'x':0, 'y':0, '$t^,$':0}
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
    gan_data =load_gan("11_all")
    roo_data = roo_data[0:gan_data.shape[0], :]
    data_min = [-6.86, -3.1415926,5.206e-05,-120,-120,-0.032]
    data_max = [7.05,3.1415926,3.14,120,120,1.05]
    for j in range(6):
        gan_data[:,j] = Antinormalization(gan_data[:,j],data_max[j],data_min[j])
    gan_data = gan_data[np.where(np.sqrt(gan_data[:, 3]**2+gan_data[:, 4]**2)<120)]
    list_gan = ['$p^,$','$\phi$',r'$\theta$','x','y','$t^,$']
    list_geant4 = ['$p^,$','$\phi$',r'$\theta$','x','y','$t^,$']
    list1 = [1,2,3,4,5,6]
    list_range=[[-7,7.05],[-3.142,3.142],[5.206e-05,3.14],[-120, 120],[-120, 120],[-0.032, 1.05]]
    #gan vs geant4 1d
    fig1 = plt.figure(figsize=(20, 15))
    for index,g4_key, gan_key in zip(list1,list_geant4, list_gan):
        plt.subplot(3, 3, index)
        my_plot_hist_compare(roo_data[:, index-1],gan_data[:, index-1],100,g4_key,'SimG4','GAN',list_range[index-1])
    plt.savefig(fname="11_1d.svg", format="svg")
    #plt.subplot(3, 3, 7)
    #time_arr_G4 = np.array(dic_Geant4['log(log(log(t)))'])
    #time_arr_GAN = np.array(dic_GAN['log(log(log(t)))'])
    #my_plot_hist_compare(time_arr_G4[np.where((time_arr_G4>0)&(time_arr_G4<200))],
    #                     time_arr_GAN[np.where((time_arr_GAN>0)&(time_arr_GAN<200))],100,'0<time<200','G4_time','GAN_time')
    #plt.subplot(3, 3, 8)
    #my_plot_hist_compare(time_arr_G4[np.where((time_arr_G4>200)&(time_arr_G4<1e7))],
    #                     time_arr_GAN[np.where((time_arr_GAN>200)&(time_arr_GAN<1e7))],100,'200<time<1e7','G4_time','GAN_time')

    #gan vs geant4 2d
    dic_GAN, dic_Geant4 = np2dict(gan_data, roo_data)
    list1 = [[1,2,3,4,5],[6,7,8,9,10]]
    #list2 = [['$p^,$','$p^,$','$\phi$'],['x'],['$p^,$','$p^,$'],['$\phi$','$\phi$'],[r'$\theta$',r'$\theta$'],['$t^,$','$t^,$','$t^,$'],['$t^,$','$t^,$']]
    #list3 = [['$\phi$',r'$\theta$',r'$\theta$'],['y'],['x','y'],['x','y'],
    #             ['x','y'],['$p^,$','$\phi$',r'$\theta$'],['x','y']]
    list2 = [['$p^,$','$p^,$','$p^,$','$p^,$','$p^,$'], ['$\phi$','$\phi$','$\phi$','$\phi$',r'$\theta$'],[r'$\theta$',r'$\theta$','x','x','y']]
    list3 = [['$\phi$',r'$\theta$','x','y','$t^,$'], [r'$\theta$','x','y','$t^,$','x'],['y','$t^,$','y','$t^,$','$t^,$']]
    list4 = [0,1,2,3,4]
    #range_2d=[[[-7,7.05,-3.14,3.14],[-7,7.05,0,3.14],[-3.14,3.14,0,3.14]],
    #              [[-125,125,-125,125]],
    #              [[-7,7.05,-125,125],[-7,7.05,-125,125]],
    #              [[-3.14,3.14,-150,150],[-3.14,3.14,-150,150]],
    #              [[0,3.14,-150,150],[0,3.14,-150,150]],
    #              [[-0.032,1.05,-7,7.05],[-0.032,1.05,-3.14,3.14],[-0.032,1.05,0,3.14]],
    #              [[-0.032, 1.05, -125, 125],[-0.032, 1.05, -125, 125]]]
    range_2d=[[[-7,7.05,-3.142,3.1415926],[-7,7.05,5.206e-05,3.14],[-7,7.05,-120, 120],[-7,7.05,-120, 120],[-7,7.05,-0.032, 1.05]],
              [[-3.142,3.142,5.206e-05,3.14],[-3.142,3.142,-120,120],[-3.142,3.142,-120,120],[-3.142,3.142,-0.032, 1.05],[5.206e-05,3.14,-120,120]],
              [[5.206e-05,3.14,-120, 120],[5.206e-05,3.14,-0.032, 1.05],[-120, 120,-120, 120],[-120, 120,-0.032, 1.05],[-120, 120,-0.032, 1.05]]
              ]
    for i in range(len(list2)):
        fig = plt.figure(figsize=(90, 30))
        for j,k,l,m in zip(list1[0], list2[i], list3[i],list4):
            plt.subplot(2,5,j)
            range_xy = range_2d[i][m]
            my_plot_2d_hist(dic_Geant4[k], dic_Geant4[l], 30,'SimG4 '+k+' vs '+l,k,l,range_xy)
        for j,k,l,m in zip(list1[1], list2[i],list3[i],list4):
            plt.subplot(2,5,j)
            range_xy = range_2d[i][m]
            my_plot_2d_hist(dic_GAN[k], dic_GAN[l], 30,'GAN '+k+' vs '+l, k,l,range_xy)
        plt.savefig("%i.svg"%i, format="svg")

if __name__=='__main__':
    main(sys.argv)
