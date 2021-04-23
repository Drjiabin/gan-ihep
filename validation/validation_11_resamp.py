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

def load_resamp(pdgid):
    time_extract = time.time()
    path="/hpcfs/bes/mlgpu/cuijb/gan/resamp"
    file_name = os.path.join(path, "MC4o_Resampling_"+pdgid+"_Z_4175.root")
    events = uproot.open(file_name)['Particle']
    df_all = events.pandas.df(['momx','momy','momz','posx','posy','posz','time','phi'], flatten=True)
    print("before select shape: %d,%d" % df_all.shape)
    df_all = df_all[(df_all['posz']>4175.0027)&(df_all['posz']<4175.003)&(np.sqrt(df_all['posx']**2+df_all['posy']**2)<120)]
    print("after select shape: %d,%d" % df_all.shape)

    #calculate p,p_phi,p_theta
    realmom = np.sqrt(np.square(df_all['momx'])+np.square(df_all['momy'])+np.square(df_all['momz']))
    total_mom = (realmom+6.86)/13.91
    realtheta = np.arccos(df_all['momz']/realmom)
    phi = df_all['phi']
    #if((df_all['momx']<0.0)&(df_all['momy']>0.0)):
    #    realphi = realphi + 3.1415926
    #if((df_all['momx']<0.0)&(df_all['momy']<0.0)):
    #    realphi = realphi - 3.1415926

    df_new = df_all.rename(columns={'momx':'p', 'momy':'p_phi', 'momz':'p_theta', 'posx':'x','posy':'y','posz':'z','time':'time'})
    df_new['p'] = realmom
    df_new['p_phi'] = phi
    df_new['p_theta'] = realtheta
    df_new = df_new.drop(['z'], axis=1)
    time_complete = time.time()-time_extract
    print("The time to extract rootracker data is:", time_complete)
    #p,time log transformation
    df_new['t']=np.log(np.log(np.log(df_new['time'])))
    df_new['p']=np.log(df_new['p'])

    df_new = np.array(df_new)
    return df_new


def load_mc4o(pdgid):
    time_extract = time.time()
    path="/hpcfs/bes/mlgpu/cuijb/gan/MC4o"
    file_name = os.path.join(path, "MC4o_"+pdgid+"_Z_4175.root")
    events = uproot.open(file_name)['Particle']
    df_all = events.pandas.df(['momx','momy','momz','posx','posy','posz','time','phi'], flatten=True)
    print("before select shape: %d,%d" % df_all.shape)
    df_all = df_all[(df_all['posz']>4175.0027)&(df_all['posz']<4175.003)&(np.sqrt(df_all['posx']**2+df_all['posy']**2)<120)]
    print("after select shape: %d,%d" % df_all.shape)

    #calculate p,p_phi,p_theta
    realmom = np.sqrt(np.square(df_all['momx'])+np.square(df_all['momy'])+np.square(df_all['momz']))
    #print('momx: %d', df_all['momx'])
    #print('momy: %d', df_all['momy'])
    total_mom = (realmom+6.86)/13.91
    realtheta = np.arccos(df_all['momz']/realmom)
    #p_theta = (realtheta - 5.206e-5)/(3.14 - 5.206e-5)
    phi = df_all['phi']
    #p_phi = (realphi + 3.1415926)/(2*3.1415926)
    #phi = df_all['phi']
    #if((df_all['momx']<0.0)&(df_all['momy']>0.0)):
    #    realphi = realphi + 3.1415926
    #    print('test if')
    #if(df_all(df_all['momx']<0.0)&(df_all['momy']<0.0)):
    #    realphi = realphi - 3.1415926
    #    print('test if2')


    df_new = df_all.rename(columns={'momx':'p', 'momy':'p_phi', 'momz':'p_theta', 'posx':'x','posy':'y','posz':'z','time':'time'})
    df_new['p'] = realmom
    df_new['p_phi'] = phi
    df_new['p_theta'] = realtheta
    df_new = df_new.drop(['z'], axis=1)
    time_complete = time.time()-time_extract
    print("The time to extract rootracker data is:", time_complete)
    #p,time log transformation
    df_new['time']=np.log(np.log(np.log(df_new['time'])))
    df_new['p']=np.log(df_new['p'])

    df_new = np.array(df_new)
    return df_new


def np2dict(data1, data2):
    dic_resamp = {'log(p)':0, 'p_phi':0, 'p_theta':0,'x':0, 'y':0, 'log(log(log(t)))':0}
    dic_mc = {'log(p)':0, 'p_phi':0, 'p_theta':0,'x':0, 'y':0, 'log(log(log(t)))':0}
    list1 = list(dic_resamp.keys())
    for i, attribute in enumerate(list1):
        dic_resamp[attribute]=data1[:, i]
        dic_mc[attribute]=data2[:, i]
    return dic_resamp, dic_mc

def main(argv):
    #-------------------------------------------------------------------
    #extract Geant4 data,global_x,global_y,beam_z,px beam,py beam,pz beam,time
    resamp_data = load_resamp('2212')
    mc4o_data = load_mc4o('2212')
    mc4o_data = mc4o_data[0:resamp_data.shape[0], :]
    data_min = [-6.86, -3.1415926,5.206e-05,-120,-120,-0.032]
    data_max = [7.05,3.1415926,3.14,120,120,1.05]


    with PdfPages('11.pdf') as pdf:
        list_resamp = ['log(p)','p_phi','p_theta','x','y','log(log(log(t)))']
        list_mc4o = ['log(p)','p_phi','p_theta','x','y','log(log(log(t)))']
        list1 = [1,2,3,4,5,6]
        list_range=[[-7,7.05],[-3.1415926,3.1415926],[5.206e-05,3.14],[-120, 120],[-120, 120],[-0.032, 1.05]]
        #gan vs geant4 1d
        fig1 = plt.figure(figsize=(20, 15))
        for index,resamp_key, mc_key in zip(list1,list_resamp, list_mc4o):
            plt.subplot(3, 3, index)
            my_plot_hist_compare(resamp_data[:, index-1],mc4o_data[:, index-1],100,resamp_key,'resamp_'+mc_key,'mc4o_'+mc_key,list_range[index-1])
        pdf.savefig(fig1)
        fig2 = plt.figure(figsize=(20, 15))
        for index,resamp_key, mc_key in zip(list1,list_resamp, list_mc4o):
            plt.subplot(3, 3, index)
            my_plot_hist_log_compare(resamp_data[:, index-1],mc4o_data[:, index-1],100,resamp_key,'resamp_'+mc_key,'mc4o_'+mc_key,list_range[index-1])
        pdf.savefig(fig2)
        #plt.subplot(3, 3, 7)
        #time_arr_G4 = np.array(dic_Geant4['log(log(log(t)))'])
        #time_arr_GAN = np.array(dic_GAN['log(log(log(t)))'])
        #my_plot_hist_compare(time_arr_G4[np.where((time_arr_G4>0)&(time_arr_G4<200))],
        #                     time_arr_GAN[np.where((time_arr_GAN>0)&(time_arr_GAN<200))],100,'0<time<200','G4_time','GAN_time')
        #plt.subplot(3, 3, 8)
        #my_plot_hist_compare(time_arr_G4[np.where((time_arr_G4>200)&(time_arr_G4<1e7))],
        #                     time_arr_GAN[np.where((time_arr_GAN>200)&(time_arr_GAN<1e7))],100,'200<time<1e7','G4_time','GAN_time')

        #gan vs geant4 2d
        dic_resamp, dic_mc = np2dict(resamp_data, mc4o_data)
        list1 = [[1,2,3],[4,5,6]]
        list2 = [['log(p)','log(p)','p_phi'],['x'],['log(p)','log(p)'],['p_phi','p_phi'],['p_theta','p_theta'],['log(log(log(t)))','log(log(log(t)))','log(log(log(t)))'],['log(log(log(t)))','log(log(log(t)))']]
        list3 = [['p_phi','p_theta','p_theta'],['y'],['x','y'],['x','y'],
                     ['x','y'],['log(p)','p_phi','p_theta'],['x','y']]
        list4 = [0,1,2]
        range_2d=[[[-7,7.05,-3.14,3.14],[-7,7.05,0,3.14],[-3.14,3.14,0,3.14]],
                      [[-125,125,-125,125]],
                      [[-7,7.05,-125,125],[-7,7.05,-125,125]],
                      [[-3.14,3.14,-150,150],[-3.14,3.14,-150,150]],
                      [[0,3.14,-150,150],[0,3.14,-150,150]],
                      [[-0.032,1.05,-7,7.05],[-0.032,1.05,-3.14,3.14],[-0.032,1.05,0,3.14]],
                      [[-0.032, 1.05, -125, 125],[-0.032, 1.05, -125, 125]]]
        for i in range(len(list2)):
            fig = plt.figure(figsize=(27, 15))
            for j,k,l,m in zip(list1[0], list2[i], list3[i],list4):
                plt.subplot(2,3,j)
                range_xy = range_2d[i][m]
                my_plot_2d_hist(dic_mc[k], dic_mc[l], 30,'mc '+k+' vs '+l,range_xy)
            for j,k,l,m in zip(list1[1], list2[i],list3[i],list4):
                plt.subplot(2,3,j)
                range_xy = range_2d[i][m]
                my_plot_2d_hist(dic_resamp[k], dic_resamp[l], 30,'resamp '+k+' vs '+l, range_xy)
            pdf.savefig(fig)

if __name__=='__main__':
    main(sys.argv)
