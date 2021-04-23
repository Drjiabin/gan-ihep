import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import ks_2samp
from scipy import stats

import time
import uproot

def load_gan(pdgid):
    path="/hpcfs/bes/mlgpu/cuijb/gan/root_gan"
    file_name = os.path.join(path, pdgid+".root")
    events = uproot.open(file_name)['ttree']
    data_arr = events.arrays(['momentum','phi','theta','x','y','time'], library="pd")
    data_arr = np.array(data_arr)
    print("gan data shape is :", data_arr.shape)
    return data_arr

def Antinormalization(data,max,min):
    antinorm_data = data * (max-min) + min 
    return antinorm_data


def load_roodata(file_num_start, file_num_end):
    time_extract = time.time()
    roo_data = pd.DataFrame()
    for file_id in range(file_num_start, file_num_end):
        print(file_id)
        path="/hpcfs/bes/mlgpu/cuijb/gan/data/Geant4/eight_particles"
        id_str = "%04d" % file_id
        file_name = os.path.join(path, "extracted_13_rootracker_"+id_str+".txt.root")
        events = uproot.open(file_name)['T']
        df_all = events.arrays(['p','p_phi','p_theta','x','y','z','t'], library="pd")
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
    data_min = [-1.2, -3.2, 9.4e-05, -120, -120, -0.03]
    data_max = [7.1, 3.2, 2, 120, 120, 0.61]

    path = "/hpcfs/bes/mlgpu/cuijb/gan/root_gan"
    gan_data = load_gan('13_onetenth')[0:100, :]
    start = 1
    end = 2
    roo_data = load_roodata(start, end)
    roo_data = roo_data[0:gan_data.shape[0],:]
    print(gan_data.shape)
    print(roo_data.shape)
    for j in range(6):
        gan_data[:,j] = Antinormalization(gan_data[:,j],data_max[j],data_min[j])

    for i in range(6):
        #print("t test is: ")
        #print(stats.ttest_ind(gan_data[:, i],roo_data[:, i]))
        #print('chisquare is: ')
        #print(stats.chisquare(roo_data[:, i], gan_data[:, i]))
        print('ttest rel is: ')
        print(stats.ttest_rel(roo_data[:, i], gan_data[:, i]))

if __name__=='__main__':
    main(sys.argv)

