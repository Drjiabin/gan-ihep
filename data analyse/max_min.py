#Find the maximum and minimum values of all the features in rootracker files(no log)
#Author: CUI Jiabin<cuijiabin@ihep.ac.cn>
#Data: 2020 July 06

import uproot
import numpy as np
import pandas as pd
import os
import sys
from inspect import currentframe

pdgs=[-11,11,-13,13,22,-211,211,2212]
#pdgs=[11]
feature_names = ['p','p_phi','p_theta','x','y','t']

def debug_print(arg):
    frameinfo = currentframe()
    print(frameinfo.f_back.f_lineno,":",arg)

#define and initialize min and max array
#Fill the minimum and maximum arrays with the maximum and minimum values of floating-point numbers, respectively
pdg_features_min = np.full((len(pdgs),len(feature_names)),sys.float_info.max)
pdg_features_max = np.full((len(pdgs),len(feature_names)),sys.float_info.min)

#Use uproot to extract the data corresponding to pdgid in the .rootracker file
def get_feature_from_rootracker_file(pdg,file_num):
    #root_path=os.path.join(file_name)
    path="/hpcfs/bes/mlgpu/cuijb/gan/data/Geant4/eight_particles"
    id_str = "%04d" % file_num
    file_name = os.path.join(path, "extracted_"+str(pdg)+"_rootracker_"+id_str+".txt.root")
    events = uproot.open(file_name)['T']
    df_all = events.pandas.df(['p','p_phi','p_theta','x','y','z','t'],flatten=True)
    df_all = df_all[(df_all['z']>4175.0027)&(df_all['z']<4175.003)&(np.sqrt(df_all['x']**2+df_all['y']**2)<120)]
    feature_by_pdg = df_all.drop(['z'], axis=1)
    return feature_by_pdg


def find_min_max(pdgs):
    #loop over files
    for i_pdg,pdg in enumerate(pdgs):
        #loop over pdgs
        for file_num in range(1, 13):
            feature_by_pdg = get_feature_from_rootracker_file(pdg,file_num)
            #if feature_by_pdg
            #loop over features
            for i_feature,key in enumerate(feature_names):
                feature=feature_by_pdg[key]
                #get min and max
                if feature.min()<pdg_features_min[i_pdg][i_feature]:
                    pdg_features_min[i_pdg][i_feature] = feature.min()
                if feature.max()>pdg_features_max[i_pdg][i_feature]:
                    pdg_features_max[i_pdg][i_feature] = feature.max()
                #print(i_pdg,i_feature,'min %f' % feature.min())
                #print(i_pdg,i_feature,'max %f' % feature.max())
    print(pdg_features_min)
    print(pdg_features_max)
    np.savez("eight_particle_max_min.npz",pdg_features_min=pdg_features_min,pdg_features_max=pdg_features_max)



def main(argv):
    file_end=13
    #pdgs=[-11,11,-13,13,22,-211,211,2212]
    # find max min of all features in all the files
    find_min_max(pdgs)
#plot
if __name__ == '__main__':
    main(sys.argv)

