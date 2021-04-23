import pandas as pd
import numpy as np
import uproot
import os
import time
pdg_list = [-11,11,-13,13,22,-211,211,2212]
for pdg_id in pdg_list:
    shape = 0
    print(pdg_id)
    for file_id in range(1, 13):
        #path="./data/Geant4/circle_1"
        print(file_id)
        path="./data/Geant4/eight_particles"
        id_str = "%04d" % file_id
        file_name = os.path.join(path, "extracted_"+str(pdg_id)+"_rootracker_"+id_str+".txt.root")
        events = uproot.open(file_name)['T']
        #df_all = events.pandas.df(['beam_px','beam_py','beam_pz','global_x','global_y', 'global_z', "beam_t"], flatten=True)
        df_all = events.pandas.df(['p','p_phi','p_theta','x','y','z','t'], flatten=True)
        df_all = df_all[(df_all['z']>4175.0027)&(df_all['z']<4175.003)&(np.sqrt(df_all['x']**2+df_all['y']**2)<120)]
        df_all = df_all.drop(['z'], axis=1)
        shape = df_all.shape[0]+shape
    print("the number of pdg_id %d is %d"%(pdg_id, shape))
