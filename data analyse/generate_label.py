#取各种类型粒子数量的1/10,暂不考虑gamma
import numpy as np
#labels_minus_11 = np.zeros((31678901, 8))
#labels_minus_11[:, 0] = labels_minus_11[:, 0] + 1
labels_11 = np.zeros((242238380, 8))
labels_11[:, 1] = labels_11[:, 1] + 1
#labels_minus_13 = np.zeros((15373496, 8))
#labels_minus_13[:, 2] = labels_minus_13[:, 2] + 1
#labels_13 = np.zeros((10950522, 8))
#labels_13[:, 3] = labels_13[:, 3] + 1
#先不考虑gamma
#labels_22 = np.zeros((319123652, 8))
#labels_22[:, 4] = labels_22[:, 4] + 1
#labels_minus_211 = np.zeros((8038389, 8))
#labels_minus_211[:, 4] = labels_minus_211[:, 4] + 1
#labels_211 = np.zeros((11121593, 8))
#labels_211[:, 5] = labels_211[:, 5] + 1
#labels_2212 = np.zeros((2414370, 8))
#labels_2212[:, 6] = labels_2212[:, 6] + 1
#labels_num = np.concatenate((labels_minus_11, labels_11))
#labels_num = np.concatenate((labels_num, labels_minus_13))
#labels_num = np.concatenate((labels_num, labels_13))
#先不考虑gamma
#labels_num = np.concatenate((labels_num, labels_22))
#labels_num = np.concatenate((labels_num, labels_minus_211))
#labels_num = np.concatenate((labels_num, labels_211))
#labels_num = np.concatenate((labels_num, labels_2212))
#np.save("./data/Geant4/eight_particles/labels_onetenth.npy", labels_num)
np.save("./data/Geant4/eight_particles/label_11_all.npy", labels_11)
print("completed !!")
