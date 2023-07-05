import numpy as np


list_feat_0_10 = np.load("Datasets/PERCLOS_EARopen_EAR_Blink/010_115/PERCLOS_EARopen_EAR_BLINKduration_01_48.npy")

list_feat_12_17 = np.load("Datasets/PERCLOS_EARopen_EAR_Blink/120_180/PERCLOS_EARopen_EAR_BLINKduration_01_48.npy")

list_feat_19_33 = np.load("Datasets/PERCLOS_EARopen_EAR_Blink/190_3310/PERCLOS_EARopen_EAR_BLINKduration_01_48.npy")

list_feat_35_43 = np.load("Datasets/PERCLOS_EARopen_EAR_Blink/350_440/PERCLOS_EARopen_EAR_BLINKduration_01_48.npy")

list_feat_37_48 = np.load("Datasets/PERCLOS_EARopen_EAR_Blink/370_4810/PERCLOS_EARopen_EAR_BLINKduration_37_48.npy")

list_feat_11_18_34_36 = np.load("Datasets/PERCLOS_EARopen_EAR_Blink/11_18_33_36/PERCLOS_EARopen_EAR_BLINKduration_37_48.npy")

list_feat_0_10 = list_feat_0_10[0:120] # 120
list_feat_11 = list_feat_11_18_34_36[0:12] # 12
list_feat_12_17 = list_feat_12_17[0:72] # 72
list_feat_18 = list_feat_11_18_34_36[12:24] # 12
list_feat_19_33 = list_feat_19_33 # 168 ohne 25
list_feat_34_36 = list_feat_11_18_34_36[24:60] # 36
list_feat_37_48 = list_feat_37_48 # 144

list_all = np.concatenate((list_feat_0_10,list_feat_11,list_feat_12_17,list_feat_18,list_feat_19_33,list_feat_34_36,list_feat_37_48))
np.save("Datasets/PERCLOS_EARopen_EAR_BLINKduration_list.npy", list_all)

list_lengths = {
    'list_feat_0_10': len(list_feat_0_10),
    'list_feat_11': len(list_feat_11),
    'list_feat_12_17': len(list_feat_12_17),
    'list_feat_18': len(list_feat_18),
    'list_feat_19_33': len(list_feat_19_33),
    'list_feat_34_36': len(list_feat_34_36),
    'list_feat_37_48': len(list_feat_37_48)
}

for list_name, length in list_lengths.items():
    print(f"Liste: {list_name}, Länge: {length}")

input_list = [2.97222222e-01, 2.89189189e+00, 2.29545478e-01, 2.28563396e-01,
              6.22222222e-01, 2.94736842e+00, 2.28644572e-01, 2.26397551e-01,
              5.28333333e+00, 6.62717770e+00, 1.90927718e-01, 1.78454619e-01,
              6.11111111e-02, 2.00000000e+00, 3.18036829e-01, 3.17707035e-01,
              2.27777778e-01, 3.03703704e+00, 3.16376723e-01, 3.15052307e-01,
              4.50000000e-01, 3.44680851e+00, 2.88998352e-01, 2.86790891e-01]

output_list = []
sublist1 = []
sublist2 = []
sublist3 = []
sublist4 = []
for i in range(0, len(input_list), 4):
    sublist1.append(input_list[i])
    sublist2.append(input_list[i+1])
    sublist3.append(input_list[i+2])
    sublist4.append(input_list[i+3])

list_features = [sublist1, sublist2, sublist3, sublist4]

output_list = []

for i in range(len(list_features[0])):
    sublist = [row[i] for row in list_features]
    output_list.append(sublist)


list_fps = [30, 30, 30, 30, 30, 30]

for i in range(len(list_fps)):
    output_list[i][0] = output_list[i][0]/list_fps[i]
    output_list[i][1] = (output_list[i][1]/list_fps[i])*1000
fps = np.load("Datasets/fps.npy")

print("FPS")
print(fps)
print(len(fps))