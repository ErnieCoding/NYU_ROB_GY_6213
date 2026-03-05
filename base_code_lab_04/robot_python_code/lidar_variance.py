import data_handling

data = { # pkl file : list of distances at angles 0, 90, 180, 270
    "robot_data_0_0_04_03_26_20_02_22.pkl": [46, 214, 160, 177],
    "robot_data_0_0_04_03_26_20_18_39.pkl":[114, 82, 93, 312],
    "robot_data_0_0_04_03_26_20_21_58.pkl": [203, 43, 190, 167],
}

n = 0
sum_sq_diff = 0
for filename in data.keys():
    pf_data = data_handling.get_file_data_for_pf("./data_LiDAR_variance/"+filename)

    for t in range(1, len(pf_data)):
        row = pf_data[t]
        z_t = row[2]

        target_angles = [0, 90, 180, 270]
        for i in range(len(target_angles)):
            if target_angles[i] in z_t.angles:
                distance_val = z_t.distances[z_t.angles.index(target_angles[i])]
                n += 1
                
                sum_sq_diff += (data[filename][i]/100 - distance_val/1000) ** 2

var = sum_sq_diff / (n-1)

print(var)

# VARIANCE: 0.000210764044943819