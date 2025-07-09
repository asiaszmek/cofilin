import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fname = "Model_Cof-HFGlu_jRGECO1.h5"
exp_res_spine = "ca_spine_HFGlu_jRGECO1.csv"
exp_res_dend = "ca_dend_HFGlu_jRGECO1.csv"

t_init = 10000

def get_fluo_sig(signal, t_init, dt):
    min_len = min([len(dat) for dat in signal])
    out = np.array([data[:min_len] for data in signal]).mean(axis=0)
    basal = np.mean(out[:int(t_init/dt)])
    out = (out-basal)/basal
    return out, min_len


def region_volumes(my_file):
    if isinstance(my_file, str):
        my_file = h5py.File(my_file)
    grid_list = get_grid_list(my_file)
    regions = get_regions(my_file)
    volumes = {}
    for region in regions:
        volumes[region] = 0
    for cell in grid_list:
        key = get_key(cell)
        volumes[key] += float(cell[12])

    return volumes


def get_grid_list(My_file):
    return np.array(My_file['model']['grid'])


def nano_molarity(N, V):
    return 10 * N / V / NA


def pico_sd(N, S):
    return 10 * N / S / NA


def get_populations(my_file, trial='trial0', output='__main__'):
    return np.array(my_file[trial]['output'][output]['population'])


def get_times(My_file, trial, output):
    return np.array(My_file[trial]['output'][output]['times'])


if __name__ == "__main__":
    data_spine = []
    data_dend = []
    my_file = h5py.File(fname, "r")


    for key in my_file.keys():
        if not key.startswith("trial"):
            continue

        psd = get_populations(my_file, trial=key,
                             output="fluo_PSD")
        head = get_populations(my_file, trial=key,
                              output="fluo_head")
        neck = get_populations(my_file, trial=key,
                              output="fluo_neck")
        spine = psd.sum(axis=1) + head.sum(axis=1) + neck.sum(axis=1)
        data_spine.append(spine)
        dend = get_populations(my_file, trial=key,
                                         output="fluo_dend")

        data_dend.append(dend.sum(axis=1))

        

    time = get_times(my_file, key, "__main__")-t_init      

    dt = time[1] - time[0]
    out_spine, min_len = get_fluo_sig(data_spine, t_init, dt)
    out_dend, min_len = get_fluo_sig(data_dend, t_init, dt)
    spine_res = pd.read_csv(exp_res_spine)
    dend_res = pd.read_csv(exp_res_dend)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(time[:min_len]/1000, out_spine, "tab:green", label="Spine model")
    ax.plot(time[:min_len]/1000, out_dend, "tab:blue", label="Dendrite model")
    ax.plot(spine_res["x"], spine_res["Curve1"], "g", label="Spine experiment")
    ax.plot(dend_res["x"], dend_res["Curve1"], "b", label="Dendrite experiment")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Fluorescence change")
    ax.legend()
    
    plt.show()
