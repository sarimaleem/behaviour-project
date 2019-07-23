import numpy as np
import matplotlib.pyplot as plt

def plot_miss_offset(x, df):
    num_hits = int(df.sum()['miss'])
    missoffset = np.zeros(shape=(num_hits, 500))
    missoffset[:, :] = np.nan
    dt = 0.01

    miss_trace = x[df['miss']]
    hit_rt = np.asarray(df[df['miss']]['rt'])
    for i in range(num_hits):
        rt_index = int(hit_rt[i] / dt) + 1
        missoffset[i, 500 - rt_index:] = miss_trace[i, 0:rt_index]

    missoffset = np.nanmean(missoffset, axis=0)[~np.isnan(np.nanmean(missoffset, axis=0))]
    plt.plot(missoffset)
    plt.title('hit offset')