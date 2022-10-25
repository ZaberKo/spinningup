#%%
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from pathlib import Path
def smooth(y, radius, mode='two_sided', valid_only=False):
    '''Smooth signal y, where radius is determines the size of the window.

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / \
            np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / \
            np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out


def plot(path, r=0):
    df=pd.read_csv(os.path.join(path,"progress.txt"),delimiter="\t")
    plt.plot(dpi=300,figsize=(300,600))
    plt.plot(df["Epoch"],smooth(df["AverageTestEpRet"],r),label="return",lw=0.4)
    plt.fill_between(df["Epoch"],df["MinTestEpRet"],df["MaxTestEpRet"],alpha=0.2)
    plt.legend()
    plt.show()

def plot_all(path, r=0):
    
    plt.plot(dpi=300,figsize=(300,600))
    for exp_path in Path(path).expanduser().iterdir():
        print(exp_path)
        df=pd.read_csv(os.path.join(exp_path,"progress.txt"),delimiter="\t")
        plt.plot(df["Epoch"],smooth(df["AverageTestEpRet"],r),lw=0.4)
        plt.fill_between(df["Epoch"],smooth(df["MinTestEpRet"],r),smooth(df["MaxTestEpRet"],r),alpha=0.2)
    plt.legend()
    plt.show()
#%%
path="~/workspace/spinningup/data/sac-gpu-hopper/sac-gpu_s0"
plot(path)
# %%
path="~/workspace/spinningup/data/td3-hopper"
plot_all(path,r=5)
# %%
path="~/workspace/spinningup/data/td3-ant"
plot_all(path,r=5)
# %%
