# %%
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from pathlib import Path


from record_utils import plot, plot_all, plot_all_alpha



# %%
path = "~/workspace/spinningup/data/sac-gpu-hopper/sac-gpu_s0"
plot(path,r=2)
# %%
path = "~/workspace/spinningup/data/td3-hopper"
plot_all(path, r=5)
# %%
path = "~/workspace/spinningup/data/td3-ant"
plot_all(path, r=2)
# %%
path = "~/workspace/spinningup/data/td3-mod-ant"
plot_all(path, r=2)
# %%
path = "~/workspace/spinningup/data/td3-mod-ant2"
plot_all(path, r=2)
# %%
path = "~/workspace/spinningup/data/sac-ant"
plot_all(path, r=2)

# %%
path = "~/workspace/spinningup/data/sac-alpha-ant-qonly"
plot_all(path, r=2, plot_avg=True)
plot_all_alpha(path, r=0)
# %%
path = "~/workspace/spinningup/data/sac-hopper"
plot_all(path, r=0)
# %%
