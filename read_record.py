# %%
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from pathlib import Path


from record_utils import plot, plot_all, plot_all_alpha




# %%
path = "~/workspace/workspace/spinningup/data/sac-fix-mod-hopper-256"
plot_all(path, r=2)

# %%
