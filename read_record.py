# %%
from record_utils import plot, plot_all, plot_all_alpha, plotly_plot_all


# %%
path = "./data/sac-fix-mod-hopper-256"
plot_all(path, r=2)

# %%
# %matplotlib widget
path = "./data/rl-cpu/sac_tune_alpha_HalfCheetah-v3_2022_11_11_02_28_01"
plot_all(path, r=2)
# %%
path = "./data/rl-cpu/sac_HalfCheetah-v3_2022_11_11_04_53_15"
plot_all(path, r=2)
# %%
path = "./data/rl-cpu/sac_tune_alpha_HalfCheetah-v3_2022_11_11_02_28_01"
plotly_plot_all(path, r=2)
# %%
