# %% [markdown]
# # EDA starter — open as notebook via Jupytext or copy cells into .ipynb
# %%
# %load_ext autoreload
# %autoreload 2
from wimbledon.data import load_points_with_direction, load_serve_direction

# %%
sd = load_serve_direction("m")
grass = sd[sd["Surface"] == "Grass"]
grass.head()
