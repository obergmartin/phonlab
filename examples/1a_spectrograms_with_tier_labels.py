#!/usr/bin/env python

# %%
import importlib
import phonlab as phon

# from IPython.display import Audio
import matplotlib.pyplot as plt
from matplotlib.text import Text

print(phon.__name__, phon.__version__)

# %%

example_file = (
    importlib.resources.files("phonlab") / "data" / "example_audio" / "stereo.wav"
)
local_file = "dimex/s09003.wav"

x, fs = phon.loadsig(
    example_file, chansel=[0]
)  # taking just the left channel, with 'chansel'
print(f"number of samples = {len(x)}, sampling rate = {fs}, duration = {len(x)/fs}")

y, fs = phon.prep_audio(x, fs, target_fs=16000)
print(f"number of samples = {len(y)}, sampling rate = {fs}, duration = {len(y)/fs}")

# %%


def plot_tier_times(times=None, labels=None, ax=None):
    if ax is None:
        ax = plt.gca()
    for t1, t2 in times:
        ax.axvline(t1, color="k")
        ax.axvline(t2, color="k")
    for (t1, _), lab in zip(times, labels):
        ax.text(t1, 0, lab, size=16)


# %%


def plot_tier_spans(spans=None, labels=None, ax=None, **kwargs):
    if "alpha" not in kwargs.keys():
        kwargs["alpha"] = 0.3
    if "color" not in kwargs.keys():
        kwargs["color"] = "b"
    breakpoint()
    if ax is None:
        ax = plt.gca()
    for s1, s2 in spans:
        ax.axvspan(s1, s2, **kwargs)
    for (s1, _), lab in zip(spans, labels):
        ax.text(s1, 0, lab, size=16)


# %%


def make_figure(df, n_plots=1, n_tiers=1):
    fig = plt.figure(figsize=(5, 2), dpi=72)
    height_ratios = [1] * n_tiers + [10] * n_plots
    gs = fig.add_gridspec(
        nrows=len(height_ratios), ncols=1, height_ratios=height_ratios
    )
    ax = [fig.add_subplot(x) for x in gs]
    for i in range(n_tiers):
        # hide axes for tiers
        ax[i].set_axis_off()
    for i in range(1, len(height_ratios)):
        # share x axis across all plots
        ax[0].sharex(ax[i])

    return fig, ax


# %%
fig, ax = make_figure([], 1, 1)
plot_tier_times(times=[[1, 1.2], [1.3, 1.5]], labels=["a", "b"], ax=ax[0])
plot_tier_times(times=[[1, 1.2], [1.3, 1.5]], labels=[], ax=ax[1])
ret = phon.sgram(y, fs, ax=ax[1])
plt.show()
# %%
fig, ax = make_figure([], 1, 1)
plot_tier_spans(spans=[[1, 1.2], [1.3, 1.5]], labels=["a", "b"], ax=ax[0], color="g")
plot_tier_spans(spans=[[1, 1.2], [1.3, 1.5]], labels=[], ax=ax[1])
ret = phon.sgram(y, fs, ax=ax[1])
plt.show()

# %%
