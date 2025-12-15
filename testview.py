#!/usr/bin/env python
# %%
import numpy as np
import importlib
import phonlab as phon
from IPython.display import Audio
import matplotlib.pyplot as plt


import phonlab
from phonlab import Viewer



# %%

fn = "./phonlab/data/example_audio/im_twelve.wav"

df = phonlab.tg_to_df("./phonlab/data/example_audio/im_twelve.TextGrid")
v = Viewer(fn, df[0])
# %%
