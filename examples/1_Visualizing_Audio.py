#!/usr/bin/env python
# %%
import numpy as np
import importlib
import phonlab as phon
from IPython.display import Audio
import matplotlib.pyplot as plt

print(phon.__name__,phon.__version__)

# %%

example_file = importlib.resources.files('phonlab') / 'data' / 'example_audio' / 'stereo.wav'
local_file = 'dimex/s09003.wav'

x,fs = phon.loadsig(example_file,chansel=[0])  # taking just the left channel, with 'chansel'
print(f'number of samples = {len(x)}, sampling rate = {fs}, duration = {len(x)/fs}')

y,fs = phon.prep_audio(x,fs,target_fs=16000)
print(f'number of samples = {len(y)}, sampling rate = {fs}, duration = {len(y)/fs}')

# %%

def df2lines(df):
    lines = np.concat([df.t1.to_numpy(), [df.t2.iloc[-1]]])
    return lines

df2lines(df0)

# %%
import phonlab
from phonlab import Viewer


v = Viewer(fn)

# %%

import subprocess
start_time = .1
end_time = 1.0
fn = "./phonlab/data/example_audio/im_twelve.wav"

df = phonlab.tg_to_df("./phonlab/data/example_audio/im_twelve.TextGrid")
# fn = "./phonlab/data/example_audio/the_soviet_union.wav"
subprocess.run(["ffplay", "-ss", f"{start_time}", "-t", f"{end_time}", "-nodisp", f"-autoexit", f"{fn}"])
subprocess.run("pwd")
# %%
