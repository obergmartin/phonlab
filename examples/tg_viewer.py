#!/usr/bin/env python

import phonlab as phon

print(phon.__name__,phon.__version__)

# %%

fn = "./phonlab/data/example_audio/im_twelve.wav"
df = phonlab.tg_to_df("./phonlab/data/example_audio/im_twelve.TextGrid")

import phonlab
v = phonlab.Viewer(fn, df[0:2])
# %%
