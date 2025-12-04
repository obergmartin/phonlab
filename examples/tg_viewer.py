#!/usr/bin/env python

import phonlab as phon

print(phon.__name__,phon.__version__)

# %%

fn = "./phonlab/data/example_audio/im_twelve.wav"
df = phon.tg_to_df("./phonlab/data/example_audio/im_twelve.TextGrid")

v = phon.Viewer(fn, df[0])
# %%
