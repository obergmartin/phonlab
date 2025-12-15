import pandas as pd

# %%

@pd.api.extensions.register_dataframe_accessor("phon")
class DataFrameAccessor:
    def __init__(self, pandas_obj):
        # self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # make assertions about shared boundaries?
        pass

    def add_segment(self, t1, text):
        """Add segment to textgrid dataframe.

        Inserting a segment effectively splits one segment into two.  The
        original segment will keep its t1 and text label and take the t1 input
        for its new t2 value.  The inserted segment only needs a t1 and will
        use t2 from the segment it is inserted into.
        """
        idx = self._obj['t1'].searchsorted(t1)
        t2 = self._obj['t2'][idx]
        x = dict(t1=t1, t2=t2, text=text)
        df = pd.concat([self._obj, pd.DataFrame([x])], ignore_index=True)
        df = df.sort_values(by='t1', ignore_index=True)
        return df

# %%

df = pd.DataFrame()
df['t1'] = [1,2]
df['t2'] = [2,3]
df['text'] = ['a', 'b']
df = df.phon.add_segment(t1=1.1, text='foo')
print(df)

# %%
# from pandas.core.base import PandasObject
# def add_segment(df, x):
#     df = pd.concat([df, pd.DataFrame([x])])
#     df = df.sort_values(by='t1')
#     return df
#
# PandasObject.add_segment = add_segment
# df.add_segment(x=dict(t1=1.1, t2=2, text='foo'))

# %%
