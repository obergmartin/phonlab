__all__=['Viewer']

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import subprocess
from .signal import loadsig
from phonlab import prep_audio
from phonlab import sgram
from time import time
import matplotlib.style as mplstyle
mplstyle.use('fast')


def df2lines(df):
    lines = np.concat([df.t1.to_numpy(), [df.t2.iloc[-1]]])
    return lines

def df2texobj(df):
    return zip(
        df[['t1','t2']].mean(axis=1),
        df.iloc[:, 2]
    )

def make_linecollection(lines):
    lpts = [[(x,0), (x,1)] for x in lines]
    return LineCollection(lpts)


class Viewer:
    def __init__(self, fn, df):
        self.start_x = None
        self.n_rows = 3
        lines = df2lines(df)
        self.lines = lines

        self.fig, self.axs = plt.subplots(nrows=self.n_rows, ncols=1)
        for i in range(1, self.n_rows):
            self.axs[i].sharex(self.axs[0])

        self.fn = fn
        wavdata, fs = loadsig(fn, chansel=[0])  # taking just the left channel, with 'chansel'
        self.axs[0].plot(np.arange(0, wavdata.size)/fs, wavdata, c='k')
        self.axs[0].set_xlim([0,wavdata.size/fs])
        sgram(wavdata, fs, tf=8000, ax=self.axs[1])
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0)

        cid_keypress = self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        l, b, w, h =  self.axs[self.n_rows-1].get_position().extents
        print(l, b, w, h)
        self.axprev = self.fig.add_axes([l, b, 0, 0.05])
        self.axcur = self.fig.add_axes([l, b, w-l,  0.05])
        self.axnext = self.fig.add_axes([l, b, 0, 0.05])
        self.bnext = Button(self.axnext, '')
        self.bnext.on_clicked(self.play_segment)
        self.bcur = Button(self.axcur, '')
        self.bcur.on_clicked(self.play_segment)
        self.bprev = Button(self.axprev, '')
        self.bprev.on_clicked(self.play_segment)

        self.tier_kwargs = dict(
            alpha=1,
            c="b",
            ls="-",
        )
        self.position_kwargs = dict(
            alpha=0.7,
            c="r",
            ls="--",
        )
        self.line_kwargs = dict(
            color="b",
            ls="-",
            lw=4,
        )
        self.span_kwargs = dict(
            alpha=0.3,
            ls="-",
        )
        self.seg_span_kwargs = dict(
            alpha=0.3,
            fill='g',
            ls="-",
        )
        self.seg_kwargs = dict(
            fill=False,
            edgecolor="b",
            lw=2,
            ls="-",
        )

        self.line_axs = list(range(len(self.axs)))
        self.cursor_lines = [
            a.axvline(1, 0, 1, **self.position_kwargs)
            for i, a in enumerate(self.axs)
            if i in self.line_axs
        ]
        # collection of handles for selection span on axes
        self.current_span = [
            a.axvspan(1, 2, **self.span_kwargs)
            for i, a in enumerate(self.axs)
        ]
        # selected tier segment
        self.seg_span = self.axs[2].axvspan(-2,-1, **self.seg_span_kwargs)
        # format lims for tier axes
        self.axs[2].get_yaxis().set_visible(False)
        self.axs[2].get_yaxis().set_ticks([])
        # 
        # for t1 in self.lines:
        #     # self.axs[2].axvspan(t1, t2, -0.1, 1.02, **self.seg_kwargs)
        #     self.axs[2].axvline(t1, -0.1, 1.02, **self.line_kwargs)
        self.tier_lines = make_linecollection(self.lines)
        self.axs[2].add_collection(self.tier_lines)
        for t, txt in df2texobj(df):
            self.axs[2].text(t, 0.5, txt, ha="center")


        plt.show()

    ## methods for updating visuals when user interacts with figure

    def play_segment(self, event):
        xlims = self.axs[0].get_xlim()
        s1 = self.current_span[0].get_x()
        s2 = s1+self.current_span[0].get_width()
        if event.inaxes == self.axprev:
            print("prev")
            start_time = xlims[0]
            end_time = s1
        elif event.inaxes == self.axcur:
            print("cur")
            start_time = s1
            end_time = s2
        elif event.inaxes == self.axnext:
            print("next")
            start_time = s2
            end_time = xlims[1]
        else:
            return
        print(f"{start_time=}")
        print(f"{end_time=}")
        # this has problems because of seek position?
        subprocess.run(["ffplay", "-loglevel", "quiet", "-ss", f"{start_time}", "-t", f"{end_time}", "-nodisp", f"-autoexit", f"{self.fn}"])

    def set_active_tier_boundary(self, ind):
        """Click on a tier boundary to move it.
        """
        # clear selection line
        lw = np.ones(len(self.lines), dtype=int) *1
        if ind >= 0:
            lw[ind] = 4
        self.tier_lines.set(linewidths=lw)

    def set_active_tier_segment(self, ind):
        """Click in a tier segment to higlight it.
        """
        # clear selection line
        # show tier selection
        if ind >= 0:
            self.seg_span.set_visible(True)
            self.seg_span.set_x(self.lines[ind])
            self.seg_span.set_width(self.lines[ind+1] - self.lines[ind])
        else:
            self.seg_span.set_visible(False)

    def set_active_span(self, l, w, visible=True):
        """Update active span across all axes.
        """
        # draw spans
        for a in self.current_span:
            a.set_visible(visible)
            a.set_x(l)
            a.set_width(w)

    def set_active_line(self, p1):
        """Click on a signal to get value.
        Also allows adding/splitting segments.
        """
        # clear selection line
        for cur_line in self.cursor_lines:
            cur_line.set_xdata([p1])

    def resize_play_buttons(self, s1, s2):
        l, b, w, h =  self.axs[self.n_rows-1].get_position().extents
        xsz = np.diff(self.axs[self.n_rows-1].get_xlim())[0]
        p1 =  ((s1/xsz)* (w-l))
        p2 =  ((s2/xsz)* (w-l))
        w3 = (xsz-s2)/xsz * (w-l)

        self.axprev.set_position([l, b, p1, .05])
        self.axcur.set_position([l+p1, b, p2-p1, .05])
        self.axnext.set_position([l+p2, b, w3, .05])

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        self.epsilon = 5  # in n_pixels, make dependant on zoom level?

        eventxt = self.tier_lines.get_transform().inverted().transform((event.x,event.y))
        xy = [(x,0) for x in self.lines]
        xyt = self.tier_lines.get_transform().transform(xy)  # to display coords
        d = np.array([x[0] for x in xyt]) - event.x
        ind = int(abs(d).argmin())
        if abs(d[ind]) < self.epsilon:
            return ind
        else:
            ind =  np.searchsorted([x[0] for x in xyt], event.x)
            return (ind-1, ind)

    ## event callbacks

    def on_keypress(self, event):
        print('press', event.key)
        sys.stdout.flush()
        if event.key == 'x':
            pass
            # visible = xl.get_visible()
            # xl.set_visible(not visible)
            # fig.canvas.draw()
        elif event.key == "ctrl+1":
            print(f"adding sement on tier 1 at {self.start_x=}")
            pos = np.searchsorted(self.lines, self.start_x)
            self.lines = np.insert(self.lines, pos, self.start_x)
            self.tier_lines.set_segments(lines)
            self.set_tier_labels(labels)

    def on_press(self, event):
        """Record the starting x-coordinate on button press."""
        # is click on existing:
        # tier axes
        if event.inaxes == self.axs[2]:
            ind = self.get_ind_under_point(event)
            # span boundary
            if isinstance(ind, int):
                p1 = self.lines[ind]
                self.set_active_tier_boundary(ind)
                self.set_active_line(event.xdata)
                self.set_active_tier_segment(-1)
                self.set_active_span(0, 0, False)
                self.resize_play_buttons(p1, p1)
            # tier span
            else:
                p1, p2 = self.lines[ind[0]], self.lines[ind[1]]
                # setspanhere
                self.set_active_tier_boundary(-1)
                self.set_active_tier_segment(ind[0])
                self.set_active_span(p1, p2-p1)
                self.resize_play_buttons(p1, p2)
        # sgram or signal
        elif event.inaxes == self.axs[0] or event.inaxes == self.axs[1]: 
            # print("setting line", event.xdata)
            # clear span
            self.set_active_tier_boundary(-1)
            self.set_active_span(0, 0, False)
            # draw selection line
            self.set_active_line(event.xdata)
            self.start_x = event.xdata
            self.resize_play_buttons(event.xdata, event.xdata)
        # for a in self.current_span:
        #     a.set_x(event.xdata)
        #     a.set_width(0)
        self.fig.canvas.draw_idle() # Redraw the canvas efficiently

    def on_motion(self, event):
        """Dynamically update the axvspan as the mouse moves (dragging)."""
        # if:
        # moving span
        # moving edge/point
        if self.start_x is None or event.inaxes != self.axs[0]:
            return

        end_x = event.xdata
        # Draw the new temporary vertical span
        x = min(self.start_x, end_x)
        w = abs(self.start_x - end_x)
        self.set_active_span(x, w)
        s1 = self.current_span[0].get_x()
        s2 = s1+self.current_span[0].get_width()
        self.resize_play_buttons(s1, s2)
        self.set_active_tier_segment(-1)
        # self.fig.canvas.draw_idle() # Redraw the canvas efficiently
        self.fig.canvas.draw_idle() # Redraw the canvas efficiently

    def on_release(self, event):
        """Finalize the axvspan on button release."""
        if self.start_x is None or event.inaxes != self.axs[0]: 
            return
        # self.selection_span.get_xy()[i, 0] = new_xmin if i in [0, 3] else new_xmax
        # s1 = self.current_span.get_x()
        # s2 = self.current_span.get_width()
        # p = self.current_span.get_xy()
        # print(f"{s1=} {s2=}")
        # print(f"{p=}")
        # The final span is left on the plot by the last on_motion call
        self.start_x = None


