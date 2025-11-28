__all__=['Viewer']

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
import subprocess
from .signal import loadsig
from phonlab import prep_audio
from phonlab import sgram


class Viewer:
    def __init__(self, fn):
        self.start_x = None
        self.current_span = None
        self.fig, self.axs = plt.subplots(nrows=3, ncols=1)
        self.fn = fn
        wavdata,fs = loadsig(fn, chansel=[0])  # taking just the left channel, with 'chansel'
        self.axs[0].plot(np.arange(0, wavdata.size)/fs, wavdata)
        self.axs[0].set_xlim([0,wavdata.size/fs])
        sgram(wavdata, ax=self.axs[0])


        cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        l, b, w, h =  self.axs[0].get_position().extents
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
        plt.show()

    def play_segment(self, event):
        xlims = self.axs[0].get_xlim()
        s1 = self.current_span.get_x()
        s2 = s1+self.current_span.get_width()
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

    def resize_play_buttons(self):
        l, b, w, h =  self.axs[0].get_position().extents
        xsz = np.diff(self.axs[0].get_xlim())[0]
        s1 = self.current_span.get_x()
        s2 = s1+self.current_span.get_width()
        p1 =  ((s1/xsz)* (w-l))
        p2 =  ((s2/xsz)* (w-l))
        w3 = (xsz-s2)/xsz * (w-l)
            
        self.axprev.set_position([l, b, p1, .05])
        self.axcur.set_position([l+p1, b, p2-p1, .05])
        self.axnext.set_position([l+p2, b, w3, .05])

    def on_press(self, event):
        """Record the starting x-coordinate on button press."""
        if event.inaxes != self.axs[0]: return
        # Remove previous span if exists to allow drawing a new one
        if self.current_span:
            self.current_span.remove()
            self.current_span = None
        self.start_x = event.xdata

    def on_motion(self, event):
        """Dynamically update the axvspan as the mouse moves (dragging)."""
        if self.start_x is None or event.inaxes != self.axs[0]: return

        # Remove the previous temporary span before drawing the new one
        if self.current_span:
            self.current_span.remove()

        end_x = event.xdata
        # Draw the new temporary vertical span
        self.current_span = self.axs[0].axvspan(min(self.start_x, end_x), max(self.start_x, end_x), color='gray', alpha=0.5)
        self.resize_play_buttons()
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


