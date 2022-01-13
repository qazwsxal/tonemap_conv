import torch
from torch.nn import Module
from math import floor, ceil
import matplotlib.pyplot as plt
import matplotlib
import numpy
import inspect
import tonemappers


PIXELS = 512
MAX_BRIGHTNESS = 1.0
ASPECT_RATIO = 8
rep_count = floor(PIXELS/(ASPECT_RATIO*4)) # x4 because we have 4 colours

tm_members = {name: cls() for name, cls in inspect.getmembers(tonemappers)
              if inspect.isclass(cls) and issubclass(cls, Module)
              }
names = sorted(tm_members.keys())
names.remove('Linear')
names.insert(0,'Linear')
rows = len(names)


slope = torch.linspace(0, MAX_BRIGHTNESS, 512)

rgbk_slopes = torch.zeros((4,PIXELS, 3))
for i in range(3):
    rgbk_slopes[i,:,i] = slope
rgbk_slopes[3,:,:] = slope[...,None]

with plt.rc_context({'axes.edgecolor':'#888888', 'xtick.color':'#888888', 'ytick.color':'#888888','axes.labelcolor':'#888888'}):
    fig, axes = plt.subplots(rows, 1, sharex=True)
    axes[-1].set_xlabel("Brightness")
    for ax, tm in zip(axes, names):
        val = tm_members[tm](rgbk_slopes)
        r_val = torch.repeat_interleave(val, rep_count, dim=0)
        ax.imshow(r_val.detach().numpy(),
                  extent=(0,MAX_BRIGHTNESS,0,1),
                  aspect=1/ASPECT_RATIO)
        ax.set_ylabel(tm, rotation=0, ha='right')
        ax.tick_params(
            axis='y',
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelleft=False
        )
    fig.show()

print('aaa')

