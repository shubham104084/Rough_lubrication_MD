import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# -----------------------------
# Global style controls
# -----------------------------
label_fontsize = 18
title_fontsize = 20
tick_fontsize  = 16
legend_fontsize = 18

major_tick_length = 6
major_tick_width  = 1.5
minor_tick_length = 4
minor_tick_width  = 1.0

show_minor_ticks = True         # True / False
n_minor_intervals_x = 2         # number of minor intervals between major ticks
n_minor_intervals_y = 2

# -----------------------------
# Data for each subplot
# Each entry contains:
# x positions, x labels, stacked components, title
# -----------------------------
plot_data = [
    {
        "x": [0, 1, 2, 3,4,5,6],
        "xlabels": ['20', '30', '54', '100', '130', '170', '200'],
        "y1": [3,6,8,13,38,37,27],
        "y2": [64,86,129,199,142,147,196],
        "ylim": (0, 230),
        "title": "water: 50 m/s"
    },
    {
        "x": [0, 1, 2, 3,4,5,6],
        "xlabels": ['20', '30', '54', '100', '130', '170','200'],
        "y1": [18,18,18,24,24,32,34],
        "y2": [45,66,60,63,63,62,62],
        "ylim": (0, 230),
        "title": r"$n$-dodecane: 50 m/s"
    },
    {
        "x": [0, 1, 2, 3,4,5],
        "xlabels": ['24', '54', '80', '100', '130', '200'],
        "y1": [57, 57, 25,25,6,6],
        "y2": [3,14,21,21,27,31],
        "ylim": (0, 230),
        "title": "two-fluid: 50 m/s"
    },
    {
        "x": [0, 1, 2, 3, 4],
        "xlabels": ['17.6', '28', '51.2', '80', '100'],
        "y1": [44, 50, 50, 35, 38],
        "y2": [52, 72, 79, 71, 76],
        "ylim": (0, 155),
        "title": "water: 10 m/s"
    },
    {
        "x": [0, 1, 2, 3, 4, 5],
        "xlabels": ['20', '30', '54', '68', '80', '100'],
        "y1": [14, 14, 31, 54, 54, 56],
        "y2": [65, 128, 76, 45, 69, 46],
        "ylim": (0, 155),
        "title": r"$n-$dodecane: 10 m/s"
    },
    {
        "x": [0, 1, 2, 3, 4],
        "xlabels": ['18', '50', '68', '80', '100'],
        "y1": [24, 24, 14, 14, 14],
        "y2": [26, 40, 49,49,49],
        "ylim": (0, 155),
        "title": "two-fluid: 10 m/s"
    }
]

# -----------------------------
# Create figure
# -----------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, (ax, d) in enumerate(zip(axes, plot_data)):
    x = np.array(d["x"])
    y1 = np.array(d["y1"])
    y2 = np.array(d["y2"])
    #y3 = np.array(d["y3"])
    
    if len(x) > 1:
        bar_width = 0.6 * np.min(np.diff(np.sort(x)))
    else:
        bar_width = 0.5   # fallback if only one bar
    
    # Stacked bars
    bar_bottom=ax.bar(x, y2, width=bar_width, label='Bottom',color="C1")
    bar_top=ax.bar(x, y1, width=bar_width, bottom=y2, label='Top',color="C0")
    #ax.bar(x, y3, width=bar_width, bottom=y1+y2, label='Part 3')

    for spine in ax.spines.values():
        spine.set_linewidth(minor_tick_width)


    # Titles and labels
    ax.set_title(d["title"], fontsize=title_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(d["xlabels"], fontsize=tick_fontsize)
    if i ==0 or i==3:
        ax.set_ylabel("net number of lost-atoms", fontsize=label_fontsize)
    ax.set_ylim(d["ylim"])
    
    if i==1 or i==2 or i==4 or i==5:
        ax.tick_params(axis='y', labelleft=False)
    
    if i<3:
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(10))

    # Tick style: inward ticks
    ax.tick_params(axis='both', which='major',
                   direction='in',
                   length=major_tick_length,
                   width=major_tick_width,
                   labelsize=tick_fontsize,
                   top=True, right=True)

    ax.tick_params(axis='both', which='minor',
                   direction='in',
                   length=minor_tick_length,
                   width=minor_tick_width,
                   top=True, right=True)

    # Minor ticks control
    # if show_minor_ticks:
    #     ax.xaxis.set_minor_locator(AutoMinorLocator(n_minor_intervals_x + 1))
    #     ax.yaxis.set_minor_locator(AutoMinorLocator(n_minor_intervals_y + 1))
        
    

   
# Show legend only once
#axes[0].legend(fontsize=legend_fontsize)
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles[::-1], labels[::-1], fontsize=legend_fontsize)
axes[3].set_xlabel(r"$d_{\mathrm{slid}}$ (nm)", fontsize=label_fontsize)
axes[4].set_xlabel(r"$d_{\mathrm{slid}}$ (nm)", fontsize=label_fontsize)
axes[5].set_xlabel(r"$d_{\mathrm{slid}}$ (nm)", fontsize=label_fontsize)

plt.tight_layout()
plt.savefig("wear.png", bbox_inches="tight", dpi=500)
plt.show()