import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


# ----------------------------
# Data
# ----------------------------
speeds = ['1 m/s', '10 m/s', '50 m/s']

mu_water    = [0.225, 0.309, 0.316]
mu_dodecane = [0.25, 0.24, 0.25]
#mu_mixed    = [114/226., 0.388, 0.41]
mu_mixed    = [ 0.242, 0.223, 0.259]

# ----------------------------
# Bar positions
# ----------------------------
x = np.arange(len(speeds))   # positions for speed groups
width = 0.22                 # width of each bar

fig, ax = plt.subplots(figsize=(6, 5))

ax.bar(x - width, mu_water,    width, label='Water', color='#3399FF')
ax.bar(x,         mu_dodecane, width, label='Dodecane', color='#ffab33')
ax.bar(x + width, mu_mixed,    width, label='Two-fluid0.4',color='#48a43f')

#ax.bar(x + width, mu_mixed,    width, label='Mixed',color='#CCCC66')

# ----------------------------
# Formatting
# ----------------------------
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))
# Tick style
# Tick settings
line_width = 1.2
label_fontsize = 14
tick_labelsize = 12
title_fontsize = 14
legend_fontsize = 11

major_tick_length = 6
minor_tick_length = 4
major_tick_width = 1.5
minor_tick_width = 1.0
tick_direction = "in"

ax.tick_params(
    axis="both",
    which="major",
    direction=tick_direction,
    length=major_tick_length,
    width=major_tick_width,
    labelsize=tick_labelsize,
    top=True,
    right=True
)
ax.tick_params(
    axis="both",
    which="minor",
    direction=tick_direction,
    length=minor_tick_length,
    width=minor_tick_width,
    top=True,
    right=True
)

ax.set_xlabel('Sliding speed', fontsize=16)
ax.set_ylabel(r'Friction coefficient', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(speeds, fontsize=14)
ax.legend(fontsize=14)
ax.set_ylim([0,0.36])

plt.tight_layout()

plt.savefig("friction_coeff.pdf", bbox_inches="tight")
plt.show()



