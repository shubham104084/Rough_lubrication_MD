import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# =========================================================
# USER SETTINGS
# =========================================================

# Input files
file1 = r"/Volumes/HPCDATA/hpcshag/lubrication/step_4_slide_water_50mps/force_displacement.dat"
#file2 = r"/Users/shubagra/Downloads/lubrication/Mixed_new/slide/force_displacement.dat"
file2 = r"/Volumes/HPCDATA/hpcshag/lubrication/step_4_slide_water_50mps_2/force_displacement.dat"
file3 = r"/Volumes/HPCDATA/hpcshag/lubrication/step_4_slide_water_50mps_3/force_displacement.dat"

#file2 = r"/Volumes/HPCDATA/hpcshag/lubrication/step_4_slide_dod_50mps/force_displacement.dat"
#file3 = r"/Volumes/HPCDATA/hpcshag/lubrication/step_4_slide_mixed_50mps/force_displacement.dat"

# Which columns to read: x = col 0, y = col 1 by default
xcol1, ycol1 = 0, 9
xcol2, ycol2 = 0, 7

# Plot appearance
figsize = (10, 4)
dpi = 300
line_width = 1.0
label_fontsize = 14
tick_labelsize = 12
title_fontsize = 14
legend_fontsize = 11

# Tick settings
major_tick_length = 6
minor_tick_length = 4
major_tick_width = 1.5
minor_tick_width = 1.0
tick_direction = "in"

# Major/minor spacing
xmajor1 = 20.0
xminor1 = 5.0
ymajor1 = 200
yminor1 = 50

xmajor2 = 20.0
xminor2 = 5.0
ymajor2 = 100 #0.5
yminor2 = 20# 0.1

# Axis labels / titles
xlabel1 = r"$d_{\mathrm{slid}}$ (nm)"
ylabel1 = r"$p_{zz}$ (MPa)"
title1 = ""

xlabel2 = r"$d_{\mathrm{slid}}$ (nm)"
ylabel2 = r"$\tau_{xz}$ (MPa)"
title2 = ""

# Limits (set to None to use automatic)
xlim1 =None #[-0.1,98] # None
ylim1 =None #[-0.1,942] # None
xlim2 =None #[-0.1,98] # None
ylim2 =[-0.01,276]#None

# Output
save_name = "friction_mixed.png"
save_name_pdf = "friction_mixed.pdf"

# =========================================================
# READ DATA
# =========================================================

data1 = np.loadtxt(file1)
data2 = np.loadtxt(file2)
data3 = np.loadtxt(file3)

x11 = -0.1*data1[:, xcol1]
y11 = 0.018*1000*data1[:, ycol1]
y11_mean = 0.018*1000*np.mean(data1[:, ycol1])

x12 = -0.1*data2[:, xcol1]
y12 = 0.018*1000*data2[:, ycol1]
y12_mean = 0.018*1000*np.mean(data2[:, ycol1])

x13 = -0.1*data3[:, xcol1]
y13 = 0.018*1000*data3[:, ycol1]
y13_mean = 0.018*1000*np.mean(data3[:, ycol1])


x21 = -0.1*data1[:, xcol2]
y21 = 0.018*1000*data1[:, ycol2]
#y21 = data1[:, ycol2]/data1[:, ycol1]
y21_mean = 0.018*1000*np.mean(data1[:, ycol2])

x22 = -0.1*data2[:, xcol2]
y22 = 0.018*1000*data2[:, ycol2]
#y22 = data2[:, ycol2]/data2[:, ycol1]
y22_mean = 0.018*1000*np.mean(data2[:, ycol2])

x23 = -0.1*data3[:, xcol2]
y23 = 0.018*1000*data3[:, ycol2]
#y23 = data3[:, ycol2]/data3[:, ycol1]
y23_mean = 0.018*1000*np.mean(data3[:, ycol2])

# =========================================================
# MAKE FIGURE
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi, constrained_layout=True)

# ---------------- Plot 1 ----------------
ax = axes[0]
ax.plot(x11, y11, lw=line_width+0,c="#b2182b",ls="-", label="Water "+ "("+str(round(y11_mean))+")")

ax.plot(x12, y12, lw=line_width+0.5,c="#E69F00",ls='-', label=r"$n$-dodecane "+ "("+str(round(y12_mean))+")")
#ax.plot(x13, 0.018*1000*data3[:, ycol2], lw=line_width, color="blue")
ax.plot(x13, y13, lw=line_width+0.8,c='#0072B2',ls='-', label="two-fluid "+ "("+str(round(y13_mean))+")")

ax.set_xlabel(xlabel1, fontsize=label_fontsize)
ax.set_ylabel(ylabel1, fontsize=label_fontsize)
ax.set_title(title1, fontsize=title_fontsize)

if xlim1 is not None:
    ax.set_xlim(xlim1)
if ylim1 is not None:
    ax.set_ylim(ylim1)

# Major/minor tick spacing
ax.xaxis.set_major_locator(MultipleLocator(xmajor1))
ax.xaxis.set_minor_locator(MultipleLocator(xminor1))
ax.yaxis.set_major_locator(MultipleLocator(ymajor1))
ax.yaxis.set_minor_locator(MultipleLocator(yminor1))

ax.legend(fontsize=label_fontsize,loc='upper right')

# Tick style
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

# Spine width
for spine in ax.spines.values():
    spine.set_linewidth(minor_tick_width)

# ---------------- Plot 2 ----------------
ax = axes[1]
ax.plot(x21, y21, lw=line_width+0,c="#b2182b",ls="-", label="Water "+ "("+str(round(y21_mean))+")")
ax.plot(x22, y22, lw=line_width+0.5,c="#E69F00",ls='-', label=r"$n$-dodecane"+ "("+str(round(y22_mean))+")")
ax.plot(x23, y23, lw=line_width+0.8,c='#0072B2',ls='-', label="two-fluid "+ "("+str(round(y23_mean))+")")

ax.set_xlabel(xlabel2, fontsize=label_fontsize)
ax.set_ylabel(ylabel2, fontsize=label_fontsize)
ax.set_title(title2, fontsize=title_fontsize)

if xlim2 is not None:
    ax.set_xlim(xlim2)
if ylim2 is not None:
    ax.set_ylim(ylim2)

# Major/minor tick spacing
ax.xaxis.set_major_locator(MultipleLocator(xmajor2))
ax.xaxis.set_minor_locator(MultipleLocator(xminor2))
ax.yaxis.set_major_locator(MultipleLocator(ymajor2))
ax.yaxis.set_minor_locator(MultipleLocator(yminor2))

ax.legend(fontsize=label_fontsize,loc='upper right')
# Tick style
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

# Spine width
for spine in ax.spines.values():
    spine.set_linewidth(minor_tick_width)

# =========================================================
# SAVE / SHOW
# =========================================================

#plt.savefig(save_name, bbox_inches="tight")
#plt.savefig(save_name_pdf, bbox_inches="tight")
plt.show()