#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:03:16 2026

@author: shubagra
"""

import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageChops
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

from ovito.io import import_file
from ovito.vis import Viewport, OpenGLRenderer
from ovito.modifiers import (
    ExpressionSelectionModifier,
    DeleteSelectedModifier,
    CalculateDisplacementsModifier,
    ComputePropertyModifier,
    ColorCodingModifier,
)


def hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def crop_white_margins(img, padding=0):
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img.convert("RGB"), bg)
    bbox = diff.getbbox()

    if bbox is None:
        return img

    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(img.width, right + padding)
    bottom = min(img.height, bottom + padding)

    return img.crop((left, top, right, bottom))


def get_font(font_size):
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            pass
    return ImageFont.load_default()


def add_frame_label(img, frame_number, title_height, title_font_size,
                    title_text_color=(0, 0, 0), title_bg_color=(255, 255, 255),
                    crop_before_label=True, panel_padding=8):
    if crop_before_label:
        img = crop_white_margins(img, padding=panel_padding)

    w, h = img.size
    labeled = Image.new("RGB", (w, h + title_height), color=title_bg_color)
    labeled.paste(img, (0, title_height))

    scale = 0.1 * (2e-3) * 20000 * frame_number * 0.1

    draw = ImageDraw.Draw(labeled)
    font = get_font(title_font_size)

    text = f"d = {scale} nm"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_x = (w - text_w) // 2
    text_y = (title_height - text_h) // 2

    draw.text((text_x, text_y), text, fill=title_text_color, font=font)
    return labeled

def add_side_label(
    img,
    frame_number,
    label_width,
    label_font_size,
    label_text_color=(0, 0, 0),
    label_bg_color=(255, 255, 255),
    crop_before_label=True,
    panel_padding=0,
    rotate_clockwise=False,
):
    """
    Add vertical 'd = ... nm' label on the left side of each rendered snapshot.
    """

    if crop_before_label:
        img = crop_white_margins(img, padding=panel_padding)

    w, h = img.size

    labeled = Image.new(
        "RGB",
        (w + label_width, h),
        color=label_bg_color,
    )

    # Snapshot goes to the right of the label strip
    labeled.paste(img, (label_width, 0))

    scale = 0.1 * (2e-3) * 20000 * frame_number * 0.1
    text = f"d = {scale:g} nm"

    font = get_font(label_font_size)

    dummy = Image.new("RGB", (10, 10), color=label_bg_color)
    draw_dummy = ImageDraw.Draw(dummy)
    bbox = draw_dummy.textbbox((0, 0), text, font=font)

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    margin = 30

    text_img = Image.new(
        "RGB",
        (text_w + 2 * margin, text_h + 2 * margin),
        color=label_bg_color,
    )

    draw = ImageDraw.Draw(text_img)
    draw.text(
        (margin, margin),
        text,
        fill=label_text_color,
        font=font,
    )

    # Rotate text parallel to vertical axis
    if rotate_clockwise:
        text_img = text_img.rotate(-90, expand=True)
    else:
        text_img = text_img.rotate(90, expand=True)

    tx = (label_width - text_img.width) // 2
    ty = (h - text_img.height) // 2

    labeled.paste(text_img, (tx, ty))

    return labeled


##############################################################################
# USER SETTINGS
##############################################################################

dump_file = "dump_slide.lammpstrj"
#frames_to_plot = [7, 15, 18, 25]   # do not include frame 0 with frame_offset=-1
frames_to_plot = [30, 60, 75, 110]

img_size = (3000, 2200)
stack_direction = "vertical"   # "horizontal" or "vertical"
output_name = "stacked_dod_10mps_flow_vm_test.png"
temp_dir = "tmp_ovito_frames"

label_width = 220
label_font_size = 144
label_text_color = (0, 0, 0)
label_bg_color = (255, 255, 255)

title_height = 200
title_font_size = 144
title_text_color = (0, 0, 0)
title_bg_color = (255, 255, 255)

crop_before_label = True
panel_padding = 0
gap_between_panels = 0

# Change to True if you want opposite text direction
rotate_label_clockwise = False

neutral_color = np.array([0.7, 0.7, 0.7])  # for Cu atoms, etc.


# for Cu particles 1 and 2
#stress_prop_name = "f_avestress[3]"
vm_min, vm_max = 0.0, 40000.0
# Color-coding range for vbot and vtop
# Set these to the range you want in the Hot colormap.
vflow_min = 0
vflow_max = 0.5

##############################################################################
# OVITO PIPELINE
##############################################################################

pipeline = import_file(dump_file)

# Remove particle types 3 and 4
pipeline.modifiers.append(
    ExpressionSelectionModifier(expression="ParticleType == 3 || ParticleType == 4")
)
pipeline.modifiers.append(DeleteSelectedModifier())

# Compute displacements relative to previous frame
disp_mod = CalculateDisplacementsModifier()
disp_mod.use_frame_offset = True
disp_mod.frame_offset = -5
pipeline.modifiers.append(disp_mod)


def prepare_visuals(frame, data):
    if data.cell is not None:
        data.cell.vis.enabled = False

    n = data.particles.count
    ptype = np.asarray(data.particles["Particle Type"])

    radius = data.particles_.create_property(
        "Radius", data=np.full(n, 0.8, dtype=float)
    )
    radius[ptype == 1] = 1.32
    radius[ptype == 2] = 1.32
    radius[ptype == 5] = 0.76*2.2  # C
    radius[ptype == 6] = 0.31*2.2  # H

    colors = np.tile(neutral_color, (n, 1))
    colors[ptype == 5] = hex_to_rgb01("#83f6fa")
    colors[ptype == 6] = hex_to_rgb01("#43b3e5")
    data.particles_.create_property("Color", data=colors)

    data.particles_.create_property("vbot", data=np.full(n, np.nan, dtype=float))
    data.particles_.create_property("vtop", data=np.full(n, np.nan, dtype=float))
    
    

pipeline.modifiers.append(prepare_visuals)

# ---------------------------------------------------------------------------
# Bottom-half fluid atoms: (ReducedPosition.Z < 0.5) && (ParticleType >= 5)
# vbot = abs(Displacement.X - 2) / 2
# Color-code with Hot
# ---------------------------------------------------------------------------

##############################################################################
# COARSE-GRAIN FLOW
##############################################################################

def coarse_grain_xz_full_y(
    frame,
    data,
    dx=5.0,
    dz=5.0,
    x_range=None,
    z_range=None,
):
    pos = np.asarray(data.particles["Position"])
    ptype = np.asarray(data.particles["Particle Type"])
    disp = np.asarray(data.particles["Displacement"])

    n = data.particles.count

    is_56 = (ptype == 5) | (ptype == 6)

    # T = 20000 * 2e-3 = 40 ps
    # 50 m/s = 0.5 Angstrom/ps
    flow_raw =  -disp[:, 0] / (5*40.0) / (10.0 / 100.0)

    x = pos[:, 0]
    z = pos[:, 2]

    if x_range is None:
        x_min = x[is_56].min()
        x_max = x[is_56].max()
    else:
        x_min, x_max = x_range

    if z_range is None:
        z_min = z[is_56].min()
        z_max = z[is_56].max()
    else:
        z_min, z_max = z_range

    nx = int(np.ceil((x_max - x_min) / dx))
    nz = int(np.ceil((z_max - z_min) / dz))

    ix = np.floor((x[is_56] - x_min) / dx).astype(int)
    iz = np.floor((z[is_56] - z_min) / dz).astype(int)

    ix = np.clip(ix, 0, nx - 1)
    iz = np.clip(iz, 0, nz - 1)

    bin_id = ix + nx * iz

    sums = np.zeros(nx * nz)
    counts = np.zeros(nx * nz)

    np.add.at(sums, bin_id, flow_raw[is_56])
    np.add.at(counts, bin_id, 1)

    avg = np.full(nx * nz, np.nan)

    valid = counts > 0
    avg[valid] = sums[valid] / counts[valid]

    flow_cg = np.full(n, np.nan)
    flow_cg[is_56] = avg[bin_id]

    data.particles_.create_property("flow_xz_cg", data=flow_cg)


pipeline.modifiers.append(
    lambda frame, data: coarse_grain_xz_full_y(
        frame,
        data,
        dx=5.0,
        dz=5.0,
    )
)

# Select fluid particles and color by coarse-grained flow
pipeline.modifiers.append(
    ExpressionSelectionModifier(
        expression="ParticleType == 5 || ParticleType == 6"
    )
)

flow_norm = Normalize(vmin=-0.5, vmax=0.5)

flow_cmap = LinearSegmentedColormap.from_list(
    "blue_gray_red",
    ["#0000ff", "#e0e0e0", "#ff0000"],
    N=256
)

flow_cmap = LinearSegmentedColormap.from_list(
    "classic_jet",
    ["#000080", "#0000ff", "#00ffff", "#ffff00", "#ff0000", "#800000"],
    N=256
)

flow_cmap = LinearSegmentedColormap.from_list(
    "classic_jet",
    [ "#0000ff", "#00ffff", "#ffff00", "#ff0000"],
    N=256
)
flow_cmap = LinearSegmentedColormap.from_list(
    "jet_no_yellow",
    [
        (0.0, "#0000ff"),  # blue
        (0.250, "#00ffff"),  # cyan
        (0.50, "#e0e0e0"),  # neutral gray
        (0.75, "#ff0000"),  # orange
        (1.00, "#800000"),  # red
    ],
    N=256
)

flow_cmap = LinearSegmentedColormap.from_list(
    "jet_no_yellow",
    [
        (0.0, "#1f3cff"),  # blue
        (0.250, "#00bcd4"),  # cyan
        (0.5, "#d9d9d9"),  # neutral gray
        (0.750, "#ff6b4a"),  # orange
        (1.00, "#540000")
    ],
    N=256
)

def color_fluid_flow_with_matplotlib(frame, data):
    ptype = np.asarray(data.particles["Particle Type"])
    flow = np.asarray(data.particles["flow_xz_cg"], dtype=float)

    colors = np.asarray(data.particles["Color"], dtype=float).copy()

    is_fluid = (ptype == 5) | (ptype == 6)
    valid = is_fluid & np.isfinite(flow)

    rgba = flow_cmap(flow_norm(flow[valid]))
    colors[valid] = rgba[:, :3]

    data.particles_.create_property("Color", data=colors)


pipeline.modifiers.append(color_fluid_flow_with_matplotlib)

##############################################################################
# WALL VON MISES STRESS
##############################################################################


pipeline.modifiers.append(
    ExpressionSelectionModifier(expression="ParticleType == 1 || ParticleType == 2")
)

pipeline.modifiers.append(
    ComputePropertyModifier(
        output_property="von_mises",
        expressions=[
            "sqrt(0.5*((f_avestress_1_-f_avestress_2_)*(f_avestress_1_-f_avestress_2_) + "
            "(f_avestress_2_-f_avestress_3_)*(f_avestress_2_-f_avestress_3_) + "
            "(f_avestress_3_-f_avestress_1_)*(f_avestress_3_-f_avestress_1_) + "
            "6*(f_avestress_4_*f_avestress_4_ + f_avestress_5_*f_avestress_5_ + f_avestress_6_*f_avestress_6_)))"
        ],
        only_selected=True
    )
)

# pipeline.modifiers.append(
#     ColorCodingModifier(
#         property="von_mises",
#         gradient=ColorCodingModifier.Magma(),#BlueWhiteRed(), #ColorCodingModifier.Magma(),
#         start_value=vm_min,
#         end_value=vm_max,
#         only_selected=True
#     )
# )


stress_cmap = plt.get_cmap("viridis") 
stress_norm = Normalize(vmin=vm_min, vmax=vm_max)

def color_stress_with_matplotlib(frame, data):
    ptype = np.asarray(data.particles["Particle Type"])
    stress = np.asarray(data.particles["von_mises"], dtype=float)

    colors = np.asarray(data.particles["Color"], dtype=float).copy()

    is_wall = (ptype == 1) | (ptype == 2)
    valid = is_wall & np.isfinite(stress)

    rgba = stress_cmap(stress_norm(stress[valid]))
    colors[valid] = rgba[:, :3]

    data.particles_.create_property("Color", data=colors)


pipeline.modifiers.append(color_stress_with_matplotlib)

##############################################################################
# RENDER FRAMES
##############################################################################

pipeline.add_to_scene()

vp = Viewport(type=Viewport.Type.Ortho)
vp.camera_dir = (0, -1, 0)   # front view for xy slab
vp.zoom_all()

os.makedirs(temp_dir, exist_ok=True)

##############################################################################
# RENDER FRAMES
##############################################################################

rendered_files = []
for fr in frames_to_plot:
    fname = os.path.join(temp_dir, f"frame_{fr:05d}.png")
    vp.render_image(
        filename=fname,
        frame=fr,
        size=img_size,
        background=(1, 1, 1),
        renderer=OpenGLRenderer()
    )
    rendered_files.append(fname)

##############################################################################
# LABEL IMAGES
##############################################################################

images = []
for fr, f in zip(frames_to_plot, rendered_files):
    img = Image.open(f).convert("RGB")
    img = add_side_label(
        img=img,
        frame_number=fr,
        label_width=label_width,
        label_font_size=label_font_size,
        label_text_color=label_text_color,
        label_bg_color=label_bg_color,
        crop_before_label=crop_before_label,
        panel_padding=panel_padding,
        rotate_clockwise=rotate_label_clockwise,
    )
    images.append(img)

##############################################################################
# STACK IMAGES
##############################################################################

if stack_direction == "vertical":
    total_width = max(img.width for img in images)
    total_height = (
        sum(img.height for img in images)
        + gap_between_panels * (len(images) - 1)
    )

    stacked = Image.new(
        "RGB",
        (total_width, total_height),
        color=(255, 255, 255),
    )

    y = 0

    for img in images:
        x = (total_width - img.width) // 2
        stacked.paste(img, (x, y))
        y += img.height + gap_between_panels

elif stack_direction == "horizontal":
    total_width = (
        sum(img.width for img in images)
        + gap_between_panels * (len(images) - 1)
    )

    total_height = max(img.height for img in images)

    stacked = Image.new(
        "RGB",
        (total_width, total_height),
        color=(255, 255, 255),
    )

    x = 0

    for img in images:
        y = (total_height - img.height) // 2
        stacked.paste(img, (x, y))
        x += img.width + gap_between_panels

else:
    raise ValueError("stack_direction must be 'horizontal' or 'vertical'.")


stacked.save(output_name)

print(f"Saved combined image as: {output_name}")