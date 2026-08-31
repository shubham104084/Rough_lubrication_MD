#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple LAMMPS data reader → (optional) post-process → writer.
Designed to be easy to read and hack for quick workflows.
Supports files produced by `write_data` with:
  - Header counts (atoms, atom types, etc.)
  - Orthogonal or triclinic boxes (xy xz yz optional)
  - Sections: Masses, Atoms (styles: atomic or full), Velocities (optional)
Other sections are ignored (left untouched).
"""

from pathlib import Path
import numpy as np

def parse_data_file(path):
    """
    Minimal parser. Returns a dict with:
      data["counts"]  -> {label: int}
      data["box"]     -> {"xlo":..., "xhi":..., "ylo":..., "yhi":..., "zlo":..., "zhi":..., "xy":..., "xz":..., "yz":... (optional)}
      data["masses"]  -> {type(int): mass(float)}
      data["atoms"]   -> list of dict rows
                         For atomic: {"id","type","x","y","z", optionally "ix","iy","iz"}
                         For full:   {"id","mol","type","q","x","y","z", optionally "ix","iy","iz"}
      data["atom_style"] -> "atomic" | "full" | "unknown"
      data["velocities"] -> list of dict rows (optional) {"id","vx","vy","vz"}
    Anything not recognized is skipped.
    """
    text = Path(path).read_text().splitlines()

    data = {
        "counts": {},
        "box": {},
        "masses": {},
        "atoms": [],
        "atom_style": "unknown",
        "velocities": []
    }

    # Helper flags
    section = None  # e.g. "Masses", "Atoms", "Velocities"
    atom_style_hint = None  # e.g. "full" from "Atoms # full"

    # Utility functions
    def strip_comment(line):
        # Keep "Atoms # full" intact for header detection; for data lines drop trailing '# ...'
        return line.split('#', 1)[0].rstrip()

    # First pass: read header + sections
    i = 0
    while i < len(text):
        raw = text[i].rstrip()
        line = raw.strip()

        # Identify new section headers
        if line.lower().startswith("masses"):
            section = "Masses"
            i += 1;  # skip header line
            # skip a blank line after header, if present
            if i < len(text) and not text[i].strip():
                i += 1
            continue

        if line.lower().startswith("pair coeffs") or \
           line.lower().startswith("bond coeffs") or \
           line.lower().startswith("angle coeffs") or \
           line.lower().startswith("dihedral coeffs") or \
           line.lower().startswith("improper coeffs"):
            # We ignore coefficients in this simple script. Skip their blocks.
            section = "SkipCoeffs"
            i += 1
            if i < len(text) and not text[i].strip():
                i += 1
            continue

        if line.lower().startswith("atoms"):
            section = "Atoms"
            # Detect style in header like "Atoms # full"
            if "#" in raw:
                after = raw.split("#", 1)[1].strip().lower()
                if after:
                    atom_style_hint = after.split()[0]
                    data["atom_style"] = atom_style_hint
            i += 1
            if i < len(text) and not text[i].strip():
                i += 1
            continue

        if line.lower().startswith("velocities"):
            section = "Velocities"
            i += 1
            if i < len(text) and not text[i].strip():
                i += 1
            continue

        # End section if we hit another header
        if any(line.lower().startswith(h) for h in [
            "masses", "pair coeffs", "bond coeffs", "angle coeffs",
            "dihedral coeffs", "improper coeffs", "atoms", "velocities"
        ]):
            # handled above; keep loop going
            pass

        # Header lines: counts and box
        if section is None and line:
            # counts like "100 atoms", "1 atom types"
            toks = line.split()
            if len(toks) >= 2 and toks[0].isdigit():
                try:
                    val = int(toks[0])
                    key = " ".join(toks[1:]).lower()
                    data["counts"][key] = val
                    i += 1
                    continue
                except ValueError:
                    pass

            # bounds: "xlo xhi", etc.
            if len(toks) == 4 and toks[2].lower() in ("xlo", "ylo", "zlo"):
                try:
                    lo = float(toks[0]); hi = float(toks[1])
                    keylo, keyhi = toks[2].lower(), toks[3].lower()
                    data["box"][keylo] = lo
                    data["box"][keyhi] = hi
                    i += 1
                    continue
                except ValueError:
                    pass

            # tilt factors line: "xy xz yz"
            if len(toks) == 6 and toks[2].lower() == "xy" and toks[3].lower() == "xz" and toks[4].lower() == "yz":
                try:
                    data["box"]["xy"] = float(toks[0])
                    data["box"]["xz"] = float(toks[1])
                    data["box"]["yz"] = float(toks[2])  # will be overwritten below; fix quickly
                except Exception:
                    pass
                # Actually toks should be: val1 val2 val3 xy xz yz
                try:
                    data["box"]["xy"] = float(toks[0])
                    data["box"]["xz"] = float(toks[1])
                    data["box"]["yz"] = float(toks[2])
                except ValueError:
                    pass
                i += 1
                continue

        # Read section bodies
        if section == "Masses":
            s = strip_comment(raw)
            if s.strip():
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        itype = int(parts[0]); mass = float(parts[1])
                        data["masses"][itype] = mass
                    except ValueError:
                        pass

        elif section == "Atoms":
            s = strip_comment(raw).strip()
            if s:
                parts = s.split()
                # Try "full" first if hinted; otherwise infer by column count
                if (atom_style_hint == "full") or (len(parts) >= 7 and "." in parts[3] and "." in parts[4]):
                    # id mol type q x y z [ix iy iz]
                    row = {
                        "id": int(parts[0]),
                        "mol": int(parts[1]),
                        "type": int(parts[2]),
                        "q": float(parts[3]),
                        "x": float(parts[4]), "y": float(parts[5]), "z": float(parts[6]),
                    }
                    if len(parts) >= 10:
                        row["ix"], row["iy"], row["iz"] = int(parts[7]), int(parts[8]), int(parts[9])
                    data["atom_style"] = "full"
                else:
                    # atomic: id type x y z [ix iy iz]
                    row = {
                        "id": int(parts[0]),
                        "type": int(parts[1]),
                        "x": float(parts[2]), "y": float(parts[3]), "z": float(parts[4]),
                    }
                    if len(parts) >= 8:
                        row["ix"], row["iy"], row["iz"] = int(parts[5]), int(parts[6]), int(parts[7])
                    data["atom_style"] = "atomic"
                data["atoms"].append(row)

        elif section == "Velocities":
            s = strip_comment(raw).strip()
            if s:
                parts = s.split()
                if len(parts) >= 4:
                    try:
                        data["velocities"].append({
                            "id": int(parts[0]), "vx": float(parts[1]), "vy": float(parts[2]), "vz": float(parts[3])
                        })
                    except ValueError:
                        pass

        i += 1

    return data


def write_data_file(path, data):
    """
    Writes a minimal, clean LAMMPS data file from the parsed dict.
    Only writes sections we have (Masses, Atoms, Velocities).
    """
    lines = []
    lines.append("LAMMPS data file (simple_rw)\n")

    # counts
    for k, v in data.get("counts", {}).items():
        lines.append(f"{v} {k}")
    lines.append("")

    # box
    b = data.get("box", {})
    for axis in ("x", "y", "z"):
        lo = b.get(f"{axis}lo"); hi = b.get(f"{axis}hi")
        if lo is not None and hi is not None:
            lines.append(f"{lo:.16g} {hi:.16g} {axis}lo {axis}hi")
    if all(key in b for key in ("xy","xz","yz")):
        lines.append(f"{b['xy']:.16g} {b['xz']:.16g} {b['yz']:.16g} xy xz yz")
    lines.append("")

    # Masses
    if data.get("masses"):
        lines.append("Masses\n")
        for itype in sorted(data["masses"].keys()):
            lines.append(f"{itype} {data['masses'][itype]:.8f}")
        lines.append("")

    # Atoms
    style = data.get("atom_style", "atomic")
    if style == "full":
        lines.append("Atoms # full\n")
        for a in sorted(data["atoms"], key=lambda r: r["id"]):
            base = f"{a['id']} {a.get('mol',0)} {a['type']} {a.get('q',0.0):.8f} {a['x']:.8f} {a['y']:.8f} {a['z']:.8f}"
            if all(k in a for k in ("ix","iy","iz")):
                base += f" {a['ix']} {a['iy']} {a['iz']}"
            lines.append(base)
    else:
        lines.append("Atoms # atomic\n")
        for a in sorted(data["atoms"], key=lambda r: r["id"]):
            base = f"{a['id']} {a['type']} {a['x']:.8f} {a['y']:.8f} {a['z']:.8f}"
            if all(k in a for k in ("ix","iy","iz")):
                base += f" {a['ix']} {a['iy']} {a['iz']}"
            lines.append(base)
    lines.append("")

    # Velocities (optional)
    if data.get("velocities"):
        lines.append("Velocities\n")
        for v in sorted(data["velocities"], key=lambda r: r["id"]):
            lines.append(f"{v['id']} {v['vx']:.8f} {v['vy']:.8f} {v['vz']:.8f}")
        lines.append("")

    Path(path).write_text("\n".join(lines))


# ------------------ Example "post-process" operations ------------------
def translate_positions(data, dx=0.0, dy=0.0, dz=0.0):
    """Shift all atoms by a constant vector (no PBC wrapping here; keep it simple)."""
    for a in data["atoms"]:
        a["x"] += dx; a["y"] += dy; a["z"] += dz


def remap_types(data, mapping):
    """Change atom types: mapping is a dict old_type->new_type, others unchanged."""
    for a in data["atoms"]:
        t = a["type"]
        if t in mapping:
            a["type"] = mapping[t]


def set_uniform_charge(data, q=0.0):
    """If style is full, set all charges to a constant q."""
    if data.get("atom_style") == "full":
        for a in data["atoms"]:
            a["q"] = float(q)


# ------------------ CLI-like demo (edit these lines as needed) ------------------

    # EDIT THESE: quick, readable workflow
in_file  = "equil_out.data"        # <- change to your input data file
filename = "equil_out_spring.data"       # <- change to your desired output file

# 1) Read
data = parse_data_file(in_file)

#read counts
atoms_total = data.get('counts')['atoms']
atype_total = data.get('counts')['atom types']

# read box
xlo = data.get('box')['xlo']
xhi = data.get('box')['xhi']
ylo = data.get('box')['ylo']
yhi = data.get('box')['yhi']
zlo = data.get('box')['zlo']
zhi = data.get('box')['zhi']

# read masse
m = data.get('masses')[1]

# Read atoms
ids  = np.array([data.get("atoms")[r]["id"]   for r in range(len(data.get("atoms")))], dtype=int)
mols  = np.array([data.get("atoms")[r]["mol"]   for r in range(len(data.get("atoms")))], dtype=int)
types  = np.array([data.get("atoms")[r]["type"]   for r in range(len(data.get("atoms")))], dtype=int)
#ids  = np.array([data.get("atoms")[r]["id"]   for r in range(len(data.get("atoms")))], dtype=int)
qs  = np.array([data.get("atoms")[r]["q"]   for r in range(len(data.get("atoms")))], dtype=int)
X  = np.array([data.get("atoms")[r]["x"]   for r in range(len(data.get("atoms")))], dtype=float)
Y = np.array([data.get("atoms")[r]["y"]   for r in range(len(data.get("atoms")))], dtype=float)
Z  = np.array([data.get("atoms")[r]["z"]   for r in range(len(data.get("atoms")))], dtype=float)
iX = np.array([data.get("atoms")[r]["ix"]   for r in range(len(data.get("atoms")))], dtype=int)
iY = np.array([data.get("atoms")[r]["iy"]   for r in range(len(data.get("atoms")))], dtype=int)
iZ = np.array([data.get("atoms")[r]["iz"]   for r in range(len(data.get("atoms")))], dtype=int)

# Read atoms
v_ids  = np.array([data.get("velocities")[r]["id"]   for r in range(len(data.get("velocities")))], dtype=int)
vX  = np.array([data.get("velocities")[r]["vx"]   for r in range(len(data.get("velocities")))], dtype=float)
vY = np.array([data.get("velocities")[r]["vy"]   for r in range(len(data.get("velocities")))], dtype=float)
vZ  = np.array([data.get("velocities")[r]["vz"]   for r in range(len(data.get("velocities")))], dtype=float)

### Post processing begins
# Making indenter id
indenter_idx = np.where(Z>0.0)[0]
substrate_idx = np.where(Z<0.0)[0]

types[substrate_idx] = 2

## Shift the indenter down by 19 Ang
downshift = 19.0
Z[indenter_idx] -= downshift

# select the layers to be duplicated and label them as type 3
shift = 20

top_layer_idx = np.where(Z>73.0)[0]
bot_layer_idx = np.where(Z<-93.7)[0]

types[top_layer_idx] = 3
types[bot_layer_idx] = 3

# Building toplayer

Xtop_new = X[top_layer_idx]
Ytop_new = Y[top_layer_idx]
Ztop_new = Z[top_layer_idx] + shift
type_top_new = types[top_layer_idx]*0 +4
id_top_new = ids[top_layer_idx]
# Building bottomlayer
Xbot_new = X[bot_layer_idx]
Ybot_new = Y[bot_layer_idx]
Zbot_new = Z[bot_layer_idx] - shift
type_bot_new = types[bot_layer_idx]*0 +4
id_bot_new = ids[bot_layer_idx]

## Shifting Z_top_layer and indenter up by the same amount
top_shift =  20.0 #20.0
Ztop_new += downshift+top_shift
Z[indenter_idx] += downshift+top_shift

bot_shift =  20.0 #20.0
Zbot_new -= bot_shift
Z[substrate_idx] -= bot_shift




mols_new = np.append(mols[top_layer_idx],mols[bot_layer_idx])
qs_new = np.append(qs[top_layer_idx],qs[bot_layer_idx])
iX_new = np.append(iX[top_layer_idx],iX[bot_layer_idx])
iY_new = np.append(iY[top_layer_idx],iY[bot_layer_idx])
iZ_new = np.append(iZ[top_layer_idx],iZ[bot_layer_idx])
X_new = np.append(Xtop_new,Xbot_new)
Y_new = np.append(Ytop_new,Ybot_new)
Z_new = np.append(Ztop_new,Zbot_new)
id_new = np.append(id_top_new, id_bot_new)
types_new = np.append(type_top_new, type_bot_new)

# updating and bonding loop
nbonds = len(X_new)
bond_id = np.arange(1, 1+len(X_new))
btype = 0*bond_id +1
batom1 = id_new
id_new = np.arange(1+atoms_total, atoms_total+1+len(X_new))  
batom2 = id_new

atoms_total += len(X_new)

vX_new = X_new*0
vY_new = X_new*0
vZ_new = X_new*0

####### UPDATING
ids_full = np.append(ids, id_new)
mols_full = np.append(mols, mols_new)
types_full = np.append(types, types_new)
qs_full = np.append(qs, qs_new)
X_full = np.append(X, X_new)
Y_full = np.append(Y, Y_new)
Z_full = np.append(Z, Z_new)
iX_full = np.append(iX, iX_new)
iY_full = np.append(iY, iY_new)
iZ_full = np.append(iZ, iZ_new)

v_ids_full = np.append(v_ids, id_new)
vX_full = np.append(vX, vX_new)
vY_full = np.append(vY, vY_new)
vZ_full = np.append(vZ, vZ_new)

zlo_new =min(np.floor(min(Z_full))-10,zlo)
zhi_new =max(np.ceil(max(Z_full))+10,zhi)

with open(filename, 'w') as f:
    f.write("LAMMPS data file with\n\n")
    f.write(f"{atoms_total} atoms\n{nbonds} bonds\n\n")
    f.write("6 atom types\n2 bond types\n1 angle types\n\n")
    #box = r_max + 5
    f.write(f"{xlo:.1f} {xhi:.1f} xlo xhi\n")
    f.write(f"{ylo:.1f} {yhi:.1f} ylo yhi\n")
    f.write(f"{zlo_new:.1f} {zhi_new:.1f} zlo zhi\n\n")
    f.write(f"Masses\n\n1 {m}\n2 {m}\n3 {m}\n4 {m}\n5 15.999\n6 1.0\n\nAtoms #full\n\n")
    for a in range(len(ids_full)):
        f.write(f"{ids_full[a]} {mols_full[a]} {types_full[a]} {qs_full[a]} {X_full[a]:.16f} {Y_full[a]:.16f} {Z_full[a]:.16f} {iX_full[a]} {iY_full[a]} {iZ_full[a]}\n")
    f.write("\nVelocities\n\n")
    for a in range(len(ids_full)):
        f.write(f"{v_ids_full[a]} {vX_full[a]:.16f} {vY_full[a]:.16f} {vZ_full[a]:.16f} \n")
    f.write("\nBonds\n\n")
    for b in range(len(bond_id)):
        f.write(f"{bond_id[b]} {btype[b]} {batom1[b]} {batom2[b]}\n")

    # 2) Post-process (examples – comment/uncomment as needed)
    # translate_positions(data, dx=0.0, dy=0.0, dz=0.0)
    # remap_types(data, {1:2})          # turn type 1 atoms into type 2
    # set_uniform_charge(data, q=0.0)   # only if Atoms # full

    # 3) (Optional) update counts if you changed number of types etc.
    # data["counts"]["atom types"] = max(a["type"] for a in data["atoms"])

    # 4) Write back a clean LAMMPS data file
#write_data_file(out_file, data)

#print(f"Done. Wrote: {out_file}")
