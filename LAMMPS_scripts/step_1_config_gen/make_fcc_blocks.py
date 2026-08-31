#
# Set paramaters in the first few lines of main()
# Some fixed default setting in carve_profile()
#

import os
from dataclasses import dataclass, field, replace

# import ase.build
from numba import njit
import numpy as np
import ase
from numpy.typing import NDArray
FloatArray = NDArray[np.float64]


def unit_cell_fcc_111(
        r_0: float = 1.0,
        scal_latt: FloatArray = np.ones(3, dtype=np.float64),
        f_flip_xy: bool = False
) -> tuple[FloatArray, FloatArray]:

    # Cell dimensions
    a_unit = np.array([1., np.sqrt(3.), np.sqrt(6.)])

    # Atomic positions in fractional coordinates
    r_unit = np.array([
        # A layer
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        # B layer
        [0.5, 1./6, 1./3],
        [0.0, 1./6+0.5, 1./3],
        # C layer
        [0.0, 1./3, 2./3],
        [0.5, 1./3+0.5, 2./3],
    ])

    # Optionally flip x and y
    if f_flip_xy:
        a_unit[[0, 1]] = a_unit[[1, 0]]
        r_unit[:, [0, 1]] = r_unit[:, [1, 0]]

    # Scale cell dimensions and atomic positions
    a_unit *= r_0 * scal_latt
    r_unit *= a_unit

    return a_unit, r_unit


@njit
def make_crystal(
        unit_cell: FloatArray,
        r_in_cell: FloatArray,
        n_repeat: FloatArray
) -> FloatArray:

    n_atom, n_dim = r_in_cell.shape
    total_atoms = n_atom * n_repeat[0] * n_repeat[1] * n_repeat[2]

    r_crystal = np.zeros((total_atoms, n_dim), dtype=np.float64)
    idx = 0

    for i in range(n_repeat[0]):
        for j in range(n_repeat[1]):
            for k in range(n_repeat[2]):
                shift = np.zeros(n_dim, dtype=np.float64)
                shift[0] = i * unit_cell[0]
                shift[1] = j * unit_cell[1]
                shift[2] = k * unit_cell[2]

                for atom in range(n_atom):
                    for d in range(n_dim):
                        r_crystal[idx, d] = r_in_cell[atom, d] + shift[d]
                    idx += 1

    return r_crystal


def make_inc_blocks(l_x: float, a_0: float, l_z_rel: float = 1./3, file_name="topo_info.dat"):
    # l_z_rel: height of cell in units of l_x

    file = open(file_name, "a")
    r_0 = a_0 / np.sqrt(2.)
    scal_fac = np.sqrt(np.sqrt(3) * 4 / 7)

    scal_latt_top = np.array([scal_fac, 1./scal_fac, 1])
    scal_latt_bot = np.array([1. / scal_fac, scal_fac, 1])

    n_top_unit = np.array([7, 4, 1])
    n_bot_unit = np.array([4, 7, 1])

    # h-matrix of unit cell and (true) positions of atoms
    a_top, r_top = unit_cell_fcc_111(r_0=r_0, scal_latt=scal_latt_top)
    a_bot, r_bot = unit_cell_fcc_111(
        r_0=r_0, scal_latt=scal_latt_bot, f_flip_xy=True)

    # construct unstrained unit cell
    a_big_cell = np.array([7*a_top[0], 4*a_top[1], a_top[2]])
    file.write("# square unit cell dimensions:\t")
    write_array(a_big_cell, file)

    if l_x < a_big_cell[0]:
        raise ValueError("square unit cell must not exceed l_x")

    n_x = int(0.5 + l_x / a_big_cell[0])
    n_z = int(0.5 + a_big_cell[0] * n_x / a_big_cell[2] * l_z_rel)
    n_repeat = np.array([n_x, 1, n_z])

    file.write("# number of square unit cells:\t")
    write_array(n_repeat, file)

    n_top = n_repeat * n_top_unit
    n_repeat = np.array([n_x, 1, n_z+1])
    n_bot = n_repeat * n_bot_unit

    r_top_all = make_crystal(a_top, r_top, n_top)
    r_bot_all = make_crystal(a_bot, r_bot, n_bot)
    print(r_top_all.shape, r_bot_all.shape)

    scal_factor = l_x / (7 * n_x * a_top[0])
    r_top_all[:, 0] *= scal_factor
    r_bot_all[:, 0] *= scal_factor

    file.write("\n")
    file.close()
    return (r_top_all, r_bot_all)


@dataclass
class HeightTopography():

    # parameters defining the height topography
    name: str            # identifier
    a_0: float           # unit of length            [Angstrom]
    l_x: float           # length of simulation cell [Angstrom]
    hurst: float        # Hurst exponent
    lambda_r: float     # roll-off wavelength       [l_x]
    lambda_s: float     # short wavelength cuoff    [a_0]
    rms_slope: float     # rms slope
    h_shift: float      # height shift (down)       [a_0]
    r_c: float = np.inf  # radius of curvature       [l_x]
    n_x: int = 1_001    # real-space grid points

    x: FloatArray = field(
        default_factory=lambda: np.array([]), repr=False, init=False)
    height: FloatArray = field(
        default_factory=lambda: np.array([]), repr=False, init=False)

    def __post_init__(self):

        # unit conversions
        self.lambda_s *= self.a_0
        self.lambda_r *= self.l_x
        self.h_shift *= self.a_0
        self.r_c *= self.l_x

        # sanity checks
        if self.lambda_s > self.lambda_r or self.lambda_s > self.l_x:
            raise ValueError("lambda short is too short")

        # make spectrum and compute properties
        n_q = int(self.l_x / self.lambda_s + 0.5)
        q = 2 * np.pi * np.linspace(1, n_q, n_q) / self.l_x
        q_r = 2 * np.pi / self.lambda_r
        spectrum = 1. / (1 + (q/q_r)**2) ** ((1+2*self.hurst)/2)
        ms_slope = np.sum(q**2*spectrum) / 2
        spectrum *= self.rms_slope**2 / ms_slope
        rms_height = np.sqrt(np.sum(spectrum) / 2) / self.a_0
        rms_curvat = np.sqrt(np.sum(q**4 * spectrum) / 2)

        with open("topo_info.dat", "a") as file:
            file.write(f"# {self.name}\n")
            file.write(f"# {rms_height=:.6g} a_0\n")
            file.write(f"# {rms_curvat=:.6g} A\n\n")

        self.x = np.zeros(self.n_x)
        self.x[:] = np.linspace(0, self.l_x, self.n_x)
        self.height = np.zeros(self.n_x)

        # generate random roughness
        amplitude = np.sqrt(spectrum)
        phase = 2 * np.pi * np.random.rand(q.shape[0])
        for idx in range(len(self.x)):
            self.height[idx] = np.sum(amplitude * np.cos(q*self.x[idx]+phase))

        # add curvature
        if self.r_c != np.inf:
            self.height += (self.x-self.l_x/2)**2 / (2*self.r_c**1)

        self.height -= self.h_shift + np.min(self.height)
        self.height = np.maximum(self.height, 0.)

        # undo unit conversions
        self.lambda_s /= self.a_0
        self.lambda_r /= self.l_x
        self.h_shift /= self.a_0
        self.r_c /= self.l_x


def report_rigid_gaps(x, h_top, h_bot, f_movie=True):

    n_x = x.shape[0]
    if h_top.shape[0] != n_x or h_bot.shape[0] != n_x:
        raise ValueError("shapes do not match for movie")

    dir_name = "Topo_Movie"
    prepare_directory(dir_name)

    h_top_loc = h_top[:-1].copy()
    h_bot_loc = h_bot[:-1].copy()
    n_x -= 1

    gap_min = np.inf
    gap_max = 0.0

    n = range(n_x)
    for d_x in n:

        gap = np.array([h_top_loc[i_x] - h_bot_loc[i_x - d_x] for i_x in n])
        min_gap = np.min(gap)
        mean_gap = np.mean(gap) - min_gap
        gap_min = min(gap_min, mean_gap)
        gap_max = max(gap_max, mean_gap)

        if not f_movie:
            continue

        file_name_top = f"{dir_name}/top{d_x}.dat"
        file_name_bot = f"{dir_name}/bot{d_x}.dat"
        with open(file_name_top, 'w') as f_top, open(file_name_bot, 'w') as f_bot:
            for i_x in n:
                x_val = x[i_x]
                h_top_shifted = h_top_loc[(i_x + d_x) % n_x] - min_gap
                f_top.write(f"{x_val:.6g} {h_top_shifted:.6g}\n")
                f_bot.write(f"{x_val:.6g} {h_bot[i_x]:.6g}\n")

    with open("topo_info.dat", "a") as file:
        file.write(f"# {gap_min=:.6g}\n")
        file.write(f"# {gap_max=:.6g}\n\n")


def carve_profile(pos, x, y, f_flip_z=False, buffer=30):
    # buffer is the margin [Angstrom] of atoms kept above the highest point

    y_interp = np.interp(pos[:, 0], x, y)
    mask = (pos[:, 2] > y_interp) & (pos[:, 2] < np.max(y) + buffer)
    pos_new = pos[mask]

    names = np.full(pos_new.shape[0], "Cu", dtype="<U2")
    # TODO: change
    # names[pos_new[:, 2] > np.max(pos_new[:, 2]) - 2.5] = "Au"

    if f_flip_z:
        pos_new[:, 2] *= -1

    return pos_new, names


def write_config(r, names, file_name):
    with open(file_name, "a") as f:
        for r, n in zip(r, names):
            f.write(f"\n{n}")
            for i in range(r.shape[0]):
                f.write(f"\t{r[i]:.6g}")

# # #  U t i l i t y   F u n c t i o n s


@njit
def get_squared_distances(r):
    """
    Prints the squared distance between all atom pairs.

    Parameters:
        r (ndarray): Atomic positions, shape (n_atom, n_dim)
    """
    n_atom = r.shape[0]
    d2 = np.zeros((n_atom, n_atom))
    for i_atom in range(n_atom-1):
        for j_atom in range(i_atom+1, n_atom):
            d2[i_atom, j_atom] = np.sum((r[i_atom,] - r[j_atom,])**2)
    return d2


def write_array(arr, f, precision=6):
    arr = np.atleast_2d(arr)

    if np.issubdtype(arr.dtype, np.integer):
        fmt = "{:_}"
    else:
        fmt = f"{{:.{precision}f}}"

    for row in arr:
        f.write("\t".join(fmt.format(val) for val in row) + "\n")


def prepare_directory(dir_name: str) -> None:
    """
    Creates a directory with the name dir_name and deletes all *.dat files in it
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        for fname in os.listdir(dir_name):
            if '.dat' in fname:
                fpath = os.path.join(dir_name, fname)
                if os.path.isfile(fpath):
                    os.remove(fpath)


def main():

    # random_seed = 305853 (26/46)
    random_seed = 305853
    np.random.seed(random_seed)
    with open("topo_info.dat", "w") as f:
        f.write("")

    # define parameters for rough surface and indenter
    substrate = HeightTopography(
        name="substrate",
        a_0=3.62,
        l_x=500.0,
        hurst=0.8,
        lambda_r=0.25,
        lambda_s=6.0,
        rms_slope=1.0,
        h_shift=0.5)
    
    print(substrate)
    
    indenter = replace(substrate,
                       name="indenter",
                       r_c=1.2)
    print(indenter)
    
    # write info to file
    with open("topo_info.dat", "a") as f:
        f.write(f"# {indenter=}\n")
        f.write(f"# {random_seed=}\n\n")

    report_rigid_gaps(indenter.x, indenter.height, -
                      substrate.height, f_movie=True)

    r_top, r_bot = make_inc_blocks(l_x=substrate.l_x, a_0=substrate.a_0)

    r_top, names_top = carve_profile(
        r_top, indenter.x, indenter.height, f_flip_z=False)
    r_bot, names_bot = carve_profile(
        r_bot, substrate.x, substrate.height, f_flip_z=True, buffer=40)

    r_top[:, 2] += 5
    r_bot[:, 2] -= 5

    file_name = "config2.dat"
    # with open(file_name, "w") as f:
    #     f.write(f"{r_top.shape[0] + r_bot.shape[0]}\n")
    # write_config(r_top, names_top, file_name)
    # write_config(r_bot, names_bot, file_name)


if __name__ == "__main__":
    main()
