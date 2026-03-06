import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import galsim
from astropy.table import Table

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from superbit_lensing.utils import get_admoms, get_admoms_of_best_fit_em5_image

from superbit_lensing.medsmaker.superbit.psf_extender import PSFWrapper
from dataclasses import dataclass
from typing import Any, Dict

@dataclass(frozen=True)
class EM5PsfexMaps:
    # sampled grid + geometry
    x: np.ndarray
    y: np.ndarray
    xx: np.ndarray
    yy: np.ndarray
    extent: list
    image_xsize: int
    image_ysize: int
    step: int
    margin: int
    Ny: int
    Nx: int

    # maps (observed + model)
    e1: np.ndarray
    e2: np.ndarray
    T:  np.ndarray
    e1_em: np.ndarray
    e2_em: np.ndarray
    T_em:  np.ndarray

    # config that affects interpretation/plotting
    smooth: bool
    interpolation: str
    scale: float
    mode: str
    reduced: bool
    

    def to_npz(self, path: str, **extra_meta: Any) -> None:
        """
        Save maps + metadata to a compressed npz.
        extra_meta can store strings like psfex_file, image_file, git hash, etc.
        """
        d: Dict[str, Any] = {
            # geometry
            "x": self.x, "y": self.y, "xx": self.xx, "yy": self.yy,
            "extent": np.array(self.extent, dtype=float),
            "image_xsize": np.int64(self.image_xsize),
            "image_ysize": np.int64(self.image_ysize),
            "step": np.int64(self.step),
            "margin": np.int64(self.margin),
            "Ny": np.int64(self.Ny),
            "Nx": np.int64(self.Nx),

            # maps
            "e1": self.e1, "e2": self.e2, "T": self.T,
            "e1_em": self.e1_em, "e2_em": self.e2_em, "T_em": self.T_em,

            # config
            "smooth": np.bool_(self.smooth),
            "interpolation": np.array(self.interpolation),
            "scale": np.float64(self.scale),
            "mode": np.array(self.mode),
            "reduced": np.bool_(self.reduced),
        }

        # Add any user-provided metadata (strings/ints/floats/arrays)
        for k, v in extra_meta.items():
            d[f"meta__{k}"] = np.array(v)

        np.savez_compressed(path, **d)

    @staticmethod
    def from_npz(path: str) -> "EM5PsfexMaps":
        """Load maps + metadata from a compressed npz created by to_npz()."""
        z = np.load(path, allow_pickle=True)

        def _scalar(name, cast):
            return cast(z[name].item()) if z[name].shape == () else cast(z[name])

        extent = z["extent"].astype(float).tolist()

        return EM5PsfexMaps(
            x=z["x"], y=z["y"], xx=z["xx"], yy=z["yy"],
            extent=extent,
            image_xsize=_scalar("image_xsize", int),
            image_ysize=_scalar("image_ysize", int),
            step=_scalar("step", int),
            margin=_scalar("margin", int),
            Ny=_scalar("Ny", int),
            Nx=_scalar("Nx", int),

            e1=z["e1"], e2=z["e2"], T=z["T"],
            e1_em=z["e1_em"], e2_em=z["e2_em"], T_em=z["T_em"],

            smooth=bool(_scalar("smooth", bool)),
            interpolation=str(z["interpolation"].item() if z["interpolation"].shape == () else z["interpolation"]),
            scale=float(_scalar("scale", float)),
            mode=str(z["mode"].item() if z["mode"].shape == () else z["mode"]),
            reduced=bool(_scalar("reduced", bool)),
        )
        
    def to_table(self) -> Table:
        """
        Convert grid maps into a row-wise Astropy Table.

        Each row corresponds to a sampled detector position (x, y).
        """

        tab = Table()

        # flatten coordinates
        tab["x"] = self.xx.ravel()
        tab["y"] = self.yy.ravel()
        tab["r"] = np.sqrt((tab["x"]-4800)**2 + (tab["y"]-3200)**2)
        iy, ix = np.indices(self.e1.shape)
        tab["ix"] = ix.ravel()
        tab["iy"] = iy.ravel()

        # observed quantities
        tab["e1"] = self.e1.ravel()
        tab["e2"] = self.e2.ravel()
        tab["T"]  = self.T.ravel()

        # model quantities
        tab["e1_em"] = self.e1_em.ravel()
        tab["e2_em"] = self.e2_em.ravel()
        tab["T_em"]  = self.T_em.ravel()

        # residuals (optional but usually useful)
        tab["de1"] = tab["e1"] - tab["e1_em"]
        tab["de2"] = tab["e2"] - tab["e2_em"]
        tab["dT"]  = tab["T"]  - tab["T_em"]

        return tab
    
    def save_table(self, path):
        self.to_table().write(path, overwrite=True)


def compute_residuals(m: EM5PsfexMaps):
    r1 = m.e1 - m.e1_em
    r2 = m.e2 - m.e2_em
    rT = m.T  - m.T_em
    return r1, r2, rT

def format_colorbar(im, ax, vmin, vmax):
    """Helper function for colorbars"""
    import matplotlib.pyplot as plt
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{vmin:.2f}", "", f"{((vmin+vmax)/2):.2f}", "", f"{vmax:.2f}"])
    return cbar

def compute_em5_psfex_maps(
    psfex_file,
    image_file=None,
    image_xsize=9600,
    image_ysize=6400,
    step=200,
    margin=0,
    smooth=True,
    scale=0.141,
    mode="ngmix",
    reduced=True,
):
    seed = 12345
    interpolation = "bicubic" if smooth else "nearest"

    model = PSFWrapper(psf_file=psfex_file, image_file=image_file)
    # wcs = PSFWrapper.galsim_wcs(image_file=image_file)

    x = np.arange(margin, image_xsize - margin, step)
    y = np.arange(margin, image_ysize - margin, step)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    Ny, Nx = xx.shape

    extent = [margin, image_xsize - margin, margin, image_ysize - margin]

    e1_map = np.full((Ny, Nx), np.nan, dtype=float)
    e2_map = np.full((Ny, Nx), np.nan, dtype=float)
    T_map  = np.full((Ny, Nx), np.nan, dtype=float)

    e1_em_map = np.full((Ny, Nx), np.nan, dtype=float)
    e2_em_map = np.full((Ny, Nx), np.nan, dtype=float)
    T_em_map  = np.full((Ny, Nx), np.nan, dtype=float)

    for i in range(Ny):
        for j in range(Nx):
            y_im = int(yy[i, j])
            x_im = int(xx[i, j])
            try:
                psf_im = model.get_rec(y_im, x_im)
                galsim_pos = galsim.PositionD(x_im + 1.0, y_im + 1.0)
                # wcs_local = wcs.local(galsim_pos)
                # scale = np.sqrt(wcs_local.pixelArea())
                res = get_admoms(psf_im, scale=scale, mode=mode, reduced=reduced, seed=seed+1)
                em_res, em_im = get_admoms_of_best_fit_em5_image(image=psf_im, scale=scale, seed=seed)
                if res["flags"] == 0 and em_res["flags"] == 0:
                    e1_map[i, j] = res["e1"]
                    e2_map[i, j] = res["e2"]
                    T_map[i, j]  = res["T"]
                    e1_em_map[i, j] = em_res["e1"]
                    e2_em_map[i, j] = em_res["e2"]
                    T_em_map[i, j]  = em_res["T"]
            except Exception:
                continue

    return EM5PsfexMaps(
        x=x, y=y, xx=xx, yy=yy,
        extent=extent,
        image_xsize=image_xsize, image_ysize=image_ysize,
        step=step, margin=margin,
        Ny=Ny, Nx=Nx,
        e1=e1_map, e2=e2_map, T=T_map,
        e1_em=e1_em_map, e2_em=e2_em_map, T_em=T_em_map,
        smooth=smooth, interpolation=interpolation,
        scale=scale, mode=mode, reduced=reduced,
    )

def plot_em5_psfex_maps(m, show=True):
    """
    Plot observed/model/residual PSF moment maps from an EM5PsfexMaps result.

    Parameters
    ----------
    m : EM5PsfexMaps
        Output of compute_em5_psfex_maps.
    show : bool
        If True, call plt.show().

    Returns
    -------
    fig, axes
    """

    # ---- colormaps (NaNs -> grey) ----
    cmap_shape = cm.RdBu_r.copy()
    cmap_shape.set_bad(color="lightgray")

    cmap_T = cm.viridis.copy()
    cmap_T.set_bad(color="lightgray")

    # ---- plot ----
    SHOW_MODEL_ROW = True
    nrows = 3 if SHOW_MODEL_ROW else 2
    fig_width = 15
    cell_aspect = m.Ny / m.Nx
    fig_height = fig_width * (nrows / 3) * cell_aspect

    fig, axes = plt.subplots(nrows, 3, figsize=(fig_width, fig_height), constrained_layout=True)
    axes = np.array(axes).reshape(nrows, 3)

    COLOR_LIMITS = {
        "e_lim": 0.08,       # for e1 and e2
        "e_res_lim": 0.01,   # for e1 and e2 residuals
        "T_res_lim": 0.01,   # for T residuals
    }
    e_lim = COLOR_LIMITS["e_lim"]
    e_res_lim = COLOR_LIMITS["e_res_lim"]
    T_res_lim = COLOR_LIMITS["T_res_lim"]

    # Residuals (derived; not stored)
    residual_1_map, residual_2_map, residual_T_map = compute_residuals(m)

    interpolation = 'nearest'
    extent = m.extent

    # Observed row
    im1 = axes[0, 0].imshow(
        m.e1, origin="lower", cmap="RdBu_r",
        vmin=-e_lim, vmax=e_lim, aspect="equal", interpolation=interpolation,
        extent=extent,
    )
    axes[0, 0].set_aspect("equal")
    format_colorbar(im1, axes[0, 0], -e_lim, e_lim)

    im2 = axes[0, 1].imshow(
        m.e2, origin="lower", cmap="RdBu_r",
        vmin=-e_lim, vmax=e_lim, aspect="equal", interpolation=interpolation,
        extent=extent,
    )
    axes[0, 1].set_aspect("equal")
    format_colorbar(im2, axes[0, 1], -e_lim, e_lim)

    im3 = axes[0, 2].imshow(
        m.T, origin="lower", cmap="viridis",
        aspect="equal", interpolation=interpolation,
        extent=extent,
    )
    axes[0, 2].set_aspect("equal")
    format_colorbar(im3, axes[0, 2], np.nanmin(m.T), np.nanmax(m.T))

    # Model row (if enabled)
    if SHOW_MODEL_ROW:
        im4 = axes[1, 0].imshow(
            m.e1_em, origin="lower", cmap="RdBu_r",
            vmin=-e_lim, vmax=e_lim, aspect="equal", interpolation=interpolation,
            extent=extent,
        )
        axes[1, 0].set_aspect("equal")
        format_colorbar(im4, axes[1, 0], -e_lim, e_lim)

        im5 = axes[1, 1].imshow(
            m.e2_em, origin="lower", cmap="RdBu_r",
            vmin=-e_lim, vmax=e_lim, aspect="equal", interpolation=interpolation,
            extent=extent,
        )
        axes[1, 1].set_aspect("equal")
        format_colorbar(im5, axes[1, 1], -e_lim, e_lim)

        im6 = axes[1, 2].imshow(
            m.T_em, origin="lower", cmap="viridis",
            aspect="equal", interpolation=interpolation,
            extent=extent,
        )
        axes[1, 2].set_aspect("equal")
        format_colorbar(im6, axes[1, 2], np.nanmin(m.T_em), np.nanmax(m.T_em))

    # Residual row
    row_idx = 2 if SHOW_MODEL_ROW else 1

    im7 = axes[row_idx, 0].imshow(
        residual_1_map, origin="lower", cmap="RdBu_r",
        vmin=-e_res_lim, vmax=e_res_lim, aspect="equal", interpolation=interpolation,
        extent=extent,
    )
    axes[row_idx, 0].set_aspect("equal")
    format_colorbar(im7, axes[row_idx, 0], -e_res_lim, e_res_lim)

    im8 = axes[row_idx, 1].imshow(
        residual_2_map, origin="lower", cmap="RdBu_r",
        vmin=-e_res_lim, vmax=e_res_lim, aspect="equal", interpolation=interpolation,
        extent=extent,
    )
    axes[row_idx, 1].set_aspect("equal")
    format_colorbar(im8, axes[row_idx, 1], -e_res_lim, e_res_lim)

    im9 = axes[row_idx, 2].imshow(
        residual_T_map, origin="lower", cmap="RdBu_r",
        vmin=-T_res_lim, vmax=T_res_lim, aspect="equal", interpolation=interpolation,
        extent=extent,
    )
    axes[row_idx, 2].set_aspect("equal")
    format_colorbar(im9, axes[row_idx, 2], -T_res_lim, T_res_lim)

    # Axis labels
    nx_ticks = 5
    ny_ticks = 5
    xt = np.linspace(m.x.min(), m.x.max(), nx_ticks)
    yt = np.linspace(m.y.min(), m.y.max(), ny_ticks)

    for i in range(nrows):
        for j in range(3):
            ax = axes[i, j]

            ax.set_xlim(0, m.image_xsize)
            ax.set_ylim(0, m.image_ysize)
            ax.set_facecolor("lightgray")

            if i == nrows - 1:
                ax.set_xlabel("X [pixels]")
                ax.set_xticks(xt)
                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            else:
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel("Y [pixels]")
                ax.set_yticks(yt)
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            else:
                ax.set_yticks([])

    # Column titles
    col_titles = [r"$e_1$", r"$e_2$", r"$T$"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, pad=10)

    # Row labels
    if SHOW_MODEL_ROW:
        row_labels = ["Observed", "Model", "Residual"]
        y_positions = [0.83, 0.50, 0.17]
    else:
        row_labels = ["Observed", "Residual"]
        y_positions = [0.765, 0.28]

    for label, ypos in zip(row_labels, y_positions):
        fig.text(
            1.00, ypos, label,
            rotation=270, va="center", ha="center",
            fontsize=16, transform=fig.transFigure,
            clip_on=False,
        )

    if show:
        plt.show()

    return fig, axes

def assert_compatible_maps(maps_list):
    if len(maps_list) == 0:
        raise ValueError("maps_list is empty")

    m0 = maps_list[0]
    for k, m in enumerate(maps_list[1:], start=1):
        if (m.Ny != m0.Ny) or (m.Nx != m0.Nx):
            raise ValueError(f"Incompatible Ny/Nx at index {k}: {(m.Ny, m.Nx)} vs {(m0.Ny, m0.Nx)}")
        if (m.step != m0.step) or (m.margin != m0.margin):
            raise ValueError(f"Incompatible step/margin at index {k}")
        if (m.image_xsize != m0.image_xsize) or (m.image_ysize != m0.image_ysize):
            raise ValueError(f"Incompatible image size at index {k}")
        # these should match if you want “same grid”
        if not (np.array_equal(m.x, m0.x) and np.array_equal(m.y, m0.y)):
            raise ValueError(f"Incompatible x/y grid at index {k}")
        if m.extent != m0.extent:
            raise ValueError(f"Incompatible extent at index {k}")
    return True


def average_em5_maps(maps_list, *, smooth=None, interpolation=None) -> EM5PsfexMaps:
    """
    Average a list of EM5PsfexMaps on the same grid using nanmean.

    Returns a new EM5PsfexMaps with averaged fields.
    """
    assert_compatible_maps(maps_list)
    m0 = maps_list[0]

    def nanmean_stack(attr):
        stack = np.stack([getattr(m, attr) for m in maps_list], axis=0)
        return np.nanmean(stack, axis=0)

    e1 = nanmean_stack("e1")
    e2 = nanmean_stack("e2")
    T  = nanmean_stack("T")
    e1_em = nanmean_stack("e1_em")
    e2_em = nanmean_stack("e2_em")
    T_em  = nanmean_stack("T_em")

    # keep config from first unless overridden
    _smooth = m0.smooth if smooth is None else bool(smooth)
    _interp = m0.interpolation if interpolation is None else str(interpolation)

    return EM5PsfexMaps(
        x=m0.x, y=m0.y, xx=m0.xx, yy=m0.yy,
        extent=m0.extent,
        image_xsize=m0.image_xsize, image_ysize=m0.image_ysize,
        step=m0.step, margin=m0.margin,
        Ny=m0.Ny, Nx=m0.Nx,

        e1=e1, e2=e2, T=T,
        e1_em=e1_em, e2_em=e2_em, T_em=T_em,

        smooth=_smooth,
        interpolation=_interp,
        scale=m0.scale,
        mode=m0.mode,
        reduced=m0.reduced,
    )
    
class EM5PsfexMapCollection:
    def __init__(self, maps_list=None):
        self.maps_list = list(maps_list) if maps_list is not None else []

    def add(self, m: EM5PsfexMaps):
        self.maps_list.append(m)

    def save_all(self, out_dir, stem="psf_maps"):
        import os
        os.makedirs(out_dir, exist_ok=True)
        for i, m in enumerate(self.maps_list):
            m.to_npz(os.path.join(out_dir, f"{stem}_{i:04d}.npz"))

    @staticmethod
    def load_many(paths):
        return EM5PsfexMapCollection([EM5PsfexMaps.from_npz(p) for p in paths])

    def mean(self):
        return average_em5_maps(self.maps_list)

    def plot_mean(self, show=True):
        m_avg = self.mean()
        return plot_em5_psfex_maps(m_avg, show=show)
    



def em5_psfex_shape_maps(
    psfex_file,
    image_file = None,
    image_xsize=9600,
    image_ysize=6400,
    step=200,
    margin=0,
    smooth=True,
    scale=0.141,
    mode="ngmix",
    reduced=True,
    show=True,
    return_vals=False
):
    """
    Sample PSFEx model across the detector on a coarse grid and plot e1, e2, T maps.

    Returns
    -------
    (e1_map, e2_map, T_map, xx, yy)
      where maps have shape (Ny, Nx) and xx,yy are the coordinate grids.
    """
    seed = 12345
    interpolation = "bicubic" if smooth else "nearest"

    # ---- load PSFEx model ----
    model = PSFWrapper(psf_file=psfex_file, image_file=image_file)
    #model = psfex.PSFEx(psfex_file)

    # ---- grid of sample points ----
    x = np.arange(margin, image_xsize - margin, step)
    y = np.arange(margin, image_ysize - margin, step)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    Ny, Nx = xx.shape

    dx = step
    dy = step
    extent = [
        x.min() - dx/2, x.max() + dx/2,   # x from left to right
        y.min() - dy/2, y.max() + dy/2    # y from bottom to top
    ]

    extent = [margin, image_xsize - margin, margin, image_ysize - margin]

    e1_map = np.full((Ny, Nx), np.nan, dtype=float)
    e2_map = np.full((Ny, Nx), np.nan, dtype=float)
    T_map  = np.full((Ny, Nx), np.nan, dtype=float)
    
    e1_em_map = np.full((Ny, Nx), np.nan, dtype=float)
    e2_em_map = np.full((Ny, Nx), np.nan, dtype=float)
    T_em_map  = np.full((Ny, Nx), np.nan, dtype=float)
    
    residual_1_map = np.full((Ny, Nx), np.nan, dtype=float)
    residual_2_map = np.full((Ny, Nx), np.nan, dtype=float)
    residual_T_map = np.full((Ny, Nx), np.nan, dtype=float)

    # ---- evaluate PSF + moments ----
    for i in range(Ny):
        for j in range(Nx):
            y_im = int(yy[i, j])
            x_im = int(xx[i, j])

            try:
                psf_im = model.get_rec(y_im, x_im)
                #psf_im = psf_im/np.sum(psf_im)
                res = get_admoms(psf_im, scale=scale, mode=mode, reduced=reduced, seed=seed+1)
                em_res, em_im = get_admoms_of_best_fit_em5_image(image=psf_im, scale=scale, seed=seed)
                if res['flags'] == 0 and em_res['flags'] == 0:
                    e1_map[i, j] = res["e1"]
                    e2_map[i, j] = res["e2"]
                    T_map[i, j]  = res["T"]
                    e1_em_map[i, j] = em_res["e1"]
                    e2_em_map[i, j] = em_res["e2"]
                    T_em_map[i, j]  = em_res["T"]
                    residual_1_map[i, j] = res["e1"] - em_res["e1"]
                    residual_2_map[i, j] = res["e2"] - em_res["e2"]
                    residual_T_map[i, j]  = res["T"] - em_res["T"]
            except Exception:
                # keep NaNs if anything fails
                continue

    # ---- colormaps (NaNs -> grey) ----
    cmap_shape = cm.RdBu_r.copy()
    cmap_shape.set_bad(color="lightgray")

    cmap_T = cm.viridis.copy()
    cmap_T.set_bad(color="lightgray")

    # ---- plot ----
    # Plotting
    SHOW_MODEL_ROW = True
    nrows = 3 if SHOW_MODEL_ROW else 2
    fig_width = 15
    cell_aspect = Ny / Nx
    fig_height = fig_width * (nrows / 3) * cell_aspect

    fig, axes = plt.subplots(nrows, 3, figsize=(fig_width, fig_height), constrained_layout=True)
    axes = np.array(axes).reshape(nrows, 3)
    
    COLOR_LIMITS = {
        'e_lim': 0.08,        # for e1 and e2
        'e_res_lim': 0.06,    # for e1 and e2 residuals
        'T_res_lim': 0.06     # for T residuals
    }
    e_lim = COLOR_LIMITS['e_lim']
    e_res_lim = COLOR_LIMITS['e_res_lim']
    T_res_lim = COLOR_LIMITS['T_res_lim']
    
    
    # Observed row
    im1 = axes[0,0].imshow(e1_map, origin='lower', cmap='RdBu_r', 
                           vmin=-e_lim, vmax=e_lim, aspect='equal', interpolation=interpolation,
                           extent=extent)
    axes[0,0].set_aspect('equal')
    format_colorbar(im1, axes[0,0], -e_lim, e_lim)

    im2 = axes[0,1].imshow(e2_map, origin='lower', cmap='RdBu_r',
                           vmin=-e_lim, vmax=e_lim, aspect='equal', interpolation=interpolation,
                           extent=extent)
    axes[0,1].set_aspect('equal')
    format_colorbar(im2, axes[0,1], -e_lim, e_lim)

    im3 = axes[0,2].imshow(T_map, origin='lower', cmap='viridis', aspect='equal', interpolation=interpolation,
                           extent=extent)
    axes[0,2].set_aspect('equal')
    format_colorbar(im3, axes[0,2], np.nanmin(T_map), np.nanmax(T_map))
    
    # Model row (if enabled)
    if SHOW_MODEL_ROW:
        im4 = axes[1,0].imshow(e1_em_map, origin='lower', cmap='RdBu_r',
                            vmin=-e_lim, vmax=e_lim, aspect='equal', interpolation=interpolation,
                           extent=extent)
        axes[1,0].set_aspect('equal')
        format_colorbar(im4, axes[1,0], -e_lim, e_lim)

        im5 = axes[1,1].imshow(e2_em_map, origin='lower', cmap='RdBu_r',
                            vmin=-e_lim, vmax=e_lim, aspect='equal', interpolation=interpolation,
                            extent=extent)
        axes[1,1].set_aspect('equal')
        format_colorbar(im5, axes[1,1], -e_lim, e_lim)

        im6 = axes[1,2].imshow(T_em_map, origin='lower', cmap='viridis', aspect='equal', interpolation=interpolation,
                            extent=extent)
        axes[1,2].set_aspect('equal')
        format_colorbar(im6, axes[1,2], np.nanmin(T_em_map), np.nanmax(T_em_map))

    # Residual row
    row_idx = 2 if SHOW_MODEL_ROW else 1

    im7 = axes[row_idx,0].imshow(residual_1_map, origin='lower', cmap='RdBu_r', 
                                vmin=-e_res_lim, vmax=e_res_lim, aspect='equal', interpolation=interpolation,
                                extent=extent)
    axes[row_idx,0].set_aspect('equal')
    format_colorbar(im7, axes[row_idx,0], -e_res_lim, e_res_lim)

    im8 = axes[row_idx,1].imshow(residual_2_map, origin='lower', cmap='RdBu_r', 
                                vmin=-e_res_lim, vmax=e_res_lim, aspect='equal', interpolation=interpolation,
                                extent=extent)
    axes[row_idx,1].set_aspect('equal')
    format_colorbar(im8, axes[row_idx,1], -e_res_lim, e_res_lim)

    im9 = axes[row_idx,2].imshow(residual_T_map, origin='lower', cmap='RdBu_r',
                                vmin=-T_res_lim, vmax=T_res_lim, aspect='equal', interpolation=interpolation,
                                extent=extent)
    axes[row_idx,2].set_aspect('equal')
    format_colorbar(im9, axes[row_idx,2], -T_res_lim, T_res_lim)

    # Axis labels
    # choose ~5 ticks across the detector for readability
    nx_ticks = 5
    ny_ticks = 5
    xt = np.linspace(x.min(), x.max(), nx_ticks)
    yt = np.linspace(y.min(), y.max(), ny_ticks)

    for i in range(nrows):
        for j in range(3):
            ax = axes[i, j]

            # lock limits so all panels match (important once using extent)
            ax.set_xlim(0, image_xsize)
            ax.set_ylim(0, image_ysize)
            ax.set_facecolor("lightgray")



            if i == nrows - 1:
                ax.set_xlabel("X [pixels]")
                ax.set_xticks(xt)
                ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            else:
                ax.set_xticks([])  # <-- instead of set_xticklabels([])

            if j == 0:
                ax.set_ylabel("Y [pixels]")
                ax.set_yticks(yt)
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            else:
                ax.set_yticks([])  # <-- instead of set_yticklabels([])

    # Column titles
    col_titles = [r"$e_1$", r"$e_2$", r"$T$"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, pad=10)

    # Row labels
    if SHOW_MODEL_ROW:
        row_labels = ["Observed", "Model", "Residual"]
        y_positions = [0.83, 0.50, 0.17]
    else:
        row_labels = ["Observed", "Residual"]
        y_positions = [0.765, 0.28]

    for label, ypos in zip(row_labels, y_positions):
        fig.text(1.00, ypos, label,
                rotation=270, va="center", ha="center",
                fontsize=16, transform=fig.transFigure,
                clip_on=False)
    if show:
        plt.show()
    if return_vals:
        return e1_map, e2_map, T_map, xx, yy