"""
Free-Floating Planet (FFP) kick-velocity / Galactic-dynamics simulation.
this is basically a simulation where kicks r happening 24/7 over 10^8 myr

"""

import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# ═══════════
# CONSTANTS 
# ═════════════
G            = 4.30091e-6      # kpc (km/s)^2 / Msun
M_halo       = 7.25e11         # Msun            
R_d          = 3.5             # kpc  disk scale length   
z_d          = 0.3             # kpc  disk scale height  
M_earth      = 3.003e-6        # Msun  (Earth mass, used only for M_e = M/M_earth)

R_EPS        = 1e-3            # kpc  guard for R=0 
R0_HALO      = 1e-8            # kpc 
GRAD_H       = 1e-2            # kpc 
KM_PER_KPC   = 3.0857e16
SEC_PER_YEAR = 3.154e7
DT_FAC       = SEC_PER_YEAR / KM_PER_KPC   # unit conversion: yr -> kpc/(km/s)

# ══════════════════
# BULGE PARAMETERS  
# ═══════════════════
BULGE_RHO0   = 1e10             # Msun/kpc^3 
BULGE_A_SOFT = 0.5              # kpc         
BULGE_THETA  = 25.0             # deg        
BULGE_X0     = 1.49             # kpc         
BULGE_Y0     = 0.58
BULGE_Z0     = 0.40

# ═════════════════
# HALO PARAMETERS  
# ══════════════════
R_S_HALO = 10.0                 

# ═════════════════════
# EJECTION CHANNEL PARAMETERS  
# ══════════════════════════

# Circumbinary (CB) channel — factorized source term
CB_ALPHA          = 1.3   
CB_M_MIN_EARTH    = 1.0
CB_M_MAX_EARTH    = 317.0
CB_M_TURNOVER     = 3.0   
CB_GAMMA_SUPPRESS = 2.0    
SIGMA_CB          = 10.0   # km/s

# Single-star planet-planet scattering (PL) channel — constant-kick model
PL_ALPHA          = 1.3   
PL_M_MIN_EARTH    = 1.0
PL_M_MAX_EARTH    = 317.0
PL_V_KICK         = 4.0    # km/s --> v_k = 4 km/s (range 2-6)
PL_V_KICK_SIGMA   = 1.0    # km/s —> narrow Gaussian standing in for the delta function
# NOTE: PL_V_KICK_SIGMA gives a Gaussian speed distribution conditional on
# mass, v ~ Normal(v_base(M), PL_V_KICK_SIGMA). That is only a *population-
# level* Gaussian when MASS_COUPLED = False (v_base = PL_V_KICK, constant).
# When MASS_COUPLED = True, v_base(M) varies with mass, so the marginal
# (mass-integrated) PL speed distribution is a mixture over P(M) and is NOT
# Gaussian -- don't call it "Gaussian kick distribution" in that case.

# Channel mix — matches the 500,000 / 500,000 CB/PL split used in the paper's tables
F_CB = 0.5

# ══════════════════════════════════════════════
# MASS_COUPLED = False -> NULL MODEL (factorized f(v) ):
#     CB: Maxwellian(sigma_CB = 10 km/s) for every particle, mass-independent
#     PL: Gaussian(v_k = 4 km/s, sigma = 1 km/s) for every particle, mass-independent
#
# MASS_COUPLED = True  -> COUPLED MODEL f(v|M):
#     CB: sigma_eff(M) = sigma_CB * (M/M_earth)^-0.15
#     PL: v_base(M)   = v_k * (M/M_earth)^-0.5
# ══════════════════════════════════════════════
MASS_COUPLED = True

# Disk escape threshold — Sec. 5.4.3 / Table (tab:quantities): Disk escape = |z| > z_esc
Z_ESCAPE_KPC = 2.0     # kpc

# ══════════════════════════════════════════════
# MASS-RESOLVED ANALYSIS / NO-KICK CONTROL
# ══════════════════════════════════════════════
# Mass-bin edges [M_earth] used by mass_resolved_kick_report() and
# plot_mass_resolved_spatial(). Five roughly half-decade bins spanning the
# sampled 1-317 M_earth range; the first three edges (1, 3, 10) are the same
# ones compute_disk_departure_fraction() uses. Edit to match the paper's bins.
MASS_BIN_EDGES_EARTH = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 317.0])

# Evolve a matched no-kick control population alongside the kicked one
# (same initial positions / masses / labels, velocity = local circular
# velocity only). Roughly doubles the integration cost and memory; set to
# False to skip the control and the kicked-vs-control spatial figures.
RUN_NOKICK_CONTROL = True

# Per-panel random subsample cap for the mass-resolved (R, z) KDE panels
# (there are 2 x n_mass_bins KDEs per snapshot, so this is kept below
# PLOT_MAX_POINTS to keep plotting time reasonable).
MASS_RES_PLOT_MAX_POINTS = 100000


# ══════════════════════════════════════════════
# GALACTIC POTENTIAL 
# ══════════════════════════════════════════════

def _disk_potential_v(R, z):
    """Softened exponential disk potential 'a phenomenological softened exponential potential')."""
    return -G * (6e10) * np.exp(-R / R_d) / np.sqrt(
        R**2 + (z_d + np.abs(z))**2 + 1e-12)


def _bulge_potential_v(x, y, z):
    """Bissantz-Gerhard triaxial bulge, Eq. (bulge_potential)."""
    ct = np.cos(np.radians(BULGE_THETA))
    st = np.sin(np.radians(BULGE_THETA))
    xp = x * ct + y * st
    yp = -x * st + y * ct
    r_s = np.sqrt((xp / BULGE_X0)**2 + (yp / BULGE_Y0)**2 + (z / BULGE_Z0)**2)
    return (-4.0 * np.pi * G * BULGE_RHO0
            * BULGE_X0 * BULGE_Y0 * BULGE_Z0
            / (r_s + BULGE_A_SOFT))


def _halo_potential_v(r):
    """NFW-like logarithmic halo potential, Eq. (halo_potential)."""
    r_eff = np.sqrt(r * r + R0_HALO**2)
    return -G * M_halo * np.log(1.0 + r_eff / R_S_HALO) / r_eff


def total_potential_v(x, y, z):
    """Phi_tot = Phi_disk + Phi_bulge + Phi_halo """
    R = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)
    return _disk_potential_v(R, z) + _bulge_potential_v(x, y, z) + _halo_potential_v(r)


# ════════════════
# ACCELERATIONS  
# ═════════════════

def compute_accelerations(positions):
    h = GRAD_H
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    ax = -(total_potential_v(x + h, y, z) - total_potential_v(x - h, y, z)) / (2 * h)
    ay = -(total_potential_v(x, y + h, z) - total_potential_v(x, y - h, z)) / (2 * h)
    az = -(total_potential_v(x, y, z + h) - total_potential_v(x, y, z - h)) / (2 * h)
    return np.column_stack([ax, ay, az])


# ════════════════════
# CIRCULAR VELOCITY  
# ════════════════════

# NOTE: an azimuthally-*averaged* v_circ(R) (tried first, kept in git
# history / previous version of this comment) only fixes the BIAS -- it
# re-centers the reference curve on the correct mean at each R, but a
# single number per R still cannot match the true per-particle circular
# velocity, which genuinely differs at fixed R depending on azimuthal
# angle in a triaxial potential. That leaves the SPREAD (the std that
# actually inflates sigma_phi) untouched.
#
# subtract each particle's own local circular velocity,
# evaluated at its ACTUAL (x, y) position (not just its R), via
# circular_velocity_xy() below. This is fully vectorized -- no
# interpolation table is needed since it's a direct O(1) evaluation of
# the potential gradient per particle, done for all ~1e6 particles at
# once with numpy, so the old R-only lookup-table machinery
# (_VCIRC_TABLE / circular_velocity_fast) is no longer needed and has
# been removed. circular_velocity(R) is kept only as a scalar
# convenience wrapper (e.g. for printing a reference number at a given
# radius); it is NOT used anywhere in the per-particle pipeline anymore.

def circular_velocity(R, h=GRAD_H):
    """Scalar v_circ(R) evaluated along y=0 -- kept only as a quick
    reference value (e.g. for print statements); per-particle code uses
    circular_velocity_xy() instead, which is angle-aware."""
    R = max(R, R_EPS)
    dPhidR = (total_potential_v(R + h, 0.0, 0.0)
              - total_potential_v(R - h, 0.0, 0.0)) / (2 * h)
    return float(np.sqrt(max(R * dPhidR, 0.0)))


def circular_velocity_xy(x, y, h=GRAD_H):
    """Vectorized, ANGLE-AWARE circular velocity: for each particle,
    evaluates the radial potential gradient along the true direction
    from the Galactic center to that particle's (x, y), at z = 0 (the
    midplane rotation curve,
    generalized to be well-defined at every azimuthal angle rather than
    only along y = 0). x and y may be scalars or arrays."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    R = np.sqrt(x**2 + y**2)
    R_safe = np.maximum(R, R_EPS)
    ux, uy = x / R_safe, y / R_safe
    xp, yp = x + h * ux, y + h * uy
    xm, ym = x - h * ux, y - h * uy
    zeros = np.zeros_like(R_safe)
    dPhidR = (total_potential_v(xp, yp, zeros) - total_potential_v(xm, ym, zeros)) / (2 * h)
    return np.sqrt(np.maximum(R_safe * dPhidR, 0.0))


# circular_velocity_fast(R) / the R-only interpolation table have been

# ~1e6 particles since it's pure numpy array math, not a Python loop.


# ══════════════════════════════════════════════
# MASS SAMPLERS  (Phenomenological Source Term)
# ══════════════════════════════════════════════

_trapz = getattr(np, "trapezoid", None) or np.trapz

def _inverse_cdf_sample(f_unnorm, M_grid, n, rng):
    f = f_unnorm / _trapz(f_unnorm, M_grid)
    dM = M_grid[1] - M_grid[0]
    cdf = np.cumsum(f) * dM
    cdf /= cdf[-1]
    return np.interp(rng.uniform(size=n), cdf, M_grid) * M_earth


def sample_cb_masses(n, rng):
    """dN/dM ~ M^-alpha * [1 + (M/M_b)^gamma]^-1 """
    M = np.linspace(CB_M_MIN_EARTH, CB_M_MAX_EARTH, 5000000)
    f = M**(-CB_ALPHA) / (1.0 + (M / CB_M_TURNOVER)**CB_GAMMA_SUPPRESS)
    return _inverse_cdf_sample(f, M, n, rng)


def sample_pl_masses(n, rng):
    """dN/dM ~ M^-alpha, constant-kick model mass spectrum """
    M = np.linspace(PL_M_MIN_EARTH, PL_M_MAX_EARTH, 5000000)
    f = M**(-PL_ALPHA)
    return _inverse_cdf_sample(f, M, n, rng)


# ══════════════════════════
# ISOTROPIC KICK DIRECTIONS
# ══════════════════════════

def sample_isotropic_directions(n, rng):
    phi = rng.uniform(0, 2 * np.pi, n)
    costh = rng.uniform(-1, 1, n)
    sinth = np.sqrt(1.0 - costh**2)
    return np.column_stack((sinth * np.cos(phi), sinth * np.sin(phi), costh))


# ══════════════════════════════════════════════
# INITIALIZATION  (Kick Model Implementation)
# ══════════════════════════════════════════════

def initialize_ffps(n, f_cb=F_CB, rng=None):
    """
    Positions: R sampled from Gamma(shape=2, scale=R_d), which follows from
    the disk surface density Sigma(R) ~ exp(-R/R_d) (Sec. 5.1, disk density)
    combined with the cylindrical area element dA = 2*pi*R*dR.
    z sampled from a Laplace distribution with scale z_d, matching the
    disk density's exp(-|z|/z_d) vertical profile exactly (Sec. 5.1).

    Velocities: v_init = (0, v_phi, 0) purely azimuthal (Sec. 4),
    then v = (v_R,kick, v_phi,circ + v_phi,kick, v_z,kick) after the kick
    

    Returns positions, velocities, masses, labels, and speeds (the sampled
    kick-speed magnitude per particle, before the isotropic direction is
    applied). `speeds` is what should be used for any "mean kick" diagnostic
    -- in the mass-coupled model (MASS_COUPLED = True) sigma_eff / v_base
    vary per particle, so there is no single analytic mean to fall back on;
    the sampled values must be averaged directly.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_cb = int(round(n * f_cb))
    n_pl = n - n_cb

    # Positions
    R = rng.gamma(shape=2.0, scale=R_d, size=n)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = rng.laplace(0.0, z_d, n)
    positions = np.vstack([x, y, z]).T

    # Circular velocity -> purely azimuthal initial velocity
    # (vectorized, angle-aware via circular_velocity_xy; a per-particle
    # Python loop here would be prohibitively slow at n ~ 1e6, and using
    # only R -- ignoring the particle's actual angle -- would be wrong
    # given the triaxial bulge; see notes above circular_velocity_xy)
    v_circ = circular_velocity_xy(x, y)
    v_star = np.vstack([-np.sin(phi) * v_circ,
                         np.cos(phi) * v_circ,
                         np.zeros(n)]).T

    # Masses and kick speeds
    masses = np.empty(n)
    speeds = np.empty(n)

    if n_cb > 0:
        masses[:n_cb] = sample_cb_masses(n_cb, rng)
        if MASS_COUPLED:
            M_e_cb = masses[:n_cb] / M_earth
            sigma_eff = SIGMA_CB * M_e_cb**(-0.15)
            vx_cb = rng.normal(0, sigma_eff, n_cb)
            vy_cb = rng.normal(0, sigma_eff, n_cb)
            vz_cb = rng.normal(0, sigma_eff, n_cb)
        else:
            vx_cb = rng.normal(0, SIGMA_CB, n_cb)
            vy_cb = rng.normal(0, SIGMA_CB, n_cb)
            vz_cb = rng.normal(0, SIGMA_CB, n_cb)
        speeds[:n_cb] = np.sqrt(vx_cb**2 + vy_cb**2 + vz_cb**2)

    if n_pl > 0:
        masses[n_cb:] = sample_pl_masses(n_pl, rng)
        if MASS_COUPLED:
            M_e_pl = masses[n_cb:] / M_earth
            v_base = PL_V_KICK * M_e_pl**(-0.5)
        else:
            v_base = PL_V_KICK
        speeds[n_cb:] = np.abs(rng.normal(v_base, PL_V_KICK_SIGMA, n_pl))

    # Isotropic kick direction -> total velocity 
    kicks = speeds[:, None] * sample_isotropic_directions(n, rng)
    velocities = v_star + kicks

    labels = np.array(['CB'] * n_cb + ['PL'] * n_pl)
    return positions, velocities, masses, labels, speeds


# ══════════════════════════════════════════════
# NO-KICK CONTROL POPULATION
# ══════════════════════════════════════════════

def initialize_control_population(positions):
    """
    Matched no-kick control for the kicked FFP population.

    The control starts from EXACTLY the same initial positions as the
    kicked population (row i of the control is the same particle as row i
    of the kicked arrays, so the masses / labels returned by
    initialize_ffps apply to both), but with v = v_star only, i.e. the
    purely azimuthal local circular velocity and NO ejection kick. It
    represents the stellar tracer population (planets that stayed with
    their host stars) against which the kicked distribution is compared.

    v_star is rebuilt from the positions with the same formula used inside
    initialize_ffps (angle-aware circular_velocity_xy, direction from
    arctan2(y, x)), so it is identical to the kicked population's pre-kick
    velocity. Because the particles are test particles (no mass dependence
    in the dynamics), the evolved control is independent of MASS_COUPLED
    and of the kick model; only its assignment to mass bins uses `masses`.

    Returns (positions_control, velocities_control).
    """
    x, y = positions[:, 0], positions[:, 1]
    phi = np.arctan2(y, x)
    v_circ = circular_velocity_xy(x, y)
    v_star = np.column_stack([-np.sin(phi) * v_circ,
                               np.cos(phi) * v_circ,
                               np.zeros(len(x))])
    return positions.copy(), v_star


# ══════════════════════════════════════════════
# ESCAPE VELOCITY DIAGNOSTIC 
# v_esc(r) = sqrt(-2 Phi_tot(r)) 
# ══════════════════════════════════════════════

def check_escape_velocity(positions, speeds, labels):
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    phi_vals = total_potential_v(x, y, z)
    v_esc = np.sqrt(-2.0 * phi_vals)   # km/s
    R = np.sqrt(x**2 + y**2)

    print("\n  Escape velocity at birth locations:")
    print(f"    median v_esc = {np.median(v_esc):.1f} km/s")
    print(f"    min v_esc    = {np.min(v_esc):.1f} km/s")
    print(f"    max v_esc    = {np.max(v_esc):.1f} km/s")
    print(f"    median R     = {np.median(R):.2f} kpc")

    # Actual mean kick speeds, measured directly from the sampled
    # kick-speed distribution rather than an analytic Maxwellian formula.
    # SIGMA_CB * sqrt(8/pi) (and PL_V_KICK alone) are only the true means
    # when MASS_COUPLED = False; in the coupled model sigma_eff / v_base
    # vary per particle with mass, so the population mean must come from
    # the data itself.
    cb_mask = (labels == 'CB')
    pl_mask = (labels == 'PL')
    cb_mean_kick = np.mean(speeds[cb_mask]) if cb_mask.any() else float('nan')
    pl_mean_kick = np.mean(speeds[pl_mask]) if pl_mask.any() else float('nan')
    print(f"\n    CB mean kick = {cb_mean_kick:.1f} km/s  "
          f"({cb_mean_kick / np.median(v_esc) * 100:.1f}% of v_esc)")
    print(f"    PL mean kick = {pl_mean_kick:.1f} km/s  "
          f"({pl_mean_kick / np.median(v_esc) * 100:.1f}% of v_esc)")


# ══════════════════════════════════════════════
# GALACTIC ESCAPE FRACTION  
# E_i = 0.5*v^2 + Phi_tot ;  unbound if E_i > 0
# ══════════════════════════════════════════════

def compute_galactic_escape_fraction(positions, velocities, labels, tag=""):
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    vx, vy, vz = velocities[:, 0], velocities[:, 1], velocities[:, 2]
    E = 0.5 * (vx**2 + vy**2 + vz**2) + total_potential_v(x, y, z)
    unbound = E > 0.0

    print(f"\n  Galactic escape fractions (E > 0) ({tag}):")
    for ch in ['CB', 'PL', 'ALL']:
        mask = np.ones(len(E), dtype=bool) if ch == 'ALL' else (labels == ch)
        if mask.any():
            print(f"    {ch}: f_galactic_esc = {unbound[mask].mean():.4f}")
    return unbound


# ══════════════════════════════════════════════
# DISK DEPARTURE FRACTION
# Vertical-excursion criterion: |z| > Z_ESCAPE_KPC
# NOTE: this is NOT gravitational escape -- a particle at |z| slightly
# above Z_ESCAPE_KPC can still be perfectly bound (E < 0). It only marks
# departure from the thin-disk region. True escape (E > 0) is computed
# separately in compute_galactic_escape_fraction().
# ══════════════════════════════════════════════

def compute_disk_departure_fraction(positions, labels, masses, tag=""):
    z_abs = np.abs(positions[:, 2])
    is_departed = z_abs > Z_ESCAPE_KPC
    M_e = masses / M_earth
    mass_bins = np.array([1.0, 3.0, 10.0, 317.0])   

    print(f"\n  Disk departure fractions |z| > {Z_ESCAPE_KPC} kpc ({tag}):")
    print(f"  {'Mass bin [Me]':>16s}  {'Ch':>3s}  {'N':>6s}  {'f_dep':>7s}")
    print(f"  {'-'*42}")

    for i in range(len(mass_bins) - 1):
        m_lo, m_hi = mass_bins[i], mass_bins[i + 1]
        m_mask = (M_e >= m_lo) & (M_e < m_hi)
        for ch in ['CB', 'PL']:
            combined = (labels == ch) & m_mask
            if combined.sum() > 0:
                f_dep = is_departed[combined].mean()
                print(f"  {m_lo:6.1f} - {m_hi:6.1f}   {ch}  "
                      f"{combined.sum():6d}  {f_dep:7.3f}")

    print()
    for ch in ['CB', 'PL', 'ALL']:
        mask = np.ones(len(z_abs), dtype=bool) if ch == 'ALL' else (labels == ch)
        if mask.any():
            print(f"  Overall {ch}: f_disk_dep = {is_departed[mask].mean():.3f}")

    return is_departed


# ══════════════════════════════════════════════
# MASS-BIN HELPERS  (used by the mass-resolved report / plots)
# ══════════════════════════════════════════════

def _safe_filename_tag(label):
    """Sanitize a run label (e.g. 'coupled_f(v|M)') for use in a filename.

    Any run of characters other than letters, digits, '.', '_' and '-'
    (spaces, parentheses, '|', slashes, ...) is collapsed to a single '_'
    and leading/trailing '_' are stripped, so e.g.
        'coupled_f(v|M)' -> 'coupled_f_v_M'
        'null_f(v)'      -> 'null_f_v'
    Falls back to 'run' if nothing is left."""
    tag = re.sub(r'[^A-Za-z0-9._-]+', '_', str(label)).strip('_')
    return tag or "run"


def _mass_bin_masks(masses_mearth, edges=None):
    """Split particles into mass bins (masses given in Earth masses).

    Returns (masks, bin_labels): one boolean mask and one label string
    per bin, defined by `edges` (default MASS_BIN_EDGES_EARTH). Every
    particle lands in exactly one bin (np.digitize, clipped), so the bins
    partition the population: masses below edges[0] fall in the first bin
    and masses >= edges[-1] fall in the last, which also keeps a particle
    sitting exactly on the 317 M_earth upper limit (or off it by
    floating-point round-off after the M / M_earth division) from being
    dropped."""
    if edges is None:
        edges = MASS_BIN_EDGES_EARTH
    edges = np.asarray(edges, dtype=float)
    n_bins = len(edges) - 1
    idx = np.clip(np.digitize(masses_mearth, edges) - 1, 0, n_bins - 1)
    masks = [idx == i for i in range(n_bins)]
    bin_labels = [f"{edges[i]:g}-{edges[i + 1]:g} Me" for i in range(n_bins)]
    return masks, bin_labels


def plot_mass_resolved_spatial(pos_k, pos_c, masses, time_label, run_label=""):
    """Save mass-binned spatial KDE contours in (R,z).

    Each output file contains five mass-bin rows and two columns: the kicked
    population on the left and the matched no-kick control on the right.
    This directly visualizes P(R,z|M,t) and the corresponding control
    distribution used in the paper's stellar-tracing comparison.
    """
    if pos_c is None:
        return

    masses_mearth = masses / M_earth
    masks, bin_labels = _mass_bin_masks(masses_mearth)

    Rk = np.sqrt(pos_k[:, 0]**2 + pos_k[:, 1]**2)
    zk = pos_k[:, 2]
    Rc = np.sqrt(pos_c[:, 0]**2 + pos_c[:, 1]**2)
    zc = pos_c[:, 2]

    z_all = np.concatenate([zk, zc])
    z_max = max(2.0, np.percentile(np.abs(z_all), 95))
    xrange = (0, 15)
    yrange = (-z_max, z_max)

    n_bins = len(masks)
    fig, axes = plt.subplots(n_bins, 2, figsize=(10, 2.6 * n_bins), sharex=True, sharey=True)
    if n_bins == 1:
        axes = np.array([axes])

    fig.suptitle(f"Mass-resolved spatial density (R, z) - {time_label} [{run_label}]", fontsize=12)

    for i, (mask, blabel) in enumerate(zip(masks, bin_labels)):
        entries = [
            (axes[i, 0], Rk[mask], zk[mask], 'Oranges', f"kicked: {blabel}"),
            (axes[i, 1], Rc[mask], zc[mask], 'Greys', f"control: {blabel}"),
        ]
        for ax, xvals, yvals, cmap, title in entries:
            cf = _kde_contour(ax, xvals, yvals, xrange=xrange, yrange=yrange, cmap=cmap,
                              max_points=MASS_RES_PLOT_MAX_POINTS)
            if cf is not None:
                plt.colorbar(cf, ax=ax, label=r"$\log_{10}$(density)", pad=0.01)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("R [kpc]")
            ax.set_ylabel("z [kpc]")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    model_tag = _safe_filename_tag(run_label)
    safe_time = time_label.replace(' ', '_')
    fname = f"MassResolved_Rz_{safe_time}_{model_tag}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    saved {fname}")


# ══════════════════════════════════════════════
# LEAPFROG INTEGRATOR  (kick-drift-kick, dt = 1e4 yr, 10000 steps)
# ══════════════════════════════════════════════

def leapfrog_step(positions, velocities, dt_years, accels):
    dt = dt_years * DT_FAC
    v_half = velocities + 0.5 * dt * accels
    pos_new = positions + dt * v_half
    acc_new = compute_accelerations(pos_new)
    vel_new = v_half + 0.5 * dt * acc_new
    return pos_new, vel_new, acc_new


# ══════════════════════════════════════════════
# CYLINDRICAL COORDINATES  ( v_R, v_phi)
# ══════════════════════════════════════════════

def to_cylindrical(positions, velocities):
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    vx, vy, vz = velocities[:, 0], velocities[:, 1], velocities[:, 2]
    R = np.sqrt(x**2 + y**2)
    R_safe = np.maximum(R, R_EPS)
    vR = (x * vx + y * vy) / R_safe
    vphi = (x * vy - y * vx) / R_safe
    return R, z, vR, vphi, vz


# ══════════════════════════════════════════════
# SNAPSHOT STATISTICS  (Tables: tab:snap_initial, tab:snap_mid, tab:snap_final)
# sigma_R, sigma_z = std of the velocity components per channel.
# sigma_phi is computed on vphi_kick = vphi - v_circ(x,y): the *residual*
# after subtracting each particle's own local circular (rotation-curve)
# velocity, so it reflects the dispersion of the ejection kicks (plus
# genuine orbital heating) rather than being dominated by ordered
# Galactic rotation (raw sigma_phi ~ 120 km/s from rotation alone would
# otherwise swamp a ~few km/s kick).

# sigma_R/sigma_z ratio is reported in place of the spherical Jeans beta
# anisotropy parameter, which isn't meaningful for a rotating-disk
# geometry (it produces unphysical values like beta ~ -500 here since
# sigma_phi, uncorrected, is dominated by rotation, not velocity
# anisotropy).
# z_std = std(z); z_d_est = <|z|>, which for a Laplace(0, z_d) profile is
# the unbiased estimator of the disk scale height z_d itself
# (std(z) = sqrt(2)*z_d for a Laplace distribution, a different quantity).
# ══════════════════════════════════════════════

def print_snapshot_statistics(positions, velocities, labels, tag=""):
    x, y = positions[:, 0], positions[:, 1]
    R, z, vR, vphi, vz = to_cylindrical(positions, velocities)
    vphi_kick = vphi - circular_velocity_xy(x, y)

    print(f"\n{'='*74}")
    print(f"Snapshot statistics: {tag}")
    print(f"{'='*74}")
    print(f"{'Ch':5s}  {'N':>6s}  "
          f"{'sig_R':>7s}  {'sig_phi':>7s}  {'sig_z':>7s}  "
          f"{'z_std':>7s}  {'z_d_est':>7s}  {'sR/sz':>8s}")
    print("-" * 74)

    for ch in ['CB', 'PL', 'ALL']:
        mask = np.ones(len(R), dtype=bool) if ch == 'ALL' else (labels == ch)
        if not mask.any():
            continue
        n_in_ch = mask.sum()

        sig_R = np.std(vR[mask])
        sig_phi = np.std(vphi[mask])
        sig_z = np.std(vz[mask])
        z_std = np.std(z[mask])
        z_d_est = np.mean(np.abs(z[mask]))

        ratio_str = f"{sig_R / (sig_z + 1e-30):8.2f}"

        print(f"{ch:5s}  {n_in_ch:>6d}  "
              f"{sig_R:7.2f}  {sig_phi:7.2f}  {sig_z:7.2f}  "
              f"{z_std:7.3f}  {z_d_est:7.3f}  {ratio_str:>8s}")


# ══════════════════════════════════════════════
# PHASE SPACE CONTOUR PLOTS  (four diagnostic projections)
#   (R, z)      spatial density slice
#   (R, v_R)    radial velocity slice
#   (R, |v_z|)  vertical heating slice  
#   (v_R, v_z)  velocity ellipsoid projection
# ══════════════════════════════════════════════

PLOT_MAX_POINTS = 500000

def _kde_contour(ax, x, y, xrange, yrange, cmap, n_grid=120, n_levels=12, rng=None,
                 max_points=None):
    # max_points: optional per-call override of the random-subsample cap
    # (defaults to PLOT_MAX_POINTS, i.e. unchanged behaviour for callers
    # that don't pass it).
    cap = PLOT_MAX_POINTS if max_points is None else max_points
    mask = ((x >= xrange[0]) & (x <= xrange[1]) &
            (y >= yrange[0]) & (y <= yrange[1]))
    xc, yc = x[mask], y[mask]
    if len(xc) < 20:
        ax.text(0.5, 0.5, 'insufficient data',
                transform=ax.transAxes, ha='center', va='center', fontsize=8)
        return None
    if np.std(xc) < 1e-8 or np.std(yc) < 1e-8:
        ax.text(0.5, 0.5, 'no dispersion',
                transform=ax.transAxes, ha='center', va='center', fontsize=8)
        return None
    if len(xc) > cap:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(len(xc), size=cap, replace=False)
        xc, yc = xc[idx], yc[idx]
    kde = gaussian_kde(np.vstack([xc, yc]))
    xi = np.linspace(xrange[0], xrange[1], n_grid)
    yi = np.linspace(yrange[0], yrange[1], n_grid)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    log_Zi = np.log10(np.maximum(Zi, Zi.max() * 1e-5))
    levels = np.linspace(log_Zi.min(), log_Zi.max(), n_levels)
    cf = ax.contourf(Xi, Yi, log_Zi, levels=levels, cmap=cmap)
    ax.contour(Xi, Yi, log_Zi, levels=levels[::3],
               colors='white', linewidths=0.4, alpha=0.4)
    return cf


def plot_snapshot(positions, velocities, labels, time_label, run_label=""):
    R, z, vR, vphi, vz = to_cylindrical(positions, velocities)
    coords = {"R": R, "z": z, "vR": vR, "vz": vz, "abs_vz": np.abs(vz)}

    z_max = max(2.0, np.percentile(np.abs(z), 95))
    vR_max = max(25.0, np.percentile(np.abs(vR), 95))
    vz_max = max(25.0, np.percentile(np.abs(vz), 95))

    plot_defs = [
        dict(name="Spatial_Rz", xkey="R", ykey="z",
             xrange=(0, 15), yrange=(-z_max, z_max),
             xlabel="R [kpc]", ylabel="z [kpc]",
             title="Spatial Density (R, z)"),
        dict(name="Velocity_RvR", xkey="R", ykey="vR",
             xrange=(0, 15), yrange=(-vR_max, vR_max),
             xlabel="R [kpc]", ylabel=r"$v_R$ [km/s]",
             title=r"Velocity Slice (R, $v_R$)"),
        dict(name="Velocity_Rabsvz", xkey="R", ykey="abs_vz",
             xrange=(0, 15), yrange=(0, vz_max),
             xlabel="R [kpc]", ylabel=r"$|v_z|$ [km/s]",
             title=r"Vertical Heating Slice (R, $|v_z|$)"),
        dict(name="Ellipsoid_vRvz", xkey="vR", ykey="vz",
             xrange=(-vR_max, vR_max), yrange=(-vz_max, vz_max),
             xlabel=r"$v_R$ [km/s]", ylabel=r"$v_z$ [km/s]",
             title=r"Velocity Ellipsoid ($v_R$, $v_z$)"),
    ]

    for pdef in plot_defs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        fig.suptitle(f"{pdef['title']}  -  {time_label}  [{run_label}]", fontsize=12)
        for ax, (mask, cmap, ch_label) in zip(
                axes,
                [((labels == 'PL'), 'Blues', 'PL channel'),
                 ((labels == 'CB'), 'Oranges', 'CB channel')]):
            cf = _kde_contour(ax, coords[pdef['xkey']][mask],
                               coords[pdef['ykey']][mask],
                               xrange=pdef['xrange'], yrange=pdef['yrange'],
                               cmap=cmap)
            if cf is not None:
                plt.colorbar(cf, ax=ax, label=r"$\log_{10}$(density)", pad=0.02)
            ax.set_xlabel(pdef['xlabel'])
            ax.set_ylabel(pdef['ylabel'])
            ax.set_title(ch_label)
        plt.tight_layout()
        safe_label = run_label.replace(' ', '_')
        fname = f"{pdef['name']}_{time_label.replace(' ', '_')}_{safe_label}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    saved {fname}")


def mass_resolved_kick_report(masses, speeds, labels, tag=""):
    """Report the injected kick speed as a function of FFP mass."""
    masses_mearth = masses / M_earth
    masks, bin_labels = _mass_bin_masks(masses_mearth)
    print(f"\n{'='*100}")
    print(f"Injected kick speed versus FFP mass [{tag}]")
    print(f"{'='*100}")
    print(f"{'Population':<8s} {'Mass bin':<18s} {'N':>8s} {'median v [km/s]':>18s} {'mean v [km/s]':>16s} {'90% [km/s]':>14s}")
    print("-" * 100)
    for ch in ['CB', 'PL', 'ALL']:
        ch_mask = np.ones(len(labels), dtype=bool) if ch == 'ALL' else (labels == ch)
        for bmask, blabel in zip(masks, bin_labels):
            mask = ch_mask & bmask
            n_bin = int(mask.sum())
            if n_bin == 0:
                continue
            vals = speeds[mask]
            print(f"{ch:<8s} {blabel:<18s} {n_bin:8d} {np.median(vals):18.3f} {np.mean(vals):16.3f} {np.percentile(vals,90):14.3f}")



# ══════════════════════════════════════════════
# MAIN SIMULATION RUNNER  (1e9 yr, 10000 steps, dt = 1e4 yr)
# ══════════════════════════════════════════════

def run_simulation(n_particles=1000000, n_steps=10000, total_time_yr=1e9):
    import time

    run_label = "coupled_f(v|M)" if MASS_COUPLED else "null_f(v)"
    dt = total_time_yr / n_steps
    rng = np.random.default_rng(seed=42)

    print("=== FFP Simulation ===")
    print(f"  Model: {run_label}")
    print(f"  n={n_particles}  steps={n_steps}  dt={dt:.2e} yr  total={total_time_yr:.1e} yr")
    if MASS_COUPLED:
        print(f"  Kick coupling: CB sigma_eff ~ M^-0.15,  PL v_base ~ M^-0.5")
    else:
        print(f"  Kick coupling: NONE - factorized f(v), mass-independent kicks")
    print(f"  F_CB={F_CB:.0%}  |  Z_ESCAPE={Z_ESCAPE_KPC} kpc")
    print(f"  No-kick control population: {'ON' if RUN_NOKICK_CONTROL else 'OFF'}")

    t0 = time.time()
    positions, velocities, masses, labels, speeds = initialize_ffps(n_particles, f_cb=F_CB, rng=rng)
    accels = compute_accelerations(positions)

    # Matched no-kick control: same initial positions (and, by index, the
    # same masses / labels), velocity = local circular velocity only.
    if RUN_NOKICK_CONTROL:
        pos_c, vel_c = initialize_control_population(positions)
        acc_c = compute_accelerations(pos_c)
    else:
        pos_c = vel_c = acc_c = None
    print(f"  Initialization: {time.time()-t0:.1f}s")

    check_escape_velocity(positions, speeds, labels)

    # Injected kick speeds are fixed at t = 0, so this report is made once
    # per run rather than once per snapshot.
    mass_resolved_kick_report(masses, speeds, labels, tag=run_label)

    snapshots = {'initial': (positions.copy(), velocities.copy(), labels.copy())}
    # Control positions per snapshot (only positions are needed for the
    # mass-resolved spatial comparison); empty when the control is off.
    control_snapshots = {'initial': pos_c.copy()} if RUN_NOKICK_CONTROL else {}
    midway = n_steps // 2

    print(f"\nRunning leapfrog ({n_steps} steps)...")
    t0 = time.time()
    for step in range(1, n_steps + 1):
        positions, velocities, accels = leapfrog_step(positions, velocities, dt, accels)
        if RUN_NOKICK_CONTROL:
            pos_c, vel_c, acc_c = leapfrog_step(pos_c, vel_c, dt, acc_c)
        if step == midway:
            snapshots['midway'] = (positions.copy(), velocities.copy(), labels.copy())
            if RUN_NOKICK_CONTROL:
                control_snapshots['midway'] = pos_c.copy()
        elif step == n_steps:
            snapshots['final'] = (positions.copy(), velocities.copy(), labels.copy())
            if RUN_NOKICK_CONTROL:
                control_snapshots['final'] = pos_c.copy()
    print(f"  done in {time.time()-t0:.0f}s")

    print("\nGenerating statistics, escape fractions, and plots...")
    for time_label, (pos, vel, lbl) in snapshots.items():
        tag = f"{time_label} [{run_label}]"
        print_snapshot_statistics(pos, vel, lbl, tag=tag)
        compute_disk_departure_fraction(pos, lbl, masses, tag=tag)
        compute_galactic_escape_fraction(pos, vel, lbl, tag=tag)
        plot_snapshot(pos, vel, lbl, time_label, run_label=run_label)
        # Kicked vs no-kick control, mass-binned (returns immediately if
        # the control is disabled, since .get() then yields None).
        plot_mass_resolved_spatial(pos, control_snapshots.get(time_label), masses,
                                   time_label, run_label=run_label)

    print(f"\nDone. [{run_label}]")


if __name__ == "__main__":
    # (null model) and (coupled model) are the only two
    # kick-model configurations
    for coupled in (False, True):
        MASS_COUPLED = coupled
        run_simulation(n_particles=1000000, n_steps=10000, total_time_yr=1e9)
