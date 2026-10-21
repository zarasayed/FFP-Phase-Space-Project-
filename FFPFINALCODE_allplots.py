"""
Free-Floating Planet (FFP) kick-velocity / Galactic-dynamics simulation.
Current implementation gives each FFP one birth kick and then integrates it for the requested time.
Continuous formation is a separate source-process model and is not represented by repeatedly kicking the same particle.

"""

import argparse
import time

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
#
# BUGFIX (post-referee read): when MASS_COUPLED = True, v_base(M) shrinks
# with mass (v_base ~ M^-0.5) but PL_V_KICK_SIGMA was previously left at a
# FIXED 1.0 km/s regardless of mass. speeds[] is drawn as
# np.abs(rng.normal(v_base, sigma)), and once sigma is comparable to or
# larger than v_base, a large share of the underlying (pre-abs) draws are
# negative and get folded back to positive. That folding systematically
# biases the mean/median speed *above* v_base -- it does not average out.
# This was verified to reproduce, almost exactly, the 40-130% deviation
# from the M^-1/2 scaling seen in the two highest mass bins (30-100 Me and
# 100-317 Me), where v_base drops to ~0.5-0.2 km/s while sigma stayed at
# 1.0 km/s. Confirmed root cause, not merely a plausible guess.
#
# Fix: in the coupled model, sigma is now RELATIVE to v_base(M) rather than
# a fixed absolute km/s value, so the "narrow Gaussian standing in for a
# delta function" stays narrow (relative to the kick speed) at every mass,
# not just at M ~ 1 Earth mass. The old absolute PL_V_KICK_SIGMA is kept
# and still used, unmodified, in the null (MASS_COUPLED = False) model,
# where v_base is constant and this was never an issue.
PL_V_KICK_SIGMA_REL = 0.25   # fractional width of v_base; coupled model only

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

# ---------------------------------------------------------------------------
# INITIAL-DISK KINEMATICS
#
# The old initializer used v_R=v_z=0 for every particle.  That is a cold,
# non-equilibrium disk, especially because particles are sampled at z != 0.
# The control population then undergoes a real dynamical transient before
# the FFP kicks can be interpreted.
#
# The default below uses a Jeans-like warm-disk initializer.  It is not an
# exact DF for the triaxial potential, but it removes the artificial "cold
# disk collapse" and gives the control population a physically motivated
# baseline.  For publication-grade work, an action-based DF/orbit-library
# initializer in the full potential is preferable.
# ---------------------------------------------------------------------------
EQUILIBRIUM_INIT = True

# Fiducial disk dispersions at R=R_SIGMA_REF.
R_SIGMA_REF      = 8.2       # kpc
SIGMA_R_REF      = 35.0      # km/s
SIGMA_Z_REF      = 20.0      # km/s
SIGMA_R_SCALE    = 2.0 * R_d
SIGMA_Z_SCALE    = 2.0 * R_d

# Approximate epicyclic relation used only for the azimuthal random motion.
SIGMA_PHI_RATIO  = 0.70      # sigma_phi / sigma_R

# Optional short diagnostic integration before production.  Keep OFF by
# default: a pure "relaxation run" is not an equilibrium generator.
CHECK_CONTROL_TRANSIENT = True

# Disk escape threshold — Sec. 5.4.3 / Table (tab:quantities): Disk escape = |z| > z_esc
Z_ESCAPE_KPC = 2.0     # kpc

# Mass bins in Earth masses, shared by every mass-resolved diagnostic
# (disk-departure table, kick-speed table, mass-resolved (R,z) plots) so the
# same bins are reported everywhere.
MASS_BIN_EDGES = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 317.0])

# Assumed elapsed times after the birth/ejection kick at which to evaluate
# the mass-resolved spatial displacement. These are NOT measured ejection
# times; they are fiducial post-ejection timescales used to connect the
# kick model to Galactic propagation.
DISTANCE_TIMES_MYR = np.array([1.0, 10.0, 30.0, 100.0])


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


def circular_velocity_xy(x, y, z=0.0, h=GRAD_H):
    """Vectorized, ANGLE-AWARE local circular velocity: for each particle,
    evaluates the cylindrical-radial potential gradient along the true
    direction from the Galactic center to that particle's (x, y), at the
    particle's own height z (z = 0 gives the midplane rotation curve).
    Generalized to be well-defined at every azimuthal angle rather than
    only along y = 0. x, y, z may be scalars or arrays.

    Using the particle's actual z matters: the radial force weakens with
    height, so a midplane v_circ applied to a particle at z != 0 puts it on
    a non-circular orbit and injects spurious radial energy that swamps the
    few-km/s ejection kicks this simulation is meant to measure."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    R = np.sqrt(x**2 + y**2)
    R_safe = np.maximum(R, R_EPS)
    z = np.broadcast_to(np.asarray(z, dtype=float), R_safe.shape)
    ux, uy = x / R_safe, y / R_safe
    xp, yp = x + h * ux, y + h * uy
    xm, ym = x - h * ux, y - h * uy
    dPhidR = (total_potential_v(xp, yp, z) - total_potential_v(xm, ym, z)) / (2 * h)
    return np.sqrt(np.maximum(R_safe * dPhidR, 0.0))


# circular_velocity_fast(R) / the R-only interpolation table have been
# removed: circular_velocity_xy() is cheap enough to call directly for
# ~1e6 particles since it's pure numpy array math, not a Python loop.


# ══════════════════════════════════════════════
# SHARED HELPERS  (mass binning / filename tags)
# ══════════════════════════════════════════════

def _mass_bin_masks(masses_mearth, edges=MASS_BIN_EDGES):
    """Boolean masks + printable labels for the shared MASS_BIN_EDGES bins.

    The last bin is closed on the right so the heaviest planets
    (M = M_max exactly) are not silently dropped.
    """
    masses_mearth = np.asarray(masses_mearth, dtype=float)
    masks, labels = [], []
    n_bins = len(edges) - 1
    for i in range(n_bins):
        m_lo, m_hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (masses_mearth >= m_lo) & (masses_mearth <= m_hi)
        else:
            mask = (masses_mearth >= m_lo) & (masses_mearth < m_hi)
        masks.append(mask)
        labels.append(f"{m_lo:g}-{m_hi:g} Me")
    return masks, labels


def _safe_filename_tag(label):
    """Turn a run/time label such as 'coupled_f(v|M)' into something safe to
    embed in a filename ('coupled_f_v_M')."""
    cleaned = ''.join(c if (c.isalnum() or c in '-_') else '_' for c in str(label))
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    return cleaned.strip('_') or 'run'


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
    M = np.linspace(CB_M_MIN_EARTH, CB_M_MAX_EARTH, 500000)
    f = M**(-CB_ALPHA) / (1.0 + (M / CB_M_TURNOVER)**CB_GAMMA_SUPPRESS)
    return _inverse_cdf_sample(f, M, n, rng)


def sample_pl_masses(n, rng):
    """dN/dM ~ M^-alpha, constant-kick model mass spectrum """
    M = np.linspace(PL_M_MIN_EARTH, PL_M_MAX_EARTH, 500000)
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

def _jeans_disk_kinematics(R, v_circ, rng):
    """Return a Jeans-like warm-disk velocity realization.

    Assumptions:
      rho(R) ~ exp(-R/R_d)
      sigma_R^2(R) ~ exp(-R/SIGMA_R_SCALE)
      sigma_z^2(R) ~ exp(-R/SIGMA_Z_SCALE)

    The radial Jeans equation then gives the mean streaming speed
    approximately as

        v_c^2 - <v_phi>^2 =
            sigma_R^2 [R/R_d + R/SIGMA_R_SCALE - 1].

    This is an axisymmetric/epicyclic approximation.  It is deliberately
    preferable to the old cold initialization because it does not start
    the control population with zero random motion.
    """
    R_safe = np.maximum(R, R_EPS)

    sigma_R = SIGMA_R_REF * np.exp(
        -(R_safe - R_SIGMA_REF) / SIGMA_R_SCALE
    )
    sigma_z = SIGMA_Z_REF * np.exp(
        -(R_safe - R_SIGMA_REF) / SIGMA_Z_SCALE
    )

    # Prevent the outer exponential tail from producing unreasonably large
    # dispersions while retaining a smooth radial profile.
    sigma_R = np.clip(sigma_R, 5.0, 80.0)
    sigma_z = np.clip(sigma_z, 5.0, 50.0)

    asymmetric_drift = sigma_R**2 * (
        R_safe / R_d + R_safe / SIGMA_R_SCALE - 1.0
    )
    vphi_mean_sq = np.maximum(v_circ**2 - asymmetric_drift, 0.0)
    vphi_mean = np.sqrt(vphi_mean_sq)

    v_R = rng.normal(0.0, sigma_R)
    v_z = rng.normal(0.0, sigma_z)
    v_phi = vphi_mean + rng.normal(
        0.0, SIGMA_PHI_RATIO * sigma_R
    )

    return v_R, v_phi, v_z, sigma_R, sigma_z


def initialize_ffps(n, f_cb=F_CB, rng=None):
    """
    Initialize FFPs and a matched no-kick control.

    Positions follow the same exponential-disk radial and Laplace vertical
    distributions as the original model.  The important change is that the
    stellar/FFP birth velocities are no longer a cold v_R=v_z=0 population.
    A Jeans-like warm disk is used instead, so the control population starts
    close to dynamical equilibrium rather than undergoing an artificial
    cold-disk relaxation.

    The ejection kick is still applied exactly once at birth.  If the intended
    physical model is continuous FFP formation over 1e8 yr, that requires a
    cohort/source-process implementation rather than repeatedly kicking each
    existing FFP.
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

    # Reference circular speed.  For the full triaxial potential there is no
    # exact circular orbit at arbitrary azimuth.  We therefore use the old
    # local radial-force speed as the streaming reference and add warm-disk
    # random motions.  The control run below measures any residual bar/shape
    # driven evolution that remains.
    v_circ = circular_velocity_xy(x, y, z)

    if EQUILIBRIUM_INIT:
        v_R, v_phi, v_z, _, _ = _jeans_disk_kinematics(R, v_circ, rng)
    else:
        # Legacy cold initializer, retained as an explicit null option.
        v_R = np.zeros(n)
        v_phi = v_circ
        v_z = np.zeros(n)

    v_star = np.column_stack([
        v_R * np.cos(phi) - v_phi * np.sin(phi),
        v_R * np.sin(phi) + v_phi * np.cos(phi),
        v_z
    ])

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
            # Relative width (see PL_V_KICK_SIGMA_REL note above): keeps the
            # "narrow Gaussian" narrow relative to v_base at every mass,
            # instead of a fixed absolute km/s value that becomes larger
            # than v_base itself at high mass and biases the folded (abs)
            # speed upward.
            pl_sigma = PL_V_KICK_SIGMA_REL * v_base
        else:
            v_base = PL_V_KICK
            pl_sigma = PL_V_KICK_SIGMA
        speeds[n_cb:] = np.abs(rng.normal(v_base, pl_sigma, n_pl))

    # Isotropic kick direction -> total velocity
    kicks = speeds[:, None] * sample_isotropic_directions(n, rng)
    velocities = v_star + kicks

    labels = np.array(['CB'] * n_cb + ['PL'] * n_pl)

    # Matched no-kick control: exactly the same positions, masses, and
    # pre-kick stellar velocities.
    velocities_control = v_star.copy()

    return positions, velocities, velocities_control, masses, labels, speeds


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
    masks, bin_labels = _mass_bin_masks(M_e)

    print(f"\n  Disk departure fractions |z| > {Z_ESCAPE_KPC} kpc ({tag}):")
    print(f"  {'Mass bin [Me]':>16s}  {'Ch':>3s}  {'N':>8s}  {'f_dep':>7s}")
    print(f"  {'-'*44}")

    for m_mask, blabel in zip(masks, bin_labels):
        for ch in ['CB', 'PL']:
            combined = (labels == ch) & m_mask
            if combined.sum() > 0:
                f_dep = is_departed[combined].mean()
                print(f"  {blabel:>16s}  {ch:>3s}  "
                      f"{combined.sum():8d}  {f_dep:7.3f}")

    print()
    for ch in ['CB', 'PL', 'ALL']:
        mask = np.ones(len(z_abs), dtype=bool) if ch == 'ALL' else (labels == ch)
        if mask.any():
            print(f"  Overall {ch}: f_disk_dep = {is_departed[mask].mean():.3f}")

    return is_departed

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
            cf = _kde_contour(ax, xvals, yvals, xrange=xrange, yrange=yrange, cmap=cmap)
            if cf is not None:
                plt.colorbar(cf, ax=ax, label=r"$\log_{10}$(density)", pad=0.01)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("R [kpc]")
            ax.set_ylabel("z [kpc]")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fname = (f"MassResolved_Rz_{_safe_filename_tag(time_label)}"
             f"_{_safe_filename_tag(run_label)}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    saved {fname}")    

# ══════════════════════════════════════════════
# LEAPFROG INTEGRATOR  (kick-drift-kick, dt = 1e4 yr, 10000 steps)
# ══════════════════════════════════════════════

def leapfrog_step(positions, velocities, dt_years, accels):
    dt_kpc = dt_years * DT_FAC
    v_half = velocities + 0.5 * dt_kpc * accels
    pos_new = positions + dt_kpc * v_half
    acc_new = compute_accelerations(pos_new)
    vel_new = v_half + 0.5 * dt_kpc * acc_new
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
    vphi_kick = vphi - circular_velocity_xy(x, y, z)

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
        # residual azimuthal dispersion (rotation subtracted), as documented above
        sig_phi = np.std(vphi_kick[mask])
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

PLOT_MAX_POINTS = 50000

def _kde_contour(ax, x, y, xrange, yrange, cmap, n_grid=120, n_levels=12, rng=None):
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
    if len(xc) > PLOT_MAX_POINTS:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(len(xc), size=PLOT_MAX_POINTS, replace=False)
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
        fname = (f"{pdef['name']}_{_safe_filename_tag(time_label)}"
                 f"_{_safe_filename_tag(run_label)}.png")
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





def _radial_distance_kpc(positions):
    """Galactocentric spherical radius r = sqrt(x^2+y^2+z^2), in kpc."""
    return np.sqrt(np.sum(positions**2, axis=1))


def distance_displacement_report(pos0, positions_by_time, masses, labels, speeds,
                                 tag="", times_myr=DISTANCE_TIMES_MYR,
                                 positions_control_by_time=None):
    """Report mass-resolved propagation after the birth/ejection kick.

    If positions_control_by_time is given, the primary metric is the
    separation between the kicked particle and its matched no-kick
    control counterpart AT THE SAME TIME -- this isolates the kick's
    effect from the bulk galactic-orbital motion both share.

    If no control is available, falls back to chord displacement from the
    birth position |r(t)-r(0)|. Be aware that this fallback metric is
    NOT monotonic for bound orbits: it rises and falls with orbital phase
    and does not by itself indicate how far the kick has carried the FFP.
    """
    masses_mearth = masses / M_earth
    masks, bin_labels = _mass_bin_masks(masses_mearth)
    r0 = _radial_distance_kpc(pos0)
    use_control = positions_control_by_time is not None

    print(f"\n{'='*128}")
    print(f"Mass-resolved propagation after the birth kick [{tag}]")
    if use_control:
        print("Metric: |r_kicked(t) - r_control(t)| relative to a matched no-kick "
              "twin -- isolates the kick, removing shared orbital motion.")
    else:
        print("Metric: |r(t)-r(0)| relative to birth position -- NOTE: not "
              "monotonic for bound orbits (see docstring); no control was run.")
    print(f"{'='*128}")
    header = (
        f"{'Mass bin':<18s} {'N':>8s} {'t [Myr]':>9s} "
        f"{'med kick [km/s]':>17s} {'med sep [kpc]':>17s} "
        f"{'90% sep [kpc]':>17s} {'med r_GC [kpc]':>18s}"
    )
    print(header)
    print("-" * len(header))

    for bmask, blabel in zip(masks, bin_labels):
        n_bin = int(bmask.sum())
        if n_bin == 0:
            continue
        med_kick = float(np.median(speeds[bmask]))
        for i, t_myr in enumerate(times_myr):
            pos_t = positions_by_time[i]
            if use_control:
                sep = np.linalg.norm(pos_t - positions_control_by_time[i], axis=1)
            else:
                sep = np.linalg.norm(pos_t - pos0, axis=1)
            r_t = _radial_distance_kpc(pos_t)
            print(
                f"{blabel:<18s} {n_bin:8d} {t_myr:9.1f} "
                f"{med_kick:17.3f} {np.median(sep[bmask]):17.4f} "
                f"{np.percentile(sep[bmask], 90):17.4f} "
                f"{np.median(r_t[bmask]):18.3f}"
            )

    for i, t_myr in enumerate(times_myr):
        pos_t = positions_by_time[i]
        sep = (np.linalg.norm(pos_t - positions_control_by_time[i], axis=1)
               if use_control else np.linalg.norm(pos_t - pos0, axis=1))
        print(
            f"  Overall t={t_myr:.1f} Myr: median sep={np.median(sep):.4f} kpc, "
            f"90%={np.percentile(sep,90):.4f} kpc, "
            f"median r_GC={np.median(_radial_distance_kpc(pos_t)):.3f} kpc"
        )


def plot_mass_resolved_displacement(pos0, positions_by_time, masses, labels,
                                    speeds, tag="", times_myr=DISTANCE_TIMES_MYR):
    """Plot median net displacement versus elapsed post-kick time by mass bin."""
    masses_mearth = masses / M_earth
    masks, bin_labels = _mass_bin_masks(masses_mearth)

    fig, ax = plt.subplots(figsize=(8, 5))
    for bmask, blabel in zip(masks, bin_labels):
        if not bmask.any():
            continue
        medians = []
        p90s = []
        for pos_t in positions_by_time:
            disp = np.linalg.norm(pos_t - pos0, axis=1)[bmask]
            medians.append(np.median(disp))
            p90s.append(np.percentile(disp, 90))
        ax.plot(times_myr, medians, marker="o", label=blabel)
        ax.plot(times_myr, p90s, marker="x", linestyle="--", alpha=0.7)

    ax.set_xlabel("Time after ejection [Myr]")
    ax.set_ylabel(r"Net displacement $|\mathbf{r}(t)-\mathbf{r}(0)|$ [kpc]")
    ax.set_title(f"Mass-resolved FFP propagation [{tag}]")
    ax.legend(title="Solid: median; dashed: 90th percentile", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fname = f"Displacement_vs_time_{_safe_filename_tag(tag)}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    saved {fname}")


def compare_kicked_to_control(pos_k, vel_k, pos_c, vel_c, labels, tag=""):
    """Report the differential heating relative to the matched control.

    Raw final sigma is not the clean observable when the background disk
    itself evolves.  This reports sigma_kicked, sigma_control, and the
    variance excess sqrt(max(sigma_k^2 - sigma_c^2, 0)).
    """
    if pos_c is None or vel_c is None:
        return

    _, _, vRk, _, vzk = to_cylindrical(pos_k, vel_k)
    _, _, vRc, _, vzc = to_cylindrical(pos_c, vel_c)

    print(f"\nDifferential heating relative to no-kick control [{tag}]")
    print(f"{'Ch':5s} {'sigR_k':>9s} {'sigR_c':>9s} {'excess_R':>10s} "
          f"{'sigz_k':>9s} {'sigz_c':>9s} {'excess_z':>10s}")
    print("-" * 76)

    for ch in ['CB', 'PL', 'ALL']:
        mask = np.ones(len(labels), dtype=bool) if ch == 'ALL' else (labels == ch)
        if not mask.any():
            continue

        srk, src = np.std(vRk[mask]), np.std(vRc[mask])
        szk, szc = np.std(vzk[mask]), np.std(vzc[mask])

        excess_r = np.sqrt(max(srk**2 - src**2, 0.0))
        excess_z = np.sqrt(max(szk**2 - szc**2, 0.0))

        print(f"{ch:5s} {srk:9.2f} {src:9.2f} {excess_r:10.2f} "
              f"{szk:9.2f} {szc:9.2f} {excess_z:10.2f}")

# ══════════════════════════════════════════════
# MAIN SIMULATION RUNNER  (1e8 yr, 10000 steps, dt = 1e4 yr)
# ══════════════════════════════════════════════

def set_mass_coupling(flag):
    """Switch between the null model f(v) and the coupled model f(v|M).

    The samplers read the module-level MASS_COUPLED flag, so it has to be
    rebound on the module (a bare assignment inside a function would only
    create a local name and silently leave the model unchanged).
    """
    global MASS_COUPLED
    MASS_COUPLED = bool(flag)


def run_simulation(n_particles=1000000, n_steps=10000, total_time_yr=1e8,
                   mass_coupled=None, with_control=True, seed=42,
                   distance_times_myr=DISTANCE_TIMES_MYR):
    if mass_coupled is not None:
        set_mass_coupling(mass_coupled)

    run_label = "coupled_f(v|M)" if MASS_COUPLED else "null_f(v)"
    dt = total_time_yr / n_steps
    rng = np.random.default_rng(seed=seed)

    print("=== FFP Simulation ===")
    print(f"  Model: {run_label}")
    print(f"  n={n_particles}  steps={n_steps}  dt={dt:.2e} yr  total={total_time_yr:.1e} yr")
    if MASS_COUPLED:
        print("  Kick coupling: CB sigma_eff ~ M^-0.15,  PL v_base ~ M^-0.5")
    else:
        print("  Kick coupling: NONE - factorized f(v), mass-independent kicks")
    print(f"  F_CB={F_CB:.0%}  |  Z_ESCAPE={Z_ESCAPE_KPC} kpc")

    t0 = time.time()
    (positions, velocities, velocities_control,
     masses, labels, speeds) = initialize_ffps(n_particles, f_cb=F_CB, rng=rng)
    accels = compute_accelerations(positions)
    print(f"  Initialization: {time.time()-t0:.1f}s")
    print(f"  Control population (no kick): {'ON' if with_control else 'OFF'}")
    print(f"  Initialization: {'Jeans-like warm disk' if EQUILIBRIUM_INIT else 'LEGACY COLD DISK'}")

    check_escape_velocity(positions, speeds, labels)
    mass_resolved_kick_report(masses, speeds, labels, tag=run_label)

    # Matched no-kick control: same birth positions and masses, purely
    # circular velocities, integrated in the same potential.
    if with_control:
        pos_c = positions.copy()
        vel_c = velocities_control.copy()
        acc_c = accels.copy()
    else:
        pos_c = vel_c = acc_c = None

    snapshots = {'initial': (
        positions.copy(), velocities.copy(), labels.copy(),
        None if pos_c is None else pos_c.copy(),
        None if vel_c is None else vel_c.copy()
    )}
    midway = max(1, n_steps // 2)

    # Store only the requested propagation times; this avoids retaining the
    # full 1e6-particle trajectory in memory.
    distance_times_myr = np.asarray(distance_times_myr, dtype=float)
    if np.any(distance_times_myr <= 0) or np.any(np.diff(distance_times_myr) <= 0):
        raise ValueError("distance_times_myr must be strictly increasing and positive.")
    if distance_times_myr[-1] > total_time_yr / 1e6:
        raise ValueError("Requested distance time exceeds total integration time.")
    # --- inside run_simulation(), replace the distance_snapshots block ---
    distance_steps = {max(1, int(round(t * 1e6 / dt))): float(t)
                      for t in distance_times_myr}
    distance_snapshots = {}
    distance_snapshots_control = {}   # NEW

    for step in range(1, n_steps + 1):
        positions, velocities, accels = leapfrog_step(positions, velocities, dt, accels)
        if with_control:
            pos_c, vel_c, acc_c = leapfrog_step(pos_c, vel_c, dt, acc_c)
        if step in distance_steps:
            distance_snapshots[distance_steps[step]] = positions.copy()
            if with_control:
                distance_snapshots_control[distance_steps[step]] = pos_c.copy()   # NEW

    # --- after the loop ---
    ordered_times = [float(t) for t in distance_times_myr]
    ordered_positions = [distance_snapshots[t] for t in ordered_times]
    ordered_positions_control = (
        [distance_snapshots_control[t] for t in ordered_times] if with_control else None
    )  # NEW

    distance_displacement_report(
        snapshots['initial'][0], ordered_positions, masses, labels, speeds,
        tag=run_label, times_myr=ordered_times,
        positions_control_by_time=ordered_positions_control,   # NEW
    )
    plot_mass_resolved_displacement(
        snapshots['initial'][0], ordered_positions, masses, labels, speeds,
        tag=run_label, times_myr=ordered_times,
    )

    print(f"\nDone. [{run_label}]")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="FFP kick-velocity / Galactic-dynamics simulation.")
    parser.add_argument("-n", "--n-particles", type=int, default=1000000,
                        help="particles per run (default: 1e6, as in the paper)")
    parser.add_argument("-s", "--n-steps", type=int, default=10000,
                        help="leapfrog steps (default: 10000)")
    parser.add_argument("-T", "--total-time-yr", type=float, default=1e8,
                        help="integration time in years (default: 1e8)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=("null", "coupled", "both"), default="both",
                        help="which kick model(s) to run (default: both)")
    parser.add_argument("--cold-init", action="store_true",
                        help="use the legacy cold v_R=v_z=0 initializer (diagnostic only)")
    parser.add_argument("--no-control", action="store_true",
                        help="skip the matched no-kick control population "
                             "(also skips the mass-resolved (R,z) plots)")
    parser.add_argument("--distance-times-myr", type=float, nargs="+",
                        default=DISTANCE_TIMES_MYR.tolist(),
                        help="post-kick times [Myr] for mass-resolved displacement "
                             "diagnostics (default: 1 10 30 100)")
    args = parser.parse_args(argv)

    global EQUILIBRIUM_INIT
    EQUILIBRIUM_INIT = not args.cold_init

    models = {"null": (False,), "coupled": (True,), "both": (False, True)}[args.model]
    for coupled in models:
        run_simulation(n_particles=args.n_particles,
                       n_steps=args.n_steps,
                       total_time_yr=args.total_time_yr,
                       mass_coupled=coupled,
                       with_control=not args.no_control,
                       seed=args.seed,
                       distance_times_myr=args.distance_times_myr)


if __name__ == "__main__":
    main()