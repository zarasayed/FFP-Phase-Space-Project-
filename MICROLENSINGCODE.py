import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
# constants
# ══════════════════════════════════════════════════════════════════
G            = 4.30091e-6      # kpc (km/s)^2 / Msun
M_halo       = 7.25e11         # Msun
R_d          = 3.5             # kpc  disk scale length
z_d          = 0.3             # kpc  disk scale height
M_earth      = 3.003e-6        # Msun

R_EPS        = 1e-3            # kpc
R0_HALO      = 1e-8            # kpc
GRAD_H       = 1e-2            # kpc
KM_PER_KPC   = 3.0857e16
SEC_PER_YEAR = 3.154e7
DT_FAC       = SEC_PER_YEAR / KM_PER_KPC

BULGE_RHO0   = 1e10
BULGE_A_SOFT = 0.5
BULGE_THETA  = 25.0
BULGE_X0     = 1.49
BULGE_Y0     = 0.58
BULGE_Z0     = 0.40

R_S_HALO = 10.0

CB_ALPHA          = 1.3
CB_M_MIN_EARTH    = 1.0
CB_M_MAX_EARTH    = 317.0
CB_M_TURNOVER     = 3.0
CB_GAMMA_SUPPRESS = 2.0
SIGMA_CB          = 10.0

PL_ALPHA          = 1.3
PL_M_MIN_EARTH    = 1.0
PL_M_MAX_EARTH    = 317.0
PL_V_KICK         = 4.0
PL_V_KICK_SIGMA   = 1.0

F_CB = 0.5
MASS_COUPLED = True
Z_ESCAPE_KPC = 2.0

# speed of light -- physical constant, needed to evaluate the paper's own
# R_E formula; not a piece of "outside" astrophysical data.
C_KMS = 299792.458


R0_KPC = 8.2

# Source distance: Roman's Galactic Bulge Time Domain Survey fields lie
# within a few degrees of the Galactic Center (Penny et al. 2019, ApJS,
# 241, 3), so D_S = R0 is adopted as the characteristic bulge-field
# source distance for a line of sight through the bulge.
D_S_KPC = R0_KPC

def _disk_potential_v(R, z):
    return -G * (6e10) * np.exp(-R / R_d) / np.sqrt(
        R**2 + (z_d + np.abs(z))**2 + 1e-12)


def _bulge_potential_v(x, y, z):
    ct = np.cos(np.radians(BULGE_THETA))
    st = np.sin(np.radians(BULGE_THETA))
    xp = x * ct + y * st
    yp = -x * st + y * ct
    r_s = np.sqrt((xp / BULGE_X0)**2 + (yp / BULGE_Y0)**2 + (z / BULGE_Z0)**2)
    return (-4.0 * np.pi * G * BULGE_RHO0
            * BULGE_X0 * BULGE_Y0 * BULGE_Z0
            / (r_s + BULGE_A_SOFT))


def _halo_potential_v(r):
    r_eff = np.sqrt(r * r + R0_HALO**2)
    return -G * M_halo * np.log(1.0 + r_eff / R_S_HALO) / r_eff


def total_potential_v(x, y, z):
    R = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)
    return _disk_potential_v(R, z) + _bulge_potential_v(x, y, z) + _halo_potential_v(r)


def compute_accelerations(positions):
    h = GRAD_H
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    ax = -(total_potential_v(x + h, y, z) - total_potential_v(x - h, y, z)) / (2 * h)
    ay = -(total_potential_v(x, y + h, z) - total_potential_v(x, y - h, z)) / (2 * h)
    az = -(total_potential_v(x, y, z + h) - total_potential_v(x, y, z - h)) / (2 * h)
    return np.column_stack([ax, ay, az])


def circular_velocity_xy(x, y, h=GRAD_H):
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


_trapz = getattr(np, "trapezoid", None) or np.trapz


def _inverse_cdf_sample(f_unnorm, M_grid, n, rng):
    f = f_unnorm / _trapz(f_unnorm, M_grid)
    dM = M_grid[1] - M_grid[0]
    cdf = np.cumsum(f) * dM
    cdf /= cdf[-1]
    return np.interp(rng.uniform(size=n), cdf, M_grid) * M_earth


def sample_cb_masses(n, rng):
    M = np.linspace(CB_M_MIN_EARTH, CB_M_MAX_EARTH, 500000)
    f = M**(-CB_ALPHA) / (1.0 + (M / CB_M_TURNOVER)**CB_GAMMA_SUPPRESS)
    return _inverse_cdf_sample(f, M, n, rng)


def sample_pl_masses(n, rng):
    M = np.linspace(PL_M_MIN_EARTH, PL_M_MAX_EARTH, 500000)
    f = M**(-PL_ALPHA)
    return _inverse_cdf_sample(f, M, n, rng)


def sample_isotropic_directions(n, rng):
    phi = rng.uniform(0, 2 * np.pi, n)
    costh = rng.uniform(-1, 1, n)
    sinth = np.sqrt(1.0 - costh**2)
    return np.column_stack((sinth * np.cos(phi), sinth * np.sin(phi), costh))


def initialize_ffps(n, f_cb=F_CB, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n_cb = int(round(n * f_cb))
    n_pl = n - n_cb

    R = rng.gamma(shape=2.0, scale=R_d, size=n)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = rng.laplace(0.0, z_d, n)
    positions = np.vstack([x, y, z]).T

    v_circ = circular_velocity_xy(x, y)
    v_star = np.vstack([-np.sin(phi) * v_circ,
                         np.cos(phi) * v_circ,
                         np.zeros(n)]).T

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

    kicks = speeds[:, None] * sample_isotropic_directions(n, rng)
    velocities = v_star + kicks

    labels = np.array(['CB'] * n_cb + ['PL'] * n_pl)
    return positions, velocities, masses, labels, speeds


def leapfrog_step(positions, velocities, dt_years, accels):
    dt = dt_years * DT_FAC
    v_half = velocities + 0.5 * dt * accels
    pos_new = positions + dt * v_half
    acc_new = compute_accelerations(pos_new)
    vel_new = v_half + 0.5 * dt * acc_new
    return pos_new, vel_new, acc_new


def to_cylindrical(positions, velocities):
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    vx, vy, vz = velocities[:, 0], velocities[:, 1], velocities[:, 2]
    R = np.sqrt(x**2 + y**2)
    R_safe = np.maximum(R, R_EPS)
    vR = (x * vx + y * vy) / R_safe
    vphi = (x * vy - y * vx) / R_safe
    return R, z, vR, vphi, vz


# ══════════════════════════════════════════════════════════════════
# PART 1 — run the N-body simulation and take a final snapshot
# ══════════════════════════════════════════════════════════════════

def run_nbody(n_particles=500_000, n_steps=2000, total_time_yr=1e8, seed=42):
    dt = total_time_yr / n_steps
    rng = np.random.default_rng(seed=seed)

    print("=== FFP N-body engine (Sec. 4-6 of the paper; NEWFFPCODE.py physics) ===")
    print(f"  n={n_particles}  steps={n_steps}  dt={dt:.2e} yr  total={total_time_yr:.1e} yr")
    print(f"  MASS_COUPLED={MASS_COUPLED}  F_CB={F_CB:.0%}")

    t0 = time.time()
    positions, velocities, masses, labels, speeds = initialize_ffps(n_particles, f_cb=F_CB, rng=rng)
    accels = compute_accelerations(positions)
    print(f"  init: {time.time()-t0:.1f}s")

    t0 = time.time()
    for step in range(1, n_steps + 1):
        positions, velocities, accels = leapfrog_step(positions, velocities, dt, accels)
    print(f"  leapfrog ({n_steps} steps): {time.time()-t0:.0f}s")

    speed_final = np.sqrt((velocities**2).sum(axis=1))
    print(f"  sanity: max|v|={speed_final.max():.0f} km/s, "
          f"NaNs={np.isnan(positions).any() or np.isnan(velocities).any()}")

    return positions, velocities, masses, labels


# ══════════════════════════════════════════════════════════════════
# PART 2 — calibrate the qDF (Eq. 18) shape directly against the
# relaxed N-body snapshot. Nothing here is assumed a priori; every
# number quoted is a fit to the simulated final-time particle ensemble.
# ══════════════════════════════════════════════════════════════════

def _fit_exp_profile(R_centers, sigma_vals, weights):
    """Fit sigma(R) = sigma0 * exp(-R/Rsig) via weighted log-linear regression."""
    good = (sigma_vals > 0) & np.isfinite(sigma_vals) & (weights > 0)
    if good.sum() < 3:
        s0 = np.average(sigma_vals[weights > 0], weights=weights[weights > 0])
        return s0, np.inf
    x = R_centers[good]
    y = np.log(sigma_vals[good])
    w = weights[good]
    # weighted least squares: y = a + b*x  (a=ln(sigma0), b=-1/Rsig)
    W = np.sum(w)
    xbar = np.sum(w * x) / W
    ybar = np.sum(w * y) / W
    b = np.sum(w * (x - xbar) * (y - ybar)) / np.sum(w * (x - xbar) ** 2)
    a = ybar - b * xbar
    sigma0 = np.exp(a)
    Rsig = -1.0 / b if b < 0 else np.inf   # if profile doesn't decline, treat as flat
    return sigma0, Rsig


def calibrate_qdf(positions, velocities, labels, masses, R_bins=None):
   
    x, y = positions[:, 0], positions[:, 1]
    R, z, vR, vphi, vz = to_cylindrical(positions, velocities)
    vphi_kick = vphi - circular_velocity_xy(x, y)
    M_e = masses / M_earth

    if R_bins is None:
  
        R_bins = np.linspace(2.0, 14.0, 13)
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])

    mass_bin_edges = np.array([1.0, 3.0, 10.0, 317.0])
    mass_bin_names = ["Earth-like (1-3 Me)", "Neptune-like (3-10 Me)", "Jupiter-like (10-317 Me)"]
    mass_bin_repr = np.sqrt(mass_bin_edges[:-1] * mass_bin_edges[1:])  # geometric-mean representative mass

    params = {"R_bins": R_bins, "R_centers": R_centers,
              "mass_bin_edges": mass_bin_edges, "mass_bin_names": mass_bin_names,
              "mass_bin_repr_Me": mass_bin_repr, "channels": {}}

    for ch in ["CB", "PL", "ALL"]:
        mask_ch = np.ones(len(R), dtype=bool) if ch == "ALL" else (labels == ch)

        # --- density profile nu(R,z) ---
        counts, _ = np.histogram(R[mask_ch], bins=R_bins)
        shell_area = np.pi * (R_bins[1:]**2 - R_bins[:-1]**2)   # kpc^2 (area of annulus, all z)
        Sigma_R = counts / shell_area                             # particles / kpc^2 (surface density)
        Sigma_R_err = np.sqrt(np.maximum(counts, 1)) / shell_area  # Poisson counting error
        Sigma_amp, R_d_fit = _fit_exp_profile(R_centers, Sigma_R, counts.astype(float))
        z_d_fit = np.mean(np.abs(z[mask_ch]))                     # Laplace-profile estimator (per code comment)

        # --- velocity ellipsoid vs R ---
        sig_R = np.full(len(R_centers), np.nan)
        sig_phi = np.full(len(R_centers), np.nan)
        sig_z = np.full(len(R_centers), np.nan)
        sig_R_err = np.full(len(R_centers), np.nan)
        sig_phi_err = np.full(len(R_centers), np.nan)
        sig_z_err = np.full(len(R_centers), np.nan)
        vphi_mean = np.full(len(R_centers), np.nan)
        counts_v = np.zeros(len(R_centers))
        for i in range(len(R_centers)):
            m = mask_ch & (R >= R_bins[i]) & (R < R_bins[i + 1])
            counts_v[i] = m.sum()
            if m.sum() >= 30:
                n_i = m.sum()
                sig_R[i] = np.std(vR[m])
                sig_phi[i] = np.std(vphi_kick[m])
                sig_z[i] = np.std(vz[m])
                # standard error of a sample standard deviation, sigma/sqrt(2(n-1))
                sig_R_err[i] = sig_R[i] / np.sqrt(2 * (n_i - 1))
                sig_phi_err[i] = sig_phi[i] / np.sqrt(2 * (n_i - 1))
                sig_z_err[i] = sig_z[i] / np.sqrt(2 * (n_i - 1))
                vphi_mean[i] = np.mean(vphi[m])

        sigR0, RsigR = _fit_exp_profile(R_centers, sig_R, counts_v)
        sigPhi0, RsigPhi = _fit_exp_profile(R_centers, sig_phi, counts_v)
        sigZ0, RsigZ = _fit_exp_profile(R_centers, sig_z, counts_v)

        # mean rotation curve: keep as a direct (R_centers, vphi_mean) interpolant
        good = counts_v >= 30
        vphi_R_table = (R_centers[good], vphi_mean[good])

        # --- mass-dependent dispersion scaling (aggregated over all R) ---
        mass_scale = {"repr_Me": mass_bin_repr, "sigma_R": [], "sigma_phi": [], "sigma_z": [], "n": []}
        for i in range(3):
            lo, hi = mass_bin_edges[i], mass_bin_edges[i + 1]
            m = mask_ch & (M_e >= lo) & (M_e < hi)
            mass_scale["n"].append(int(m.sum()))
            if m.sum() >= 30:
                mass_scale["sigma_R"].append(float(np.std(vR[m])))
                mass_scale["sigma_phi"].append(float(np.std(vphi_kick[m])))
                mass_scale["sigma_z"].append(float(np.std(vz[m])))
            else:
                mass_scale["sigma_R"].append(np.nan)
                mass_scale["sigma_phi"].append(np.nan)
                mass_scale["sigma_z"].append(np.nan)

        # fit power-law index sigma_eff(M) ~ M^p (log-log slope across the 3 bins)
        def _powerlaw_index(vals):
            v = np.array(vals)
            good = np.isfinite(v) & (v > 0)
            if good.sum() < 2:
                return 0.0
            lx = np.log(mass_bin_repr[good])
            ly = np.log(v[good])
            p = np.polyfit(lx, ly, 1)[0]
            return float(p)

        mass_scale["p_sigma_R"] = _powerlaw_index(mass_scale["sigma_R"])
        mass_scale["p_sigma_phi"] = _powerlaw_index(mass_scale["sigma_phi"])
        mass_scale["p_sigma_z"] = _powerlaw_index(mass_scale["sigma_z"])
        # reference mass = center of bin 0 (Earth-like), scaling is relative to this
        mass_scale["M_ref_Me"] = float(mass_bin_repr[0])

        params["channels"][ch] = dict(
            n=int(mask_ch.sum()),
            Sigma_amp=float(Sigma_amp),
            R_d_fit=float(R_d_fit), z_d_fit=float(z_d_fit),
            sigR0=float(sigR0), RsigR=float(RsigR),
            sigPhi0=float(sigPhi0), RsigPhi=float(RsigPhi),
            sigZ0=float(sigZ0), RsigZ=float(RsigZ),
            R_centers=R_centers, sig_R_data=sig_R, sig_phi_data=sig_phi, sig_z_data=sig_z,
            sig_R_err=sig_R_err, sig_phi_err=sig_phi_err, sig_z_err=sig_z_err,
            Sigma_data=Sigma_R, Sigma_err=Sigma_R_err,
            vphi_R_table=vphi_R_table,
            mass_scale=mass_scale,
        )

    return params


# ══════════════════════════════════════════════════════════════════
# PART 3 — microlensing event-rate formalism 
#
# Eq. (21):
#   dGamma/dt_E = int_0^{D_S} dD_L (2 R_E^2 / t_E^2)
#                  int d^3v f(x,v) delta(v_perp - R_E/t_E)
# ══════════════════════════════════════════════════════════════════

SEC_PER_DAY = 86400.0


def sigma_of_R(R, sigma0, Rsig):
    if np.isinf(Rsig):
        return np.full_like(np.atleast_1d(R), sigma0, dtype=float)
    return sigma0 * np.exp(-np.asarray(R) / Rsig)


def vphi_bar_of_R(R, table):
    Rc, vphi_mean = table
    return np.interp(R, Rc, vphi_mean, left=vphi_mean[0], right=vphi_mean[-1])


def nu0_of_R(R, ch):
    """Volume number density at z=0, nu(R,0) = Sigma(R)/(2 z_d) for a
    Laplace vertical profile [int_{-inf}^{inf} exp(-|z|/zd) dz = 2 zd]."""
    Sigma_R = ch["Sigma_amp"] * np.exp(-np.asarray(R) / ch["R_d_fit"])
    return Sigma_R / (2.0 * ch["z_d_fit"])


def mass_scaled_sigma(R, ch, M_Me):

    ms = ch["mass_scale"]
    Mref = ms["M_ref_Me"]
    sig_phi_R = sigma_of_R(R, ch["sigPhi0"], ch["RsigPhi"])
    sig_z_R = sigma_of_R(R, ch["sigZ0"], ch["RsigZ"])
    sig_phi_eff = sig_phi_R * (M_Me / Mref) ** ms["p_sigma_phi"]
    sig_z_eff = sig_z_R * (M_Me / Mref) ** ms["p_sigma_z"]
    return sig_phi_eff, sig_z_eff


def transverse_speed_density(s, R, ch, M_Me, n_theta=90):
    
    sig_phi, sig_z = mass_scaled_sigma(np.atleast_1d(R), ch, M_Me)
    sig_phi, sig_z = float(sig_phi[0]), float(sig_z[0])
    vbar = float(vphi_bar_of_R(R, ch["vphi_R_table"]))

    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    vy = s * np.cos(theta)
    vz_ = s * np.sin(theta)
    p2d = (1.0 / (2 * np.pi * sig_phi * sig_z) *
           np.exp(-((vy - vbar) ** 2) / (2 * sig_phi ** 2)
                  - (vz_ ** 2) / (2 * sig_z ** 2)))
    return s * np.mean(p2d) * 2 * np.pi


def einstein_radius_kpc(M_Msun, D_L, D_S):
    """Screenshot 3: R_E = sqrt(4 G M / c^2 * D_L(D_S-D_L)/D_S)."""
    return np.sqrt(np.clip(4 * G * M_Msun / C_KMS**2 * D_L * (D_S - D_L) / D_S, 0, None))


def dGamma_dtE(tE_days, M_Me, channel_params, R_obs, D_S, n_DL=400):
    
    M_Msun = M_Me * M_earth
    D_L = np.linspace(1e-4 * D_S, D_S * (1 - 1e-4), n_DL)
    R_line = R_obs - D_L
    R_line = np.clip(R_line, 1e-3, None)

    RE_kpc = einstein_radius_kpc(M_Msun, D_L, D_S)
    RE_km = RE_kpc * KM_PER_KPC
    tE_sec = tE_days * SEC_PER_DAY
    v_req = RE_km / tE_sec   # km/s, transverse speed needed for this t_E at this D_L

    nu0 = nu0_of_R(R_line, channel_params)
    Q = np.array([transverse_speed_density(v_req[i], R_line[i], channel_params, M_Me)
                  for i in range(n_DL)])

    # CORRECTED: RE_km**3 / tE_sec**3 (was RE_km**2 / tE_sec**2)
    integrand = (2.0 * RE_km**3 / tE_sec**3) * nu0 * Q
    return _trapz(integrand, D_L)

# ══════════════════════════════════════════════════════════════════
# PART 4 — diagnostic + result plots
# ══════════════════════════════════════════════════════════════════

CH_COLORS = {"CB": "tab:orange", "PL": "tab:blue"}
MASS_BIN_COLORS = ["#2a9d8f", "#e76f51", "#264653"]  # Earth / Neptune / Jupiter-like


def plot_qdf_calibration(params, fname):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))

    for ch in ["CB", "PL"]:
        p = params["channels"][ch]
        Rc = p["R_centers"]
        c = CH_COLORS[ch]

        # sigma_R, sigma_phi, sigma_z panels
        for ax, key, errkey, (k0, k1), ylab in zip(
                axes[:3], ["sig_R_data", "sig_phi_data", "sig_z_data"],
                ["sig_R_err", "sig_phi_err", "sig_z_err"],
                [("sigR0", "RsigR"), ("sigPhi0", "RsigPhi"), ("sigZ0", "RsigZ")],
                [r"$\sigma_R$ [km/s]", r"$\sigma_\phi$ (kick residual) [km/s]", r"$\sigma_z$ [km/s]"]):
            data = p[key]
            err = p[errkey]
            good = np.isfinite(data)
            ax.errorbar(Rc[good], data[good], yerr=err[good], fmt="o", ms=4,
                        color=c, ecolor=c, elinewidth=1, capsize=2,
                        label=f"{ch} (sim, R-binned)")
            Rline = np.linspace(Rc.min(), Rc.max(), 100)
            ax.plot(Rline, sigma_of_R(Rline, p[k0], p[k1]), "--", color=c,
                     label=f"{ch} fit: $\\sigma_0$={p[k0]:.1f}, $R_\\sigma$={p[k1]:.1f} kpc")
            ax.set_xlabel("R [kpc]")
            ax.set_ylabel(ylab)

        # density profile panel (log), with Poisson error bars
        ax = axes[3]
        good = p["Sigma_data"] > 0
        ax.errorbar(Rc[good], p["Sigma_data"][good], yerr=p["Sigma_err"][good],
                    fmt="o", ms=4, color=c, ecolor=c, elinewidth=1, capsize=2,
                    label=f"{ch} (sim, R-binned)")
        Rline = np.linspace(Rc.min(), Rc.max(), 100)
        ax.plot(Rline, p["Sigma_amp"] * np.exp(-Rline / p["R_d_fit"]), "--", color=c,
                 label=f"{ch} fit: $R_d$={p['R_d_fit']:.2f} kpc, $z_d$={p['z_d_fit']:.2f} kpc")
        ax.set_yscale("log")
        ax.set_xlabel("R [kpc]")
        ax.set_ylabel(r"$\Sigma(R)$ [sim particles / kpc$^2$]")

    for ax, title in zip(axes, [r"$\sigma_R(R)$", r"$\sigma_\phi(R)$", r"$\sigma_z(R)$",
                                  r"Surface density $\Sigma(R)$"]):
        ax.set_title(title)
        ax.legend(fontsize=7.5)
    fig.suptitle("qDF (Eq. 18) shape calibrated against the relaxed N-body snapshot "
                  "(disk fit restricted to R >= 2 kpc)", fontsize=11)
    plt.tight_layout()
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close()


def plot_timescale_distributions(results, fname):
    """results: dict[(channel, bin_idx)] -> (tE_grid_days, dGamma_dtE_vals, weight, label)"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for (ch, i), (tE, vals, w, label) in results.items():
        ls = "-" if ch == "CB" else "--"
        ax.plot(tE, vals, ls, color=MASS_BIN_COLORS[i], alpha=0.9,
                 label=f"{label} [{ch}]")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(bottom=1e-10)
    ax.set_xlabel(r"$t_E$ [days]")
    ax.set_ylabel(r"$d\Gamma/dt_E$  [relative units]")
    ax.set_title("Per-mass-bin timescale distribution (unweighted by mass-function)")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    for (ch, i), (tE, vals, w, label) in results.items():
        ls = "-" if ch == "CB" else "--"
        ax.plot(tE, vals * w, ls, color=MASS_BIN_COLORS[i], alpha=0.9,
                 label=f"{label} [{ch}] (w={w:.3f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(bottom=1e-10)
    ax.set_xlabel(r"$t_E$ [days]")
    ax.set_ylabel(r"$w_{\rm bin}\times d\Gamma/dt_E$  [relative units]")
    ax.set_title("Contribution to the total observable rate\n"
                  "(weighted by each bin's share of the channel's mass function)")
    ax.legend(fontsize=7.5)

    plt.tight_layout()
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close()


def plot_relative_abundance(summary_rows, fname):
    fig, ax = plt.subplots(figsize=(12, 10))
    labels = [f"{r['bin']}\n[{r['channel']}]" for r in summary_rows]
    vals = [r["Gamma_weighted"] for r in summary_rows]
    colors = [MASS_BIN_COLORS[r["bin_idx"]] for r in summary_rows]
    bars = ax.bar(labels, vals, color=colors,
                   hatch=["" if r["channel"] == "CB" else "//" for r in summary_rows],
                   edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\Gamma$ (Eq. 22), mass-function-weighted [relative units]")
    ax.set_title("Predicted relative event-rate contribution by FFP mass bin & channel")
    plt.xticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# PART 5 — main driver
# ══════════════════════════════════════════════════════════════════

def main(n_particles=500_000, n_steps=2000, total_time_yr=1e8, seed=42,
         out_dir="."):
    positions, velocities, masses, labels = run_nbody(
        n_particles=n_particles, n_steps=n_steps, total_time_yr=total_time_yr, seed=seed)

    print("\n=== Calibrating qDF (Eq. 18) shape against the relaxed snapshot ===")
    params = calibrate_qdf(positions, velocities, labels, masses)
    for ch in ["CB", "PL", "ALL"]:
        p = params["channels"][ch]
        print(f"\n[{ch}]  n={p['n']}")
        print(f"  density:  R_d_fit={p['R_d_fit']:.2f} kpc  z_d_fit={p['z_d_fit']:.3f} kpc")
        print(f"  sigma_R(R)   = {p['sigR0']:6.2f} * exp(-R/{p['RsigR']:.2f})  km/s")
        print(f"  sigma_phi(R) = {p['sigPhi0']:6.2f} * exp(-R/{p['RsigPhi']:.2f})  km/s  (kick residual)")
        print(f"  sigma_z(R)   = {p['sigZ0']:6.2f} * exp(-R/{p['RsigZ']:.2f})  km/s")
        ms = p["mass_scale"]
        print(f"  mass-scaling (measured, sim final state): "
              f"sigma_phi ~ M^{ms['p_sigma_phi']:.3f}, sigma_z ~ M^{ms['p_sigma_z']:.3f}  "
              f"(M_ref={ms['M_ref_Me']:.2f} Me)")

    # ---- geometry: observationally motivated, not derived from the sim ----
    R_obs = R0_KPC
    D_S = D_S_KPC
    print(f"\n=== Line-of-sight geometry (observationally motivated; "
          f"see R0_KPC/D_S_KPC definitions) ===")
    print(f"  R_obs = {R_obs:.2f} kpc (GRAVITY Collaboration 2019 / "
          f"Bland-Hawthorn & Gerhard 2016)")
    print(f"  D_S   = {D_S:.2f} kpc (Roman GBTDS bulge fields, Penny et al. 2019)")

    # ---- microlensing event-rate predictions, Eqs. 20-22 ----
    print("\n=== Computing microlensing timescale distributions (Eq. 21) ===")
    tE_grid = np.geomspace(2e-4, 300, 90)   # days
    results = {}
    summary_rows = []
    for ch in ["CB", "PL"]:
        p = params["channels"][ch]
        ms = p["mass_scale"]
        n_total_ch = sum(ms["n"])
        for i, (M_Me, name, n_bin) in enumerate(zip(ms["repr_Me"], params["mass_bin_names"], ms["n"])):
            t0 = time.time()
            vals = np.array([dGamma_dtE(t, M_Me, p, R_obs, D_S, n_DL=600) for t in tE_grid])
            w = n_bin / n_total_ch if n_total_ch > 0 else 0.0
            results[(ch, i)] = (tE_grid, vals, w, name)
            Gamma_w = _trapz(vals * w, tE_grid * SEC_PER_DAY)
            Gamma_unw = _trapz(vals, tE_grid * SEC_PER_DAY)
            tE_peak = tE_grid[np.argmax(vals)]
            summary_rows.append(dict(channel=ch, bin=name, bin_idx=i, M_repr_Me=M_Me,
                                      weight=w, n_sim=n_bin,
                                      Gamma_weighted=Gamma_w, Gamma_unweighted=Gamma_unw,
                                      tE_peak_days=tE_peak))
            print(f"  [{ch}] {name:24s} M={M_Me:7.2f} Me  w={w:.3f}  "
                  f"peak t_E={tE_peak:8.4f} d   ({time.time()-t0:.1f}s)")

    # ---- plots ----
    print("\n=== Saving plots ===")
    plot_qdf_calibration(params, f"{out_dir}/qdf_calibration.png")
    print(f"  saved {out_dir}/qdf_calibration.png")
    plot_timescale_distributions(results, f"{out_dir}/microlensing_timescale_distributions.png")
    print(f"  saved {out_dir}/microlensing_timescale_distributions.png")
    plot_relative_abundance(summary_rows, f"{out_dir}/relative_abundance_by_mass_bin.png")
    print(f"  saved {out_dir}/relative_abundance_by_mass_bin.png")

    # ---- summary table (CSV + JSON) ----
    import csv
    with open(f"{out_dir}/event_rate_summary.csv", "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["channel", "mass_bin", "M_repr_Me", "mass_fn_weight", "n_sim_particles",
                      "tE_peak_days", "Gamma_weighted_relative", "Gamma_unweighted_relative"])
        for r in summary_rows:
            wri.writerow([r["channel"], r["bin"], f"{r['M_repr_Me']:.3f}", f"{r['weight']:.4f}",
                          r["n_sim"], f"{r['tE_peak_days']:.5f}",
                          f"{r['Gamma_weighted']:.5e}", f"{r['Gamma_unweighted']:.5e}"])
    print(f"  saved {out_dir}/event_rate_summary.csv")

    total_Gamma = sum(r["Gamma_weighted"] for r in summary_rows)
    for r in summary_rows:
        r["Gamma_fraction_of_total"] = r["Gamma_weighted"] / total_Gamma if total_Gamma > 0 else 0.0

    print("\n=== Relative contribution to total predicted event rate ===")
    for r in sorted(summary_rows, key=lambda r: -r["Gamma_fraction_of_total"]):
        print(f"  [{r['channel']}] {r['bin']:24s}  "
              f"{r['Gamma_fraction_of_total']*100:5.1f}%   (peak t_E={r['tE_peak_days']:.3f} d)")

    calib_out = {ch: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in params["channels"][ch].items()
                       if k not in ("vphi_R_table",)}
                 for ch in params["channels"]}
    with open(f"{out_dir}/qdf_calibration_params.json", "w") as f:
        json.dump({"R_obs_kpc": R_obs, "D_S_kpc": D_S, "channels": calib_out}, f, indent=2, default=str)
    print(f"  saved {out_dir}/qdf_calibration_params.json")

    return params, results, summary_rows, R_obs, D_S


if __name__ == "__main__":
    main()