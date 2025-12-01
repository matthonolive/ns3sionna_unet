import numpy as np

'''
    Constants and helper functions
    
    author: Zubow
'''
c = 299792458 # speed of light in m/s

# in ns
MICROSECOND = 1000
MILLISECOND = 1000000
SECOND      = 1000000000

# important constants
MAX_COHERENCE_TIME = 10 * SECOND

def subcarrier_frequencies(num_subcarriers: int,
                           subcarrier_spacing: int):
    """
    Compute the vector of frequencies corresponding to each subcarrier
    :param num_subcarriers: number of subcarriers
    :param subcarrier_spacing: subcarrier spacing in Hz
    :return:
    """

    if num_subcarriers % 2 == 0:
        start = int(-num_subcarriers/2)
        limit = int(num_subcarriers/2)
    else:
        start = int(-(num_subcarriers-1)/2)
        limit = int((num_subcarriers-1)/2+1)

    frequencies = np.arange(start, limit, dtype=int)
    frequencies *= subcarrier_spacing

    return frequencies

def compute_coherence_time(v, fc, model='rappaport2'):
    """
    Computes the channel coherence time
    :param v: mobile speed in m/s
    :param fc: center frequency in Hz
    :param model: model for computation
    :return: Tc in ns
    """
    assert 0 <= v <= 100
    assert fc >= 1e6 # > 1MHz

    if model == 'rappaport':
        Tc = int(np.ceil(9 * c * 1e9 / (16 * np.pi * 2 * v * fc)))
    elif model == 'rappaport2':
        Tc = int(np.ceil(0.423 * c * 1e9 / (v * fc)))
    return Tc


def coherence_from_velocities(v_tx, v_rx, f_c,
                              pos_tx=None, pos_rx=None,
                              path_dirs=None, path_powers=None):
    """
    v_tx, v_rx : arrays (3,) velocities in m/s
    f_c        : carrier freq (Hz)
    pos_tx,pos_rx : optional positions (3,) to define LOS direction
    path_dirs  : optional array (N,3) of path direction vectors (not necessarily unit)
                 If None and pos_tx/pos_rx provided, LOS is used.
                 If None and no positions, worst-case (direction || v_rel) is used.
    path_powers: optional array (N,) for power-weighted RMS; else equal weights.
    Returns dict with per-path f_D, T_c and aggregates.
    """
    v_tx = np.asarray(v_tx, dtype=float)
    v_rx = np.asarray(v_rx, dtype=float)
    v_rel = v_rx - v_tx

    eps = np.finfo(float).eps
    if np.linalg.norm(v_rel) <= eps:
        return MAX_COHERENCE_TIME

    # Determine path directions
    if path_dirs is not None:
        u = np.atleast_2d(path_dirs).astype(float)
        norms = np.linalg.norm(u, axis=1)
        if np.any(norms == 0):
            raise ValueError("A path direction is zero.")
        u_hat = (u.T / norms).T
    elif (pos_tx is not None) and (pos_rx is not None):
        vec = np.asarray(pos_rx, dtype=float) - np.asarray(pos_tx, dtype=float)
        n = np.linalg.norm(vec)
        if n == 0:
            raise ValueError("TX and RX positions coincide; LOS undefined.")
        u_hat = np.atleast_2d(vec / n)
    else:
        # worst-case: use direction parallel to v_rel (so radial speed = |v_rel|)
        vr_norm = np.linalg.norm(v_rel)
        if vr_norm == 0:
            # no motion -> zero Doppler path (pick arbitrary unit)
            u_hat = np.atleast_2d(np.array([1.0, 0.0, 0.0]))
        else:
            u_hat = np.atleast_2d(v_rel / vr_norm)

    # compute
    v_rad = u_hat.dot(v_rel)           # signed radial speeds (N,)
    v_rad_abs = np.abs(v_rad)
    f_D = (v_rad_abs / c) * f_c        # per-path Doppler (Hz)
    T_c_per = np.where(f_D > 0, 0.423 / f_D, np.inf)

    f_D_max = np.max(f_D)
    T_c_worst = 0.423 / f_D_max if f_D_max > 0 else np.inf
    f_D_rms = np.sqrt(np.mean(f_D**2))
    T_c_rms = 0.423 / f_D_rms if f_D_rms > 0 else np.inf

    if path_powers is not None:
        p = np.asarray(path_powers, dtype=float)
        if p.shape[0] != f_D.shape[0]:
            raise ValueError("path_powers length mismatch")
        p_norm = p / np.sum(p)
        f_D_rms_pw = np.sqrt(np.sum(p_norm * f_D**2))
        T_c_rms_pw = 0.423 / f_D_rms_pw if f_D_rms_pw > 0 else np.inf
    else:
        f_D_rms_pw = None
        T_c_rms_pw = None

    # out = {
    #     "v_rel": v_rel,
    #     "path_unit_vectors": u_hat,
    #     "v_rad_signed": v_rad,
    #     "v_rad_abs": v_rad_abs,
    #     "f_D_per_path": f_D,
    #     "T_c_per_path": T_c_per,
    #     "f_D_max": f_D_max,
    #     "T_c_worst": T_c_worst,
    #     "f_D_rms": f_D_rms,
    #     "T_c_rms": T_c_rms,
    #     "f_D_rms_pw": f_D_rms_pw,
    #     "T_c_rms_pw": T_c_rms_pw
    # }

    return int(T_c_worst * 1e9)


if __name__ == '__main__':
    v = 1.0 # m/s
    fc = 5210e6 # center freq

    Tc1 = compute_coherence_time(v, fc, model='rappaport')
    print(f'{Tc1}ns -> {round(Tc1/1e6,2)}ms')

    Tc2 = compute_coherence_time(v, fc, model='rappaport2')
    print(f'{Tc2}ns -> {round(Tc2/1e6,2)}ms')

    # Example usage:
    v_tx = [-3.0, 0.0, 0.0]  # m/s
    v_rx = [5.0, 0.0, 0.0]  # m/s
    pos_tx = [0.0, 0.0, 0.0]
    pos_rx = [100.0, 0.0, 0.0]  # LOS along x
    Tc3 = coherence_from_velocities(v_tx, v_rx, fc, pos_tx=pos_tx, pos_rx=pos_rx)
    #for k, v in out.items():
    #    print(k, ":", v)
    print(f'{Tc3}ns -> {round(Tc3/1e6,2)}ms')