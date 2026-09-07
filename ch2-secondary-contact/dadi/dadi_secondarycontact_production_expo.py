import dadi, numpy as np, matplotlib.pyplot as plt, demes, demesdraw, pysam, pathlib, random, json, datetime
def split_mig_model(params, ns, pts):
    import dadi
    # params = (nu_mainland, nu_island, T_iso, T_mig, m)
    nu_mainland, nu_island, T_iso, T_mig, m = params

    xx = dadi.Numerics.default_grid(pts)
    deme_ids = ('Mainland', 'Island')

    # Equilibrium → split to 2D with stable deme labels (good for CUDA)
    phi = dadi.PhiManip.phi_1D(xx, deme_ids=(deme_ids[0],))
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi, deme_ids=deme_ids)

    # Isolation epoch (no migration)
    if T_iso > 0:
        phi = dadi.Integration.two_pops(
            phi, xx, T_iso, nu_mainland, nu_island, deme_ids=deme_ids
        )

    # Secondary contact epoch (symmetric migration)
    if T_mig > 0:
        phi = dadi.Integration.two_pops(
            phi, xx, T_mig, nu_mainland, nu_island,
            m12=m, m21=m, deme_ids=deme_ids
        )

    return dadi.Spectrum.from_phi(phi, ns, (xx, xx), mask_corners=True)


def split_one_way_mig_model(params, ns, pts):
    """
    Split model with initial isolation, followed by secondary contact and 
    one-way migration from Mainland to Island.
    
    params: List of parameters [nu_mainland, nu_island, T_split, T_mig, m]
    ns: List of sample sizes [n_mainland, n_island]
    pts: Number of grid points
    """
    nu_mainland, nu_island, T_split, T_mig, m = params
    
    # Create a grid of points
    xx = dadi.Numerics.default_grid(pts)
    
    # Ancestral population equilibrium
    phi = dadi.PhiManip.phi_1D(xx)
    
    # Split the population at time T_split
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    
    phi = dadi.Integration.two_pops(
        phi, xx, T_split - T_mig,
        nu_mainland, nu_island,
        deme_ids=(0,1)
    )
    phi = dadi.Integration.two_pops(
        phi, xx, T_mig,
        nu_mainland, nu_island,
        m12=m, m21=m,
        deme_ids=(0,1)
    )
    # Calculate the frequency spectrum
    fs = dadi.Spectrum.from_phi(phi, ns, (xx, xx))
    
    return fs

def split_one_way_mig_model_flipped(params, ns, pts):
    """
    Split model with initial isolation, followed by secondary contact and 
    one-way migration from Island to Mainland.
    
    params: List of parameters [nu_mainland, nu_island, T_split, T_mig, m]
    ns: List of sample sizes [n_mainland, n_island]
    pts: Number of grid points
    """
    nu_mainland, nu_island, T_split, T_mig, m = params
    
    # Create a grid of points
    xx = dadi.Numerics.default_grid(pts)
    
    # Ancestral population equilibrium
    phi = dadi.PhiManip.phi_1D(xx)
    
    # Split the population at time T_split
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    
    # Integrate populations in isolation with sizes nu_mainland and nu_island for T_split - T_mig generations
    phi = dadi.Integration.two_pops(phi, xx, T_split - T_mig, nu_mainland, nu_island)
    
    # Integrate with one-way migration from Island to Mainland from T_mig generations ago until present
    phi = dadi.Integration.two_pops(phi, xx, T_mig, nu_mainland, nu_island, m12=0, m21=m)
    
    # Calculate the frequency spectrum
    fs = dadi.Spectrum.from_phi(phi, ns, (xx, xx))
    
    return fs

import pycuda.driver as drv
drv.init()
print("Device count:", drv.Device.count())
print("First device:", drv.Device(0).name())
ctx = drv.Device(0).make_context()
ctx.pop()
import skcuda.cublas as cublas
handle = cublas.cublasCreate()
cublas.cublasDestroy(handle)
import dadi
dadi.cuda_enabled(True)

import pathlib
import pysam
import dadi

# ── User inputs ──────────────────────────────────────────────────────
VCF = pathlib.Path(
    "/nas/longleaf/home/adaigle/johri/projects/ghist_2025/data/demography/"
    "GHIST_2025_secondary_contact.testing.fixedheader.vcf.gz"
)

#VCF = pathlib.Path(
#    "/nas/longleaf/home/adaigle/johri/projects/ghist_2025/data/demography/"
#    "GHIST_2025_secondary_contact.final.vcf.gz"
#)
REGION_LENGTH = 100_000_000  # total callable bases
pop_ids       = ["mainland", "island"]
ns            = [48, 22]     # diploid sample sizes

# ── 1) make a .popfile.txt if needed ────────────────────────────────
popfile = VCF.with_suffix(".popfile.txt")
if not popfile.exists():
    with popfile.open("w") as fh:
        for sample in pysam.VariantFile(str(VCF)).header.samples:
            pop = sample.split("_", 1)[0]
            fh.write(f"{sample}\t{pop}\n")

# ── 2) build data-dict & raw SFS (unmasked corners) ─────────────────
dd = dadi.Misc.make_data_dict_vcf(str(VCF), str(popfile))
fs = dadi.Spectrum.from_data_dict(
    dd, pop_ids, ns,
    polarized=True,       # or False if you want folded
    mask_corners=False,   # allow corner bins to be unmasked
)

# ── 3) inject monomorphic sites into [0,0] ─────────────────────────
observed_sites = fs.sum()
mono = REGION_LENGTH - observed_sites
fs[0, 0] += mono

# ── 4) explicitly unmask the corners you care about ────────────────
fs.mask[0, 0]   = True   # invariant
fs.mask[-1, -1] = True   # fixed-derived
fs.mask[0, -1]  = True  # private-derived on island
fs.mask[-1, 0]  = True  # private-derived on mainland

# ── 5) sanity checks ────────────────────────────────────────────────
print("Total sites in fs:", fs.sum(), "(should equal REGION_LENGTH)")
print(fs)

import numpy as np, dadi
import matplotlib.pyplot as plt
dadi.cuda_enabled(True)

# ---- grid: a bit larger than max sample size
nmax = int(np.max(fs.sample_sizes))
pts_l = [nmax + 15, nmax + 25, nmax + 35]
def split_mig_sym_exp_indep(params, ns, pts):
    """
    Symmetric migration, independent size-change start times.

    params = (nuM0, nuI0, nuM1, nuI1, T_iso, T_mig, f1, f2, m)
      - nuM0,nuI0 : sizes in isolation and at contact start
      - nuM1,nuI1 : present-day sizes
      - T_iso     : isolation duration (split→contact)
      - T_mig     : contact duration (contact→present)
      - f1,f2     : fractions in [0,1] when each deme begins changing within contact
      - m         : symmetric migration during contact
    """
    import numpy as np
    import dadi

    (nuM0, nuI0, nuM1, nuI1, T_iso, T_mig, f1, f2, m) = params
    eps = 1e-8
    T_iso = max(T_iso, eps)
    T_mig = max(T_mig, eps)

    # Clamp fractions and convert to “time since contact start”
    f1 = min(max(f1, 0.0), 1.0)
    f2 = min(max(f2, 0.0), 1.0)
    t1 = f1 * T_mig                 # Mainland change begins t1 before present
    t2 = f2 * T_mig                 # Island change begins t2 before present
    startM = T_mig - t1             # time since contact start when Mainland starts changing
    startI = T_mig - t2

    xx = dadi.Numerics.default_grid(pts)
    deme_ids = ('Mainland', 'Island')

    # --- CUDA hint: set deme_ids globally for the CUDA backend
    try:
        import dadi.cuda as _dc
        _dc.Integration.deme_ids = deme_ids
    except Exception:
        pass

    # Build phi
    phi = dadi.PhiManip.phi_1D(xx, deme_ids=(deme_ids[0],))
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi, deme_ids=deme_ids)

    # 1) Isolation (constant sizes, no migration)
    phi = dadi.Integration.two_pops(phi, xx, T_iso, nuM0, nuI0, deme_ids=deme_ids)

    # Helper for exponential schedule
    def _exp_lambda(s0, s1, total_dur):
        if total_dur <= eps:
            return (lambda t, v=s0: v)
        g = np.log(max(s1, eps) / max(s0, eps))
        if abs(g) < 1e-12:
            return (lambda t, v=s0: v)
        return lambda t, s0=s0, g=g, total_dur=total_dur: s0 * np.exp(g * (t / total_dur))

    # Segment the contact epoch, clamp to nonnegative
    smin = max(min(startM, startI), 0.0)
    smax = max(startM, startI)
    smax = min(max(smax, 0.0), T_mig)

    d0 = max(smin, 0.0)                  # before any change
    d1 = max(smax - smin, 0.0)           # first deme changing
    d2 = max(T_mig - smax, 0.0)          # both possibly changing

    # 2) Contact segment 1: both constant
    if d0 > eps:
        phi = dadi.Integration.two_pops(
            phi, xx, d0, nuM0, nuI0, m12=m, m21=m, deme_ids=deme_ids
        )

    # Track sizes at current boundary
    curM, curI = nuM0, nuI0

    # 3) Contact segment 2: whichever starts first
    if d1 > eps:
        if startM < startI:
            T_remain_M = max(t1, 0.0)
            nuM_seg2 = _exp_lambda(curM, nuM1, max(T_remain_M, eps))
            phi = dadi.Integration.two_pops(
                phi, xx, d1, nu1=nuM_seg2, nu2=(lambda t, v=curI: v),
                m12=m, m21=m, deme_ids=deme_ids
            )
            # update Mainland size at the boundary
            curM = nuM_seg2(d1)
        else:
            T_remain_I = max(t2, 0.0)
            nuI_seg2 = _exp_lambda(curI, nuI1, max(T_remain_I, eps))
            phi = dadi.Integration.two_pops(
                phi, xx, d1, nu1=(lambda t, v=curM: v), nu2=nuI_seg2,
                m12=m, m21=m, deme_ids=deme_ids
            )
            curI = nuI_seg2(d1)

    # 4) Contact segment 3: both changing (or one continues)
    if d2 > eps:
        if startM < startI:
            T_left_M = max(t1 - d1, 0.0)   # remaining time for Mainland’s exp
            T_left_I = max(t2, 0.0)        # Island starts now
        else:
            T_left_I = max(t2 - d1, 0.0)
            T_left_M = max(t1, 0.0)
        nuM_seg3 = _exp_lambda(curM, nuM1, max(T_left_M, eps))
        nuI_seg3 = _exp_lambda(curI, nuI1, max(T_left_I, eps))
        phi = dadi.Integration.two_pops(
            phi, xx, d2, nu1=nuM_seg3, nu2=nuI_seg3,
            m12=m, m21=m, deme_ids=deme_ids
        )

    return dadi.Spectrum.from_phi(phi, ns, (xx, xx))

# ---- wrap to match data mask/fold
def model_for_opt(params, ns, pts):
    mfs = split_mig_sym_exp_indep(params, ns, pts)
    if fs.folded and not mfs.folded:
        mfs = mfs.fold()
    mfs.mask = fs.mask.copy()
    return mfs

func = dadi.Numerics.make_extrap_log_func(model_for_opt)

# ---- bounds & a reasonable seed
lower_bound = [1e-2, 1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6]  # sizes, sizes, times, f1,f2,m
upper_bound = [15,  15,  50,  50,   5.0,  5.0,  1-1e-6, 1-1e-6, 2.0]

# Seed guess (tweak as desired)
p0 = [2.5, 0.5, 3.0, 0.3, 0.6, 0.05, 0.7, 0.4, 3.0]

# ---- multi-start optimize_log in log-space
best_ll, best_popt = -np.inf, None
nstarts = 80
for i in range(nstarts):
    pstart = dadi.Misc.perturb_params(p0, fold=2.0,
                                      lower_bound=lower_bound,
                                      upper_bound=upper_bound)
    try:
        popt = dadi.Inference.optimize_log(
            pstart, fs, func, pts_l,
            lower_bound=lower_bound, upper_bound=upper_bound, verbose=0
        )
        model = func(popt, fs.sample_sizes, pts_l)
        ll = dadi.Inference.ll_multinom(model, fs)
        if ll > best_ll:
            best_ll, best_popt = ll, popt
        print(f"[{i:02d}] ll={ll:.2f}  params={popt}")
    except Exception as e:
        print(f"[{i:02d}] failed: {e}")

print("\nBest log-likelihood:", best_ll)
print("Best params:", best_popt)

import demes, demesdraw

# ---- mutation/length/gen time (edit as needed)
mu = 8.4e-8
L  = 100_000_000
generation_time = 1.0

# theta, Nref
model_best = func(best_popt, fs.sample_sizes, pts_l)
theta = dadi.Inference.optimal_sfs_scaling(model_best, fs)
Nref  = theta / (4 * mu * L)

(nuM0, nuI0, nuM1, nuI1, T_iso, T_mig, f1, f2, m) = best_popt

# sizes
N_M_iso = nuM0 * Nref
N_I_iso = nuI0 * Nref
N_M_now = nuM1 * Nref
N_I_now = nuI1 * Nref

# absolute times (generations before present)
T_split_gen   = (T_iso + T_mig) * 2 * Nref
T_contact_gen =  T_mig          * 2 * Nref
t1_gen        = (f1 * T_mig)    * 2 * Nref
t2_gen        = (f2 * T_mig)    * 2 * Nref

# per-generation migration (symmetric)
m_per_gen = m / (2 * Nref)

print(f"\nθ={theta:.3f}  Nref={Nref:.1f}")
print(f"Mainland size: isolation={N_M_iso:.1f} → present={N_M_now:.1f}, time={t1_gen:.1f}")
print(f"Island   size: isolation={N_I_iso:.1f} → present={N_I_now:.1f}, time={t2_gen:.1f}")
print(f"Split time: {T_split_gen:.1f} gens, contact start: {T_contact_gen:.1f} gens")
print(f"Symmetric migration m per gen: {m_per_gen:.6g}")

# ---- Demes graph: piecewise with independent size-change times
epsg = 1e-6  # tiny to avoid zero-length epochs
yaml_model = f"""
description: Split→isolation→contact with independent exponential size changes (sym migration)
time_units: generations
generation_time: {generation_time}
demes:
  - name: Ancestral
    start_time: .inf
    epochs:
      - end_time: {max(T_split_gen, epsg):.8f}
        start_size: {max(Nref, epsg):.8f}

  - name: Mainland
    start_time: {max(T_split_gen, epsg):.8f}
    ancestors: [Ancestral]
    epochs:
      - end_time: {max(T_contact_gen, epsg):.8f}
        start_size: {max(N_M_iso, epsg):.8f}
      - end_time: {max(t1_gen, epsg):.8f}
        start_size: {max(N_M_iso, epsg):.8f}
      - end_time: 0
        start_size: {max(N_M_iso, epsg):.8f}
        end_size:   {max(N_M_now, epsg):.8f}
        size_function: exponential

  - name: Island
    start_time: {max(T_split_gen, epsg):.8f}
    ancestors: [Ancestral]
    epochs:
      - end_time: {max(T_contact_gen, epsg):.8f}
        start_size: {max(N_I_iso, epsg):.8f}
      - end_time: {max(t2_gen, epsg):.8f}
        start_size: {max(N_I_iso, epsg):.8f}
      - end_time: 0
        start_size: {max(N_I_iso, epsg):.8f}
        end_size:   {max(N_I_now, epsg):.8f}
        size_function: exponential

migrations:
  - source: Mainland
    dest: Island
    start_time: {max(T_contact_gen, epsg):.8f}
    end_time: 0
    rate: {max(min(m_per_gen, 0.999999), 0.0):.12f}
  - source: Island
    dest: Mainland
    start_time: {max(T_contact_gen, epsg):.8f}
    end_time: 0
    rate: {max(min(m_per_gen, 0.999999), 0.0):.12f}
"""
g = demes.loads(yaml_model)
fig, ax = plt.subplots(1,1, figsize=(7.4,4.2), dpi=150)
demesdraw.tubes(g, ax=ax)
ax.set_title("Estimated history: independent size-change times, symmetric migration")
plt.show()

# # Optional: save YAML/PNG
# import pathlib, demes
out_dir = pathlib.Path("/nas/longleaf/home/adaigle/ghist_2024/workflow/plots/demography_stuff"); out_dir.mkdir(exist_ok=True, parents=True)
fig.savefig(out_dir / "modelB_testing.png", dpi=200); plt.close(fig)
demes.dump(g, str(out_dir / "modelB_testing.demes.yaml"))
