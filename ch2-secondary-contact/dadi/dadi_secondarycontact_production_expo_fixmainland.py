import dadi, numpy as np, matplotlib.pyplot as plt, demes, demesdraw, pysam, pathlib, random, json, datetime
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
    Symmetric migration; Mainland kept constant (fixed size), Island may change.

    params = (nuM0, nuI0, nuM1, nuI1, T_iso, T_mig, f1, f2, m)
      - NOTE: Mainland is held constant by ignoring (nuM1, f1).
    """
    import numpy as np
    import dadi

    (nuM0, nuI0, nuM1, nuI1, T_iso, T_mig, f1, f2, m) = params

    # --- FIX MAINLAND: keep Mainland constant across contact ---
    nuM1 = nuM0
    f1   = 0.0
    constM = (lambda t, v=nuM0: v)

    eps = 1e-8
    T_iso = max(T_iso, eps)
    T_mig = max(T_mig, eps)

    # Island timing
    f2 = min(max(f2, 0.0), 1.0)
    t2 = f2 * T_mig
    startM = T_mig           # Mainland never changes
    startI = T_mig - t2

    xx = dadi.Numerics.default_grid(pts)
    deme_ids = ('Mainland', 'Island')

    # CUDA hint to keep named demes consistent
    try:
        import dadi.cuda as _dc
        _dc.Integration.deme_ids = deme_ids
    except Exception:
        pass

    # Build phi with named demes
    phi = dadi.PhiManip.phi_1D(xx, deme_ids=(deme_ids[0],))
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi, deme_ids=deme_ids)

    # 1) Isolation (constant sizes, no migration)
    phi = dadi.Integration.two_pops(
        phi, xx, T_iso, nuM0, nuI0, deme_ids=deme_ids
    )

    # Helper for Island exponential schedule
    def _exp_lambda(s0, s1, total_dur):
        if total_dur <= eps:
            return (lambda t, v=s0: v)
        g = np.log(max(s1, eps) / max(s0, eps))
        if abs(g) < 1e-12:
            return (lambda t, v=s0: v)
        return lambda t, s0=s0, g=g, total_dur=total_dur: s0 * np.exp(g * (t / total_dur))

    # Segment contact epoch (Mainland constant)
    smin = max(min(startM, startI), 0.0)
    smax = min(max(startM, startI), T_mig)
    d0 = max(smin, 0.0)           # both constant
    d1 = max(smax - smin, 0.0)    # Island starts/continues changing
    d2 = max(T_mig - smax, 0.0)   # Island may keep changing

    # 2) Contact segment 1: both constant
    if d0 > eps:
        phi = dadi.Integration.two_pops(
            phi, xx, d0, nu1=constM, nu2=(lambda t, v=nuI0: v),
            m12=m, m21=m, deme_ids=deme_ids
        )

    curI = nuI0

    # 3) Contact segment 2: Island begins changing
    if d1 > eps:
        nuI_seg2 = _exp_lambda(curI, nuI1, max(t2, eps))
        phi = dadi.Integration.two_pops(
            phi, xx, d1, nu1=constM, nu2=nuI_seg2,
            m12=m, m21=m, deme_ids=deme_ids
        )
        curI = nuI_seg2(d1)

    # 4) Contact segment 3: Island continues changing (Mainland still constant)
    if d2 > eps:
        T_left_I = max(t2 - d1, 0.0)
        nuI_seg3 = _exp_lambda(curI, nuI1, max(T_left_I, eps))
        phi = dadi.Integration.two_pops(
            phi, xx, d2, nu1=constM, nu2=nuI_seg3,
            m12=m, m21=m, deme_ids=deme_ids
        )

    return dadi.Spectrum.from_phi(phi, ns, (xx, xx), mask_corners=True)

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
nstarts = 40
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
fig.savefig(out_dir / "modelBmainlandfix_testing.png", dpi=200); plt.close(fig)
demes.dump(g, str(out_dir / "modelBmainlandfix_testing.demes.yaml"))
