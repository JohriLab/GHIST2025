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

def split_mig_sym_step_indep(params, ns, pts):
    """
    Params:
      (nuM0, nuI0, nuM1, nuI1,  T_iso, T_mig,  f1, f2,  m)

      nu*_0 : size during isolation and at contact start
      nu*_1 : size after an instantaneous change within contact
      T_iso : duration split→contact (coalescent units)
      T_mig : duration contact→present (coalescent units)
      f1,f2 : fractions in [0,1]; change occurs at t1=f1*T_mig, t2=f2*T_mig before present
      m     : symmetric migration during contact epoch
    """
    import numpy as np
    import dadi

    nuM0, nuI0, nuM1, nuI1, T_iso, T_mig, f1, f2, m = params
    eps = 1e-8
    T_iso = max(T_iso, eps)
    T_mig = max(T_mig, eps)

    # Clamp fractions; convert to "time since contact start" when change happens
    f1 = min(max(f1, 0.0), 1.0)
    f2 = min(max(f2, 0.0), 1.0)
    t1 = f1 * T_mig   # Mainland change happens t1 before present
    t2 = f2 * T_mig   # Island   change happens t2 before present
    startM = T_mig - t1   # time since contact start when Mainland changes
    startI = T_mig - t2

    xx = dadi.Numerics.default_grid(pts)
    deme_ids = ('Mainland', 'Island')

    # CUDA hint so GPU backend keeps the same ordering
    try:
        import dadi.cuda as _dc
        _dc.Integration.deme_ids = deme_ids
    except Exception:
        pass

    # Build phi
    phi = dadi.PhiManip.phi_1D(xx, deme_ids=(deme_ids[0],))
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi, deme_ids=deme_ids)

    # 1) Isolation: constant sizes, no migration
    phi = dadi.Integration.two_pops(phi, xx, T_iso, nuM0, nuI0, deme_ids=deme_ids)

    # Segment the contact epoch
    smin = max(min(startM, startI), 0.0)
    smax = min(max(startM, startI), T_mig)
    d0 = max(smin, 0.0)          # before any size change: both at *_0
    d1 = max(smax - smin, 0.0)   # only the early-changer has switched to *_1
    d2 = max(T_mig - smax, 0.0)  # both have switched to *_1

    # 2) Contact seg1: both constant at *_0
    if d0 > eps:
        phi = dadi.Integration.two_pops(
            phi, xx, d0, nuM0, nuI0, m12=m, m21=m, deme_ids=deme_ids
        )

    # 3) Contact seg2: whichever changes first flips to *_1 (instantaneous)
    if d1 > eps:
        if startM < startI:
            # Mainland changed; Island still *_0
            phi = dadi.Integration.two_pops(
                phi, xx, d1, nuM1, nuI0, m12=m, m21=m, deme_ids=deme_ids
            )
            curM, curI = nuM1, nuI0
        else:
            # Island changed; Mainland still *_0
            phi = dadi.Integration.two_pops(
                phi, xx, d1, nuM0, nuI1, m12=m, m21=m, deme_ids=deme_ids
            )
            curM, curI = nuM0, nuI1
    else:
        curM, curI = (nuM0, nuI0) if d0 > eps else (nuM1 if startM==0 else nuM0,
                                                    nuI1 if startI==0 else nuI0)

    # 4) Contact seg3: both are at *_1
    if d2 > eps:
        phi = dadi.Integration.two_pops(
            phi, xx, d2, nuM1, nuI1, m12=m, m21=m, deme_ids=deme_ids
        )

    return dadi.Spectrum.from_phi(phi, ns, (xx, xx))  # default mask_corners=True

import numpy as np, matplotlib.pyplot as plt, dadi

# generous grid based on your data fs
nmax = int(np.max(fs.sample_sizes))
pts_l = [nmax + 15, nmax + 25, nmax + 35]

# wrapper to synchronize mask/folding with fs
def model_for_opt(params, ns, pts):
    mfs = split_mig_sym_step_indep(params, ns, pts)
    if fs.folded and not mfs.folded:
        mfs = mfs.fold()
    mfs.mask = fs.mask.copy()
    return mfs

func = dadi.Numerics.make_extrap_log_func(model_for_opt)

# bounds: sizes wide, times reasonable, change fractions in [0,1], symmetric m
lower_bound = [1e-2, 1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5]
upper_bound = [15,  15,  50,  50,   5.0,  5.0,  1.0-1e-6, 1.0-1e-6, 2.0]

# a reasonable seed (adjust if you like)
p0 = [2.5, 0.5, 3.0, 0.3, 0.6, 0.05, 0.7, 0.4, 3.0]

# optional: warm-up call to compile CUDA kernels
_ = func(p0, fs.sample_sizes, pts_l)

best_ll, best_popt = -np.inf, None
nstarts = 80
for i in range(nstarts):
    pstart = dadi.Misc.perturb_params(
        p0, fold=2.0, lower_bound=lower_bound, upper_bound=upper_bound
    )
    try:
        popt = dadi.Inference.optimize_log(
            pstart, fs, func, pts_l,
            lower_bound=lower_bound, upper_bound=upper_bound,
            verbose=0
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

# edit these:
mu = 8.4e-8
L  = 100_000_000
generation_time = 1.0

model_best = func(best_popt, fs.sample_sizes, pts_l)
theta = dadi.Inference.optimal_sfs_scaling(model_best, fs)
Nref  = theta / (4 * mu * L)

nuM0, nuI0, nuM1, nuI1, T_iso, T_mig, f1, f2, m = best_popt

# sizes
N_M_iso = nuM0 * Nref
N_I_iso = nuI0 * Nref
N_M_now = nuM1 * Nref
N_I_now = nuI1 * Nref

# absolute times (gens before present)
T_split_gen   = (T_iso + T_mig) * 2 * Nref   # split
T_contact_gen =  T_mig          * 2 * Nref   # contact start
t1_gen        = (f1 * T_mig)    * 2 * Nref   # Mainland step time
t2_gen        = (f2 * T_mig)    * 2 * Nref   # Island   step time

# symmetric migration per gen
m_per_gen = m / (2 * Nref)

print(f"\nθ={theta:.3f}  Nref={Nref:.1f}")
print(f"Mainland: {N_M_iso:.1f} → {N_M_now:.1f} (step at {t1_gen:.1f} gens)")
print(f"Island:   {N_I_iso:.1f} → {N_I_now:.1f} (step at {t2_gen:.1f} gens)")
print(f"Split: {T_split_gen:.1f} gens; contact start: {T_contact_gen:.1f} gens")
print(f"m per gen (sym): {m_per_gen:.6g}")

# Demes graph: piecewise-constant with 2 independent step times
epsg = 1e-6
yaml_model = f"""
description: Split→isolation→contact with independent instantaneous size changes (sym migration)
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
        start_size: {max(N_M_now, epsg):.8f}

  - name: Island
    start_time: {max(T_split_gen, epsg):.8f}
    ancestors: [Ancestral]
    epochs:
      - end_time: {max(T_contact_gen, epsg):.8f}
        start_size: {max(N_I_iso, epsg):.8f}
      - end_time: {max(t2_gen, epsg):.8f}
        start_size: {max(N_I_iso, epsg):.8f}
      - end_time: 0
        start_size: {max(N_I_now, epsg):.8f}

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
fig, ax = plt.subplots(1,1, figsize=(7.4,4.1), dpi=150)
demesdraw.tubes(g, ax=ax)
ax.set_title("Estimated history: independent step size changes, symmetric migration")
plt.show()

# # Optional: save
# import pathlib
# out_dir = pathlib.Path("dadi_fit_step"); out_dir.mkdir(exist_ok=True, parents=True)
# demes.dump(g, str(out_dir / "model_step.demes.yaml"))
# fig.savefig(out_dir / "model_step.png", dpi=200); plt.close(fig)

# # Optional: save YAML/PNG
# import pathlib, demes
out_dir = pathlib.Path("/nas/longleaf/home/adaigle/ghist_2024/workflow/plots/demography_stuff"); out_dir.mkdir(exist_ok=True, parents=True)
fig.savefig(out_dir / "modelA_testing.png", dpi=200); plt.close(fig)
demes.dump(g, str(out_dir / "modelA_testing.demes.yaml"))
