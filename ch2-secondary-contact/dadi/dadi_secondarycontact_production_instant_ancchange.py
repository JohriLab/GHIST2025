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
#VCF = pathlib.Path(
#    "/nas/longleaf/home/adaigle/johri/projects/ghist_2025/data/demography/"
#    "GHIST_2025_secondary_contact.testing.fixedheader.vcf.gz"
#)

VCF = pathlib.Path(
    "/nas/longleaf/home/adaigle/johri/projects/ghist_2025/data/demography/"
    "GHIST_2025_secondary_contact.final.vcf.gz"
)
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
fs.mask[0, -1]  = True   # private-derived on island
fs.mask[-1, 0]  = True   # private-derived on mainland

# ── 5) sanity checks ────────────────────────────────────────────────
print("Total sites in fs:", fs.sum(), "(should equal REGION_LENGTH)")
print(fs)

import numpy as np, dadi
import matplotlib.pyplot as plt
dadi.cuda_enabled(True)

# ---- grid: a bit larger than max sample size
nmax = int(np.max(fs.sample_sizes))
pts_l = [nmax + 15, nmax + 25, nmax + 35]

def split_mig_sym_step_indep_with_anc(params, ns, pts):
    """
    Model: Split → isolation (no migration) → secondary contact (sym migration),
    with independent instantaneous size steps during contact for each daughter,
    AND an ancestral size change in the common ancestor *before* the split.

    Params (order):
      (nuA0, nuA1, T_anc, fA,
       nuM0, nuI0, nuM1, nuI1,
       T_iso, T_mig, f1, f2, m)

      Ancestral (pre-split):
        - nuA0 : ancestral size before the change
        - nuA1 : ancestral size after the change (closer to split)
        - T_anc: total duration of the modeled ancestral epoch before split (2Nref units)
        - fA   : fraction in [0,1]; the change occurs fA*T_anc before the split

      Daughter pops:
        - nu*_0: size during isolation and at contact start
        - nu*_1: size after instantaneous change within contact
        - T_iso: duration split→contact (2Nref units)
        - T_mig: duration contact→present (2Nref units)
        - f1,f2: fractions in [0,1]; step times within contact for Mainland/Island
        - m    : symmetric migration rate during contact epoch (scaled, 2Nref*m_per_gen)
    """
    import numpy as np
    import dadi

    (nuA0, nuA1, T_anc, fA,
     nuM0, nuI0, nuM1, nuI1,
     T_iso, T_mig, f1, f2, m) = params

    eps = 1e-8
    T_anc = max(T_anc, 0.0)
    T_iso = max(T_iso, eps)
    T_mig = max(T_mig, eps)

    # Clamp fractions
    def _clip01(x):
        return min(max(x, 0.0), 1.0)
    fA = _clip01(fA)
    f1 = _clip01(f1)
    f2 = _clip01(f2)

    # Ancestral split-relative timing
    # Change happens fA*T_anc before split; older segment length = T_anc - fA*T_anc
    dA1 = fA * T_anc          # recent ancestral segment (after change) -> right before split
    dA0 = T_anc - dA1         # older ancestral segment (before change)

    # Daughter step times inside the contact epoch
    t1 = f1 * T_mig   # Mainland change time before present within contact
    t2 = f2 * T_mig   # Island   change time before present within contact
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

    # Build phi with an explicit ancestral epoch containing a size change
    phi = dadi.PhiManip.phi_1D(xx, deme_ids=('Ancestral',))
    if dA0 > eps:
        phi = dadi.Integration.one_pop(phi, xx, dA0, nu=nuA0, deme_ids=('Ancestral',))
    if dA1 > eps:
        phi = dadi.Integration.one_pop(phi, xx, dA1, nu=nuA1, deme_ids=('Ancestral',))

    # Split to two pops at the end of the ancestral epoch
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi, deme_ids=deme_ids)

    # 1) Isolation: constant sizes, no migration
    phi = dadi.Integration.two_pops(phi, xx, T_iso, nuM0, nuI0, deme_ids=deme_ids)

    # Segment the contact epoch (allow different within-contact step times)
    smin = max(min(startM, startI), 0.0)
    smax = min(max(startM, startI), T_mig)
    d0 = max(smin, 0.0)          # both at *_0
    d1 = max(smax - smin, 0.0)   # only early-changer at *_1
    d2 = max(T_mig - smax, 0.0)  # both at *_1

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
        else:
            # Island changed; Mainland still *_0
            phi = dadi.Integration.two_pops(
                phi, xx, d1, nuM0, nuI1, m12=m, m21=m, deme_ids=deme_ids
            )

    # 4) Contact seg3: both are at *_1
    if d2 > eps:
        phi = dadi.Integration.two_pops(
            phi, xx, d2, nuM1, nuI1, m12=m, m21=m, deme_ids=deme_ids
        )

    return dadi.Spectrum.from_phi(phi, ns, (xx, xx))  # default mask_corners=True


# wrapper to synchronize mask/folding with fs
def model_for_opt(params, ns, pts):
    mfs = split_mig_sym_step_indep_with_anc(params, ns, pts)
    if fs.folded and not mfs.folded:
        mfs = mfs.fold()
    mfs.mask = fs.mask.copy()
    return mfs

func = dadi.Numerics.make_extrap_log_func(model_for_opt)

# ── Bounds/seed ─────────────────────────────────────────────────────
# sizes wide, times reasonable, change fractions in [0,1], symmetric m
lower_bound = [
    1e-2, 1e-2, 1e-4, 1e-6,      # nuA0, nuA1, T_anc, fA
    1e-2, 1e-2, 1e-2, 1e-2,      # nuM0, nuI0, nuM1, nuI1
    1e-4, 1e-4, 1e-6, 1e-6,      # T_iso, T_mig, f1, f2
    1e-5                         # m
]
upper_bound = [
    50, 50, 5.0, 1.0 - 1e-6,     # nuA0, nuA1, T_anc, fA
    15, 15, 50, 50,              # nuM0, nuI0, nuM1, nuI1
    5.0, 5.0, 1.0 - 1e-6, 1.0 - 1e-6,  # T_iso, T_mig, f1, f2
    2.0                          # m
]

# a reasonable seed (adjust if you like)
p0 = [
    1.0, 0.5, 0.5, 0.5,          # nuA0, nuA1, T_anc, fA
    2.5, 0.5, 3.0, 0.3,          # nuM0, nuI0, nuM1, nuI1
    0.6, 0.05, 0.7, 0.4,         # T_iso, T_mig, f1, f2
    3.0                          # m
]

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

# edit these if needed:
mu = 8.4e-8
L  = 100_000_000
generation_time = 1.0

model_best = func(best_popt, fs.sample_sizes, pts_l)
theta = dadi.Inference.optimal_sfs_scaling(model_best, fs)
Nref  = theta / (4 * mu * L)

(nuA0, nuA1, T_anc, fA,
 nuM0, nuI0, nuM1, nuI1,
 T_iso, T_mig, f1, f2, m) = best_popt

# sizes
N_A0 = nuA0 * Nref
N_A1 = nuA1 * Nref
N_M_iso = nuM0 * Nref
N_I_iso = nuI0 * Nref
N_M_now = nuM1 * Nref
N_I_now = nuI1 * Nref

# absolute times (gens before present)
T_split_gen    = (T_iso + T_mig) * 2 * Nref       # split
T_contact_gen  =  T_mig          * 2 * Nref       # contact start
T_anc_total_gen = T_anc           * 2 * Nref      # total ancestral epoch modeled
tA_gen         = (fA * T_anc)    * 2 * Nref       # time between change and split
tA_abs_gen     =  T_split_gen + tA_gen            # absolute time of ancestral change

t1_gen        = (f1 * T_mig)    * 2 * Nref        # Mainland step time (before present)
t2_gen        = (f2 * T_mig)    * 2 * Nref        # Island   step time (before present)

# symmetric migration per gen
m_per_gen = m / (2 * Nref)

print(f"\nθ={theta:.3f}  Nref={Nref:.1f}")
print(f"Ancestral: {N_A0:.1f} → {N_A1:.1f} (change {tA_gen:.1f} gens before split)")
print(f"Mainland:  {N_M_iso:.1f} → {N_M_now:.1f} (step at {t1_gen:.1f} gens BP)")
print(f"Island:    {N_I_iso:.1f} → {N_I_now:.1f} (step at {t2_gen:.1f} gens BP)")
print(f"Split: {T_split_gen:.1f} gens; contact start: {T_contact_gen:.1f} gens")
print(f"m per gen (sym): {m_per_gen:.6g}")

# ── Demes graph ─────────────────────────────────────────────────────
# Piecewise-constant with 2 ancestral epochs, independent within-contact steps, symmetric migration
epsg = 1e-6
yaml_model = f"""
description: Split→isolation→contact with independent step size changes; includes ancestral size change before split (sym migration during contact)
time_units: generations
generation_time: {generation_time}
demes:
  - name: Ancestral
    start_time: .inf
    epochs:
      - end_time: {max(tA_abs_gen, epsg):.8f}
        start_size: {max(N_A0, epsg):.8f}
      - end_time: {max(T_split_gen, epsg):.8f}
        start_size: {max(N_A1, epsg):.8f}

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
fig, ax = plt.subplots(1,1, figsize=(7.6,4.2), dpi=150)
demesdraw.tubes(g, ax=ax)
ax.set_title("Estimated history: ancestral size change + independent steps, symmetric migration")
plt.tight_layout()
plt.show()

# ── Save with NEW filenames ─────────────────────────────────────────
out_dir = pathlib.Path("/nas/longleaf/home/adaigle/ghist_2024/workflow/plots/demography_stuff")
out_dir.mkdir(exist_ok=True, parents=True)
fname_prefix = "modelB_ancsize_final"
fig.savefig(out_dir / f"{fname_prefix}.png", dpi=200); plt.close(fig)
demes.dump(g, str(out_dir / f"{fname_prefix}.demes.yaml"))
