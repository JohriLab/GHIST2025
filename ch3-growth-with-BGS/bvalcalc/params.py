## Population genetic parameters for the simulated or empirical population
## Accurate estimation requires accurate and appropriate parameters
##
## e.g. Bvalcalc --pop_params path/to/ExampleParams.py
##
## Core parameters
x = 1 # Scaling factor (N,u,r), keep as 1 unless calculating for rescaled simulations
Nanc = 9826e3 / x # Ancestral population size [1]
r = 1e-8 * x # Recombination (crossover) rate per bp, per generation (sex-averaged) [2]
u = 3.36e-8 * x # Mutation rate (all types) per bp, per generation [1]
g = 0 * 1e-8 * x # Gene conversion initiation rate per bp, per generation [3]
k = 0 * 440 # Gene conversion tract length (bp) [3]
## DFE parameters for ALL sites in annotated regions (Sum must equal 1)
f0 = 0.61 # Proportion of effectively neutral mutations with 0 <= |2Ns| < 1 (Note that 2Ns<5 does not contribute to BGS) [4]
f1 = 0.19 # Proportion of weakly deleterious mutations with 1 <= |2Ns| < 10 [4]
f2 = 0.18 # Proportion of moderately deleterious mutations with 10 <= |2Ns| < 100 [4]
f3 = 0.02 # Proportion of strongly deleterious mutations with |2Ns| >= 100 [4]
## Demography parameters, using what we inferred previously
Ncur = 2.030 * Nanc # Current population size (!Requires --pop_change) [5]
time_of_change = 2.030 # Time in Nanc generations ago that effective population size went from Nanc to Ncur (!Requires --pop_change) [6]
## Advanced DFE parameters 
h = 0.5 # Dominance coefficient of selected alleles [Naive value]
mean, shape, proportion_synonymous = 2 * 6.888, 0.21, 0.3 # Gamma distribution of DFE to discretize and replace f0-f3 [mean (2Ns), shape, proportion synonymous] (!Requires --gamma_dfe) [Naive value]## Literature cited
# [1] Keightley et al 2014  doi: 10.1534/genetics.113.158758
# [2] Comeron et al 2012 doi: 10.1371/journal.pgen.1002905
# [3] Miller et al 2016 doi: 10.1534/genetics.115.186486
# [4] Johri et al 2020 doi: 10.1534/genetics.119.303002
# [5] Laurent et al 2011 doi: 10.1093/molbev/msr031
# [6] Kapopoulou et al 2018 doi: 10.1093/gbe/evy185
