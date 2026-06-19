//Parameters for the coalescence simulation program : simcoal.exe
2 samples to simulate :
//Population effective sizes (number of genes)
NCUR1$
NCUR2$
//Samples sizes and samples age
48
22
//Growth rates: negative growth implies population expansion 
0
0
//Number of migration matrices : 0 implies no migration between demes 
2
//Migration matrix 0 
0 m21$
m12$ 0
//Migration matrix 1
0 0
0 0
//historical event: time, source, sink, migrants, new deme size, growth rate, migr mat index 
4 historical event
TMIG$ 0 0 0 1 0 1
TCHANGE1$ 0 0 1 RESIZE1$ 0 1
TCHANGE2$ 1 1 1 RESIZE2$ 0 1
TDIV$ 1 0 1 1 0 1
//Number of independent loci [chromosome]
1 0
//Per chromosome: Number of contiguous linkage Block: a block is a set of contiguous loci
1
//per Block:data type, number of loci, per gen recomb and mut rates 
FREQ 1 0 8.4e-8
