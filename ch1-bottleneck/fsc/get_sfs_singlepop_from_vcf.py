#This is to get a single population SFS from a VCF:

import sys
import os

length_of_region = 100000000

def get_sfs_count(l_ac):
    d_sfs = {}
    s_seg = 0 #total number of truly segregating sites
    for x in l_ac:
        try:
            d_sfs[x] = d_sfs[x] + 1
        except:
            d_sfs[x] = 1
        #if int(x) > 0 and int(x) < int(num_indv):
        #    s_seg += 1
    #print("total number of segregating sites:" + str(s_seg))
    return(d_sfs)

def get_sfs_freq(d_sfs_count):
    d_sfs_freq = {}
    s_tot = 0
    for x in d_sfs_count.keys():
        s_tot = s_tot + int(d_sfs_count[x])
    for x in d_sfs_count.keys():
        d_sfs_freq[x] = float(d_sfs_count[x])/float(s_tot)
    return (d_sfs_freq)

num_seg_sites = 0 #this refers to sites that have the alternate allele
pi = 0.0
f_vcf = open("/nas/longleaf/home/pjohri/ghist2025/data/GHIST_2025_bottleneck.final.vcf", 'r')
l_AC = []
for line in f_vcf:
    line1 = line.strip('\n')
    line2 = line1.split('\t')
    #read column names
    if line2[0] == "#CHROM":
        d_col = {}
        col_num = 0
        for x in line2:
            d_col[x] = col_num
            col_num += 1
    #reading data:
    if line[0] != "#":
        if "," not in line2[d_col["ALT"]]: #removing biallelic variants
            #perform a check to see if reference allele is the same as the ancestral allele:
            ancestral_allele = line2[d_col["INFO"]].replace("AA=", "")
            reference_allele = line2[d_col["REF"]]
            #print(ancestral_allele + ', ' + reference_allele)
            if ancestral_allele != reference_allele:
                print ("WARNING: reference allele is not the ancestral allele here:")
                print(line)
            #store the derived allele frequency in a list
            s_allele_count = 0
            sample_size = 0
            col_num = 0
            for gt in line2:
                if col_num > d_col["FORMAT"]:
                    s_allele_count = s_allele_count + gt.count("1")
                    sample_size += 2 #for diploids
                col_num += 1
            #if s_allele_count == 0: #to check if monomorphic sites for the ancestral allele are included
                #print (line)
            if s_allele_count > 0:
                num_seg_sites += 1
                allele_freq = s_allele_count/float(sample_size)
                pi = pi + 2.0*allele_freq*(1.0 - allele_freq)
            l_AC.append(s_allele_count)

f_vcf.close()
#print (l_AC)
print ("sample size: " + str(sample_size))
print ("number of sites with ancestral alleles: " + str(num_seg_sites))
print ("nuc div per site: " + str(pi/length_of_region))

#calcualte SFS:
d_SFS_count = get_sfs_count(l_AC)
print(d_SFS_count)

#write out the full SFS:
result = open("/nas/longleaf/home/pjohri/ghist2025/data/GHIST_2025_bottleneck.final.sfs", 'w+')
result.write("1 observations" + '\n')
i = 0
while (i < int(sample_size)):
    result.write("d0_" + str(i) + '\t')
    i = i + 1
result.write(str(sample_size) + '\n')

#print the 0-class:
result.write(str(length_of_region - num_seg_sites))
i = 1
while (i <= int(sample_size)):
    result.write('\t' + str(d_SFS_count.get(i, 0)))
    i = i + 1
result.write('\n')

print("done")

