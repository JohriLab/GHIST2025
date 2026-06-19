#This is to find the fsc file with the maximum likelihood value:

import sys
import os

challenge="challenge1_final" #"challenge2_testing"
prefix="bottleneck_final"  #"bottleneck_final"/"secondary_contact_inst_growth_conditions"
num_of_reps = 10
s_column_names = ""

d_ll = {}
l_ll = []
rep = 1
while rep <= num_of_reps:
    print(rep)
    try:
        f_ll = open("/nas/longleaf/home/pjohri/ghist2025/" + challenge + "/" + prefix + "/" + prefix + str(rep) + "/" + prefix + ".bestlhoods", 'r')
        for line in f_ll:
            line1 = line.strip('\n')
            line2 = line1.split('\t')
            #store column names:
            if "MaxEstLhood" in line:
                s_column_names = line
                d_col = {}
                col_num = 0
                for x in line2:
                    d_col[x] = col_num
                    col_num += 1
            #get likelihood and make a dict so that likelohood -> line
            else:
                d_ll[float(line2[d_col["MaxEstLhood"]])] = line
                l_ll.append(float(line2[d_col["MaxEstLhood"]]))
        f_ll.close()

    except:
        print("can't read this replicate:" + '\t' + str(rep))
    
    #print(d_ll)
    #print(l_ll)
    rep += 1

max_ll = max(l_ll)
result = open("/nas/longleaf/home/pjohri/ghist2025/" + challenge + "/" + prefix + "/BEST_EST.txt", 'w+')
result.write(s_column_names)
result.write(d_ll[max_ll])
result.close()
print("done")

