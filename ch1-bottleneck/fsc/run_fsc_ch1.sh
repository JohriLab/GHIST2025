#!/bin/bash
#SBATCH --mail-user=pjohri@unc.edu
#SBATCH --mail-type=ALL
#SBATCH -p general
#SBATCH -n 1 #number of tasks
#SBATCH --time=0-5:00
#SBATCH --mem=20m
#SBATCH -o /nas/longleaf/home/pjohri/LOGFILES/fsc%j.out
#SBATCH -e /nas/longleaf/home/pjohri/LOGFILES/fsc%j.err

#run this in the relevant directory
prefix="bottleneck_final"
mkdir /nas/longleaf/home/pjohri/ghist2025/${prefix}
cd /nas/longleaf/home/pjohri/ghist2025/${prefix}

#copy the relevant files into the directory
cp /nas/longleaf/home/pjohri/bin/fsc28_linux64/fsc28 ./
cp /nas/longleaf/home/pjohri/ghist2025/data/GHIST_2025_bottleneck.final.sfs ${prefix}_DAFpop0.obs
cp /nas/longleaf/home/pjohri/ghist2025/fsc/size_change_inst_wide_priors.est ${prefix}.est
cp /nas/longleaf/home/pjohri/ghist2025/fsc/size_change_inst_wide_priors.tpl ${prefix}.tpl

#run it 50 times:
declare -i repID=1
while [ $repID -lt 51 ];
do
	echo "starting rep " $repID
	#run fsc
	./fsc28 -t ${prefix}.tpl -n 100000 -d -e ${prefix}.est -M -L 50 -q
	
	#save files and folders:
	mv ${prefix}.par ${prefix}${repID}.par
	mv seed.txt seed${repID}.txt
	mv ${prefix} ${prefix}${repID}

	echo "Finished rep " $repID
	repID=$(( $repID + 1 ))
done



