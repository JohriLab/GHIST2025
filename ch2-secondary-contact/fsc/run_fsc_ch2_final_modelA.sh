#!/bin/bash
#SBATCH --mail-user=pjohri@unc.edu
#SBATCH --mail-type=ALL
#SBATCH -p general
#SBATCH -n 1 #number of tasks
#SBATCH --time=0-24:00
#SBATCH --mem=50m
#SBATCH -o /nas/longleaf/home/pjohri/LOGFILES/fsc_ch2_modelA_%j.out
#SBATCH -e /nas/longleaf/home/pjohri/LOGFILES/fsc_ch2_modelA_%j.err

#run this in the relevant directory
prefix="secondary_contact_inst_growth"
mkdir /nas/longleaf/home/pjohri/ghist2025/challenge2_final/${prefix}
cd /nas/longleaf/home/pjohri/ghist2025/challenge2_final/${prefix}

#copy the relevant files into the directory
cp /nas/longleaf/home/pjohri/bin/fsc28_linux64/fsc28 ./
cp /proj/johrilab/projects/ghist_2025/intermediate_files/GHIST2025_secondary_contact_final_jointSFS.obs ${prefix}_jointDAFpop1_0.obs
cp /nas/longleaf/home/pjohri/ghist2025/fsc/isolation_then_migration_inst_growth.est ${prefix}.est
cp /nas/longleaf/home/pjohri/ghist2025/fsc/isolation_then_migration_inst_growth.tpl ${prefix}.tpl

#run it 10 times:
declare -i repID=1
while [ $repID -lt 11 ];
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



