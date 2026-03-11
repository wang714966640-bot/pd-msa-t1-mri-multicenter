#! /bin/bash

dir=/media/neurox/RuiJin_PD/raw/ruijin

# for grp in `cat ${dir}/subjects.list`
# do
# 	cd ${dir}/${grp}

# 	for sub in `cat ${dir}/${grp}/subjects.list`
# 	do
# 		echo "############### now processing: ${grp}/${sub} ##################"
# 		cd ${dir}/${grp}/${sub}
# 		dcm2niix ./
# 	done

# done

for grp in `cat ${dir}/subjects.list`
do
	cd ${dir}/${grp}

	for sub in `cat ${dir}/${grp}/subjects.list`
	do
 		mkdir -p /media/neurox/RuiJin_PD/data/${grp}/${sub}/raw
 		mv ${dir}/${grp}/${sub}/*.* /media/neurox/RuiJin_PD/data/${grp}/${sub}/raw
	done

done
