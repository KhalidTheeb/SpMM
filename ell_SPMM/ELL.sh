#!/bin/bash
#SBATCH -J eLLDML_Geer           # job name
#SBATCH -o eLLDML_Geer.o%j       # output and error file name (%j expands to jobID)
#SBATCH -p gpu     # queue (partition) -- normal, development, etc.
#SBATCH -n 1              # total number of mpi tasks requested
#SBATCH -t 01:10:00        # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=khalidtheeb@gmail.com
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH -A TG-ASC130053

module load cuda
#nvcc -arch=compute_35 -code=sm_35 -o ELL ELL.cu mmio.c 
./ELLD2.out ./Mats/Dong/ML_Geer.mtx
./ELLD4.out ./Mats/Dong/ML_Geer.mtx
./ELLD63.out ./Mats/Dong/ML_Geer.mtx
./ELLD66.out ./Mats/Dong/ML_Geer.mtx
./ELLD84.out ./Mats/Dong/ML_Geer.mtx
./ELLD88.out ./Mats/Dong/ML_Geer.mtx
./ELLD105.out ./Mats/Dong/ML_Geer.mtx
./ELLD1010.out ./Mats/Dong/ML_Geer.mtx

