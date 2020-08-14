#!/bin/bash
#SBATCH --job-name=pgd attack        # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=3:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=pgd_attack%j.out # output file name
#SBATCH --error=pgd_attack%j.err  # error file name
#SBATCH --array=1-10


cd ${SLURM_SUBMIT_DIR}

module purge
module load pytorch-gpu/py3/1.4.0 

IID_ENUMER=('0' '1' '2' '3' '4' '5' '6' '7' '8' '9') 
python ./attack.py  --iid ${IID_ENUMER[$SLURM_ARRAY_TASK_ID]} &
