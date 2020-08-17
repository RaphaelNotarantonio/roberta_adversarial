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


cd ${SLURM_SUBMIT_DIR}

module purge
module load pytorch-gpu/py3/1.4.0 
pip install transformers
pip install advertorch


python ./attack.py  --iid ${'0'} & --indlist ${'[5,8]'} & --eps ${'0.03'} & --epscand ${'0.03'} & --nb_iter ${'100'} & --eps_iter ${'0.5'} & --rayon ${'0.4'} & --ord ${'np.inf'} &
