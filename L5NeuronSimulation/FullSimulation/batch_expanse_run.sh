#!/bin/bash

#SBATCH --partition shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=umc113
#SBATCH --job-name=run
#SBATCH --output=run.out
#SBATCH --mem=8G
#SBATCH --time 0-48:00

module purge
module load slurm
module load cpu
module load intel
module load intel-mpi
module load ncurses
export PATH=$HOME/nrn/x86_64/bin:$PATH
export LD_LIBRARY_PATH=$HOME/nrn/x86_64/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/nrn/lib/python:$PYTHONPATH
export PATH=$HOME/nrn/x86_64/bin:$PATH

rm -rf output
rm exc_stim_spikes2.h5

echo "Running model at $(date)"

#mpirun nrniv -mpi -quiet -python3 run_network.py simulation_config.json
#ibrun nrniv -mpi -python run_network.py
python modulate_exc.py
python run_network.py

echo "Done running model at $(date)"
