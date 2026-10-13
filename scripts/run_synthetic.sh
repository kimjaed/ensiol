#!/bin/bash
#SBATCH --job-name=EnsIOL_synthetic
#SBATCH --output=EnsIOL_synthetic_%j.out
#SBATCH --error=EnsIOL_synthetic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # one sample per worker process
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --partition=compute
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=YOUR_EMAIL

# Submit from the repository root:  sbatch scripts/run_synthetic.sh [extra arguments]
# Extra arguments are passed to run_synthetic.py (see its --help).
export HOMEDIR=/user/$USER
cd "$SLURM_SUBMIT_DIR"

ml load miniconda/25.9
PYTHON="$HOMEDIR/.conda/envs/mtpyv2-env/bin/python"       # CONFIRM with `conda env list`
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$PYTHONPATH"   # use the checkout without installing it

# One worker process per sample, each single-threaded: stop BLAS from also
# multithreading inside each worker and oversubscribing the allocated cores.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Finished items are skipped on resubmission, so if the job hits the time
# limit, just submit it again. -u: unbuffered, so progress appears as it happens.
"$PYTHON" -u scripts/run_synthetic.py --data-dir data/kilauea --out-dir outputs/synthetic "$@"
