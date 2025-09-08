#!/bin/sh
#SBATCH -J Steps                                # Job name
#SBATCH -N 1                                    # Nodes requested
#SBATCH -n 1                                    # Tasks requested
#SBATCH --exclusive                             # No other jobs can share nodes with this job
#SBATCH -t 6:00:00                              # Time requested in hour:minute:second
#SBATCH --output=output/output/output_%j.txt    # Output file
#SBATCH --error=output/error/error_%j.txt       # Error file

. ../venv/bin/activate
module load Python/3.11.5-GCCcore-13.2.0
python3 -m pip install .. --upgrade
python3 RastriginStepSizes.py $*
