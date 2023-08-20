#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a job on Compute Canada cluster.
# ---------------------------------------------------------------------
#SBATCH --account=def-pbranco
#SBATCH --nodes=4
#SBATCH --cpus-per-task=40
#SBATCH --mem=35G
#SBATCH --time=3-0:0

# Emails me when job starts, ends or fails
#SBATCH --mail-user=basar092@uottawa.ca
#SBATCH --mail-type=ALL

#SBATCH --job-name=additive-batch
#SBATCH --output=outputs/output_additive-%j.out
# ---------------------------------------------------------------------
echo "Current working directory: $(pwd)"
echo "Starting run at: $(date)"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------
# Run simulation step here...
module load python/3.9.6
source ~/projects/def-pbranco/baasare/federated_learning/venv/bin/activate

python ~/projects/def-pbranco/baasare/federated_learning/seq_client_additive.py
python ~/projects/def-pbranco/baasare/federated_learning/seq_client_additive_groups.py
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: $(date)"
