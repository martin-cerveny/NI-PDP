#!/bin/bash
#SBATCH --output="%x-%J.out"
#SBATCH --error="%x-%J.err"
#SBATCH --exclusive

# Aktivace prostředí
source /etc/profile.d/zz-cray-pe.sh
module load cray-mvapich2_pmix_nogpu

export MV2_HOMOGENEOUS_CLUSTER=1
export MV2_SUPPRESS_JOB_STARTUP_PERFORMANCE_WARNING=1
export MV2_ENABLE_AFFINITY=0
export MV2_USE_THREAD_WARNING=0

srun "$1" "$3" < "$2"

exit 0
