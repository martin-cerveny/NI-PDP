#!/bin/bash

module load cray-mvapich2_pmix_nogpu

# Check if required directories exist
if [ ! -d "mapa" ] || [ ! -d "out" ]; then
    echo "Error: 'mapa/' and 'out/' directories must exist."
    exit 1
fi

# Iterate through all input files in the mapa/ directory
for input in measure-maps/; do
    if [ -f "$input" ]; then
        filename=$(basename "$input")
        echo "Scheduling measurements for: $filename"

        # 1. Sequential application (arm_serial queue)
        sbatch -p arm_serial --job-name="seq_${filename}" \
              seq_slurm.sh "out/seq.out" "$input"

	for threads in {1,2,4,12,16,32,48}; do
        # 2. OMP Task application (arm_fast queue, fixed 48 CPUs)
        sbatch -p arm_serial -c 48 --job-name="ompt_${filename}" \
               seq_slurm.sh "out/omp-task.out" "$input" "$threads"

        # 3. OMP Data application (arm_fast queue, fixed 48 CPUs)
        sbatch -p arm_serial -c 48 --job-name="ompd_${filename}" \
               seq_slurm.sh "out/omp-data.out" "$input" "$threads"

	done
        # 4. MPI + OMP application (arm_fast queue, 4 nodes, fixed 48 CPUs)
        sbatch -p arm_fast -N 4 -c 48 --job-name="mpi_${filename}" \
               parallel_slurm.sh "out/mpi-omp.out" "$input" 48


        sbatch -p arm_fast -N 5 -c 48 --job-name="mpi_${filename}" \
              parallel_slurm.sh "out/mpi-omp.out" "$input" 48

    fi
done

echo "Done. All jobs have been submitted to the queue."
