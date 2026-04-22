#! /bin/bash

mkdir -p out

module load cray-mvapich2_pmix_nogpu

CC cpp/seq.cpp -o out/seq.out -std=c++17 -O3

CC cpp/omp-task.cpp -o out/omp-task.out -fopenmp -std=c++17 -O3

CC cpp/omp-data.cpp -o out/omp-data.out -fopenmp -std=c++17 -O3

CC cpp/mpi-omp.cpp -o out/mpi-omp.out -fopenmp -std=c++17 -O3
