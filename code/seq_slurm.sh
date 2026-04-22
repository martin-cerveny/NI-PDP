#!/bin/bash
#SBATCH --output="%x-%J.out"
#SBATCH --error="%x-%J.err"



# $1 = binárka, $2 = vstupní soubor
srun "$1" "$3" < "$2"

exit 0
