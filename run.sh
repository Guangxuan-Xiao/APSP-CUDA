source /home/spack/spack/share/spack/setup-env.sh
spack load cuda
make
srun -N 1 ./benchmark 1000