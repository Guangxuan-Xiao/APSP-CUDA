source /home/spack/spack/share/spack/setup-env.sh
spack load cuda
make
for size in 1000 2500 5000 7500 10000; do
    srun -N 1 ./benchmark $size
done
