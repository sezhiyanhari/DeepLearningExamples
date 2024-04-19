numactl --cpunodebind=3 --membind=3 ./numa_cuda 0 &
numactl --cpunodebind=3 --membind=3 ./numa_cuda 1 &
numactl --cpunodebind=1 --membind=1 ./numa_cuda 2 &
numactl --cpunodebind=1 --membind=1 ./numa_cuda 3 &
numactl --cpunodebind=7 --membind=7 ./numa_cuda 4 &
numactl --cpunodebind=7 --membind=7 ./numa_cuda 5 &
numactl --cpunodebind=5 --membind=5 ./numa_cuda 6 &
numactl --cpunodebind=5 --membind=5 ./numa_cuda 7 &