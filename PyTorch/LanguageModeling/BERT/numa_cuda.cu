#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

__global__ void init_kernel(float *data, size_t size) {
    size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        data[index] = index * 1.0f;
    }
}

int main(int argc, char *argv[]) {
    // const int num_elements = 1024;
    // const int num_elements = 1879048192;
    const size_t num_elements = 18790481920;
    const size_t size = num_elements * sizeof(float);

    // Set CUDA device to 0
    cudaSetDevice(atoi(argv[1]));

    // Allocate and initialize memory on the GPU
    float *gpu_data;
    cudaMalloc(&gpu_data, size);

    cudaDeviceSynchronize();
    std::cout << "Buffer allocated on GPU " << atoi(argv[1]) << std::endl;

    init_kernel<<<(num_elements + 255) / 256, 256>>>(gpu_data, num_elements);

    cudaDeviceSynchronize();
    std::cout << "init_kernel synced" << std::endl;

    // Allocate pinned host memory
    float *cpu_data;
    cudaHostAlloc(&cpu_data, size, cudaHostAllocDefault);

    cudaDeviceSynchronize();
    std::cout << "host memory allocated" << std::endl;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Copy data from GPU to pinned host memory
    cudaMemcpy(cpu_data, gpu_data, size, cudaMemcpyDeviceToHost);

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time taken for cudaMemcpy: " << milliseconds << " milliseconds. For GPU " << argv[1] << std::endl;

    // Free memory
    cudaFree(gpu_data);
    cudaFreeHost(cpu_data);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}