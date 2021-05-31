// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

#include "apsp.h"
#include <string.h>
#include <stdio.h>
#define BLOCK_SIZE 32
const int MAX_DISTANCE = 0x3f3f3f3f;
// Brute Force APSP Implementation:
__global__ void kernel(int n, int k, int *graph)
{
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n)
    {
        graph[i * n + j] = min(graph[i * n + j], graph[i * n + k] + graph[k * n + j]);
    }
}

__global__ void stage1(const int n, const int block, int *graph)
{
    __shared__ int cache_block[BLOCK_SIZE][BLOCK_SIZE];
    const int idx = threadIdx.x;
    const int idy = threadIdx.y;
    const int v1 = BLOCK_SIZE * block + idy;
    const int v2 = BLOCK_SIZE * block + idx;

    const int cell_idx = v1 * n + v2;
    int new_path;
    if (v1 < n && v2 < n)
    {
        cache_block[idy][idx] = graph[cell_idx];
    }
    else
    {
        cache_block[idy][idx] = MAX_DISTANCE;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k)
    {
        new_path = cache_block[idy][k] + cache_block[k][idx];
        // __syncthreads();
        cache_block[idy][idx] = new_path < cache_block[idy][idx] ? new_path : cache_block[idy][idx];
    }
    if (v1 < n && v2 < n)
    {
        graph[cell_idx] = cache_block[idy][idx];
    }
}

__global__ void stage2(const int n, const int block, int *graph)
{
    if (blockIdx.x == block)
        return;
    const int idx = threadIdx.x;
    const int idy = threadIdx.y;
    int v1 = BLOCK_SIZE * block + idy;
    int v2 = BLOCK_SIZE * block + idx;
    __shared__ int cache_center_block[BLOCK_SIZE][BLOCK_SIZE];
    int cell_idx = v1 * n + v2;
    if (v1 < n && v2 < n)
    {
        cache_center_block[idy][idx] = graph[cell_idx];
    }
    else
    {
        cache_center_block[idy][idx] = MAX_DISTANCE;
    }

    if (blockIdx.y == 0)
    {
        // i-aligned
        v2 = BLOCK_SIZE * blockIdx.x + idx;
    }
    else
    {
        // j-aligned
        v1 = BLOCK_SIZE * blockIdx.x + idy;
    }

    __shared__ int cache_current_block[BLOCK_SIZE][BLOCK_SIZE];
    int current_path;
    cell_idx = v1 * n + v2;
    if (v1 < n && v2 < n)
    {
        current_path = graph[cell_idx];
    }
    else
    {
        current_path = MAX_DISTANCE;
    }
    cache_current_block[idy][idx] = current_path;
    __syncthreads();

    int new_path;

    if (blockIdx.y == 0)
    {
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            new_path = cache_center_block[idy][k] + cache_current_block[k][idx];
            current_path = new_path < current_path ? new_path : current_path;
            cache_current_block[idy][idx] = current_path;
        }
    }
    else
    {
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            new_path = cache_current_block[idy][k] + cache_center_block[k][idx];
            current_path = new_path < current_path ? new_path : current_path;
            cache_current_block[idy][idx] = current_path;
        }
    }
    if (v1 < n && v2 < n)
    {
        graph[cell_idx] = current_path;
    }
}

__global__ void stage3(const int n, const int block, int *graph)
{
    if (blockIdx.x == block || blockIdx.y == block)
        return;
    const int idx = threadIdx.x;
    const int idy = threadIdx.y;
    const int v1 = blockDim.y * blockIdx.y + idy;
    const int v2 = blockDim.x * blockIdx.x + idx;
    __shared__ int cache_center_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cache_center_col[BLOCK_SIZE][BLOCK_SIZE];

    int v1_row = BLOCK_SIZE * block + idy;
    int v2_col = BLOCK_SIZE * block + idx;
    int cell_idx;
    if (v1_row < n && v2 < n)
    {
        cell_idx = v1_row * n + v2;
        cache_center_row[idy][idx] = graph[cell_idx];
    }
    else
    {
        cache_center_row[idy][idx] = MAX_DISTANCE;
    }

    if (v1 < n && v2_col < n)
    {
        cell_idx = v1 * n + v2_col;
        cache_center_col[idy][idx] = graph[cell_idx];
    }
    else
    {
        cache_center_col[idy][idx] = MAX_DISTANCE;
    }

    __syncthreads();
    int current_path, new_path;
    if (v1 < n && v2 < n)
    {
        cell_idx = v1 * n + v2;
        current_path = graph[cell_idx];
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            new_path = cache_center_col[idy][k] + cache_center_row[k][idx];
            current_path = new_path < current_path ? new_path : current_path;
        }
        graph[cell_idx] = current_path;
    }
}

// void print_graph(int n, int *graph_device)
// {
//     int *graph_host = (int *)malloc(n * n * sizeof(int));
//     cudaMemcpy(graph_host, graph_device, n * n * sizeof(int), cudaMemcpyDeviceToHost);
//     printf("---\n");
//     for (int i = 0; i < n; ++i)
//     {
//         for (int j = 0; j < n; ++j)
//         {
//             printf("%d ", graph_host[i * n + j]);
//         }
//         printf("\n");
//     }
//     printf("---\n");
//     free(graph_host);
// }

void apsp(int n, /* device */ int *graph)
{
    int num_block = (n - 1) / BLOCK_SIZE + 1;
    dim3 grid_stage1(1, 1, 1);
    dim3 grid_stage2(num_block, 2, 1);
    dim3 grid_stage3(num_block, num_block, 1);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
    // print_graph(n, graph);
    for (int block = 0; block < num_block; ++block)
    {

        stage1<<<grid_stage1, block_size>>>(n, block, graph);
        // printf("Block id = %d, Stage = 1\n", block);
        // print_graph(n, graph);
        stage2<<<grid_stage2, block_size>>>(n, block, graph);
        // printf("Block id = %d, Stage = 2\n", block);
        // print_graph(n, graph);
        stage3<<<grid_stage3, block_size>>>(n, block, graph);
        // printf("Block id = %d, Stage = 3\n", block);
        // print_graph(n, graph);
    }
}

// void apsp(int n, /* device */ int *graph)
// {
//     for (int k = 0; k < n; k++)
//     {
//         dim3 thr(32, 32);
//         dim3 blk((n - 1) / 32 + 1, (n - 1) / 32 + 1);
//         kernel<<<blk, thr>>>(n, k, graph);
//     }
// }
