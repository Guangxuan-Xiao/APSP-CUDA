# APSP-CUDA
CUDA APSP

## 实现方法

算法上就是按照助教提供的Multi-stage算法实现，有一些细节不确定于是我去读了原论文（见参考文献1和2）。算法分为三步，每步一个kernel，代码上对应stage1~3，写的比较清楚，以下分别介绍三个stage和主函数。

### 主循环

```c++
void apsp(int n, /* device */ int *graph)
{
    int num_block = (n - 1) / BLOCK_SIZE + 1;
    dim3 grid_stage1(1, 1, 1);
    dim3 grid_stage2(num_block, 2, 1);
    dim3 grid_stage3(num_block, num_block, 1);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
    for (int block = 0; block < num_block; ++block)
    {
        stage1<<<grid_stage1, block_size>>>(n, block, graph);
        stage2<<<grid_stage2, block_size>>>(n, block, graph);
        stage3<<<grid_stage3, block_size>>>(n, block, graph);
    }
}
```

将邻接矩阵划分为大小为每块为$32\times32$的分块矩阵，外循环执行$\lceil \frac{n}{32}\rceil$步，每一步内执行三个阶段，依次处理中心块（只依赖于自己），十字块（依赖于自己和中心块），周围块（依赖于十字块）。由于每个线程块最多能开1024个线程，因此块的大小就设为了$32\times32$，每个线程负责处理分块子矩阵中的一个元素。

第一个stage只需要处理一个子矩阵，因此只用一个grid，对应`grid_stage1`。

第二个stage需要处理$2\lceil \frac{n}{32}\rceil - 1$个子矩阵，因此分配了$2\lceil \frac{n}{32}\rceil$个grid，对应`grid_stage2`，在stage2 kernel内部判断不处理中心块。

第三个stage需要处理$(\lceil \frac{n}{32}\rceil-1)^2$个子矩阵，因此分配了$\lceil \frac{n}{32}\rceil^2$个grid，对应`grid_stage2`，在stage3 kernel内部判断不处理中心块和十字块。

### Stage 1

```c++
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
        cache_block[idy][idx] = new_path < cache_block[idy][idx] ? new_path : cache_block[idy][idx];
    }
    if (v1 < n && v2 < n)
    {
        graph[cell_idx] = cache_block[idy][idx];
    }
}
```

Stage1的功能是在中心块内部执行Floyd-Warshall算法，数据依赖仅是这个块本身，因此可以将这个块整体cache到shared memory中，块内的之后所有访问就能加速很多。

在`__syncthreads()`之前的代码就是每个线程分工将自己对应元素的矩阵数值载入到shared memory中，之后调用`__syncthreads()`，保证整个块都被载入完毕后，执行块内的FW算法。

块内的FW算法的循环次数为常数32，因此可以用循环展开加速，这对于GPU执行的代码很重要，对应`#pragma unroll`。

最后每个线程算完之后，更新global memory中的数值。

### Stage 2

```c++
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
```

Stage2的功能是在十字块执行Floyd-Warshall算法，数据依赖是中心块和十字块自己，因此需要cache的块有两个，一个是中心块`cache_center_block`，一个是十字块自己`cache_current_block`。

第一个`if`是判断是否是中心块自己，如果是就跳过。

之后在第一个`__syncthreads()`之前做的事就是把中心块和十字块自己cache进shared memory，最后sync一下保证数据准备完毕。

设定`blockIdx.y`为0时，对应横着的十字块们；`blockIdx.y`为1时，对应竖着的十字块们；接下来要做的就是在每个块内进行FW算法，同样进行循环展开，算得快。

最后每个线程算完之后，更新global memory中的数值。

### Stage 3

```c++
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
```

Stage3的功能是在周围块执行Floyd-Warshall算法，数据依赖是周围块对应的纵横两个十字块，因此需要cache的块有两个，一个是同行的十字块`cache_center_row`，一个是同列的`cache_current_col`。

第一个`if`是判断是否是中心块或十字块自己，如果是就跳过。

之后在第一个`__syncthreads()`之前做的事就是把两个十字块cache进shared memory，最后sync一下保证数据准备完毕。

然后就是循环展开进行FW算法，用两个依赖的块算`new_path`，更新寄存器中的current_path，最后写回global memory。

## 性能汇报

| n           | 1000         | 2500         | 5000         | 7500         | 10000        |
| ----------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| Baseline    | 14.6406      | 377.1215     | 2970.642     | 10015.69     | 22615.13     |
| **Mine**    | **1.915569** | **21.79991** | **149.0712** | **496.0596** | **1163.741** |
| **Speedup** | **7.64x**    | **17.3x**    | **19.93x**   | **20.19x**   | **19.43x**   |

以上汇报时间单位均为毫秒，可见实现的方法获得了稳定超过7的加速比，在最大的图上获得了超过19的加速比。

## 参考

1. "A Multi-Stage CUDA Kernel for Floyd-Warshall" (Ben Lund, Justin W. Smith)

2. Katz, Gary J., and Joseph T. Kider Jr. "All-pairs shortest-paths for large graphs on the GPU." In Proceedings of the 23rd ACM SIGGRAPH/EUROGRAPHICS symposium on Graphics hardware, pp. 47-55. Eurographics Association, 2008.

3. https://github.com/MTB90/cuda-floyd_warshall
