

#include <cuda_runtime.h>
#include <stdio.h>

// Enumerate the supported torch numerical dtypes
enum class torch_dtype {
    float32 = 0,
    float64 = 1
};

// Error-checking macro
#define CHECK_CUDA_ERROR(call) {                                \
    const cudaError_t error = call;                             \
    if (error != cudaSuccess) {                                 \
        printf("Error: %s:%d, ", __FILE__, __LINE__);           \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);                                                \
    }                                                           \
};


// =================================================================================================
// ===================================== CUDA kernel functions =====================================
// =================================================================================================

template <typename numtype>
__global__ void segcumsum_kernel(numtype* values, const int64_t* segment_ids, int64_t size, int64_t max_seg_size, numtype* block_sums_out, int64_t* block_last_ids_out, bool return_next_level) {

    extern __shared__ char shared_mem[]; // Unnamed extern shared memory
    numtype* shared_data = reinterpret_cast<numtype*>(shared_mem); // Cast to appropriate type

    int64_t tid = threadIdx.x;
    int64_t index = blockIdx.x * blockDim.x + tid;
    bool ok = (index < size);

    int64_t id_curr;    

    // Load input into shared memory
    if (ok) {
        shared_data[tid] = values[index];
        id_curr = segment_ids[index];
    } 

    // Necessary to ensure all threads have loaded data into shared memory
    __syncthreads();

    // Inclusive scan (cumulative sum) within each block
    int64_t stride_limit = min(max_seg_size, (int64_t)blockDim.x);

    bool stop = false;
    
    // if not ok, the loop will not run
    for (int64_t stride = 1; stride < stride_limit; stride *= 2) {
        numtype temp;
        if ((!ok) || (stop) || (stride > tid)) {
            temp = 0.0;
        }
        else
        {            
            int64_t id_lookback = segment_ids[index-stride];
            if (id_curr != id_lookback)
            {
                temp = 0.0;                
                stop = true;
            }
            else
                temp = shared_data[tid - stride];
        }

        // Necessary to ensure all threads have computed their temp values
        __syncthreads();

        if (ok)
           shared_data[tid] += temp; 

        // Necessary to ensure all threads have updated their shared_data values   
        __syncthreads();
    }

    // Write results to output
    if (ok) {
        values[index] = shared_data[tid];
    }

    // Write block sums
    if (return_next_level && (tid == blockDim.x - 1) && (ok)) {
        block_sums_out[blockIdx.x] = shared_data[tid];
        block_last_ids_out[blockIdx.x] = id_curr;
    }
}



template <typename numtype>
__global__ void add_block_sums_kernel(numtype* output, const numtype* block_sums, const int64_t* segment_ids, const int64_t* block_last_id, int64_t size) {

    int64_t tid = threadIdx.x;
    int64_t index = blockIdx.x * blockDim.x + tid;
    bool ok = (index < size);

    int64_t id_curr;

    if (ok)
        id_curr = segment_ids[index];

    if ((ok) && (blockIdx.x >= 1) && (block_last_id[blockIdx.x-1] == id_curr))
        output[index] += block_sums[blockIdx.x-1];

}


// =================================================================================================
// =============================== Instantiations for specific types ===============================
// =================================================================================================

extern "C" {
    void launch_segcumsum_kernel_float(
        float* values,
        const int64_t* segment_ids,
        int64_t size,
        int64_t max_seg_size,
        float* block_sums_out,
        int64_t* block_last_ids_out,
        bool return_next_level,
        int64_t num_blocks,
        int64_t threads_per_block,
        int64_t shared_memory_size
    ) {
        segcumsum_kernel<<<num_blocks, threads_per_block, shared_memory_size>>>(
            values, segment_ids, size, max_seg_size, block_sums_out, block_last_ids_out, return_next_level
        );
    }

    void launch_segcumsum_kernel_double(
        double* values,
        const int64_t* segment_ids,
        int64_t size,
        int64_t max_seg_size,
        double* block_sums_out,
        int64_t* block_last_ids_out,
        bool return_next_level,
        int64_t num_blocks,
        int64_t threads_per_block,
        int64_t shared_memory_size
    ) {
        segcumsum_kernel<<<num_blocks, threads_per_block, shared_memory_size>>>(
            values, segment_ids, size, max_seg_size, block_sums_out, block_last_ids_out, return_next_level
        );
    }

    void launch_add_block_sums_kernel_float(
        float* output,
        const float* block_sums,
        const int64_t* segment_ids, 
        const int64_t* block_last_id, 
        int64_t size,
        int64_t num_blocks,
        int64_t threads_per_block
    ) {
        add_block_sums_kernel<<<num_blocks, threads_per_block>>>(output, block_sums, segment_ids, block_last_id, size);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    void launch_add_block_sums_kernel_double(
        double* output,
        const double* block_sums,
        const int64_t* segment_ids, 
        const int64_t* block_last_id, 
        int64_t size,
        int64_t num_blocks,
        int64_t threads_per_block
    ) {
        add_block_sums_kernel<<<num_blocks, threads_per_block>>>(output, block_sums, segment_ids, block_last_id, size);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

}




// =================================================================================================
// ========================== Wrapper functions to be called from Python ===========================
// =================================================================================================

extern "C" void segcumsum_wrapper(torch_dtype dtype, void* values, const int64_t* segment_ids, int64_t size, int64_t max_seg_size, void* block_sums_out, int64_t* block_last_ids_out, bool return_next_level, int64_t num_blocks, int64_t threads_per_block, size_t shared_memory_size) {
    // segcumsum_kernel<<<num_blocks, threads_per_block, shared_memory_size>>>(values, segment_ids, size, max_seg_size, block_sums_out, block_last_ids_out, return_next_level);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    switch (dtype) {
        case torch_dtype::float32:
            launch_segcumsum_kernel_float((float*)values, segment_ids, size, max_seg_size, (float*)block_sums_out, block_last_ids_out, return_next_level, num_blocks, threads_per_block, shared_memory_size); 
            break;
        case torch_dtype::float64:
            launch_segcumsum_kernel_double((double*)values, segment_ids, size, max_seg_size, (double*)block_sums_out, block_last_ids_out, return_next_level, num_blocks, threads_per_block, shared_memory_size); 
            break;
    }
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure the kernel has finished
}


extern "C" void add_block_sums_wrapper(torch_dtype dtype, void* output, const void* block_sums, const int64_t* segment_ids, const int64_t* block_last_id, int64_t size, int64_t num_blocks, int64_t threads_per_block) {
    //add_block_sums_kernel<<<num_blocks, threads_per_block>>>(output, block_sums, segment_ids, block_last_id, size);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    switch (dtype) {
        case torch_dtype::float32:
            launch_add_block_sums_kernel_float((float*) output, (const float*) block_sums, segment_ids, block_last_id, size, num_blocks, threads_per_block);
            break;

        case torch_dtype::float64:
            launch_add_block_sums_kernel_double((double*) output, (const double*) block_sums, segment_ids, block_last_id, size, num_blocks, threads_per_block);
            break;
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure the kernel has finished
}


extern "C" int get_max_threads_per_block(int device_index) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_index);
    
    return deviceProp.maxThreadsPerBlock;
}
