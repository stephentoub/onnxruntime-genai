// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <limits>
#include "beam_search_topk.h"

namespace Generators {
namespace cuda {

// kernel to compute the top k on last axis for tensor with shape: [batch, beam_size, parts_of_vocab, vocab_part_size]
// Its grid is [batch * beam_size, parts_of_vocab]
template <typename T, int max_k, int thread_block_size>
__launch_bounds__(thread_block_size) __global__ void BeamSearchOnlineTopKStage1Kernel(
    const T* input,
    int32_t k,
    int32_t vocab_size,
    int32_t vocab_part_size,
    T* output_values,
    int32_t* output_token) {
  TopK<T, max_k> top_k_thread;
  top_k_thread.Init();

  int batch_beam = blockIdx.x;
  int voc_part_id = blockIdx.y;

  int token_id_base = voc_part_id * vocab_part_size;
  const T* input_block = input + batch_beam * vocab_size;
  // voc_part_size
  for (int i = threadIdx.x + token_id_base; i < vocab_part_size + token_id_base; i += blockDim.x) {
    if (i < vocab_size) {
      top_k_thread.Insert(input_block[i], i);
    }
  }

  // reduce in thread block
  typedef cub::BlockReduce<TopK<T, max_k>, thread_block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  TopK<T, max_k> top_k_block = BlockReduce(temp_storage).Reduce(top_k_thread, reduce_topk_op<T, max_k>);
  __syncthreads();

  output_values += batch_beam * gridDim.y * k + voc_part_id * k;
  output_token += batch_beam * gridDim.y * k + voc_part_id * k;
  if (threadIdx.x == 0) {
    for (int i = 0; i < k; i++) {
      output_values[i] = top_k_block.value[i];
      output_token[i] = top_k_block.key[i];
    }
  }
}

template <typename T, int max_k, int thread_block_size>
__launch_bounds__(thread_block_size) __global__ void BeamSearchOnlineTopKStage2Kernel(
    const T* input_values,
    const int32_t* input_tokens,
    int32_t k,
    int32_t vocab_size,
    int32_t parts_per_beam,
    T* output_values,
    int32_t* output_indices) {
  const int vector_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  extern __shared__ char shared_buf_extern[];
  T* value_shared_buf = reinterpret_cast<T*>(shared_buf_extern);
  int32_t* tokens_shared_buf =
      reinterpret_cast<int32_t*>(shared_buf_extern + max_k * parts_per_beam * sizeof(int32_t));

  typedef cub::BlockReduce<TopK<T, max_k>, thread_block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  input_values += vector_id * k * parts_per_beam;
  input_tokens += vector_id * k * parts_per_beam;

  TopK<T, max_k> thread_topk;
  thread_topk.Init();

  for (int idx = thread_id; idx < k * parts_per_beam; idx += thread_block_size) {
    value_shared_buf[idx] = input_values[idx];
    tokens_shared_buf[idx] = input_tokens[idx];
  }
  __syncthreads();

  if (thread_id < parts_per_beam) {
    T* b_v = value_shared_buf + thread_id * k;
    int32_t* b_i = tokens_shared_buf + thread_id * k;
    for (int i = 0; i < k; i++) {
      thread_topk.Insert(b_v[i], b_i[i]);
    }
  }

  TopK<T, max_k> topk_block = BlockReduce(temp_storage).Reduce(thread_topk, reduce_topk_op<T, max_k>);

  if (thread_id == 0) {
    output_values += vector_id * k;
    output_indices += vector_id * k;

    for (int i = 0; i < k; ++i) {
      if (i < k) {
        output_values[i] = topk_block.value[i];
        output_indices[i] = topk_block.key[i];
      }
    }
  }
}

template <typename T, int max_k>
void LaunchBeamSearchOnlineTopKStage2Kernel(
    const T* topk_values_tmp,
    const int32_t* topk_indices_tmp,
    int32_t batch_beam_size,
    int32_t vocab_size,
    int32_t parts_per_beam,
    int32_t K,
    T* output_values,
    int32_t* output_indices,
    cudaStream_t stream) {
  assert(parts_per_beam <= 128);  // Parts per beam should not be greater than 128

  int smem_stage2_size = parts_per_beam * max_k * 2 * sizeof(int32_t);

  if (parts_per_beam <= 32) {
    BeamSearchOnlineTopKStage2Kernel<T, max_k, 32><<<batch_beam_size, 32, smem_stage2_size, stream>>>(
        topk_values_tmp, topk_indices_tmp, K, vocab_size, parts_per_beam, output_values, output_indices);
    return;
  }

  if (parts_per_beam <= 64) {
    BeamSearchOnlineTopKStage2Kernel<T, max_k, 64><<<batch_beam_size, 64, smem_stage2_size, stream>>>(
        topk_values_tmp, topk_indices_tmp, K, vocab_size, parts_per_beam, output_values, output_indices);
    return;
  }

  BeamSearchOnlineTopKStage2Kernel<T, max_k, 128><<<batch_beam_size, 128, smem_stage2_size, stream>>>(
      topk_values_tmp, topk_indices_tmp, K, vocab_size, parts_per_beam, output_values, output_indices);
  return;
}

template <typename T, int max_k>
void TopKLauncherMaxK(
    const T* input,
    int batch_beam_size,
    int vocab_size,
    int K,
    T* output_values,
    int32_t* output_indices,
    T* output_values_tmp,
    int32_t* output_indices_tmp,
    cudaStream_t stream) {
  constexpr int kThreadBlockSize = (max_k < 16) ? (max_k < 8) ? 256 : 128 : 64;

  int voc_parts = 4;
  if (batch_beam_size < 256) {
    // volta has 80 SMs, so we aim for three waves
    voc_parts = (240 + batch_beam_size - 1) / batch_beam_size;
    voc_parts = std::min(128, voc_parts);  // we implement up to 128
  }

  dim3 grid(batch_beam_size, voc_parts);

  cudaFuncSetAttribute(BeamSearchOnlineTopKStage1Kernel<T, max_k, kThreadBlockSize>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxL1);

  BeamSearchOnlineTopKStage1Kernel<T, max_k, kThreadBlockSize>
      <<<grid, kThreadBlockSize, 0, stream>>>(input, K, vocab_size, (vocab_size + voc_parts - 1) / voc_parts, output_values_tmp, output_indices_tmp);

  LaunchBeamSearchOnlineTopKStage2Kernel<T, max_k>(
      output_values_tmp,
      output_indices_tmp,
      batch_beam_size,
      vocab_size,
      voc_parts,
      K,
      output_values,
      output_indices,
      stream);
}

template <typename T, typename I, int32_t max_k, int32_t thread_block_size>
__launch_bounds__(thread_block_size) __global__ void BatchTopKKernel(
    const T* topk_scores,
    const I* topk_tokens,
    int32_t* next_indices,
    int32_t* next_tokens,
    T* next_scores,
    int32_t batch_size,
    int32_t num_beams,
    int32_t k) {
  int thread_id = threadIdx.x;
  int block_id = blockIdx.x;
  TopK<T, max_k> thread_topk;
  if (thread_id == 0) {
    thread_topk.Init();

    int index_block = block_id * num_beams * k;
    for (int32_t i = 0; i < num_beams * k; i++) {
      thread_topk.Insert(topk_scores[index_block + i], index_block + i);
    }

    int index_next = block_id * k;
    for (int i = 0; i < k; i++) {
      next_tokens[index_next + i] = topk_tokens[thread_topk.key[i]];
      next_indices[index_next + i] = (thread_topk.key[i] - index_block) / k;
      next_scores[index_next + i] = thread_topk.value[i];
    }
  }
}

template <typename T, typename I>
void LaunchBatchTopKKernel(const T* topk_scores,
                           const I* topk_tokens,
                           int32_t* next_indices,
                           int32_t* next_tokens,
                           T* next_scores,
                           int32_t batch_size,
                           int32_t num_beams,
                           int32_t k,
                           cudaStream_t stream) {
  assert(k <= 64);  // LaunchBatchTopKKernel doesn't support k >= 64

#define BatchTopKKernelLauncher(K)                                          \
  BatchTopKKernel<T, I, K, 32><<<batch_size, 32, 0, stream>>>(topk_scores,  \
                                                              topk_tokens,  \
                                                              next_indices, \
                                                              next_tokens,  \
                                                              next_scores,  \
                                                              batch_size,   \
                                                              num_beams,    \
                                                              k);

  if (k <= 4) {
    BatchTopKKernelLauncher(4);
  } else if (k <= 8) {
    BatchTopKKernelLauncher(8);
  } else if (k <= 16) {
    BatchTopKKernelLauncher(16);
  } else if (k <= 32) {
    BatchTopKKernelLauncher(32);
  } else {
    BatchTopKKernelLauncher(64);
  }
}

template void LaunchBatchTopKKernel(const float* topk_scores,
                                    const int32_t* topk_tokens,
                                    int32_t* next_indices,
                                    int32_t* next_tokens,
                                    float* next_scores,
                                    int32_t batch_size,
                                    int32_t num_beams,
                                    int32_t k,
                                    cudaStream_t stream);

void BeamSearchTopK(
    const float* input,
    int32_t batch_size,
    int32_t num_beams,
    int32_t vocab_size,
    int32_t k,
    float* tmp_values_1st_stage,
    int32_t* tmp_indices_1st_stage,
    float* tmp_values_2nd_stage,
    int32_t* tmp_indices_2nd_stage,
    float* output_values,
    int32_t* output_tokens,
    int32_t* output_indices,
    cudaStream_t stream) {
  assert(k <= 64);  // BeamSearchTopK doesn't support k > 64

#define TopKLauncher(K)                               \
  TopKLauncherMaxK<float, K>(input,                   \
                             batch_size * num_beams,  \
                             vocab_size,              \
                             k, tmp_values_2nd_stage, \
                             tmp_indices_2nd_stage,   \
                             tmp_values_1st_stage,    \
                             tmp_indices_1st_stage,   \
                             stream)

  if (k <= 4) {
    TopKLauncher(4);
  } else if (k <= 8) {
    TopKLauncher(8);
  } else if (k <= 16) {
    TopKLauncher(16);
  } else if (k <= 32) {
    TopKLauncher(32);
  } else {
    TopKLauncher(64);
  }

  LaunchBatchTopKKernel(tmp_values_2nd_stage,
                        tmp_indices_2nd_stage,
                        output_indices,
                        output_tokens,
                        output_values,
                        batch_size,
                        num_beams,
                        k,
                        stream);
}

}  // namespace cuda
}  // namespace Generators
