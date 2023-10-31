/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
Please check example 07, 08 and 17 for the basics of dense tensor op gemm kernels.  NVIDIA Ampere
architecture also supports structured sparse tensor op for tf32, fp16, int8 and int4.

Sparse GEMM kernels needs to takes an additional E matrix which stores the meta data.  The format of
meta data is different for every data types.   CUTLASS templates can automatically infer it based on
input A and B.  Check code below.

Moreover, matrix E needs to be preprocessed so that it can use ldmatrix to load into the registers
efficiently.
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/host_uncompress.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"


///////////////////////////////////////////////
///// TEST CONFIGURATION
///////////////////////////////////////////////
#define DENSE_GEMM_EN 0 // 0: disable, 1: enable
#define VEC_ADD_EN    1 // 0: disable, 1: enable
#define REF_EN        2 // 0: disable, 1: host, 2: cutlass


#if (DENSE_GEMM_EN==1 || REF_EN==2)
#include "cutlass/gemm/device/gemm.h"
#endif

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
//using ElementAccumulator = int32_t;                 // <- data type of accumulator
using ElementAccumulator = float;                 // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
//using ElementInputA = cutlass::int4b_t;             // <- data type of elements in input matrix A
//using ElementInputB = cutlass::int4b_t;             // <- data type of elements in input matrix B
//using ElementOutput = int32_t;                      // <- data type of elements in output matrix D
using ElementInputA = cutlass::half_t;             // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;             // <- data type of elements in input matrix B
using ElementOutput = float;                      // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Row Major for
// Matrix A, Column Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    //cutlass::gemm::GemmShape<128, 128, 256>;  // <- threadblock tile M = 128, N = 128, K = 256
    cutlass::gemm::GemmShape<128, 128, 64>;  // <- threadblock tile M = 128, N = 128, K = 256
// This code section describes tile size a warp will compute
//using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 256>;  // <- warp tile M = 64, N = 64, K = 256
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;  // <- warp tile M = 64, N = 64, K = 256
// This code section describes the size of MMA op
//using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 128>;  // <- MMA Op tile M = 16, N = 8, K = 128
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>;  // <- MMA Op tile M = 16, N = 8, K = 128

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 3;

using Gemm = cutlass::gemm::device::SparseGemm<ElementInputA,
                                               LayoutInputA,
                                               ElementInputB,
                                               LayoutInputB,
                                               ElementOutput,
                                               LayoutOutput,
                                               ElementAccumulator,
                                               MMAOp,
                                               SmArch,
                                               ShapeMMAThreadBlock,
                                               ShapeMMAWarp,
                                               ShapeMMAOp,
                                               EpilogueOp,
                                               SwizzleThreadBlock,
                                               NumStages>;

// Data type and layout of meta data matrix E can be inferred from template Gemm.
using ElementInputE = typename Gemm::ElementE;
using LayoutInputE = cutlass::layout::RowMajor;
using ReorderedLayoutInputE = typename Gemm::LayoutE;

// Blow property is defined in include/cutlass/arch/sp_mma_sm80.h
// 50% Sparsity on Ampere
constexpr int kSparse = Gemm::kSparse;
// How many elements of A are covered per ElementE
constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
// The size of individual meta data 
constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;

  ///////////////////////////////////////////////
  ///// FOR DENSE GEMM
  ///////////////////////////////////////////////

#if (DENSE_GEMM_EN==1 || REF_EN==2)
using DenseGemm = cutlass::gemm::device::Gemm<ElementInputA,        // Data-type of A matrix
                                              LayoutInputA,
                                              ElementInputB,
                                              LayoutInputB,
                                              ElementOutput,
                                              LayoutOutput,
                                              ElementAccumulator,
                                              MMAOp,
                                              SmArch,
                                              ShapeMMAThreadBlock,
                                              ShapeMMAWarp
                                              >;
#endif
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrixF(float **matrix, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(float) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  //result = InitializeMatrix(*matrix, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(cutlass::half_t **matrix, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(cutlass::half_t) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  //result = InitializeMatrix(*matrix, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

//__global__ void vecAdd(cutlass::half_t *a, cutlass::half_t *b, cutlass::half_t *c, int n)
__global__ void vecAdd(float *a, float *b, float *c, int n)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if(id<n)
    c[id] = a[id] + b[id];
}

__global__ void vecAddOpt(float *a, float *b, float *c, int extra_rows, int col_num)
{
  int row_id = blockIdx.x%5;
  int col_id = row_id*blockDim.x+threadIdx.x;
  //int id = blockIdx.x*blockDim.x+threadIdx.x;
  if(row_id<extra_rows && col_id<col_num)
    c[row_id*col_num+col_id] = a[row_id*col_num+col_id] + b[row_id*col_num+col_id];
    //c[id] = a[id] + b[id];
}
///////////////////////////////////////////////////////////////////////////////////////////////////


int run() {

  const int length_m_extra = 0;
  const int length_m = 2048 + length_m_extra;
  const int length_k = 20480;
  const int length_n = 5120;

  std::cout<< "M: "<<length_m<<" (+"<<length_m_extra<<"), K: "<<length_k<<", N: "<<length_n<<std::endl;
  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse));  // <- Create matrix A with dimensions M x (K / 2)
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a_uncompressed(
      problem_size.mk());  // <- Create uncompressed matrix A with dimensions M x K for reference computing

  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  // Create matrix E with dimensions M x (K / 2 / kElementsPerElementE). This one is used by reference computing.
  cutlass::HostTensor<ElementInputE, LayoutInputE> tensor_e(
      cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));
  // Same size as the above.  The above one needs to be reordered and stored in this one.
  cutlass::HostTensor<ElementInputE, ReorderedLayoutInputE> tensor_e_reordered(
      cutlass::make_Coord(problem_size.m(), problem_size.k() / kSparse / kElementsPerElementE));

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(2),
      ElementInputA(-2),
      0);  // <- Fill matrix A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(2),
      ElementInputB(-2),
      0);  // <- Fill matrix B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(2),
      ElementOutput(-2),
      0);  // <- Fill matrix C on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomSparseMeta(                                           
      tensor_e.host_view(),
      1,
      kMetaSizeInBits);   // <- Fill matrix E on host with uniform-distribution random meta data
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  // Reorder the meta data matrix so that we can use ldmatrix to load them to tensor core
  // instructions.
  cutlass::reorder_meta(tensor_e_reordered.host_ref(), tensor_e.host_ref(),                         
                        {problem_size.m(), problem_size.n(),                                        
                         problem_size.k() / kSparse / kElementsPerElementE});

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_e_reordered.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     tensor_e_reordered.device_ref(),  // <- reference to matrix E on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not 
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // Launch initialized CUTLASS kernel
  std::cout<<"Start sparse gemm"<<std::endl;
  status = gemm_op();
  std::cout<<"End sparse gemm"<<std::endl;
  CUTLASS_CHECK(status);

  ///////////////////////////////////////////////
  ///// VECTOR ADDITION
  ///////////////////////////////////////////////

#if VEC_ADD_EN==1
  std::cout<<"Start vector addition"<<std::endl;

  int numElements = length_m_extra * length_n;

  int threadsPerBlock = 1024;
  int blocksPerGrid = (numElements + threadsPerBlock -1) / threadsPerBlock;

  std::cout<<"# of extra rows: "<< length_m_extra <<", # of blocks: "<< blocksPerGrid <<", # of threads per block: "<< threadsPerBlock <<std::endl;

  //vecAddOpt<<<blocksPerGrid,threadsPerBlock>>>(tensor_d.device_data_ptr_offset(i*length_n), tensor_d.device_data_ptr_offset((length_m-i-1)*length_n), tensor_d.device_data_ptr_offset(i*length_n), length_n);

  vecAddOpt<<<blocksPerGrid,threadsPerBlock>>>(tensor_d.device_data_ptr_offset(0), tensor_d.device_data_ptr_offset(0), tensor_d.device_data_ptr_offset(0),length_m_extra, length_n);

////  for(int i=0; i<length_m_extra; i++)
////    //std::cout<<i*length_n<<", "<<(length_m-i-1)*length_n
////    //vecAdd<<<blocksPerGrid,threadsPerBlock>>>(tensor_d.device_data_ptr_offset(i*length_n), tensor_d.device_data_ptr_offset((length_m-i-1)*length_n), tensor_d.device_data_ptr_offset(i*length_n), length_n);
////    vecAdd<<<5,1024>>>(tensor_d.device_data_ptr_offset(i*length_n), tensor_d.device_data_ptr_offset((length_m-i-1)*length_n), tensor_d.device_data_ptr_offset(i*length_n), length_n);
////    //vecAdd<<<5,1024>>>(tensor_d.device_data_ptr_offset(i*length_n/2), tensor_d.device_data_ptr_offset(length_m-i+1*length_n/2), tensor_d.device_data_ptr_offset(i*length_n/2), length_n/2);

  std::cout<<"End vector addition"<<std::endl;
#endif
  
  ///////////////////////////////////////////////
  ///// FOR DENSE GEMM
  ///////////////////////////////////////////////
#if DENSE_GEMM_EN==1
  DenseGemm gemm_operator;

  // Compute leading dimensions for each matrix.
  int lda = length_m;
  int ldb = length_k;
  int ldc = length_m;
  //float alpha = 1.0;
  //float beta = 0.0;

  ElementInputA *A;
  ElementInputB *B;
  ElementOutput *C;

  cudaError_t result;

  result = AllocateMatrix(&A, length_m, length_k, 0);

  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B, length_k, length_n, 17);

  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrixF(&C, length_m, length_n, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  DenseGemm::Arguments args({length_m , length_n, length_k},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  std::cout<<"Start dense gemm"<<std::endl;
  status = gemm_operator(args);
  std::cout<<"End dense gemm"<<std::endl;

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
#endif


  ///////////////////////////////////////////////
  ///// REFERENCE in HOST
  ///////////////////////////////////////////////
#if REF_EN==1

  std::cout<<"Start reference on host"<<std::endl;
  // uncompress tensor_a based on meta data tensor_e. We need it for reference computing.
  cutlass::uncompress(tensor_a_uncompressed.host_ref(), tensor_a.host_ref(),
                      tensor_e.host_ref(), problem_size.m(), problem_size.k());
 
  // Create instantiation for host reference gemm kernel
  cutlass::reference::host::Gemm<ElementInputA,
                                 LayoutInputA,
                                 ElementInputB,
                                 LayoutInputB,
                                 ElementOutput,
                                 LayoutOutput,
                                 ElementComputeEpilogue,
                                 ElementComputeEpilogue,
                                 typename Gemm::Operator>
      gemm_host;

  // Launch host reference gemm kernel
  gemm_host(problem_size,
            alpha,
            tensor_a_uncompressed.host_ref(),
            tensor_b.host_ref(),
            beta,
            tensor_c.host_ref(),
            tensor_ref_d.host_ref());

  // Copy output data from CUTLASS host for comparison
  tensor_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::host::TensorEquals(
    tensor_d.host_view(),
    tensor_ref_d.host_view());

  std::cout << (passed ? "Passed" : "Failed") << std::endl;
  std::cout<<"End reference on host"<<std::endl;

  return (passed ? 0  : -1);
#elif REF_EN==2
  std::cout<<"Start reference on cutlass"<<std::endl;

  DenseGemm gemm_ref;

  // uncompress tensor_a based on meta data tensor_e. We need it for reference computing.
  cutlass::uncompress(tensor_a_uncompressed.host_ref(), tensor_a.host_ref(),
                      tensor_e.host_ref(), problem_size.m(), problem_size.k());

  // Copy data from host to GPU
  tensor_a_uncompressed.sync_device();

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  DenseGemm::Arguments args_ref{problem_size,  // <- problem size of matrix multiplication
                                tensor_a_uncompressed.device_ref(),  // <- reference to matrix A on device
                                tensor_b.device_ref(),  // <- reference to matrix B on device
                                tensor_c.device_ref(),  // <- reference to matrix C on device
                                tensor_ref_d.device_ref(),  // <- reference to matrix D on device
                                {alpha, beta},          // <- tuple of alpha and beta
                                split_k_slices};        // <- k-dimension split factor

  status = gemm_ref(args_ref);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Copy output data from CUTLASS host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::host::TensorEquals(
    tensor_d.host_view(),
    tensor_ref_d.host_view());

  std::cout << (passed ? "Passed" : "Failed") << std::endl;
  std::cout<<"End reference on cutlass"<<std::endl;

  return 0;
#else
  tensor_d.host_view(),
  std::cout<<"No reference"<<std::endl;
  return 0;
#endif
}

int main() {
  
  bool notSupported = false;

  // Ampere Sparse Tensor Core operations exposed with mma.sync and ldmatrix are first available
  // in CUDA 11.1. 
  //
  // CUTLASS must be compiled with CUDA 11.1 Toolkit to run these examples.
  
  if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 1))) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.1 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major * 10 + props.minor < 80) {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  return run();
}
