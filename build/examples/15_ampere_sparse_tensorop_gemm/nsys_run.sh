#! /bin/bash
if [ $# != 1 ]; then
  echo "specify output file name"
  echo "e.g. ./nsys_run.sh output_file_name "
  exit 100
fi
nsys profile -t cuda,osrt,nvtx,cudnn,cublas -o $1.qdstrm -w true ./15_ampere_sparse_tensorop_gemm
