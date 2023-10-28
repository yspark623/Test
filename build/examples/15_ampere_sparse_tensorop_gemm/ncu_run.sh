#! /bin/bash
if [ $# != 1 ]; then
  echo "specify output file name"
  echo "e.g. ./ncu_run.sh output_file_name "
  exit 100
fi
sudo /usr/local/cuda/bin/ncu --call-stack --nvtx -o $1 --set full ./15_ampere_sparse_tensorop_gemm
