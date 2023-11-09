#! /bin/bash
if [ $# != 1 ]; then
  echo "specify output file name"
  echo "e.g. ./nsys_run.sh output_file_name "
  exit 100
fi
#for i in 0 32 64 96 128 224 320 416 512
#for i in 0 32 64 96 128 224 256
#for i in 0 32 64 96 128
#for i in 0 32 64 
for i in 0 32 
do
  nsys profile -t cuda,osrt,nvtx,cudnn,cublas -o $1_$i.qdstrm --stats=true -w true ./15_ampere_sparse_tensorop_gemm $1 $i | grep -e SparseGemm -e Gemm -e vecAddOpt -e M: >>$1.log
  echo ////////////////////////////////////////////////////////// >>$1.log
  echo  >>$1.log
done
#nsys profile -t cuda,osrt,nvtx,cudnn,cublas -o $1.qdstrm --stats=true -w true ./15_ampere_sparse_tensorop_gemm 1024 0 |grep -e SparseGemm -e Gemm -e vecAddOpt -e M:
