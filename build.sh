./configure CXXFLAGS='-fPIC -m64 -Wno-sign-compare -g -O0 -Wall -Wextra' --prefix=/home/jinhai/Documents/development/faiss-1.5.3 --with-cuda-arch=-gencode=arch=compute_60,code=sm_60 --with-cuda=/usr/local/cuda
make -j
make install