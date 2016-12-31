PREFIX=$(HOME)/torch/install
CUDA=/usr/local/cuda
CFLAGS=-I$(PREFIX)/include/THC -I$(PREFIX)/include/TH -I$(PREFIX)/include
LDFLAGS_NVCC=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lluaT -lTHC -lTH -lpng
LDFLAGS_CPP=-L$(PREFIX)/lib -lluaT -lTH `pkg-config --libs opencv`

all: libcuresmatch.so libadcensus.so libcv.so

libadcensus.so: src/adcensus.cu
	$(CUDA)/bin/nvcc -arch sm_35 -O3 -DNDEBUG --compiler-options '-fPIC' -o libadcensus.so --shared src/adcensus.cu $(CFLAGS) $(LDFLAGS_NVCC)

libcuresmatch.so: src/curesmatch.cu
	$(CUDA)/bin/nvcc -arch sm_35 -O3 -DNDEBUG --compiler-options '-fPIC' -o libcuresmatch.so --shared src/curesmatch.cu $(CFLAGS) $(LDFLAGS_NVCC)

libcv.so: src/cv.cpp
	g++ -fPIC -o libcv.so -shared src/cv.cpp $(CFLAGS) $(LDFLAGS_CPP)

clean:
	rm -f libcuresmatch.so libadcensus.so libcv.so
