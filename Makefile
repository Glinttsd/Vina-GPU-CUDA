# Need to be modified according to different users
BOOST_LIB_PATH=../boost_1_77_0
OPENCL_LIB_PATH=/usr/local/cuda
GRID_DIM=-DGRID_DIM1=64 -DGRID_DIM2=128
NVCC_COMPILER=/usr/local/cuda-11.0/bin/nvcc
# Should not be modified
BOOST_INC_PATH=-I$(BOOST_LIB_PATH) -I$(BOOST_LIB_PATH)/boost 
VINA_GPU_INC_PATH=-I./lib -I./inc/ -I./inc/cuda
OPENCL_INC_PATH=
LIB1=-lboost_program_options -lboost_system -lboost_filesystem
LIB2=-lstdc++
LIB3=-lm -lpthread
LIB_PATH=-L$(BOOST_LIB_PATH)/stage/lib
SRC=./lib/*.cpp $(BOOST_LIB_PATH)/libs/thread/src/pthread/thread.cpp $(BOOST_LIB_PATH)/libs/thread/src/pthread/once.cpp #../boost_1_77_0/boost/filesystem/path.hpMACRO=-DAMD_PLATFORM -DDISPLAY_SUCCESS -DDISPLAY_ADDITION_INFO
SRC_CUDA = ./inc/cuda/kernel2.cu
MACRO=$(GRID_DIM) #-DDISPLAY_SUCCESS -DDISPLAY_ADDITION_INFO
all:out
out:./main/main.cpp
	$(NVCC_COMPILER) -o Vina-GPU $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) ./main/main.cpp -O3 $(SRC) $(SRC_CUDA) $(LIB1) $(LIB2) $(LIB3) $(LIB_PATH) $(MACRO) $(OPTION) -DBUILD_KERNEL_FROM_SOURCE
cuda:./main/main.cpp
	$(NVCC_COMPILER) -o Vina-GPU $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) ./main/main.cpp -O3 $(SRC) $(SRC_CUDA) $(LIB1) $(LIB2) $(LIB3) $(LIB_PATH) $(MACRO) $(OPTION) -DBUILD_KERNEL_FROM_SOURCE
clean:
	rm Vina-GPU
