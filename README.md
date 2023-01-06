# Vina-GPU-CUDA
A CUDA implementation of Vina-GPU

Note: this is a demo of CUDA Vina-GPU, cuda related files are in `./inc/cuda` dir

## Compilation on Linux
**Environment**: Ubuntu 18.04.6 LTS / nvcc version: 11.7 / NVIDIA-SMI 460.91.03 Driver Version: 460.91.03 CUDA Version: 11.2

**Note**: At least 8M stack size is needed. To change the stack size, use `ulimit -s 8192`.
1. install [boost library](https://www.boost.org/) (current version is 1.77.0)
2. install [CUDA Toolkit](https://developer.nvidia.com/zh-cn/cuda-toolkit) (current version is 11.5) if you are using NVIDIA GPU cards

3. set the `BOOST_LIB_PATH` in `Makefile` according to the boost library installation path
4. set `GRID_DIM` according to the costumed grid size. **Note**: the value of GRID_DIM1*GRID_DIM2 (eg. 64*128) must equal to the value of `thread`(eg. 8192) parameter(see [Usage](https://github.com/Glinttsd/Vina-GPU-CUDA/edit/master/README.md#usage))   
5. set `NVCC_COMPILER` according to the nvcc compiler installatio path. Some options are given below:
6. type `make clean` and `make cuda` to build Vina-GPU
7. after a successful compiling, `Vina-GPU` can be seen in the directory 
8. type `./Vina-GPU --config ./input_file_example/2bm2_config.txt` to run Vina-GPU

## Usage
|Arguments| Description|Default value
|--|--|--|
|--config | the config file (in .txt format) that contains all the following arguments for the convenience of use| no default
| --receptor | the recrptor file (in .pdbqt format)| no default
|--ligand| the ligand file (in .pdbqt fotmat)| no default
|--thread| the scale of parallelism (docking lanes)|8192
|--search_depth| the number of searching iterations in each docking lane| heuristically determined
|--center_x/y/z|the center of searching box in the receptor|no default
|--size_x/y/z|the volume of the searching box|no default 
 
