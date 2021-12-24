#pragma once

#include "kernel2.h"
#ifdef WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"