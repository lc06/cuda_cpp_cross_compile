#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "vectorAdd.cuh"

extern "C" void printResult();

int main()
{
    printResult();
    return 0;
}