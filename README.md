# cuda_cpp_cross_compile

> tips:
> 1. 需要使用 `MSVC` 编译器配合编译 CUDA
> 2. CUDA 核函数 `<<<>>>` 不能直接在 `.cpp` 文件中调用，需在 `.cu/.cuh` 中定义核函数及调用核函数的普通函数，普通函数声明需以 `extern "C"`修饰
> 3. 在程序入口 `.cpp` 文件中包含 cuda 头文件，并以 `extern "C"` 修饰要调用的 cuda 头文件中定义的普通函数