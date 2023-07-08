# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\liaocai\code\cplusplus\cudatest

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\liaocai\code\cplusplus\cudatest\build

# Include any dependencies generated for this target.
include cuda/CMakeFiles/vectorAdd.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include cuda/CMakeFiles/vectorAdd.dir/compiler_depend.make

# Include the progress variables for this target.
include cuda/CMakeFiles/vectorAdd.dir/progress.make

# Include the compile flags for this target's objects.
include cuda/CMakeFiles/vectorAdd.dir/flags.make

cuda/CMakeFiles/vectorAdd.dir/vectorAdd_generated_vectorAdd.cu.obj: D:/liaocai/code/cplusplus/cudatest/cuda/vectorAdd.cu
cuda/CMakeFiles/vectorAdd.dir/vectorAdd_generated_vectorAdd.cu.obj: cuda/CMakeFiles/vectorAdd.dir/vectorAdd_generated_vectorAdd.cu.obj.depend
cuda/CMakeFiles/vectorAdd.dir/vectorAdd_generated_vectorAdd.cu.obj: cuda/CMakeFiles/vectorAdd.dir/vectorAdd_generated_vectorAdd.cu.obj.Release.cmake
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=D:\liaocai\code\cplusplus\cudatest\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object cuda/CMakeFiles/vectorAdd.dir/vectorAdd_generated_vectorAdd.cu.obj"
	cd /d D:\liaocai\code\cplusplus\cudatest\build\cuda\CMakeFiles\vectorAdd.dir && "C:\Program Files\CMake\bin\cmake.exe" -E make_directory D:/liaocai/code/cplusplus/cudatest/build/cuda/CMakeFiles/vectorAdd.dir//.
	cd /d D:\liaocai\code\cplusplus\cudatest\build\cuda\CMakeFiles\vectorAdd.dir && "C:\Program Files\CMake\bin\cmake.exe" -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Release -D generated_file:STRING=D:/liaocai/code/cplusplus/cudatest/build/cuda/CMakeFiles/vectorAdd.dir//./vectorAdd_generated_vectorAdd.cu.obj -D generated_cubin_file:STRING=D:/liaocai/code/cplusplus/cudatest/build/cuda/CMakeFiles/vectorAdd.dir//./vectorAdd_generated_vectorAdd.cu.obj.cubin.txt -P D:/liaocai/code/cplusplus/cudatest/build/cuda/CMakeFiles/vectorAdd.dir//vectorAdd_generated_vectorAdd.cu.obj.Release.cmake

# Object files for target vectorAdd
vectorAdd_OBJECTS =

# External object files for target vectorAdd
vectorAdd_EXTERNAL_OBJECTS = \
"D:/liaocai/code/cplusplus/cudatest/build/cuda/CMakeFiles/vectorAdd.dir/vectorAdd_generated_vectorAdd.cu.obj"

cuda/libvectorAdd.a: cuda/CMakeFiles/vectorAdd.dir/vectorAdd_generated_vectorAdd.cu.obj
cuda/libvectorAdd.a: cuda/CMakeFiles/vectorAdd.dir/build.make
cuda/libvectorAdd.a: cuda/CMakeFiles/vectorAdd.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=D:\liaocai\code\cplusplus\cudatest\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libvectorAdd.a"
	cd /d D:\liaocai\code\cplusplus\cudatest\build\cuda && $(CMAKE_COMMAND) -P CMakeFiles\vectorAdd.dir\cmake_clean_target.cmake
	cd /d D:\liaocai\code\cplusplus\cudatest\build\cuda && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\vectorAdd.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cuda/CMakeFiles/vectorAdd.dir/build: cuda/libvectorAdd.a
.PHONY : cuda/CMakeFiles/vectorAdd.dir/build

cuda/CMakeFiles/vectorAdd.dir/clean:
	cd /d D:\liaocai\code\cplusplus\cudatest\build\cuda && $(CMAKE_COMMAND) -P CMakeFiles\vectorAdd.dir\cmake_clean.cmake
.PHONY : cuda/CMakeFiles/vectorAdd.dir/clean

cuda/CMakeFiles/vectorAdd.dir/depend: cuda/CMakeFiles/vectorAdd.dir/vectorAdd_generated_vectorAdd.cu.obj
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\liaocai\code\cplusplus\cudatest D:\liaocai\code\cplusplus\cudatest\cuda D:\liaocai\code\cplusplus\cudatest\build D:\liaocai\code\cplusplus\cudatest\build\cuda D:\liaocai\code\cplusplus\cudatest\build\cuda\CMakeFiles\vectorAdd.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : cuda/CMakeFiles/vectorAdd.dir/depend
