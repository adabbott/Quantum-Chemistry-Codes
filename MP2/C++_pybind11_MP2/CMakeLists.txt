cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(molssi_sss_pybind)

# This must be run using a conda environment
if("$ENV{CONDA_PREFIX}" STREQUAL "")
    message(FATAL_ERROR "This must be run within the MolSSI SSS conda prefix. "
                        "Delete the build directory and try again")
endif()

# CMake will sometimes find the conda python version
# (rather the the python inside the sss environment)
set(PYTHON_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python3)

#set(CMAKE_CXX_FLAGS "-std=c++11 -03 -march_native -fopenmp -DNDEBUG")

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


# Find the pybind11 in the conda path
set(PYBIND11_CPP_STANDARD -std=c++11)
find_package(pybind11 CONFIG REQUIRED
             PATHS $ENV{CONDA_PREFIX}
             NO_DEFAULT_PATH)

message(STATUS "Found pybind11: ${pybind11_CONFIG}")

# Creates a python module named "module_name"
pybind11_add_module(mp2 MODULE mp2.cpp)
