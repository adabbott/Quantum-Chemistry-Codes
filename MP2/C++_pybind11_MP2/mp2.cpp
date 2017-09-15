#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include <stdio.h>
#include <sstream>

namespace py = pybind11;

double mp2_energy(py::array_t<double> g_iajb,
             py::array_t<double> eps,
             size_t nocc)
{
    py::buffer_info eps_info = eps.request();
    py::buffer_info g_iajb_info = g_iajb.request();
    size_t nbf = eps_info.shape[0];
    size_t nvir = nbf - nocc;

    const double*eps_data = static_cast<double*>(eps_info.ptr);
    const double*g_data = static_cast<double*>(g_iajb_info.ptr);
    size_t stride1 = g_iajb_info.strides[0] / sizeof(double);
    size_t stride2 = g_iajb_info.strides[1] / sizeof(double);
    size_t stride3 = g_iajb_info.strides[2] / sizeof(double);
    size_t stride4 = g_iajb_info.strides[3] / sizeof(double);
     
    double E = 0.0;
#pragma omp parallel for num_threads(8) schedule(dynamic) reduction(+:E) 
    for(size_t i = 0; i < nocc; i++)
    {
        for(size_t j = 0; j < nocc; j++)
        {
            for(size_t a = 0; a < nvir; a++)
            {
#pragma omp simd 
                for(size_t b = 0; b < nvir; b++)
                {
                    double D = 1 / (eps_data[i] + eps_data[j] - eps_data[nocc + a] - eps_data[nocc + b]);
                    double iajb = g_data[i*stride1 + a*stride2 + j*stride3 + b*stride4];
                    double ibja = g_data[i*stride1 + b*stride2 + j*stride3 + a*stride4];
                    E += iajb * (2 * iajb - ibja) * D;
                }
            }
        }
    }
    return E;
}


PYBIND11_PLUGIN(mp2)
{
    py::module m("mp2", "mp2 module"); // define a python module that we can import
    m.def("mp2_energy", &mp2_energy, "computes mp2 correlation energy"); // define a function (name, pointer to function, description)
    return m.ptr();                    // return the memory address of the module for Python to find it
}
