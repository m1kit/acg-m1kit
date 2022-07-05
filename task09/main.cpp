#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <deque>
#include <cmath>
#include <numeric>
#include <algorithm>

// #define MIXED_SEAMLESS_CLONING

namespace py = pybind11;

// ----------------------------------------------------------

/**
 * blending src image to dist image
 * @param dist　the distination image on which we put src image
 * @param src the source image
 * @param src_mask the mask of the source image
 * @return
 */
Eigen::MatrixXd CppPoissonBlending(
    const Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>& dist,
    const Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>& src,
    const Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>& src_mask,
    const std::array<int,2> offset,
    unsigned int num_iteration){
  assert(src.cols() == src_mask.cols() );
  assert(src.cols() == src_mask.cols() );
  Eigen::MatrixXd org = dist.cast<double>();
  Eigen::MatrixXd ret = dist.cast<double>();
  for(unsigned int itr=0;itr<num_iteration;++itr) {  // Gauss-Seidel iteration
    for (unsigned int src_i = 1; src_i < src.rows() - 1; ++src_i) {
      for (unsigned int src_j = 1; src_j < src.cols() - 1; ++src_j) {
        if (src_mask(src_i, src_j) == 0) { continue; }
        unsigned int ret_i = src_i + offset[0];
        unsigned int ret_j = src_j + offset[1];
        if (ret_i <= 0 || ret_i >= dist.rows() - 1) { continue; }
        if (ret_j <= 0 || ret_j >= dist.cols() - 1) { continue; }
        // src_[nsew]: aka g_q, src_c: aka g_p
        const double src_nsew[4] = {
          static_cast<double>(src(src_i, src_j+1)), // north
          static_cast<double>(src(src_i, src_j-1)), // south
          static_cast<double>(src(src_i+1, src_j)), // east
          static_cast<double>(src(src_i-1, src_j)), // west
        };
        const double src_c = src(src_i, src_j); // center
        // ret_nsew: aka f_q, ret_c: aka f_p
        const double ret_nsew[4] = {
          static_cast<double>(ret(ret_i, ret_j+1)), // north
          static_cast<double>(ret(ret_i, ret_j-1)), // south
          static_cast<double>(ret(ret_i+1, ret_j)), // east
          static_cast<double>(ret(ret_i-1, ret_j)), // west
        };
        // write some code here to implement Poisson image editing
#ifndef MIXED_SEAMLESS_CLONING
        // 1. seamless cloning
        double ret_c = src_c + (std::accumulate(ret_nsew, ret_nsew + 4, 0.0) - std::accumulate(src_nsew, src_nsew + 4, 0.0)) / 4.0;
#else
        // 2. mixed seamless cloning
        // org_nsew: aka f^*_q, org_c: aka f^*_p
        const double org_nsew[4] = {
          static_cast<double>(org(ret_i, ret_j+1)), // north
          static_cast<double>(org(ret_i, ret_j-1)), // south
          static_cast<double>(org(ret_i+1, ret_j)), // east
          static_cast<double>(org(ret_i-1, ret_j)), // west
        };
        const double org_c = org(ret_i, ret_j);
        double ret_c = std::accumulate(ret_nsew, ret_nsew + 4, 0.0) / 4.0;
        for (int i = 0; i < 4; i++) {
          const double df = (org_c - org_nsew[i]) / 4.0;
          const double dg = (src_c - src_nsew[i]) / 4.0;
          ret_c += std::abs(df) > std::abs(dg) ? df : dg;
        }
#endif
        // no edit below
        ret_c = (ret_c>255.) ? 255. : ret_c; // clamp
        ret_c = (ret_c<0.) ? 0. : ret_c; // clamp
        ret(ret_i, ret_j) = ret_c;
      }
    }
  }
  return ret;
}

PYBIND11_MODULE(cppmodule, m) {
  m.doc() = "cppmodule";
  m.def("poisson_blending",
        &CppPoissonBlending,
        "A function to blend image using Poisson Image Editing",
        py::arg("dist"), py::arg("src"), py::arg("src_mask"),
        py::arg("offset"), py::arg("num_iteration"));
}


