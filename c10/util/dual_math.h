#if !defined(C10_INTERNAL_INCLUDE_DUAL_REMAINING_H)
#error \
    "c10/util/dual_math.h is not meant to be individually included. Include c10/util/dual.h instead."
#endif

/*
MMC - July 2021
*/

namespace c10_dual_math {

// Exponential functions

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> exp(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::exp(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#else
  return static_cast<c10::dual<T>>(
      std::exp(static_cast<duals::dual<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> log(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::log(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#else
  return static_cast<c10::dual<T>>(
      std::log(static_cast<duals::dual<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> log10(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::log10(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#else
  return static_cast<c10::dual<T>>(
      std::log10(static_cast<duals::dual<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> log2(const c10::dual<T>& x) {
  const c10::dual<T> log2 = c10::dual<T>(::log(2.0), 0.0);
  return c10_dual_math::log(x) / log2;
}

// Power functions
//
#if defined(_LIBCPP_VERSION) || \
    (defined(__GLIBCXX__) && !defined(_GLIBCXX11_USE_C99_dual))
namespace _detail {
TORCH_API c10::dual<float> sqrt(const c10::dual<float>& in);
TORCH_API c10::dual<double> sqrt(const c10::dual<double>& in);
TORCH_API c10::dual<float> acos(const c10::dual<float>& in);
TORCH_API c10::dual<double> acos(const c10::dual<double>& in);
}; // namespace _detail
#endif

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> sqrt(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::sqrt(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#elif !(                        \
    defined(_LIBCPP_VERSION) || \
    (defined(__GLIBCXX__) && !defined(_GLIBCXX11_USE_C99_dual)))
  return static_cast<c10::dual<T>>(
      std::sqrt(static_cast<duals::dual<T>>(x)));
#else
  return _detail::sqrt(x);
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> pow(
    const c10::dual<T>& x,
    const c10::dual<T>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::pow(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x),
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(y)));
#else
  return static_cast<c10::dual<T>>(std::pow(
      static_cast<duals::dual<T>>(x), static_cast<duals::dual<T>>(y)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> pow(
    const c10::dual<T>& x,
    const T& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::pow(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x), y));
#else
  return static_cast<c10::dual<T>>(
      std::pow(static_cast<duals::dual<T>>(x), y));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> pow(
    const T& x,
    const c10::dual<T>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::pow(
      x, c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(y)));
#else
  return static_cast<c10::dual<T>>(
      std::pow(x, static_cast<duals::dual<T>>(y)));
#endif
}

template <typename T, typename U>
C10_HOST_DEVICE inline c10::dual<decltype(T() * U())> pow(
    const c10::dual<T>& x,
    const c10::dual<U>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::pow(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x),
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(y)));
#else
  return static_cast<c10::dual<T>>(std::pow(
      static_cast<duals::dual<T>>(x), static_cast<duals::dual<T>>(y)));
#endif
}

template <typename T, typename U>
C10_HOST_DEVICE inline c10::dual<decltype(T() * U())> pow(
    const c10::dual<T>& x,
    const U& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::pow(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x), y));
#else
  return static_cast<c10::dual<T>>(
      std::pow(static_cast<duals::dual<T>>(x), y));
#endif
}

template <typename T, typename U>
C10_HOST_DEVICE inline c10::dual<decltype(T() * U())> pow(
    const T& x,
    const c10::dual<U>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::pow(
      x, c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(y)));
#else
  return static_cast<c10::dual<T>>(
      std::pow(x, static_cast<duals::dual<T>>(y)));
#endif
}

// Trigonometric functions

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> sin(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::sin(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#else
  return static_cast<c10::dual<T>>(
      std::sin(static_cast<duals::dual<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> cos(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::cos(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#else
  return static_cast<c10::dual<T>>(
      std::cos(static_cast<duals::dual<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> tan(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::tan(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#else
  return static_cast<c10::dual<T>>(
      std::tan(static_cast<duals::dual<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> asin(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::asin(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#else
  return static_cast<c10::dual<T>>(
      std::asin(static_cast<duals::dual<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> acos(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::acos(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#elif !defined(_LIBCPP_VERSION)
  return static_cast<c10::dual<T>>(
      std::acos(static_cast<duals::dual<T>>(x)));
#else
  return _detail::acos(x);
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> atan(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::atan(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#else
  return static_cast<c10::dual<T>>(
      std::atan(static_cast<duals::dual<T>>(x)));
#endif
}

// Hyperbolic functions

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> sinh(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::sinh(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#else
  return static_cast<c10::dual<T>>(
      std::sinh(static_cast<duals::dual<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> cosh(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::cosh(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#else
  return static_cast<c10::dual<T>>(
      std::cosh(static_cast<duals::dual<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> tanh(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::tanh(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#else
  return static_cast<c10::dual<T>>(
      std::tanh(static_cast<duals::dual<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> asinh(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::asinh(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#else
  return static_cast<c10::dual<T>>(
      std::asinh(static_cast<duals::dual<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> acosh(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::acosh(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#else
  return static_cast<c10::dual<T>>(
      std::acosh(static_cast<duals::dual<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::dual<T> atanh(const c10::dual<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::dual<T>>(thrust::atanh(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(x)));
#else
  return static_cast<c10::dual<T>>(
      std::atanh(static_cast<duals::dual<T>>(x)));
#endif
}

} // namespace c10_dual_math

using c10_dual_math::acos;
using c10_dual_math::acosh;
using c10_dual_math::asin;
using c10_dual_math::asinh;
using c10_dual_math::atan;
using c10_dual_math::atanh;
using c10_dual_math::cos;
using c10_dual_math::cosh;
using c10_dual_math::exp;
using c10_dual_math::log;
using c10_dual_math::log10;
using c10_dual_math::log2;
using c10_dual_math::pow;
using c10_dual_math::sin;
using c10_dual_math::sinh;
using c10_dual_math::sqrt;
using c10_dual_math::tan;
using c10_dual_math::tanh;

namespace std {

using c10_dual_math::acos;
using c10_dual_math::acosh;
using c10_dual_math::asin;
using c10_dual_math::asinh;
using c10_dual_math::atan;
using c10_dual_math::atanh;
using c10_dual_math::cos;
using c10_dual_math::cosh;
using c10_dual_math::exp;
using c10_dual_math::log;
using c10_dual_math::log10;
using c10_dual_math::log2;
using c10_dual_math::pow;
using c10_dual_math::sin;
using c10_dual_math::sinh;
using c10_dual_math::sqrt;
using c10_dual_math::tan;
using c10_dual_math::tanh;

} // namespace std
