#pragma once

#include <c10/util/dual>
#include <iostream>

#include <c10/macros/Macros.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/dual.h>
#endif

/*
MMC - July 2021
*/
namespace c10 {

// c10::dual is an implementation of dual numbers that aims
// to work on all devices supported by PyTorch
//
// Most of the APIs duplicates std::complex
// Reference: https://en.cppreference.com/w/cpp/numeric/complex
//
// [NOTE: Complex Operator Unification]
// Operators currently use a mix of std::complex, thrust::complex, and
// c10::dual internally. The end state is that all operators will use
// c10::complex internally.  Until then, there may be some hacks to support all
// variants.
//
//
// [Note on Constructors]
//
// The APIs of constructors are mostly copied from C++ standard:
//   https://en.cppreference.com/w/cpp/numeric/complex/complex
//
// Since C++14, all constructors are constexpr in std::complex
//
// There are three types of constructors:
// - initializing from real and imag:
//     `constexpr dual( const T& re = T(), const T& im = T() );`
// - implicitly-declared copy constructor
// - converting constructors
//
// Converting constructors:
// - duals::dual defines converting constructor between float/double/long
// double,
//   while we define converting constructor between float/double.
// - For these converting constructors, upcasting is implicit, downcasting is
//   explicit.
// - We also define explicit casting from std::complex/thrust::complex
//   - Note that the conversion from thrust is not constexpr, because
//     thrust does not define them as constexpr ????
//
//
// [Operator =]
//
// The APIs of operator = are mostly copied from C++ standard:
//   https://en.cppreference.com/w/cpp/numeric/complex/operator%3D
//
// Since C++20, all operator= are constexpr. Although we are not building with
// C++20, we also obey this behavior.
//
// There are three types of assign operator:
// - Assign a real value from the same scalar type
//   - In std, this is templated as complex& operator=(const T& x)
//     with specialization `complex& operator=(T x)` for float/double/long
//     double Since we only support float and double, on will use `complex&
//     operator=(T x)`
// - Copy assignment operator and converting assignment operator
//   - There is no specialization of converting assignment operators, which type
//   is
//     convertible is solely dependent on whether the scalar type is convertible
//
// In addition to the standard assignment, we also provide assignment operators
// with std and thrust
//
//
// [Casting operators]
//
// duals::dual does not have casting operators. We define casting operators
// casting to duals::dual and thrust::dual
//
//
// [Operator ""]
//
// std::complex has custom literals `i`, `if` and `il` defined in namespace
// `std::literals::complex_literals`. We define our own custom literals in the
// namespace `c10::dual_literals`. Our custom literals does not follow the
// same behavior as in std::complex, instead, we define _if, _id to construct
// float/double dual literals.
//
//
// [real() and imag()]
//
// In C++20, there are two overload of these functions, one it to return the
// real/imag, another is to set real/imag, they are both constexpr. We follow
// this design.
//
//
// [Operator +=,-=,*=,/=]
//
// Since C++20, these operators become constexpr. In our implementation, they
// are also constexpr.
//
// There are two types of such operators: operating with a real number, or
// operating with another dual number. For the operating with a real number,
// the generic template form has argument type `const T &`, while the overload
// for float/double/long double has `T`. We will follow the same type as
// float/double/long double in std.
//
// [Unary operator +-]
//
// Since C++20, they are constexpr. We also make them expr
//
// [Binary operators +-*/]
//
// Each operator has three versions (taking + as example):
// - dual + dual
// - dual + real
// - real + dual
//
// [Operator ==, !=]
//
// Each operator has three versions (taking == as example):
// - dual == dual
// - dual == real
// - real == dual
//
// Some of them are removed on C++20, but we decided to keep them
//
// [Operator <<, >>]
//
// These are implemented by casting to duals::dual
//
//
//
// TODO(@zasdfgbnm): c10::complex<c10::Half> is not currently supported,
// because:
//  - lots of members and functions of c10::Half are not constexpr
//  - thrust::complex only supports float and double

template <typename T>
struct alignas(sizeof(T) * 2) dual {
  using value_type = T;

  T real_ = T(0);
  T imag_ = T(0);

  constexpr dual() = default;
  C10_HOST_DEVICE constexpr dual(const T& re, const T& im = T())
      : real_(re), imag_(im) {}
  template <typename U>
  explicit constexpr dual(const duals::dual<U>& other)
      : dual(other.real(), other.imag()) {}
#if defined(__CUDACC__) || defined(__HIPCC__)
  template <typename U>
  explicit C10_HOST_DEVICE dual(const thrust::dual<U>& other)
      : real_(other.real()), imag_(other.imag()) {}
// NOTE cannot be implemented as follows due to ROCm bug:
//   explicit C10_HOST_DEVICE dual(const thrust::dual<U> &other):
//   dual(other.real(), other.imag()) {}
#endif

  // Use SFINAE to specialize casting constructor for c10::dual<float> and
  // c10::dual<double>
  template <typename U = T>
  C10_HOST_DEVICE explicit constexpr dual(
      const std::enable_if_t<std::is_same<U, float>::value, dual<double>>&
          other)
      : real_(other.real_), imag_(other.imag_) {}
  template <typename U = T>
  C10_HOST_DEVICE constexpr dual(
      const std::enable_if_t<std::is_same<U, double>::value, dual<float>>&
          other)
      : real_(other.real_), imag_(other.imag_) {}

  constexpr dual<T>& operator=(T re) {
    real_ = re;
    imag_ = 0;
    return *this;
  }

  constexpr dual<T>& operator+=(T re) {
    real_ += re;
    return *this;
  }

  constexpr dual<T>& operator-=(T re) {
    real_ -= re;
    return *this;
  }

  constexpr dual<T>& operator*=(T re) {
    real_ *= re;
    imag_ *= re;
    return *this;
  }

  constexpr dual<T>& operator/=(T re) {
    real_ /= re;
    imag_ /= re;
    return *this;
  }

  template <typename U>
  constexpr dual<T>& operator=(const dual<U>& rhs) {
    real_ = rhs.real();
    imag_ = rhs.imag();
    return *this;
  }

  template <typename U>
  constexpr dual<T>& operator+=(const dual<U>& rhs) {
    real_ += rhs.real();
    imag_ += rhs.imag();
    return *this;
  }

  template <typename U>
  constexpr dual<T>& operator-=(const dual<U>& rhs) {
    real_ -= rhs.real();
    imag_ -= rhs.imag();
    return *this;
  }

  template <typename U>
  constexpr dual<T>& operator*=(const dual<U>& rhs) {
    // (a + bi) * (c + di) = (a*c ) + (a * d + b * c) i
    T a = real_;
    T b = imag_;
    U c = rhs.real();
    U d = rhs.imag();
    real_ = a * c;
    imag_ = a * d + b * c;
    return *this;
  }

#ifdef __APPLE__
#define FORCE_INLINE_APPLE __attribute__((always_inline))
#else
#define FORCE_INLINE_APPLE
#endif
  template <typename U>
  constexpr FORCE_INLINE_APPLE dual<T>& operator/=(const dual<U>& rhs)
      __ubsan_ignore_float_divide_by_zero__ {
    // (a + bi) / (c + di) = (ac)/(c^2) + (bc - ad)/(c^2) i
    T a = real_;
    T b = imag_;
    U c = rhs.real();
    U d = rhs.imag();
    auto denominator = c * c;
    real_ = a / c;
    imag_ = (b * c - a * d) / denominator;
    return *this;
  }
#undef FORCE_INLINE_APPLE

  template <typename U>
  constexpr dual<T>& operator=(const duals::dual<U>& rhs) {
    real_ = rhs.real();
    imag_ = rhs.imag();
    return *this;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  template <typename U>
  C10_HOST_DEVICE dual<T>& operator=(const thrust::dual<U>& rhs) {
    real_ = rhs.real();
    imag_ = rhs.imag();
    return *this;
  }
#endif

  template <typename U>
  explicit constexpr operator duals::dual<U>() const {
    return duals::dual<U>(duals::dual<T>(real(), imag()));
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  template <typename U>
  C10_HOST_DEVICE explicit operator thrust::dual<U>() const {
    return static_cast<thrust::dual<U>>(thrust::dual<T>(real(), imag()));
  }
#endif

  // consistent with NumPy behavior
  explicit constexpr operator bool() const {
    return real() || imag();
  }

  C10_HOST_DEVICE constexpr T real() const {
    return real_;
  }
  constexpr void real(T value) {
    real_ = value;
  }
  constexpr T imag() const {
    return imag_;
  }
  constexpr void imag(T value) {
    imag_ = value;
  }
};

namespace dual_literals {

constexpr dual<float> operator"" _if(long double imag) {
  return dual<float>(0.0f, static_cast<float>(imag));
}

constexpr dual<double> operator"" _id(long double imag) {
  return dual<double>(0.0, static_cast<double>(imag));
}

constexpr dual<float> operator"" _if(unsigned long long imag) {
  return dual<float>(0.0f, static_cast<float>(imag));
}

constexpr dual<double> operator"" _id(unsigned long long imag) {
  return dual<double>(0.0, static_cast<double>(imag));
}

} // namespace dual_literals

template <typename T>
constexpr dual<T> operator+(const dual<T>& val) {
  return val;
}

template <typename T>
constexpr dual<T> operator-(const dual<T>& val) {
  return dual<T>(-val.real(), -val.imag());
}

template <typename T>
constexpr dual<T> operator+(const dual<T>& lhs, const dual<T>& rhs) {
  dual<T> result = lhs;
  return result += rhs;
}

template <typename T>
constexpr dual<T> operator+(const dual<T>& lhs, const T& rhs) {
  dual<T> result = lhs;
  return result += rhs;
}

template <typename T>
constexpr dual<T> operator+(const T& lhs, const dual<T>& rhs) {
  return dual<T>(lhs + rhs.real(), rhs.imag());
}

template <typename T>
constexpr dual<T> operator-(const dual<T>& lhs, const dual<T>& rhs) {
  dual<T> result = lhs;
  return result -= rhs;
}

template <typename T>
constexpr dual<T> operator-(const dual<T>& lhs, const T& rhs) {
  dual<T> result = lhs;
  return result -= rhs;
}

template <typename T>
constexpr dual<T> operator-(const T& lhs, const dual<T>& rhs) {
  dual<T> result = -rhs;
  return result += lhs;
}

template <typename T>
constexpr dual<T> operator*(const dual<T>& lhs, const dual<T>& rhs) {
  dual<T> result = lhs;
  return result *= rhs;
}

template <typename T>
constexpr dual<T> operator*(const dual<T>& lhs, const T& rhs) {
  dual<T> result = lhs;
  return result *= rhs;
}

template <typename T>
constexpr dual<T> operator*(const T& lhs, const dual<T>& rhs) {
  dual<T> result = rhs;
  return result *= lhs;
}

template <typename T>
constexpr dual<T> operator/(const dual<T>& lhs, const dual<T>& rhs) {
  dual<T> result = lhs;
  return result /= rhs;
}

template <typename T>
constexpr dual<T> operator/(const dual<T>& lhs, const T& rhs) {
  dual<T> result = lhs;
  return result /= rhs;
}

template <typename T>
constexpr dual<T> operator/(const T& lhs, const dual<T>& rhs) {
  dual<T> result(lhs, T());
  return result /= rhs;
}

// Define operators between integral scalars and c10::dual. duals::dual does
// not support this when T is a floating-point number. This is useful because it
// saves a lot of "static_cast" when operate a dual and an integer. This
// makes the code both less verbose and potentially more efficient.
#define dual_INTEGER_OP_TEMPLATE_CONDITION                           \
  typename std::enable_if_t<                                            \
      std::is_floating_point<fT>::value && std::is_integral<iT>::value, \
      int> = 0

template <typename fT, typename iT, dual_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::dual<fT> operator+(const c10::dual<fT>& a, const iT& b) {
  return a + static_cast<fT>(b);
}

template <typename fT, typename iT, dual_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::dual<fT> operator+(const iT& a, const c10::dual<fT>& b) {
  return static_cast<fT>(a) + b;
}

template <typename fT, typename iT, dual_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::dual<fT> operator-(const c10::dual<fT>& a, const iT& b) {
  return a - static_cast<fT>(b);
}

template <typename fT, typename iT, dual_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::dual<fT> operator-(const iT& a, const c10::dual<fT>& b) {
  return static_cast<fT>(a) - b;
}

template <typename fT, typename iT, dual_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::dual<fT> operator*(const c10::dual<fT>& a, const iT& b) {
  return a * static_cast<fT>(b);
}

template <typename fT, typename iT, dual_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::dual<fT> operator*(const iT& a, const c10::dual<fT>& b) {
  return static_cast<fT>(a) * b;
}

template <typename fT, typename iT, dual_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::dual<fT> operator/(const c10::dual<fT>& a, const iT& b) {
  return a / static_cast<fT>(b);
}

template <typename fT, typename iT, dual_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::dual<fT> operator/(const iT& a, const c10::dual<fT>& b) {
  return static_cast<fT>(a) / b;
}

#undef dual_INTEGER_OP_TEMPLATE_CONDITION

template <typename T>
constexpr bool operator==(const dual<T>& lhs, const dual<T>& rhs) {
  return (lhs.real() == rhs.real()) && (lhs.imag() == rhs.imag());
}

template <typename T>
constexpr bool operator==(const dual<T>& lhs, const T& rhs) {
  return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <typename T>
constexpr bool operator==(const T& lhs, const dual<T>& rhs) {
  return (lhs == rhs.real()) && (T() == rhs.imag());
}

template <typename T>
constexpr bool operator!=(const dual<T>& lhs, const dual<T>& rhs) {
  return !(lhs == rhs);
}

template <typename T>
constexpr bool operator!=(const dual<T>& lhs, const T& rhs) {
  return !(lhs == rhs);
}

template <typename T>
constexpr bool operator!=(const T& lhs, const dual<T>& rhs) {
  return !(lhs == rhs);
}

template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os,
    const dual<T>& x) {
  return (os << static_cast<duals::dual<T>>(x));
}

template <typename T, typename CharT, typename Traits>
std::basic_istream<CharT, Traits>& operator>>(
    std::basic_istream<CharT, Traits>& is,
    dual<T>& x) {
  duals::dual<T> tmp;
  is >> tmp;
  x = tmp;
  return is;
}

} // namespace c10

// std functions
//
// The implementation of these functions also follow the design of C++20

#if defined(__CUDACC__) || defined(__HIPCC__)
namespace c10_internal {
template <typename T>
C10_HOST_DEVICE constexpr thrust::dual<T>
cuda101bug_cast_c10_dual_to_thrust_dual(const c10::dual<T>& x) {
#if defined(CUDA_VERSION) && (CUDA_VERSION < 10200)
  // This is to circumvent a CUDA compilation bug. See
  // https://github.com/pytorch/pytorch/pull/38941 . When the bug is fixed, we
  // should do static_cast directly.
  return thrust::dual<T>(x.real(), x.imag());
#else
  return static_cast<thrust::dual<T>>(x);
#endif
}
} // namespace c10_internal
#endif

namespace std {

template <typename T>
constexpr T real(const c10::dual<T>& z) {
  return z.real();
}

template <typename T>
constexpr T imag(const c10::dual<T>& z) {
  return z.imag();
}

template <typename T>
C10_HOST_DEVICE T abs(const c10::dual<T>& z) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return thrust::abs(
      c10_internal::cuda101bug_cast_c10_dual_to_thrust_dual(z));
#else
  return std::abs(static_cast<duals::dual<T>>(z));
#endif
}

#ifdef __HIP_PLATFORM_HCC__
#define ROCm_Bug(x)
#else
#define ROCm_Bug(x) x
#endif

template <typename T>
C10_HOST_DEVICE T arg(const c10::dual<T>& z) {
  return ROCm_Bug(std)::atan2(std::imag(z), std::real(z));
}

#undef ROCm_Bug

template <typename T>
constexpr T norm(const c10::dual<T>& z) {
  return z.real() * z.real();
}

// For std::conj, there are other versions of it:
//   constexpr duals::dual<float> conj( float z );
//   template< class DoubleOrInteger >
//   constexpr duals::dual<double> conj( DoubleOrInteger z );
//   constexpr duals::dual<long double> conj( long double z );
// These are not implemented
// TODO(@zasdfgbnm): implement them as c10::conj
template <typename T>
constexpr c10::dual<T> conj(const c10::dual<T>& z) {
  return c10::dual<T>(z.real(), -z.imag());
}

// Thrust does not have dual --> dual version of thrust::proj,
// so this function is not implemented at c10 right now.
// TODO(@zasdfgbnm): implement it by ourselves

// There is no c10 version of std::polar, because std::polar always
// returns std::complex. Use c10::polar instead;

} // namespace std

namespace c10 {

template <typename T>
C10_HOST_DEVICE dual<T> polar(const T& r, const T& theta = T()) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<dual<T>>(thrust::polar(r, theta));
#else
  return static_cast<dual<T>>(std::polar(r, theta));
#endif
}

} // namespace c10

#define C10_INTERNAL_INCLUDE_DUAL_REMAINING_H
// math functions are included in a separate file
#include <c10/util/dual_math.h>
// utilities for dual types
#include <c10/util/dual_utils.h>
#undef C10_INTERNAL_INCLUDE_DUAL_REMAINING_H
