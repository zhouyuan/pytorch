#if !defined(C10_INTERNAL_INCLUDE_DUAL_REMAINING_H)
#error \
    "c10/util/dual_utils.h is not meant to be individually included. Include c10/util/dual.h instead."
#endif

#include <limits>

/*
MMC - July 2021
*/

namespace c10 {

template <typename T>
struct is_dual : public std::false_type {};

template <typename T>
struct is_dual<duals::dual<T>> : public std::true_type {};

template <typename T>
struct is_dual<c10::dual<T>> : public std::true_type {};

// Extract double from duals::dual<double>; is identity otherwise
// TODO: Write in more idiomatic C++17
template <typename T>
struct scalar_value_type {
  using type = T;
};
template <typename T>
struct scalar_value_type<duals::dual<T>> {
  using type = T;
};
template <typename T>
struct scalar_value_type<c10::dual<T>> {
  using type = T;
};

} // namespace c10

namespace std {

template <typename T>
class numeric_limits<c10::dual<T>> : public numeric_limits<T> {};

} // namespace std
