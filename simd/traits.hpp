#ifndef _SIMD_TRAITS_HPP_
#define _SIMD_TRAITS_HPP_

#include <type_traits>

namespace simd {

template<typename T>
struct supported
{
   using raw_type = typename std::remove_all_extents<typename std::remove_cv<T>::type>::type;
   constexpr static bool value = std::is_same<raw_type, float>::value || std::is_same<raw_type, double>::value;
};

}//simd

#endif