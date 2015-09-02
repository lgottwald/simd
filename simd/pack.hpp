#ifndef _SIMD_PACK_HPP_
#define _SIMD_PACK_HPP_
/* some portability defines */

#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <array>
#include "config.hpp"

namespace simd
{


/**
 * class that stores packed floats or doubles and enables arithmetic operators, bitwise operators and comparisons on simd types.
 * The number of elements stored changes depending on the available instruction set (sse=16 bytes, avx=32 bytes).
 * Comparison operators do not produce boolean values but bitmasks.
 */
template<typename T>
class pack
{
public:

   pack() = default;

   pack( const T* data )
   {
      for( int i = 0; i < pack_size<T>(); ++i )
         elem_[i] = data[i];
   }

   pack( T scalar )
   {
      for( int i = 0; i < pack_size<T>(); ++i )
         elem_[i] = scalar;
   }
   /* implicit conversions */
   pack( const xmm_t& xmm ) : xmm_( xmm ) {}
   pack( const xmmd_t& xmmd ) : xmmd_( xmmd ) {}

   operator xmm_t() const
   {
      return xmm_;
   }
   operator xmmd_t() const
   {
      return xmmd_;
   }

   /**
    * Access element i of packed elements
    */
   T& operator[]( std::size_t i )
   {
      return elem_[i];
   }

   /**
    * Access element i of packed elements
    */
   const T& operator[]( std::size_t i ) const
   {
      return elem_[i];
   }

   /* operators throw domain_errors for unsupported types but are specialized for floats and doubles */

   /**
    * Multiply this pack whith given packed values for supported types and throws a domain_error
    * for unsupported instantiations.
    */
   pack<T>& operator*= ( const pack<T>& );

   /**
    * Adds given packed values to this pack for supported types and throws a domain_error
    * for unsupported instantiations.
    */
   pack<T>& operator+= ( const pack<T>& );

   /**
    * Substracts given packed values from this pack for supported types and throws a domain_error
    * for unsupported instantiations.
    */
   pack<T>& operator-= ( const pack<T>& );

   /**
    * Divides this pack whith given packed values for supported types and throws a domain_error
    * for unsupported instantiations.
    */
   pack<T>& operator/= ( const pack<T>& );

   /**
    * Computes the bitwise and of this pack and the given packed values for supported types and
    * throws a domain_error for unsupported instantiations.
    */
   pack<T>& operator&= ( const pack<T>& );

   /**
    * Computes the bitwise or of this pack and the given packed values for supported types and
    * throws a domain_error for unsupported instantiations.
    */
   pack<T>& operator|= ( const pack<T>& );

   /**
    * Computes the bitwise xor of this pack and the given packed values for supported types and
    * throws a domain_error for unsupported instantiations.
    */
   pack<T>& operator^= ( const pack<T>& );

   /**
    * Just returns this pack unchanged
    */
   const pack<T> & operator+() const
   {
      return *this;
   }

   /**
    * Returns a pack with the negated values of this pack for supported types and
    * throws a domain_error for unsupported instantiations.
    */
   pack<T> operator-() const;
   /**
    * Stores this packs elements to an aligned memory location. Results in a segmentation
    * fault if memory location is unaligned. For unsupported instantiations a domain_error
    * is thrown.
    */
   void aligned_store( T* mem ) const;

private:

   union
   {
      xmm_t xmm_;
      xmmd_t xmmd_;
      T elem_[sizeof( xmm_t ) / sizeof( T )];
   };

};


/**
 * utility function to round given integer to the next multiple
 * of pack_size<T>(). i.e. the following statements are true:
 *       -> n <= next_size<T>(n) && next_size<T>(n) % pack_size<T>() == 0
 *       -> n >= prev_size<T>(n) && prev_size<T>(n) % pack_size<T>() == 0
 */
template<typename T>
constexpr std::size_t next_size( std::size_t n )
{
   //overflow does not matter as (0-1)+1 = 0 is guaranteed for unsigned types
   return ( ( n - 1u ) | ( pack_size<T>() - 1u ) ) + 1u;
}

/**
 * utility function to round given unsigned integer to the previous multiple
 * of this->size(). i.e. the following statement is true:
 *       -> n >= prev_size<T>(n) && prev_size<T>(n) % pack_size<T>() == 0
 */

template<typename T>
constexpr std::size_t prev_size( std::size_t n )
{
   return n ^ ( n & ( pack_size<T>() - 1 ) );
}


template<typename T>
inline pack<T> zero();

/**
 * Loads a simd::pack<double>::size() elements from the given memory
 * location into a simd register and returns the packed double
 * representing the values in the register.
 * Raises a segfault if given pointer is not aligned. Also the caller
 * has to make sure that *(aligned_ptr+simd::pack<double>::size()-1)
 * accesses a legal memory location.
 */

template<typename ITER, typename REAL = typename std::iterator_traits<ITER>::value_type,
        typename std::enable_if<std::is_same<double,REAL>::value, int>::type enable = 0 >
inline pack<REAL> aligned_load( ITER aligned_ptr )
{
   ASSERT_ALIGNED( aligned_ptr );
   return SIMD_INTRIN( load_pd )( &(*aligned_ptr) );
}

/**
 * Loads a simd::pack<float>::size() elements from the given memory
 * location into a simd register and returns the packed double
 * representing the values in the register.
 * Raises a segfault if given pointer is not aligned. Also the caller
 * has to make sure that *(aligned_ptr+simd::pack<float>::size()-1)
 * accesses a legal memory location.
 */
template<typename ITER, typename REAL = typename std::iterator_traits<ITER>::value_type,
         typename std::enable_if<std::is_same<float,REAL>::value, int>::type enable = 0>
inline pack<REAL> aligned_load( ITER aligned_ptr )
{
   ASSERT_ALIGNED( aligned_ptr );
   return SIMD_INTRIN( load_ps )( &(*aligned_ptr) );
}

template<typename ITER>
ITER& advance_iter(ITER& itr)
{
    using REAL = typename std::iterator_traits<ITER>::value_type;
    return itr += pack_size<REAL>();
}

template<typename KERNEL, typename OUT_ITER, typename... ARG_ITERS>
void aligned_transform(KERNEL kernel, OUT_ITER out, OUT_ITER end, ARG_ITERS... args )
{
    using REAL = typename std::iterator_traits<OUT_ITER>::value_type;
    if(out == end)
        return;
    
    kernel(aligned_load(args)... ).aligned_store( &(*out) );
    out += pack_size<REAL>();
    while(out != end)
    {        
        kernel(aligned_load(advance_iter(args))... ).aligned_store( &(*out) );
        out += pack_size<REAL>();
    }
}

template<std::size_t MAX, std::size_t POS = 0, std::size_t N, typename REAL, typename ITER, typename... ITERS,
         typename std::enable_if< (POS < MAX) ,int >::type enable = 0>
        
void aligned_multi_store(const std::array<REAL,N> &in, ITER out, ITERS... more_out )
{
    in[POS].aligned_store( &(*out) );
    aligned_multi_store<MAX,POS+1>(in, more_out...);
}

template<std::size_t MAX, std::size_t POS = 0, std::size_t N, typename REAL, typename ITER, typename... ITERS,
         typename std::enable_if< (POS == MAX) ,int >::type enable = 0>

void aligned_multi_store(const std::array<REAL,N> &in, ITER out, ITERS... more_out )
{}

template<int OUTPUT_PARAMS=1, typename KERNEL, typename ITER, typename... MORE_ITERS>
void aligned_transform(KERNEL kernel, std::size_t N, ITER itr, MORE_ITERS... itrs )
{
    using REAL = typename std::iterator_traits<ITER>::value_type;

    ITER end = itr+N;
    if(itr == end)
        return;
    
    {
        std::array< pack<REAL>, sizeof...(MORE_ITERS)+1 > packs{aligned_load(itr), aligned_load(itrs)... };
        kernel(packs);
        aligned_multi_store<OUTPUT_PARAMS>(packs, itr, itrs...);
        itr += pack_size<REAL>();
    }
    while(itr != end)
    {        
        std::array< pack<REAL>, sizeof...(MORE_ITERS)+1 > packs{aligned_load(itr), aligned_load(advance_iter(itrs))...};
        kernel(packs);
        aligned_multi_store<OUTPUT_PARAMS>(packs, itr, itrs...);
        itr += pack_size<REAL>();
    }
}

template<typename REAL, typename OUT_ITER>
void aligned_fill(const pack<REAL> &val, OUT_ITER out, OUT_ITER end)
{
    while(out != end)
    {      
        val.aligned_store( &(*out) );
        out += pack_size<REAL>();
    }
}

/* specializations for floats and doubles */
template<>
inline void pack<double>::aligned_store( double* mem ) const
{
   ASSERT_ALIGNED( mem );
   return SIMD_INTRIN( store_pd )( mem, xmmd_ );
}

template<>
inline void pack<float>::aligned_store( float* mem ) const
{
   ASSERT_ALIGNED( mem );
   return SIMD_INTRIN( store_ps )( mem, xmm_ );
}

template<>
inline pack<double> zero<double>()
{
   return SIMD_INTRIN( setzero_pd )();
}

template<>
inline pack<float> zero<float>()
{
   return SIMD_INTRIN( setzero_ps )();
}

template<>
inline pack<double>::pack( double scalar )
{
   xmmd_ = SIMD_INTRIN( set1_pd )( scalar );
}

template<>
inline pack<float>::pack( float scalar )
{
   xmm_ = SIMD_INTRIN( set1_ps )( scalar );
}

template<>
inline pack<double>& pack<double>::operator*= ( const pack<double>& other )
{
   xmmd_ = SIMD_INTRIN( mul_pd )( xmmd_, other.xmmd_ );
   return *this;
}

template<>
inline pack<float>& pack<float>::operator*= ( const pack<float>& other )
{
   xmm_ = SIMD_INTRIN( mul_ps )( xmm_, other.xmm_ );
   return *this;
}

template<>
inline pack<double>& pack<double>::operator+= ( const pack<double>& other )
{
   xmmd_ = SIMD_INTRIN( add_pd )( xmmd_, other.xmmd_ );
   return *this;
}

template<>
inline pack<float>& pack<float>::operator+= ( const pack<float>& other )
{
   xmm_ = SIMD_INTRIN( add_ps )( xmm_, other.xmm_ );
   return *this;
}

template<>
inline pack<double>& pack<double>::operator-= ( const pack<double>& other )
{
   xmmd_ = SIMD_INTRIN( sub_pd )( xmmd_, other.xmmd_ );
   return *this;
}

template<>
inline pack<float>& pack<float>::operator-= ( const pack<float>& other )
{
   xmm_ = SIMD_INTRIN( sub_ps )( xmm_, other.xmm_ );
   return *this;
}

template<>
inline pack<double>& pack<double>::operator/= ( const pack<double>& other )
{
   xmmd_ = SIMD_INTRIN( div_pd )( xmmd_, other.xmmd_ );
   return *this;
}

template<>
inline pack<float>& pack<float>::operator/= ( const pack<float>& other )
{
   xmm_ = SIMD_INTRIN( div_ps )( xmm_, other.xmm_ );
   return *this;
}

template<>
inline pack<double>& pack<double>::operator&= ( const pack<double>& other )
{
   xmmd_ = SIMD_INTRIN( and_pd )( xmmd_, other.xmmd_ );
   return *this;
}

template<>
inline pack<float>& pack<float>::operator&= ( const pack<float>& other )
{
   xmm_ = SIMD_INTRIN( and_ps )( xmm_, other.xmm_ );
   return *this;
}

template<>
inline pack<double>& pack<double>::operator|= ( const pack<double>& other )
{
   xmmd_ = SIMD_INTRIN( or_pd )( xmmd_, other.xmmd_ );
   return *this;
}

template<>
inline pack<float>& pack<float>::operator|= ( const pack<float>& other )
{
   xmm_ = SIMD_INTRIN( or_ps )( xmm_, other.xmm_ );
   return *this;
}

template<>
inline pack<double>& pack<double>::operator^= ( const pack<double>& other )
{
   xmmd_ = SIMD_INTRIN( xor_pd )( xmmd_, other.xmmd_ );
   return *this;
}

template<>
inline pack<float>& pack<float>::operator^= ( const pack<float>& other )
{
   xmm_ = SIMD_INTRIN( xor_ps )( xmm_, other.xmm_ );
   return *this;
}

template<>
inline pack<double> pack<double>::operator-() const
{
   return SIMD_INTRIN( xor_pd )( xmmd_, SIMD_INTRIN( set1_pd )( -0.0 ) );
}

template<>
inline pack<float> pack<float>::operator-() const
{
   return SIMD_INTRIN( xor_ps )( xmm_, SIMD_INTRIN( set1_ps )( -0.0f ) );
}

template<typename T>
pack<T> operator+ ( pack<T> lhs, const pack<T>& rhs )
{
   lhs += rhs;
   return lhs;
}

template<typename T>
pack<T> operator- ( pack<T> lhs, const pack<T>& rhs )
{
   lhs -= rhs;
   return lhs;
}

template<typename T>
pack<T> operator* ( pack<T> lhs, const pack<T>& rhs )
{
   lhs *= rhs;
   return lhs;
}

template<typename T>
pack<T> operator/ ( pack<T> lhs, const pack<T>& rhs )
{
   lhs /= rhs;
   return lhs;
}

template<typename T>
pack<T> operator| ( pack<T> lhs, const pack<T>& rhs )
{
   lhs |= rhs;
   return lhs;
}

template<typename T>
pack<T> operator& ( pack<T> lhs, const pack<T>& rhs )
{
   lhs &= rhs;
   return lhs;
}

template<typename T>
pack<T> operator^ ( pack<T> lhs, const pack<T>& rhs )
{
   lhs ^= rhs;
   return lhs;
}

/* Comparison operators produce bitmasks */

inline pack<double> operator< ( const pack<double>& a, const pack<double>& b )
{
   IF_SSE( return _mm_cmplt_pd( a, b ); )
   IF_AVX( return _mm256_cmp_pd( a, b, _CMP_LT_OQ ); )
}

inline pack<float> operator< ( const pack<float>& a, const pack<float>& b )
{
   IF_SSE( return _mm_cmplt_ps( a, b ); )
   IF_AVX( return _mm256_cmp_ps( a, b, _CMP_LT_OQ ); )
}

inline pack<double> operator<= ( const pack<double>& a, const pack<double>& b )
{
   IF_SSE( return _mm_cmple_pd( a, b ); )
   IF_AVX( return _mm256_cmp_pd( a, b, _CMP_LE_OQ ); )
}

inline pack<float> operator<= ( const pack<float>& a, const pack<float>& b )
{
   IF_SSE( return _mm_cmple_ps( a, b ); )
   IF_AVX( return _mm256_cmp_ps( a, b, _CMP_LE_OQ ); )
}

inline pack<double> operator> ( const pack<double>& a, const pack<double>& b )
{
   IF_SSE( return _mm_cmpgt_pd( a, b ); )
   IF_AVX( return _mm256_cmp_pd( a, b, _CMP_GT_OQ ); )
}

inline pack<float> operator> ( const pack<float>& a, const pack<float>& b )
{
   IF_SSE( return _mm_cmpgt_ps( a, b ); )
   IF_AVX( return _mm256_cmp_ps( a, b, _CMP_GT_OQ ); )
}

inline pack<double> operator>= ( const pack<double>& a, const pack<double>& b )
{
   IF_SSE( return _mm_cmpge_pd( a, b ); )
   IF_AVX( return  _mm256_cmp_pd( a, b, _CMP_GE_OQ ); )
}

inline pack<float> operator>= ( const pack<float>& a, const pack<float>& b )
{
   IF_SSE( return _mm_cmpge_ps( a, b ); )
   IF_AVX( return  _mm256_cmp_ps( a, b, _CMP_GE_OQ ); )
}

inline pack<double> operator== ( const pack<double>& a, const pack<double>& b )
{
   IF_SSE( return _mm_cmpeq_pd( a, b ); )
   IF_AVX( return _mm256_cmp_pd( a, b, _CMP_EQ_OQ ); )
}

inline pack<float> operator== ( const pack<float>& a, const pack<float>& b )
{
   IF_SSE( return _mm_cmpeq_ps( a, b ); )
   IF_AVX( return _mm256_cmp_ps( a, b, _CMP_EQ_OQ ); )
}

inline pack<double> operator!= ( const pack<double>& a, const pack<double>& b )
{
   IF_SSE( return  _mm_cmpneq_pd( a, b ); )
   IF_AVX( return  _mm256_cmp_pd( a, b, _CMP_NEQ_OQ ); )
}

inline pack<float> operator!= ( const pack<float>& a, const pack<float>& b )
{
   IF_SSE( return  _mm_cmpneq_ps( a, b ); )
   IF_AVX( return  _mm256_cmp_ps( a, b, _CMP_NEQ_OQ ); )
}

inline pack<double> sqrt( const pack<double>& x )
{
   return SIMD_INTRIN( sqrt_pd )( x );
}

inline pack<float> sqrt( const pack<float>& x )
{
   return SIMD_INTRIN( sqrt_ps )( x );
}

inline pack<double> max( const pack<double>& a, const pack<double>& b )
{
   return SIMD_INTRIN( max_pd )( a, b );
}

inline pack<float> max( const pack<float>& a, const pack<float>& b )
{
   return SIMD_INTRIN( max_ps )( a, b );
}

inline pack<double> min( const pack<double>& a, const pack<double>& b )
{
   return SIMD_INTRIN( min_pd )( a, b );
}

inline pack<float> min( const pack<float>& a, const pack<float>& b )
{
   return SIMD_INTRIN( min_ps )( a, b );
}

/**
 * Converts a given pack<double> that represents a mask, e.g. a result from
 * a comparison, to an integer, which has the bits set to 1 if the mask
 * of the corresponding element from the given pack<double> is 1 and to 0
 * otherwise. I.e. for 64bit doubles the mask 0xFFFFFFFF00000000 stored
 * in a 16byte pack<double> (sse) is converted to the integer 0x2 or written
 * in binary 00...0010.
 */
inline int movemask( const pack<double>& p )
{
   return SIMD_INTRIN( movemask_pd )( p );
}

/**
 * Converts a given pack<float> that represents a mask, e.g. a result from
 * a comparison, to an integer, which has the bits set to 1 if the mask
 * of the corresponding element from the given pack<float> is 1 and to 0
 * otherwise. I.e. for 64bit doubles the mask 0x0000FFFF0000FFFF stored
 * in a 16byte pack<float> (sse) is converted to the integer 0x5 or written
 * in binary 00...0101.
 */
inline int movemask( const pack<float>& p )
{
   return SIMD_INTRIN( movemask_ps )( p );
}

/**
 * Returns the absolute values of the given packed doubles.
 */
inline pack<double> abs( const pack<double>& x )
{
   static const pack<double> sign_mask( -0.0 );
   return SIMD_INTRIN( andnot_pd )( sign_mask, x );
}

/**
 * Returns the absolute values of the given packed floats.
 */
inline pack<float> abs( const pack<float>& x )
{
   static const pack<float> sign_mask( -0.0f );
   return SIMD_INTRIN( andnot_ps )( sign_mask, x );
}

/**
 * Returns a pack containing the i'th element of the given pack
 * in each position.
 */
inline pack<double> unpack( const pack<double> &x, int i)
{
   IF_SSE(
      return _mm_shuffle_pd(x, x, i | (i<<1));
   )
   IF_AVX(
      int j = i&1;
      xmmd_t y = _mm256_shuffle_pd(x, x, j | j<<1 | j<<2 | j<<3);
      j = i>>1;
      return _mm256_permute2f128_pd(y, y, j | (j<<4));
   )
}


/**
 * Returns a pack containing the i'th element of the given pack
 * in each position.
 */
inline pack<float> unpack( const pack<float> &x, int i)
{
   IF_SSE(
      return _mm_shuffle_ps(x, x, i | (i<<2) | (i<<4) | (i<<6));
   )
   IF_AVX(
      int j = i&3;
      xmm_t y = _mm256_shuffle_ps(x, x, j | (j<<2) | (j<<4) | (j<<6));
      j=i>>2;
      return _mm256_permute2f128_ps(y, y, j | (j<<4));
   )
}



/**
 * Returns the sum of the elements in the given packed double.
 */
inline double sum( const pack<double>& a )
{
#if defined(__AVX__)
   __m256d temp = _mm256_hadd_pd( a, a );
   __m128d hi128 = _mm256_extractf128_pd( temp, 1 );
   __m128d sum = _mm_add_pd( _mm256_castpd256_pd128( temp ), hi128 );
   return _mm_cvtsd_f64( sum );
#elif defined(__SSE4_1__) || defined(__SSE3__)
   return _mm_cvtsd_f64( _mm_hadd_pd( a, a ) );
#else
   return _mm_cvtsd_f64( _mm_add_pd( _mm_shuffle_pd( a, a, 1 ), a ) );
#endif
}

/**
 * Returns the sum of the elements in the given packed floats.
 */
inline float sum( const pack<float>& a )
{
#if defined(__AVX__)
   __m256 temp = _mm256_hadd_ps( a, a );
   temp = _mm256_hadd_ps( temp, temp );
   __m128 hi128 = _mm256_extractf128_ps( temp, 1 );
   __m128 sum = _mm_add_ps( _mm256_castps256_ps128( temp ), hi128 );
   return _mm_cvtss_f32( sum );
#elif defined(__SSE4_1__) || defined(__SSE3__)
   __m128 b = _mm_hadd_ps( a, a );
   return _mm_cvtss_f32( _mm_hadd_ps( b, b ) );
#else
   __m128 s = _mm_shuffle_ps( a, a, _MM_SHUFFLE( 2, 3, 0, 1 ) );
   __m128 as = _mm_add_ps( a, s );
   s = _mm_shuffle_ps( as, as, _MM_SHUFFLE( 0, 1, 2, 3 ) );
   as = _mm_add_ps( as, s );
   return _mm_cvtss_f32( as );
#endif
}

/**
 * Returns the dotproduct of the two given packed doubles.
 */
inline double dot( const pack<double>& a, const pack<double>& b )
{
#if defined(__SSE_4_1) && !defined(__AVX__)
   return _mm_cvtsd_f64( _mm_dp_pd( a, b, 0xF1 ) );
#else
   return sum( a * b );
#endif
}

/**
 * Returns the dotproduct of the two given packed floats.
 */
inline float dot( const pack<float>& a, const pack<float>& b )
{
#if defined(__AVX__)
   __m256 temp = _mm256_dp_ps( a, b, 0xF1 );
   __m128 hi128 = _mm256_extractf128_ps( temp, 1 );
   __m128 dotproduct = _mm_add_ps( _mm256_castps256_ps128( temp ), hi128 );
   return _mm_cvtss_f32( dotproduct );
#elif defined(__SSE4_1__)
   return _mm_cvtss_f32( _mm_dp_ps( a, b, 0xF1 ) );
#else
   return sum( a * b );
#endif
}

#ifdef __SSE4_1__

inline pack<double> floor( const pack<double>& x )
{
   return SIMD_INTRIN( floor_pd )( x );
}

inline pack<float> floor( const pack<float>& x )
{
   return SIMD_INTRIN( floor_ps )( x );
}

inline pack<double> ceil( const pack<double>& x )
{
   return SIMD_INTRIN( ceil_pd )( x );
}

inline pack<float> ceil( const pack<float>& x )
{
   return SIMD_INTRIN( ceil_ps )( x );
}

#endif

} //namespace simd

#endif
