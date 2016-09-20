#ifndef _SIMD_ALLOC_HPP_
#define _SIMD_ALLOC_HPP_

#include <cstddef>
#include <limits>
#include <cassert>
#include <cstdint>
#include <new>
#include <vector>
#include <memory>
#include "config.hpp"

namespace simd
{

/**
 * Defines a type for an aligned array of type T whith size SIZE
 * Example usage:
 * aligned_static_array<double, 20> elems;
 * for(int i=0;i<20;++i)
 * elems[i] = x+i*h;
 */

template<typename T, std::size_t SIZE>
using aligned_static_array alignas(alignment()) = T[SIZE];

/**
 * returns a pointer that is suitably aligned
 * for all simd instructions requiring alignment.
 */
inline void* aligned_alloc( std::size_t size )
{
   using std::uintptr_t;

   if( MALLOC_ALIGNMENT == alignment() )
      return ::operator new( size );

   uintptr_t ptr = reinterpret_cast<uintptr_t>( ::operator new( size + alignment() ) );

   uintptr_t* aligned_ptr = reinterpret_cast<uintptr_t*>( ( ptr | ( alignment() - 1 ) ) + 1 );
   ASSERT_ALIGNED( aligned_ptr );
   * ( aligned_ptr - 1 ) = ptr;
   return aligned_ptr;
}

/**
 * Properly deallocates memory allocated by aligned_alloc
 */
inline void aligned_free( void* aligned_ptr )
{
   using std::uintptr_t;
   
   if( MALLOC_ALIGNMENT == alignment() )
   {
      ::operator delete( aligned_ptr );
   }
   else if(aligned_ptr)
   {
      ::operator delete( reinterpret_cast<void*>( * ( reinterpret_cast<uintptr_t*>( aligned_ptr ) - 1 ) ) );
   }
}

inline void* cache_aligned_alloc( std::size_t size )
{
   using std::uintptr_t;

   uintptr_t ptr = reinterpret_cast<uintptr_t>( ::operator new( size + cache_alignment() ) );

   uintptr_t* aligned_ptr = reinterpret_cast<uintptr_t*>( ( ptr | ( cache_alignment() - 1 ) ) + 1 );

   * ( aligned_ptr - 1 ) = ptr;
   return aligned_ptr;
}

inline void cache_aligned_free( void* aligned_ptr )
{
   using std::uintptr_t;

   if(aligned_ptr)
   {
      ::operator delete( reinterpret_cast<void*>( * ( reinterpret_cast<uintptr_t*>( aligned_ptr ) - 1 ) ) );
   }
}


template<typename T>
struct aligned_allocator : public std::allocator<T>
{
   typedef typename std::allocator<T>::size_type size_type;
   typedef typename std::allocator<T>::pointer pointer;

   aligned_allocator() {}

   template<typename U>
   aligned_allocator(const aligned_allocator<U> &other) {}

   template<typename U>
   struct rebind
   {
      typedef aligned_allocator<U> other;
   };

   pointer allocate(
      size_type cnt,
      typename std::allocator<void>::const_pointer = 0
   )
   {
      return reinterpret_cast<pointer>( aligned_alloc( cnt * sizeof( T ) ) );
   }

   void deallocate( pointer p, size_type )
   {
      aligned_free( p );
   }
 
};


template<typename T>
using aligned_vector = std::vector< T, aligned_allocator<T> >;

struct aligned_deleter {
   void operator()(void *ptr) { aligned_free(ptr); }
};

template<typename T>
using aligned_array = std::unique_ptr<T[], aligned_deleter>;

template<typename T>
aligned_array<T> alloc_aligned_array(std::size_t n)
{
   return aligned_array<T>( (T*)aligned_alloc(n*sizeof(T)) );
}


} //namespace simd

#endif
