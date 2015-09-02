# simd

Wrapper for simd intrinsics. This is a header only library and requires no build step.
Just use the simd::pack\<float\> or simd::pack\<double\> class and enable sse or avx instructions.
Then the best intruction set will be automatically used by the pack class.

The cmake generated makefile will have a target to build the source code documentation:
```
make doc
```
And the install step will install a cmake config file to easily
find this package from other projects that use cmake:
```
find_package(simd REQUIRED)

include_directories(
 ${simd_INCLUDE_DIRS}
)
```

The documentation is currently in progress.
