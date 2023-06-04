# spectralGPU

[![Build Status](https://github.com/vanillabrooks/spectralGPU.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vanillabrooks/spectralGPU.jl/actions/workflows/CI.yml?query=branch%3Amain)

## FAQ

```
warning: Linking two modules of different data layouts: '' is 'e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-ni:10:11:12:13' whereas 'start' is 'e-p:64:64:64:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-ni:10:11:12:13'
```

fix:

```
julia
]
update CUDA
precompile
```
