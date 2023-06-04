base:
	sudo -E apptainer build base.sif ./base.apptainer 

build:
	sudo -E apptainer build spectralGPU.sif ./spectralGPU.apptainer 

test:
	#julia ./test/cpu_singlethread.jl
	julia ./test/cuda.jl

bench:
	#julia ./benches/cuda.jl
	#julia ./benches/fft.jl
	#julia ./benches/curl.jl
	#julia ./benches/cross.jl
	#julia ./benches/compute_rhs.jl
	julia ./benches/cpu_gpu.jl

pluto:
	cd pluto && JULIA_CUDA_SOFT_MEMORY_LIMIT="70%" julia -e "using Pluto; Pluto.run()"

jupyter:
	cd jupyter && jupyter notebook

clean-jupyter:
	nb-clean clean ./jupyter/fft_debug.py.ipynb
	nb-clean clean ./jupyter/math_check.ipynb
