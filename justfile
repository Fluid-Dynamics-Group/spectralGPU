base:
	sudo -E apptainer build base.sif ./base.apptainer 

build:
	sudo -E apptainer build spectralGPU.sif ./spectralGPU.apptainer 

test:
	julia ./test/cpu_singlethread.jl
	julia ./test/cuda.jl

bench:
	#julia ./benches/cuda.jl
	#julia ./benches/fft.jl
	#julia ./benches/curl.jl
	#julia ./benches/cross.jl
	julia ./benches/compute_rhs.jl
	#julia ./benches/cpu_gpu.jl

pluto:
	cd pluto && julia -e "using Pluto; Pluto.run()"
