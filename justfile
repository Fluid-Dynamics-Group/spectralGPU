base:
	sudo -E apptainer build base.sif ./base.apptainer 

build:
	sudo -E apptainer build spectralGPU.sif ./spectralGPU.apptainer 

test:
	#julia ./test/cpu_singlethread.jl
	julia ./test/cuda.jl

bench:
	julia ./benches/cpu_gpu.jl
