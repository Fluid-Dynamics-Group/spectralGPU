base:
	apptainer build base.sif ./base.apptainer 

build:
	apptainer build spectralGPU.sif ./spectralGPU.apptainer 
