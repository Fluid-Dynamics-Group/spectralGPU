base:
	sudo -E apptainer build base.sif ./base.apptainer 

build:
	sudo -E apptainer build spectralGPU.sif ./spectralGPU.apptainer 
