To run in docker container:
1. Run `docker build -t local-torch-geometric .` in `torch-geometric-docker`.
1. Run `docker build -t degree-project .` in root folder.
1. Run `docker run -it --entrypoint=/bin/bash degree-project`.

Then inside the container you are able to run the programs manually 
providing proper arguments.

In case of `Bus error.` problem try setting `shm-size` (shared memory) parameter of the container to a higher value (e.g 8G).

`docker run --shm-size=8G -it --entrypoint=/bin/bash degree-project`.

In case you cannot run the experiments on CPU only set the environmental variable:

`export CUDA_VISIBLE_DEVICES=""`

To make PyTorch use only 1 core per process:

`export MKL_NUM_THREADS=1`

`export NUMEXPR_NUM_THREADS=1`

`export OMP_NUM_THREADS=1`

`sudo docker volume create michal`
`sudo docker run --mount source=michal,target=/app --gpus device=1 --shm-size 8G --entrypoint=/bin/bash degree-project`