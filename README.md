To run in docker container:
1. Run `docker build -t local-torch-geometric .` in `torch-geometric-docker`.
1. Run `docker build -t degree-project .` in root folder.
1. Run `docker run -it --entrypoint=/bin/bash degree-project`.

Then inside the container you are able to run the programs manually 
providing proper arguments.
