Running ns3sionna in docker container
============

In order to run ns3sionna inside docker do:

Installation
============

1. Build the container image:

```
sudo docker build -t ns3sionna:latest .
```

2. Create and start a new container from an image:

```
docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --name ns3sionna-container ns3sionna:latest
```

Note: you need to have GPU drivers installed on the host.

3. Start the existing container:

```
docker start -ai ns3sionna-container
```

Inside the container you need to source the Python env:

```
source /opt/sionna-venv/bin/activate
```

4. To connect a second terminal:

```
docker exec -it ns3sionna-container /bin/bash
```

