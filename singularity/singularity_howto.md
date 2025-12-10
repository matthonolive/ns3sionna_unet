Running ns3sionna in singularity container
============

In order to run ns3sionna in a typical HPC cluster singularity is used as container.

Installation
============

1. Build writeable overlay:

```
singularity overlay create --size 2048 overlay.img

```

2. Build the container:

```
sudo singularity build ubuntu-sionna-python.sif ubuntu-sionna-python.def
```

3. Run the container

```
./start_ns3sionna_container.sh
```

