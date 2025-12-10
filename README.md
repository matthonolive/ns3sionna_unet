ns3sionna
============

<img src="./doc/img/ns3sionna_logo_small_sw.jpg" width="15%" alt="Logo" style="text-align: left; vertical-align: middle;">

ns3sionna is a software module that brings realistic channel simulation using ray tracing from
Sionna (https://nvlabs.github.io/sionna/) to the widely used ns-3 network simulator (https://github.com/nsnam).

<table style="width:70%">
<tr>
<td><img src="./doc/img/munich2.png" width="75%" alt="Outdoor scenario: area around Frauenkirche in Munich" style="text-align: center; vertical-align: middle;"></td>
<td><img src="./doc/img/ex2_munich_paper.jpg" width="90%" alt="Results for outdoor scenario: trajectory of STA, CSI, Prx over time, distance vs. Prx." style="text-align: center; vertical-align: middle;"></td>
</tr>
<tr>
<td>Outdoor scenario (Munich)</td>
<td>Results from simulation</td>
</tr>
</table>


More information can be found in our paper: [Ns3Sionna paper](https://arxiv.org/abs/2412.20524)

Installation
============

We recommend using Linux (e.g. Ubuntu 22 or higher) and a multi-core/GPU compute node.

0. Install all dependencies required by ns-3 and those needed by our framework: ZMQ, Protocol Buffers:

```
apt-get update
apt-get install gcc g++ python3 python3-pip cmake
apt-get install libzmq5 libzmq5-dev
apt-get install libprotobuf-dev
apt-get install protobuf-compiler
apt-get install pkg-config

```

1. Download and install ns3

```
wget https://www.nsnam.org/releases/ns-allinone-3.40.tar.bz2
tar xf ns-allinone-3.40.tar.bz2
cd ns-allinone-3.40
```

2. Clone ns3sionna repository into contrib directory of ns3:

```
cd ./ns-3.40/contrib
git clone https://github.com/tkn-tub/ns3sionna.git ./sionna
```

Note: it is important to use the sionna as the name of the ns3sionna app directory.

3. Configure and build ns-3 project:

```
cd ../
./ns3 configure --enable-examples
./ns3 build
```

Note: ns3sionna Protocol Buffer messages (C++ and Python) are build during configure.

4. Install server component of ns3sionna located in model/ns3sionna (Python3 required)

```
cd ./contrib/sionna/model/ns3sionna/
python3 -m venv sionna-venv
source sionna-venv/bin/activate
python3 -m pip install -r requirements.txt
python3 tests/test_imports.py # all packages should be correctly installed
```

or using conda:

```
cd ./contrib/sionna/model/ns3sionna/
conda create -n sionna-venv python=3.12
conda activate sionna-venv
pip install -r requirements.txt
python3 tests/test_imports.py # all packages should be correctly installed
```

Note: ns3sionna was tested with Sionna==1.2.1.

5. Start server component of ns3sionna

```
cd ./contrib/sionna/model/ns3sionna
./run.sh
```

Note: if you experience problems with Protocol Buffers use:

```
run_python_proto.sh
```


5. Start a ns-3 example script (in separate terminal)
```
cd ns-3.40/contrib/sionna/examples
./example-sionna-sensing-mobile.sh
```

Note: you need to have matplotlib installed to see the results.

Examples
========

All examples can be found [here](./examples/).

Usage in HPC cluster
============

Using Docker [here](./docker/docker_howto.md)

Using singularity [here](./singularity/singularity_howto.md)

Current limitations
========
* SISO only
* under development: MIMO

Contact
============
* Anatolij Zubow, TU-Berlin, zubow@tkn
* Sascha Roesler, TU-Berlin, zubow@tkn
* tkn = tkn.tu-berlin.de

How to reference ns3sionna?
============

Please use the following bibtex:

```
@techreport{zubow2024ns3-preprint,
    author = {Zubow, Anatolij and Pilz, Yannik and R{\"{o}}sler, Sascha and Dressler, Falko},
    doi = {10.48550/arXiv.2412.20524},
    title = {{Ns3 meets Sionna: Using Realistic Channels in Network Simulation}},
    institution = {arXiv},
    month = {12},
    number = {2412.20524},
    type = {cs.NI},
    year = {2024},
   }
```