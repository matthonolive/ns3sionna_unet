ns3sionna
======

Sionna RT meets ns-3

## Starting

To start Python server component:
```
./run.sh
```

or

```
./run_fast.sh
```
if you want faster but less accurate simulations.

Note: in case you have problems with Protocol Buffers you can try:

```
./run_python_proto.sh
```

## TODOs

Work needs to be done on:
- better receiver filtering - no need to compute link between too distant nodes
- mode MODE_P2MP_LAH: swapping mobile TX with fixed RX due to channel reciprocity
- providing support for MODE_P2MP_LAH in case all nodes are mobile
