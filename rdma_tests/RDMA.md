# Testing RDMA between two nodes

Note: you will need `apt-get update && apt-get install -y ibverbs-utils iproute2 perftest` for tools inside docker container.

```bash
`ibv_devices` # list RDMA devices (e.g. mlx5_0)
`ibv_devinfo` # PortState: PORT_ACTIVE and note the link layer (IB vs. Ethernet/ROCE)
`rdma link show` # link state per device
```

# Raw RDMA loopback between two nodes

```bash
# On node A (server):
ib_write_bw -d mlx5_0 -F --report_gbits

# On node B (client), point at node A's IP:
ib_write_bw -d mlx5_0 -F --report_gbits <NODE_A_IP>

```

You can get `<NODE_A_IP>` for a given RDMA interface like this:

```bash
# Map RDMA device -> netdev:
ibdev2netdev                        # look for a line like: "mlx5_0 port 1 ==> rdma0 (Up)"
rdma link show                      # alternatively, look here. e.g. mlx5_0/1 ... netdev rdma0

# Then get that interface's IP:
ip -4 addr show rdma0            # look for the "inet x.x.x.x" line

```

# Test RDMA using torchrun

If you have terminal access to both machines, you can use `torchrun` to double check RDMA support. Look for a line in NCCL INFO outputs that says `NET/IB : Using [0]mlx5_1:1/RoCE ... [9]mlx5_9:1/RoCE`. If it falls back to TCP you will get an output like: `NCCL INFO NET/Socket ...`.

```bash
export NCCL_DEBUG=INFO

torchrun \
  --nnodes=2 --nproc-per-node=8 --node-rank=<rank> \
  --master-addr=<NODE_A_IP> --master-port=29500 \
  test_rdma_allreduce.py
```