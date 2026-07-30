# Testing RDMA between two nodes

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