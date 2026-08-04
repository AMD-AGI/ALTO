export NCCL_DEBUG=INFO
# export NCCL_IB_HCA=mlx5        # use your device prefix from ibdev2netdev
# export NCCL_SOCKET_IFNAME=<netdev>   # the bootstrap iface (e.g. enp1s0f0 or eth0) if autodetect picks the wrong one


torchrun \
  --nnodes=2 --nproc-per-node=8 --node-rank=0 \
  --master-addr=<NODE_A_IP> --master-port=29500 \
  rdma_tests/test_rdma_allreduce.py
