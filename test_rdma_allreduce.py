import os
import time

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"[setup] world_size={world}", flush=True)

    # 1 GiB tensor (256M fp32 elements)
    numel = 256 * 1024 * 1024
    x = torch.ones(numel, device=dev)
    nbytes = x.element_size() * x.numel()

    # warmup
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    # all-reduce bus-bandwidth: 2*(n-1)/n * size / time
    algbw = nbytes / dt
    busbw = algbw * 2 * (world - 1) / world
    if rank == 0:
        print(
            f"[result] size={nbytes/1e9:.2f} GB  time={dt*1e3:.2f} ms  "
            f"algbw={algbw/1e9:.1f} GB/s  busbw={busbw/1e9:.1f} GB/s",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
