#!/bin/bash
# Start the ALTO multinode container with host-provided RDMA support.
#
# RDMA providers (the libibverbs plugins, e.g. libmlx5 / libbnxt_re) are ABI-tied
# to the host's rdma-core AND kernel, both of which vary across our machines. So
# instead of baking a provider into the image, we mount the host's entire
# libibverbs userspace -- the library, its provider modules, and the .driver
# registration files -- read-only over the same paths. The in-container RDMA
# stack then always matches whatever kernel/NIC this particular host has.
#
# Set USE_HOST_RDMA=0 to skip the mounts and use the image's own stack.
set -euo pipefail

IMAGE="${IMAGE:-alto:multinode}"
CONTAINER="${CONTAINER:-alto_multinode}"
ALTO_DIR="${ALTO_DIR:-$HOME/lpt_branch/ALTO}"

docker_args=(
    -it
    --rm
    --name "$CONTAINER"
    --network host
    --ipc host
    --shm-size=16g
    --cap-add=IPC_LOCK
    --ulimit memlock=-1:-1
    -v "$ALTO_DIR:/alto"
)

# GPU / RDMA character devices -- add only if present on this host.
for dev in /dev/infiniband /dev/dri /dev/kfd; do
    [[ -e "$dev" ]] && docker_args+=(--device="$dev")
done

# Host RDMA userspace. Mounting the host's libibverbs.so together with its
# provider modules and driver configs keeps the library ABI self-consistent
# (host lib + host providers) while matching the host kernel's uABI.
if [[ "${USE_HOST_RDMA:-1}" == "1" && -d /etc/libibverbs.d ]]; then
    for lib in $(ldconfig -p | awk '/lib(ibverbs|rdmacm|ibumad|mlx5|mlx4|bnxt_re|efa|irdma|hns|cxgb4)\.so/ {print $NF}' | sort -u); do
        [[ -e "$lib" ]] || continue
        docker_args+=(-v "$lib:$lib:ro")
        real="$(readlink -f "$lib")"
        [[ "$real" != "$lib" && -e "$real" ]] && docker_args+=(-v "$real:$real:ro")
    done
    for d in /usr/lib/x86_64-linux-gnu/libibverbs /usr/lib64/libibverbs /etc/libibverbs.d; do
        [[ -d "$d" ]] && docker_args+=(-v "$d:$d:ro")
    done
fi

docker run "${docker_args[@]}" "$IMAGE" bash
