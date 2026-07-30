docker run -it \
    --name alto_multinode \
    --device=/dev/infiniband \
    --device=/dev/dri \
    --device=/dev/kfd \
    --ulimit memlock=-1:-1 \
    --cap-add=IPC_LOCK \
    --shm-size=16g \
    --network host \
    --ipc host \
    --cap-add=IPC_LOCK \
    --ulimit memlock=-1:-1 \
    --shm-size=16g \
    -v $HOME/lpt_branch/ALTO:/alto \
    alto:multinode \
    bash