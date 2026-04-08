docker run -it --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    --shm-size=64g \
    --name model-optimizer \
    -v /home/guanchen@amd.com/Model-Optimizer:/workspace/Model-Optimizer \
    wanghanthu/torchtitan:ubuntu22.04-pytorch2.10.0dev20251102-rocm7.0