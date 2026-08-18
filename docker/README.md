# Llama3-8B MXFP8 消融实验运行手册

两个脚本：`run_llama3_8b.sh` 在容器里跑训练，`daemon_llama3_8b_suite.sh` 把训练挂到 tmux 后台。

## 在一台新机器上跑

```bash
git clone https://github.com/AMD-AGI/ALTO.git ALTO-mxfp8-attn
cd ALTO-mxfp8-attn
git checkout yue/llama3-8b-mxfp8-ablation-config
git submodule update --init --recursive

# 从已有机器拷贝模型资产（约 17MB，只有 tokenizer 和 config，没有权重）
mkdir -p ../hf_models
scp -r <旧机器>:<旧路径>/hf_models/llama3-8b ../hf_models/

wandb login
WANDB_PROJECT=<给这个集群起个新名字> ./docker/daemon_llama3_8b_suite.sh start
```

`start` 会依次跑完 bf16、attn、linear-attn 三组。日志在 `logs/llama3_8b_mxfp8_ablation.log`，
`./docker/daemon_llama3_8b_suite.sh attach` 可以进去看实时输出。

单独跑一组用 `./docker/run_llama3_8b.sh <mode>`，mode 可选
`bf16` / `attn` / `linear` / `linear-attn` / `bf16-gbs4` / `smoke` / `suite`。

## 模型资产

`hf_models/llama3-8b` 目录需要包含这 5 个文件，缺一个 `run_llama3_8b.sh` 就会拒绝启动：

```
config.json  generation_config.json  special_tokens_map.json
tokenizer_config.json  tokenizer.json
```

来源是 HuggingFace 上的 `unsloth/Llama-3.1-8B`（`meta-llama/Llama-3.1-8B` 的非 gated 镜像），
只取了 tokenizer 和 config，没有权重文件。

默认从仓库同级的 `../hf_models` 找，放在别处就设 `HF_MODELS_DIR`。

## 换集群时要改的环境变量

| 变量 | 默认值 | 什么时候要改 |
| --- | --- | --- |
| `WANDB_PROJECT` | `alto-llama3-8b-mxfp8-ablation` | **换集群必改**，否则新旧曲线混在同一张图里分不清 |
| `HF_MODELS_DIR` | 仓库同级的 `../hf_models` | 模型资产不放在仓库同级目录时 |
| `WANDB_TEAM` | `yue-sun2-amd` | 换 W&B 账号时 |
| `WANDB_RUN_GROUP` | `llama3-8b-mxfp8-c4-fsdp-ablation` | 想在同一 project 里再分组时 |

## 依赖

- docker 镜像 `wanghanthu/torchtitan:ubuntu22.04-pytorch2.12.0dev20260217-rocm7.2-patch`，
  新集群要能 pull 到
- submodule `3rdparty/torchtitan` 来自 fork `hann-wang/torchtitan`，要有访问权限
- 宿主机 `~/.netrc` 里有 W&B 凭据，即 `wandb login` 的产物
- 训练时容器要能访问 HuggingFace，C4 数据集是在线拉取的

## 注意

`llama3_8b` 系列配置的 `hf_assets_path` 只提供 tokenizer 和模型结构，
`hf_models/llama3-8b` 里没有权重文件，checkpoint 加载也是关闭的。
这几组实验都是从随机初始化开始的 from-scratch 预训练，不是微调。
