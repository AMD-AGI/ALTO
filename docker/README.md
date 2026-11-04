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

---

# GPT-OSS-20B MXFP8 attention 消融实验运行手册

分支 `yue/gpt-oss-mxfp8-attn`。两个脚本：`run_gpt_oss_20b.sh` 在容器里跑训练，
`daemon_gpt_oss_20b.sh` 把训练挂到 tmux 后台。必须 8 卡（EP=8）。

## 四组实验

| 组 | 配置 | 走 MXFP8 的模块 |
| --- | --- | --- |
| `bf16` | `gpt_oss_20b_c4` | 无 |
| `attn` | `gpt_oss_20b_c4_mxfp8_attn` | Attention（重点组） |
| `linear` | `gpt_oss_20b_c4_mxfp8_linear` | Linear + MoE 专家 |
| `linear-attn` | `gpt_oss_20b_c4_mxfp8_linear_attn` | Linear + MoE 专家 + Attention |

凡是碰 Attention 的组都打开 `attention_consistent_delta`。`suite` 依次跑 bf16、attn；
linear 和 linear-attn 需要时单独启动。

训练协议和 m355 上 20000 步 BF16/MXFP8 对比完全一致：global batch 16、seq 8192、
峰值 lr 4e-4、warmup 128、LR 调度按 1.2M 步铺开、seed 1234、在线 HF C4、
每 768 步在 `c4_validation` 上验证 64 个 batch。

## 启动

```bash
./docker/run_gpt_oss_20b.sh smoke attn      # 先跑 10 步，确认能起来（第 1 步含 Triton 编译，约 4 分钟）
./docker/daemon_gpt_oss_20b.sh start        # 后台跑 suite
./docker/daemon_gpt_oss_20b.sh start attn   # 或者只跑一组
./docker/daemon_gpt_oss_20b.sh attach attn  # 看实时输出
```

日志在 `logs/gpt_oss_20b_<组>.log`。中途挂了就重新 `start` 同一组，会从 dump 目录里最新的 checkpoint 续跑。

## 模型资产

`hf_models/gpt-oss-20b` 只放 tokenizer 和 config，没有权重：

```
chat_template.jinja  config.json  generation_config.json
special_tokens_map.json  tokenizer_config.json  tokenizer.json
```

## 时长与磁盘

- 4 卡冒烟实测吞吐：bf16 8,800 tps/卡，attn 6,200，linear-attn 3,700。m355 上 8 卡 BF16 实测
  20000 步约 11 小时，按同样比例换算：attn 约 17 小时，linear-attn 约 28 小时，suite 合计约 2.5 天。
- checkpoint 每 5000 步存一次，保留最近 2 份（torchtitan 不允许少于 2 份）。一份约 250GB
  （20B 参数的 fp32 权重加 Adam 状态，估算值），一组最多占约 500GB。磁盘不够时，跑下一组之前删掉上一组的
  `<dump 目录>/checkpoint`。

## 和旧曲线对照

m355 旧的 BF16 run（`alto_runs/bf16_mlperf1p2m_8gpu_16k_rerun/tb`）第 1 步 loss 是 12.71482。
新 `bf16` 组在 8 卡上第 1 步若是同一个数，说明数据和初始化都没变，旧的 20000 步 BF16/MXFP8 曲线可以直接拿来对照。
