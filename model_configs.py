"""
SAGE Model Configurations
===========================
预定义的模型配置，支持通过 --model_config 参数在训练/评估脚本中切换。

配置对照表:
  ┌──────────────────┬─────────┬───────┬────────┬──────┬─────────┬────────────┐
  │ Config           │ d_model │ heads │ layers │ FFN  │ ~Params │ GPU (est.) │
  ├──────────────────┼─────────┼───────┼────────┼──────┼─────────┼────────────┤
  │ small            │   256   │   8   │   6    │ 1024 │  ~10M   │   ~4 GB    │
  │ demo             │   480   │  12   │   6    │ 1920 │  ~30M   │   ~8 GB    │
  │ demo_deep        │   480   │  12   │  12    │ 1920 │  ~48M   │  ~12 GB    │
  │ demo_deep_v2     │   480   │  12   │  12    │ 1920 │  ~48M   │  ~12 GB    │
  │ large            │   512   │  16   │  12    │ 2048 │  ~80M   │  ~16 GB    │
  └──────────────────┴─────────┴───────┴────────┴──────┴─────────┴────────────┘

用法:
    from model_configs import get_model_config, apply_config_to_args
    config = get_model_config("large")
    apply_config_to_args(args, config)
"""


# ============================================================================
# 预定义配置
# ============================================================================

MODEL_CONFIGS = {
    "small": {
        # 模型架构
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 6,
        "dim_feedforward": 1024,
        "distance_window": 128,
        # 训练超参数 (小模型可用较大 lr)
        "lr": 3e-4,
        "dropout": 0.15,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        # 对比学习 projection head 输出维度
        "contrastive_proj_dim": 128,
        # 描述
        "_description": "SAGE-Small: 256d, 6L, 8H, ~10M params",
    },

    "demo": {
        # 模型架构: d_model=480 对齐 ESM-2 (esm2_t12_35M_UR50D) 输出维度
        "d_model": 480,
        "num_heads": 12,
        "num_layers": 6,
        "dim_feedforward": 1920,
        "distance_window": 128,
        # 训练超参数 (模型更大, 适当降低 lr)
        "lr": 2e-4,
        "dropout": 0.12,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        # 对比学习 projection head 输出维度
        "contrastive_proj_dim": 192,
        # 描述
        "_description": "SAGE-Demo: 480d, 6L, 12H, ~30M params (aligned with ESM-2 t12)",
    },

    "demo_deep": {
        # 模型架构: d_model=480 对齐 ESM-2, 12 层深度
        "d_model": 480,
        "num_heads": 12,
        "num_layers": 12,
        "dim_feedforward": 1920,
        "distance_window": 128,
        # 训练超参数 (12层需要更保守的lr)
        "lr": 1.5e-4,
        "dropout": 0.1,
        "batch_size": 2,                    # 单卡 batch=2 (A100 40GB 实测可跑)
        "gradient_accumulation_steps": 16,   # 有效 batch = 2×16×8GPUs = 256
        # 对比学习 projection head 输出维度
        "contrastive_proj_dim": 192,
        # 描述
        "_description": "SAGE-Demo-Deep: 480d, 12L, 12H, ~48M params (ESM-aligned, deep)",
    },

    "demo_deep_v2": {
        # 模型架构: 与 demo_deep 完全相同
        "d_model": 480,
        "num_heads": 12,
        "num_layers": 12,
        "dim_feedforward": 1920,
        "distance_window": 128,
        # 训练超参数: warm start 后降低 lr (保护已训练的 Transformer 层)
        "lr": 5e-5,
        "dropout": 0.1,
        "batch_size": 2,
        "gradient_accumulation_steps": 16,
        # 对比学习 projection head 输出维度
        "contrastive_proj_dim": 192,
        # Loss 权重 (ESM regression MLM 版本)
        "contrastive_loss_weight": 0.0,     # Phase 1: 关闭对比学习
        # 描述
        "_description": "SAGE-Demo-Deep-v2: 480d, 12L, ESM Regression MLM (warm start from v1)",
    },

    "large": {
        # 模型架构: 全面 2x 扩增
        "d_model": 512,
        "num_heads": 16,
        "num_layers": 12,
        "dim_feedforward": 2048,
        "distance_window": 128,
        # 训练超参数 (大模型应降低 lr, 增大梯度累积)
        "lr": 1.5e-4,
        "dropout": 0.1,
        "batch_size": 1,
        "gradient_accumulation_steps": 32,
        # 对比学习 projection head 输出维度
        "contrastive_proj_dim": 256,
        # 描述
        "_description": "SAGE-Large: 512d, 12L, 16H, ~80M params",
    },
}

# 默认配置名
DEFAULT_CONFIG = "small"


def get_model_config(name: str) -> dict:
    """获取指定名称的模型配置。

    Args:
        name: 配置名称 ("small", "demo", "demo_deep", "large")

    Returns:
        配置字典
    """
    name = name.lower().strip()
    if name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model config '{name}'. Available: {available}")
    return MODEL_CONFIGS[name].copy()


def apply_config_to_args(args, config: dict, parser=None):
    """将配置中的值应用到 argparse Namespace，仅覆盖用户未显式设置的字段。

    逻辑:
      - 遍历 config 中的每个 key (跳过 '_' 开头的描述字段)
      - 如果 parser 提供了, 通过对比 argparse default 判断用户是否显式传入
      - 仅覆盖用户未显式设置 (值==default) 的字段
      - 命令行显式传入的参数具有最高优先级, 不会被覆盖

    注意: 此函数应在 parse_args() 之后立即调用。
          为了正确判断用户是否显式传入了某个参数, 建议传入 parser 对象。
    """
    for key, value in config.items():
        if key.startswith("_"):
            continue
        if hasattr(args, key):
            if parser is not None:
                default_val = parser.get_default(key)
                # 只有当用户没有在命令行显式指定时 (值仍等于 default), 才用 config 覆盖
                if getattr(args, key) == default_val:
                    setattr(args, key, value)
            else:
                # 没有 parser 时回退到无条件覆盖 (向后兼容)
                setattr(args, key, value)


def add_config_argument(parser):
    """为 argparse parser 添加 --model_config 参数。"""
    parser.add_argument(
        "--model_config", type=str, default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help=f"预定义模型配置 ({', '.join(MODEL_CONFIGS.keys())}). "
             f"设置后会覆盖对应的架构和训练超参数默认值, "
             f"但命令行显式传入的参数仍具有最高优先级."
    )


def list_configs():
    """打印所有可用配置的摘要。"""
    print("Available SAGE model configurations:")
    print("-" * 60)
    for name, cfg in MODEL_CONFIGS.items():
        desc = cfg.get("_description", name)
        d = cfg["d_model"]
        h = cfg["num_heads"]
        l = cfg["num_layers"]
        f = cfg["dim_feedforward"]
        print(f"  {name:<12} → d={d}, H={h}, L={l}, FFN={f}  ({desc})")
    print("-" * 60)


if __name__ == "__main__":
    list_configs()
