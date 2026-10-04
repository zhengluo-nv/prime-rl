import importlib.util
from collections.abc import Iterator

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.tensor.parallel import parallelize_module

from prime_rl.configs.trainer import (
    BF16MoEComputeConfig,
    DeepEPMoEDispatchConfig,
    DeepGemmFP8MoEComputeConfig,
    ModelConfig,
    MoERuntimeConfig,
    MXFP8MoEComputeConfig,
    TorchMoEDispatchConfig,
)
from prime_rl.trainer.distributed.expert_parallel import ExpertWeightParallel
from prime_rl.trainer.distributed.token_dispatcher import (
    LocalTokenDispatcher,
    MXFP8TorchTokenDispatcher,
    TorchTokenDispatcher,
)
from prime_rl.trainer.models.layers.expert_compute import (
    BF16ExpertCompute,
    DeepGemmFP8ExpertCompute,
    ExpertCompute,
    MXFP8ExpertCompute,
)
from prime_rl.trainer.models.layers.moe import MoE, TokenChoiceTopKRouter
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.utils.logger import get_logger
from prime_rl.utils.vlm import get_language_model


def _resolve_expert_compute(config: ModelConfig) -> ExpertCompute:
    compute = config.moe.compute
    if isinstance(compute, BF16MoEComputeConfig):
        if compute.backend == "sonicmoe":
            from prime_rl.trainer.models.layers.sonic_moe import SonicMoEExpertCompute

            return SonicMoEExpertCompute()
        return BF16ExpertCompute()
    if isinstance(compute, DeepGemmFP8MoEComputeConfig):
        if importlib.util.find_spec("deep_gemm") is None:
            raise RuntimeError("DeepGEMM FP8 expert compute requires the deep-gemm package.")
        capability = torch.cuda.get_device_capability()
        if capability < (9, 0):
            raise RuntimeError(
                f"DeepGEMM FP8 expert compute requires SM90 or newer, but this device is SM{capability[0]}{capability[1]}."
            )
        return DeepGemmFP8ExpertCompute()
    if isinstance(compute, MXFP8MoEComputeConfig):
        import prime_kernels

        kernel = prime_kernels.load("mxfp8_moe")
        return MXFP8ExpertCompute(
            kernel=kernel,
            high_precision_wgrad=compute.recipe == "mxfp8_rceil_wgrad_with_hp",
        )
    raise TypeError(f"Unsupported MoE compute config: {type(compute).__name__}")


def configure_moe_runtime(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims) -> None:
    moe_layers = [module for module in model.modules() if isinstance(module, MoE)]
    if not moe_layers:
        if config.moe != MoERuntimeConfig():
            raise ValueError("A non-default model.moe runtime was configured, but the model has no custom MoE layers.")
        return

    selected_moes = set(moe_layers)
    if config.moe.compute.apply_to != "all":
        language_model = get_language_model(
            model, override=config.vlm.language_model_attr if config.vlm is not None else None
        )
        selected_layers = config.moe.compute.resolve_layers(len(language_model.layers))
        selected_moes = {
            module
            for index, layer in enumerate(language_model.layers.children())
            if index in selected_layers
            for module in layer.modules()
            if isinstance(module, MoE)
        }
        get_logger().debug(f"Selected model layers for MoE compute: {sorted(selected_layers)}")
    bf16_compute = BF16ExpertCompute()
    selected_compute = _resolve_expert_compute(config) if selected_moes else bf16_compute
    ep_mesh = parallel_dims.get_mesh("ep") if parallel_dims.ep_enabled else None
    dispatch = config.moe.dispatch

    for moe in moe_layers:
        compute = selected_compute if moe in selected_moes else bf16_compute
        if ep_mesh is not None and moe.experts.num_experts % parallel_dims.ep:
            raise ValueError(
                f"MoE expert count {moe.experts.num_experts} must be divisible by model.ep={parallel_dims.ep}."
            )
        moe.experts.set_compute(compute)
        if ep_mesh is None:
            token_dispatcher = LocalTokenDispatcher(
                num_experts=moe.experts.num_experts,
                top_k=moe.router.top_k,
                token_group_alignment=compute.token_group_alignment,
            )
        elif isinstance(dispatch, TorchMoEDispatchConfig):
            if dispatch.transport == "mxfp8" and isinstance(compute, MXFP8ExpertCompute):
                token_dispatcher = MXFP8TorchTokenDispatcher(
                    num_experts=moe.experts.num_experts,
                    top_k=moe.router.top_k,
                    token_group_alignment=compute.token_group_alignment,
                    group=ep_mesh.get_group(),
                )
            else:
                token_dispatcher = TorchTokenDispatcher(
                    num_experts=moe.experts.num_experts,
                    top_k=moe.router.top_k,
                    token_group_alignment=compute.token_group_alignment,
                    group=ep_mesh.get_group(),
                )
        elif isinstance(dispatch, DeepEPMoEDispatchConfig):
            from prime_rl.trainer.distributed.deepep import DeepEPTokenDispatcher

            token_dispatcher = DeepEPTokenDispatcher(
                num_experts=moe.experts.num_experts,
                token_group_alignment=compute.token_group_alignment,
                group=ep_mesh.get_group(),
                num_sms=dispatch.num_sms,
                token_chunk_size=dispatch.token_chunk_size,
            )
        else:
            raise TypeError(f"Unsupported MoE dispatch config: {type(dispatch).__name__}")
        moe.set_token_dispatcher(token_dispatcher)

        if ep_mesh is not None:
            parallelize_module(moe.experts, device_mesh=ep_mesh, parallelize_plan=ExpertWeightParallel())

    get_logger().info(
        f"Configured {len(selected_moes)}/{len(moe_layers)} MoE layers with compute={type(selected_compute).__name__}, "
        f"apply_to={config.moe.compute.apply_to}, fallback=bf16, dispatch={config.moe.dispatch.type}, ep={parallel_dims.ep}"
    )


def iter_moe_blocks(model: nn.Module) -> Iterator[MoE]:
    """Yield the MoE MLP of each decoder layer, skipping dense layers."""
    for layer in get_language_model(model).layers:
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, MoE):
            yield mlp


def freeze_moe_router(model: nn.Module) -> None:
    """Freeze MoE router parameters to maintain stable routing during training."""
    logger = get_logger()
    num_frozen = 0
    for moe in iter_moe_blocks(model):
        for param in moe.router.parameters():
            param.requires_grad = False
            num_frozen += 1

    if num_frozen == 0:
        raise ValueError("No MoE router parameters found to freeze. Is this a MoE model?")

    logger.info(f"Froze {num_frozen} MoE router parameters")


def apply_fp32_moe_router(model: nn.Module) -> None:
    """Cast MoE router gates to fp32 so routing runs in fp32 in forward and backward.

    The FSDP bf16 cast exemption is applied separately in `setup_fsdp`.
    """
    logger = get_logger()
    num_routers = 0
    for moe in iter_moe_blocks(model):
        moe.router.to(torch.float32)
        if isinstance(moe.router, TokenChoiceTopKRouter):
            moe.router.fp32_gate = True
        num_routers += 1

    # No-op for non-MoE models: moe_router_dtype='float32' is the default,
    # so absence of MoE routers is the common case, not an error.
    if num_routers > 0:
        logger.info(f"Running {num_routers} MoE router gates in fp32")


def apply_force_balanced_routing(model: nn.Module) -> None:
    """Force MoE token-choice routers into round-robin assignment for fake-data smoke tests."""
    logger = get_logger()
    num_routers = 0
    for moe in iter_moe_blocks(model):
        moe.router.force_balanced = True
        num_routers += 1

    if num_routers == 0:
        raise ValueError("No MoE routers found to force-balance. Is this a MoE model?")

    logger.warning(
        f"Forced balanced routing on {num_routers} MoE layers (debug.force_balanced_routing=True). "
        "Expert assignment is round-robin; gradient flow through the router is broken."
    )


def is_tt_moe_model(model: nn.Module) -> bool:
    config = getattr(model.config, "text_config", model.config)
    return hasattr(config, "num_experts") or hasattr(config, "n_routed_experts")


def get_load_balance_stats(
    model: nn.Module,
    group: dist.ProcessGroup | None = None,
) -> dict[str, Tensor | None]:
    """Compute routing stats after summing raw counts across the group, if given.

    Also returns this rank's raw per-layer expert counts (`[num_moe_layers, num_experts]`) for step-level stats.
    """
    per_layer_max_vio = []
    per_layer_routing_confidence = []
    block_mlps = list(iter_moe_blocks(model))
    if not block_mlps:
        return {"max_vio": None, "routing_confidence": None, "tokens_per_expert": torch.empty(0, 0)}

    local_tokens_per_expert = torch.stack([block_mlp.tokens_per_expert for block_mlp in block_mlps])
    layer_stats = [(block_mlp.tokens_per_expert, block_mlp.routing_confidence_sum) for block_mlp in block_mlps]
    if group is not None:
        sizes = [tokens_per_expert.numel() + 1 for tokens_per_expert, _ in layer_stats]
        packed_stats = torch.cat(
            [torch.cat((tokens_per_expert, confidence.reshape(1))) for tokens_per_expert, confidence in layer_stats]
        )
        dist.all_reduce(packed_stats, op=dist.ReduceOp.SUM, group=group)
        layer_stats = [(stats[:-1], stats[-1]) for stats in packed_stats.split(sizes)]

    for block_mlp, (tokens_per_expert, routing_confidence_sum) in zip(block_mlps, layer_stats):
        num_routed_tokens = tokens_per_expert.sum() / block_mlp.router.top_k
        tokens_per_expert = tokens_per_expert.sort(dim=0, descending=True).values[block_mlp.router.top_k :]
        balanced_load = tokens_per_expert.mean()
        max_vio = (tokens_per_expert.max() - balanced_load) / balanced_load
        per_layer_max_vio.append(max_vio.detach())

        routing_confidence = routing_confidence_sum / num_routed_tokens
        per_layer_routing_confidence.append(routing_confidence.detach())

        block_mlp.tokens_per_expert.zero_()
        block_mlp.routing_confidence_sum.zero_()
    return {
        "max_vio": torch.stack(per_layer_max_vio),
        "routing_confidence": torch.stack(per_layer_routing_confidence),
        "tokens_per_expert": local_tokens_per_expert,
    }


def get_global_moe_stats(
    model: nn.Module,
    ep_group: dist.ProcessGroup | None,
    dp_cp_group: dist.ProcessGroup,
) -> tuple[dict[str, Tensor], Tensor]:
    """Reduce one microstep's routing stats across EP, then DP and CP ranks.

    Also returns this rank's unreduced per-layer expert counts, to accumulate over the step for `get_expert_load_stats`.
    """
    stats = {}
    load_balance_stats = get_load_balance_stats(model, group=ep_group)
    tokens_per_expert = load_balance_stats.pop("tokens_per_expert")
    for name, values in load_balance_stats.items():
        if values is None:
            continue
        value = values.max() if name == "max_vio" else values.mean()
        if name == "max_vio":
            dist.all_reduce(value, op=dist.ReduceOp.MAX, group=dp_cp_group)
        else:
            dist.all_reduce(value, op=dist.ReduceOp.SUM, group=dp_cp_group)
            value /= dist.get_world_size(dp_cp_group)
        stats[name] = value.to("cpu")
    return stats, tokens_per_expert


def compute_expert_load_stats(tokens_per_expert: Tensor) -> dict[str, Tensor]:
    """Per-layer expert-load balance from `[num_moe_layers, num_experts]` token counts, as mean and max over layers.

    `cv` is std/mean of the expert loads, `max_mean` the busiest expert's load over the mean load, and `cold_frac`
    the fraction of experts receiving under 0.1x the mean load (the MiMo-V2.6 definition).
    """
    mean_load = tokens_per_expert.mean(dim=1)
    per_layer = {
        "cv": tokens_per_expert.std(dim=1, correction=0) / mean_load,
        "max_mean": tokens_per_expert.amax(dim=1) / mean_load,
        "cold_frac": (tokens_per_expert < 0.1 * mean_load[:, None]).float().mean(dim=1),
    }
    stats = {}
    for name, values in per_layer.items():
        stats[f"expert_load/{name}/mean"] = values.mean()
        stats[f"expert_load/{name}/max"] = values.max()
    return stats


def get_expert_load_stats(tokens_per_expert: Tensor, group: dist.ProcessGroup) -> dict[str, float]:
    """Sum a step's per-layer expert counts across the group (every rank routes distinct tokens) and compute load stats."""
    if tokens_per_expert.numel() == 0:
        return {}
    dist.all_reduce(tokens_per_expert, op=dist.ReduceOp.SUM, group=group)
    return {name: value.item() for name, value in compute_expert_load_stats(tokens_per_expert).items()}
