from types import SimpleNamespace

from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MLAAttentionSpec

from prime_rl.inference.patches import get_routed_experts_attn_gid_skipping_host_groups


def test_routed_experts_slots_skip_hisparse_host_group():
    spec = MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=576, dtype="bfloat16")
    groups = [
        KVCacheGroupSpec(["source"], spec, host_resident=True),
        KVCacheGroupSpec(["indexer"], spec),
    ]
    assert get_routed_experts_attn_gid_skipping_host_groups(SimpleNamespace(kv_cache_groups=groups)) == 1
