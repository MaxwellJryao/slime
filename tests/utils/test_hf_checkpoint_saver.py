import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from slime.backends.megatron_utils.hf_checkpoint_saver import (
    _copy_missing_weights_from_origin,
    _copy_hf_assets,
    _finalize_shard_files,
    _SafetensorShardWriter,
    _write_pending_chunk,
    save_hf_model_direct_to_path,
)

NUM_GPUS = 0


def test_copy_hf_assets_keeps_quantized_config_and_skips_weights(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    config = {"model_type": "tiny", "quantization_config": {"quant_method": "fp8"}}
    (src / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (src / "tokenizer.json").write_text("{}", encoding="utf-8")
    (src / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (src / "model-00001-of-00001.safetensors").write_bytes(b"weight")
    (src / "pytorch_model.bin").write_bytes(b"weight")

    _copy_hf_assets(str(src), dst)

    assert json.loads((dst / "config.json").read_text(encoding="utf-8")) == config
    assert (dst / "tokenizer.json").exists()
    assert not (dst / "model.safetensors.index.json").exists()
    assert not (dst / "model-00001-of-00001.safetensors").exists()
    assert not (dst / "pytorch_model.bin").exists()


def test_save_hf_model_direct_to_path_rejects_origin_checkpoint(tmp_path: Path):
    args = SimpleNamespace(hf_checkpoint=str(tmp_path))

    with pytest.raises(ValueError, match="same directory as --hf-checkpoint"):
        save_hf_model_direct_to_path(args, tmp_path, model=None)


def test_safetensor_shard_writer_writes_hf_index(tmp_path: Path):
    writer = _SafetensorShardWriter(tmp_path, enabled=True)
    writer.write([("layers.0.weight", torch.ones(2, 2)), ("layers.0.weight_scale", torch.ones(1))], shard_idx=0)
    writer.write([("layers.1.weight", torch.zeros(2, 2))], shard_idx=1)
    writer.finalize()

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["metadata"]["total_size"] == 36
    assert index["weight_map"] == {
        "layers.0.weight": "model-00001-of-00002.safetensors",
        "layers.0.weight_scale": "model-00001-of-00002.safetensors",
        "layers.1.weight": "model-00002-of-00002.safetensors",
    }

    shard0 = load_file(tmp_path / "model-00001-of-00002.safetensors")
    shard1 = load_file(tmp_path / "model-00002-of-00002.safetensors")
    assert torch.equal(shard0["layers.0.weight"], torch.ones(2, 2))
    assert torch.equal(shard1["layers.1.weight"], torch.zeros(2, 2))


def test_finalize_shard_files_merges_node_writer_states(tmp_path: Path):
    writer0 = _SafetensorShardWriter(tmp_path, enabled=True)
    writer1 = _SafetensorShardWriter(tmp_path, enabled=True)

    writer0.write([("layers.0.weight", torch.ones(2, 2))], shard_idx=0)
    writer1.write([("layers.1.weight", torch.zeros(2, 2))], shard_idx=1)

    _finalize_shard_files(tmp_path, [writer0.state(), writer1.state()])

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["metadata"]["total_size"] == 32
    assert index["weight_map"] == {
        "layers.0.weight": "model-00001-of-00002.safetensors",
        "layers.1.weight": "model-00002-of-00002.safetensors",
    }
    assert not (tmp_path / "model-00001.safetensors").exists()
    assert not (tmp_path / "model-00002.safetensors").exists()

    shard0 = load_file(tmp_path / "model-00001-of-00002.safetensors")
    shard1 = load_file(tmp_path / "model-00002-of-00002.safetensors")
    assert torch.equal(shard0["layers.0.weight"], torch.ones(2, 2))
    assert torch.equal(shard1["layers.1.weight"], torch.zeros(2, 2))


def test_pending_chunk_write_flushes_incomplete_node_group(tmp_path: Path):
    num_nodes = 3
    writers = [_SafetensorShardWriter(tmp_path, enabled=True) for _ in range(num_nodes)]
    pending_writes = [None] * num_nodes

    for chunk_idx in range(5):
        node_rank = chunk_idx % num_nodes
        pending_writes[node_rank] = (
            chunk_idx,
            [(f"layers.{chunk_idx}.weight", torch.full((1,), chunk_idx, dtype=torch.float32))],
        )

        if (chunk_idx + 1) % num_nodes == 0:
            for i, writer in enumerate(writers):
                pending_writes[i] = _write_pending_chunk(writer, pending_writes[i])

    for i, writer in enumerate(writers):
        pending_writes[i] = _write_pending_chunk(writer, pending_writes[i])

    _finalize_shard_files(tmp_path, [writer.state() for writer in writers])

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["weight_map"] == {f"layers.{i}.weight": f"model-{i + 1:05d}-of-00005.safetensors" for i in range(5)}


def test_origin_fused_experts_are_not_copied_over_complete_trained_split_experts(tmp_path: Path):
    origin = tmp_path / "origin"
    output = tmp_path / "output"
    origin.mkdir()
    output.mkdir()
    (origin / "config.json").write_text(json.dumps({"text_config": {"num_experts": 2}}), encoding="utf-8")

    prefix = "model.language_model.layers.0.mlp.experts"
    origin_weights = {
        f"{prefix}.gate_up_proj": torch.ones(2, 4, 3),
        f"{prefix}.down_proj": torch.ones(2, 3, 2),
        "model.visual.weight": torch.ones(1),
    }

    save_file(origin_weights, origin / "model.safetensors")
    saved_weight_map = {
        f"{prefix}.{expert}.{projection}.weight": "model-trained.safetensors"
        for expert in range(2)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }

    states = _copy_missing_weights_from_origin(output, origin, saved_weight_map)

    assert len(states) == 1
    assert states[0]["weight_map"] == {"model.visual.weight": "model-origin-00001.safetensors"}
    assert set(load_file(output / "model-origin-00001.safetensors")) == {"model.visual.weight"}


def test_origin_fused_expert_is_copied_when_trained_split_experts_are_incomplete(tmp_path: Path):
    origin = tmp_path / "origin"
    output = tmp_path / "output"
    origin.mkdir()
    output.mkdir()
    (origin / "config.json").write_text(json.dumps({"text_config": {"num_experts": 2}}), encoding="utf-8")

    name = "model.language_model.layers.0.mlp.experts.gate_up_proj"

    save_file({name: torch.ones(2, 4, 3)}, origin / "model.safetensors")
    saved_weight_map = {
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight": "model-trained.safetensors",
        "model.language_model.layers.0.mlp.experts.0.up_proj.weight": "model-trained.safetensors",
    }

    states = _copy_missing_weights_from_origin(output, origin, saved_weight_map)

    assert states[0]["weight_map"] == {name: "model-origin-00001.safetensors"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
