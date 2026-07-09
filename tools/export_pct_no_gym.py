from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


def _install_fake_gym() -> None:
    gym = types.ModuleType("gym")
    envs = types.ModuleType("gym.envs")
    registration = types.ModuleType("gym.envs.registration")
    registration.register = lambda *args, **kwargs: None
    envs.registration = registration
    gym.envs = envs
    sys.modules.setdefault("gym", gym)
    sys.modules.setdefault("gym.envs", envs)
    sys.modules.setdefault("gym.envs.registration", registration)


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, (tuple, list)) and len(state) == 2:
        state = state[0]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    return dict(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Online-3D-BPP-PCT")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--internal-node-holder", type=int, default=200)
    parser.add_argument("--leaf-node-holder", type=int, default=100)
    parser.add_argument("--setting", type=int, default=3)
    args = parser.parse_args()

    _install_fake_gym()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    from model import DRL_GAT
    from tools import observation_decode_leaf_node

    internal_node_length = 7 if args.setting == 3 else 6
    pct_args = SimpleNamespace(
        embedding_size=64,
        hidden_size=128,
        gat_layer_num=1,
        internal_node_holder=args.internal_node_holder,
        internal_node_length=internal_node_length,
        leaf_node_holder=args.leaf_node_holder,
        learn_finish_action=False,
    )

    policy = DRL_GAT(pct_args)
    state = _load_state_dict(Path(args.model_path))
    missing, unexpected = policy.load_state_dict(state, strict=False)
    actor_missing = [name for name in missing if name.startswith("actor.")]
    if actor_missing or unexpected:
        raise RuntimeError(f"Unexpected state dict mismatch: missing={missing}, unexpected={unexpected}")
    actor = policy.actor.eval()

    class ExportActor(nn.Module):
        def __init__(self, wrapped: nn.Module, factor: float) -> None:
            super().__init__()
            self.wrapped = wrapped
            self.factor = factor

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            a = self.wrapped
            internal_nodes, leaf_nodes, next_item, valid_flag, full_mask = observation_decode_leaf_node(
                input,
                a.internal_node_holder,
                a.internal_node_length,
                a.leaf_node_holder,
            )
            leaf_node_mask = 1 - valid_flag
            valid_length = full_mask.sum(1)
            full_mask = 1 - full_mask

            batch_size = input.size(0)
            graph_size = input.size(1)
            internal_inputs = internal_nodes.reshape(batch_size * a.internal_node_holder, a.internal_node_length) * self.factor
            leaf_inputs = leaf_nodes.reshape(batch_size * a.leaf_node_holder, 8) * self.factor
            current_inputs = next_item.reshape(batch_size * 1, 6) * self.factor

            internal_emb = a.init_internal_node_embed(internal_inputs).reshape((batch_size, -1, a.embedding_dim))
            leaf_emb = a.init_leaf_node_embed(leaf_inputs).reshape((batch_size, -1, a.embedding_dim))
            next_emb = a.init_next_embed(current_inputs).reshape((batch_size, -1, a.embedding_dim))
            init_embedding = torch.cat((internal_emb, leaf_emb, next_emb), dim=1).view(
                batch_size * graph_size,
                a.embedding_dim,
            )

            embeddings, _ = a.embedder(init_embedding, mask=full_mask, evaluate=False)
            shape = (batch_size, graph_size, embeddings.shape[-1])
            fixed = a._precompute(embeddings, shape=shape, full_mask=full_mask, valid_length=valid_length)
            probs, mask = a._get_log_p(fixed, leaf_node_mask)
            return probs * (1 - mask)

    export_model = ExportActor(actor, factor=0.8).eval()
    sample = np.zeros((1, args.internal_node_holder + args.leaf_node_holder + 1, 9), dtype=np.float32)
    sample[:, 0, -1] = 1.0
    sample[:, args.internal_node_holder, 8] = 1.0
    sample[:, args.internal_node_holder + args.leaf_node_holder, -1] = 1.0
    sample_tensor = torch.from_numpy(sample)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch_out = export_model(sample_tensor).detach().numpy()
    torch.onnx.export(
        export_model,
        sample_tensor,
        str(out),
        input_names=["obs"],
        output_names=["leaf_probs"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"obs": sample})[0]
    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    print(f"saved={out}")
    print(f"max_abs_diff={max_diff:.8g}")
    print(f"argmax_torch={int(np.argmax(torch_out[0]))} argmax_onnx={int(np.argmax(ort_out[0]))}")


if __name__ == "__main__":
    main()
