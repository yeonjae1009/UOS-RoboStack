"""
PCT GAT 정책을 ONNX로 export + onnxruntime 결과가 torch와 일치하는지 검증.

export 전용 forward: 학습 forward에서 카테고리 분포/샘플링을 제거하고,
'잎 노드별 (마스킹된) 확률' (1, leaf_node_holder) 만 반환한다. 추론 시 numpy argmax 로 선택.
normFactor 는 모델 안에 baked-in 하여 ONNX 입력은 raw observation (1, 301, 9) 으로 둔다.

사용:
  .venv-pct/bin/python export_onnx.py --model-path <ckpt.pt> --out model.onnx
"""
import sys, argparse
import numpy as np
import torch
import torch.nn as nn
# numpy 2.x compat (old env code uses np.float/np.int/np.bool)
for _a, _t in (('float', float), ('int', int), ('bool', bool), ('object', object)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)

mine = argparse.ArgumentParser()
mine.add_argument('--model-path', required=True)
mine.add_argument('--out', default='model.onnx')
mine.add_argument('--internal-node-holder', default='200')
mine.add_argument('--leaf-node-holder', default='100')
mine.add_argument('--setting', default='1')
mine.add_argument('--box-sequence', default='../templete code/box_sequence/box_sequence_0.json')
margs, _ = mine.parse_known_args()

INH = int(margs.internal_node_holder)
LNH = int(margs.leaf_node_holder)

sys.argv = ['x', '--continuous', '--setting', str(margs.setting),
            '--internal-node-holder', str(INH), '--leaf-node-holder', str(LNH),
            '--evaluate', '--load-model', '--model-path', margs.model_path]

from tools import (registration_envs, get_args, get_leaf_nodes_with_factor,
                   load_policy, observation_decode_leaf_node)
from model import DRL_GAT
from pct_envs.PctContinuous0.binCreator import BoxCreator
import gym, json

registration_envs()
args = get_args()
device = torch.device('cpu')   # export는 CPU로
FACTOR = float(args.normFactor)

policy = DRL_GAT(args)
_sd = torch.load(margs.model_path, map_location='cpu')
if isinstance(_sd, dict) and 'model' in _sd:
    _sd = _sd['model']
# critic head shape can differ (scalar bias); export uses ACTOR only -> strict=False is safe.
_missing, _unexpected = policy.load_state_dict(_sd, strict=False)
_actor_missing = [k for k in _missing if k.startswith('actor.')]
print(f"[load] strict=False  missing={len(_missing)} (actor_missing={len(_actor_missing)}) unexpected={len(_unexpected)}")
assert not _actor_missing, f"ACTOR weights missing: {_actor_missing[:5]}"
policy = policy.to(device)
policy.eval()
actor = policy.actor


class ExportActor(nn.Module):
    """ONNX 친화적 forward: 잎 노드별 마스킹 확률 반환 (deterministic 선택 = argmax)."""
    def __init__(self, a, factor):
        super().__init__()
        self.a = a
        self.factor = factor

    def forward(self, input):
        a = self.a
        f = self.factor
        internal_nodes, leaf_nodes, next_item, valid_flag, full_mask = observation_decode_leaf_node(
            input, a.internal_node_holder, a.internal_node_length, a.leaf_node_holder)
        leaf_node_mask = 1 - valid_flag
        valid_length = full_mask.sum(1)
        full_mask = 1 - full_mask

        bs = input.size(0)
        gs = input.size(1)
        internal_inputs = internal_nodes.reshape(bs * a.internal_node_holder, a.internal_node_length) * f
        leaf_inputs = leaf_nodes.reshape(bs * a.leaf_node_holder, 8) * f
        current_inputs = next_item.reshape(bs * 1, 6) * f

        ie = a.init_internal_node_embed(internal_inputs).reshape((bs, -1, a.embedding_dim))
        le = a.init_leaf_node_embed(leaf_inputs).reshape((bs, -1, a.embedding_dim))
        ne = a.init_next_embed(current_inputs).reshape((bs, -1, a.embedding_dim))
        init_embedding = torch.cat((ie, le, ne), dim=1).view(bs * gs, a.embedding_dim)

        embeddings, _ = a.embedder(init_embedding, mask=full_mask, evaluate=False)
        shape = (bs, gs, embeddings.shape[-1])
        fixed = a._precompute(embeddings, shape=shape, full_mask=full_mask, valid_length=valid_length)
        log_p, mask = a._get_log_p(fixed, leaf_node_mask)   # log_p = 확률 (1, LNH)
        masked = log_p * (1 - mask)
        return masked


export_model = ExportActor(actor, FACTOR).to(device).eval()


# --- 검증용 실제 관찰값 수집 (box_sequence_0 를 정책으로 적재하며) ---
def load_seq(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out

seq = load_seq(margs.box_sequence)
sizes = [list(map(float, b['size'])) for b in seq]


class FixedCreator(BoxCreator):
    def __init__(self, sizes):
        super().__init__(); self.sizes = sizes; self.idx = 0
    def reset(self):
        self.box_list.clear(); self.idx = 0
    def generate_box_size(self, **kw):
        if self.idx < len(self.sizes):
            self.box_list.append(tuple(self.sizes[self.idx])); self.idx += 1
        else:
            self.box_list.append((100.0, 100.0, 100.0))


env = gym.make(args.id, setting=args.setting, container_size=args.container_size,
               item_set=args.item_size_set, internal_node_holder=INH, leaf_node_holder=LNH,
               LNES=args.lnes, shuffle=False, sample_from_distribution=False,
               data_name=None, load_test_data=False)
e = env.unwrapped
e.box_creator = FixedCreator(sizes)

obs = env.reset()
samples = []
torch_sel = []
batchX = torch.arange(1)
for _ in range(60):
    t = torch.FloatTensor(obs).unsqueeze(0)
    all_nodes, leaf_nodes = get_leaf_nodes_with_factor(t, 1, INH, LNH)
    with torch.no_grad():
        _, selectedIdx, _, _ = policy(all_nodes, True, normFactor=FACTOR)
        masked = export_model(all_nodes)
    samples.append(all_nodes.numpy().astype(np.float32))
    torch_sel.append(int(selectedIdx.squeeze().item()))
    sel_leaf = leaf_nodes[batchX, selectedIdx.squeeze()]
    obs, reward, done, info = env.step(sel_leaf.cpu().numpy()[0][0:6])
    if done:
        break

print(f"수집한 관찰값: {len(samples)}개")

# --- ONNX export ---
sample_in = torch.from_numpy(samples[0])
torch.onnx.export(
    export_model, sample_in, margs.out,
    input_names=['obs'], output_names=['leaf_probs'],
    opset_version=13, do_constant_folding=True,
    dynamic_axes=None,
)
print(f"ONNX 저장: {margs.out}")

# --- 검증: onnxruntime argmax == torch argmax ? ---
import onnxruntime as ort
sess = ort.InferenceSession(margs.out, providers=['CPUExecutionProvider'])
match = 0
for s, ts in zip(samples, torch_sel):
    out = sess.run(None, {'obs': s})[0]
    onnx_sel = int(np.argmax(out[0]))
    if onnx_sel == ts:
        match += 1
print(f"argmax 일치: {match}/{len(samples)}  ({'OK ✓' if match==len(samples) else '불일치 있음!'})")
