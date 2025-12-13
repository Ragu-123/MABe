
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import polars as pl
import pyarrow.parquet as pq

# ==============================================================================
# 1. SHARED CONFIGURATION & DATA (Copied exactly from train.py)
# ==============================================================================
LAB_CONFIGS = {
    "AdaptableSnail":       {"thresh": 718.59, "window": 120, "pix_cm": 14.5},
    "BoisterousParrot":     {"thresh": 50.93,  "window": 292, "pix_cm": 5.5},
    "CRIM13":               {"thresh": 207.95, "window": 117, "pix_cm": 14.5},
    "CalMS21_supplemental": {"thresh": 206.05, "window": 196, "pix_cm": 18.3},
    "CalMS21_task1":        {"thresh": 154.32, "window": 140, "pix_cm": 18.3},
    "CalMS21_task2":        {"thresh": 177.51, "window": 122, "pix_cm": 18.3},
    "CautiousGiraffe":      {"thresh": 119.97, "window": 67,  "pix_cm": 21.0},
    "DeliriousFly":         {"thresh": 97.31,  "window": 172, "pix_cm": 16.0},
    "ElegantMink":          {"thresh": 88.58,  "window": 391, "pix_cm": 18.4},
    "GroovyShrew":          {"thresh": 254.45, "window": 115, "pix_cm": 11.3},
    "InvincibleJellyfish":  {"thresh": 249.33, "window": 158, "pix_cm": 32.0},
    "JovialSwallow":        {"thresh": 99.68,  "window": 62,  "pix_cm": 15.3},
    "LyricalHare":          {"thresh": 198.80, "window": 361, "pix_cm": 10.9},
    "NiftyGoldfinch":       {"thresh": 303.02, "window": 78,  "pix_cm": 13.5},
    "PleasantMeerkat":      {"thresh": 150.58, "window": 32,  "pix_cm": 15.8},
    "ReflectiveManatee":    {"thresh": 117.76, "window": 97,  "pix_cm": 15.0},
    "SparklingTapir":       {"thresh": 281.60, "window": 252, "pix_cm": 40.0},
    "TranquilPanther":      {"thresh": 133.98, "window": 105, "pix_cm": 12.3},
    "UppityFerret":         {"thresh": 228.77, "window": 55,  "pix_cm": 12.7},
    "DEFAULT":              {"thresh": 150.00, "window": 128, "pix_cm": 15.0}
}

ACTION_LIST = sorted([
    "allogroom", "approach", "attack", "attemptmount", "avoid", "biteobject",
    "chase", "chaseattack", "climb", "defend", "dig", "disengage", "dominance",
    "dominancegroom", "dominancemount", "ejaculate", "escape", "exploreobject",
    "flinch", "follow", "freeze", "genitalgroom", "huddle", "intromit", "mount",
    "rear", "reciprocalsniff", "rest", "run", "selfgroom", "shepherd", "sniff",
    "sniffbody", "sniffface", "sniffgenital", "submit", "tussle"
])
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}
NUM_CLASSES = len(ACTION_LIST)
BODY_PARTS = [
    "ear_left", "ear_right", "nose", "neck", "body_center",
    "lateral_left", "lateral_right", "hip_left", "hip_right",
    "tail_base", "tail_tip"
]

class BioPhysicsDataset(Dataset):
    def __init__(self, data_root, mode='test', video_ids=None):
        self.root = Path(data_root)
        self.mode = mode
        self.tracking_dir = self.root / f"{mode}_tracking"

        # Load Metadata
        self.metadata = pd.read_csv(self.root / f"{mode}.csv")
        if video_ids is not None:
            self.metadata = self.metadata[self.metadata['video_id'].astype(str).isin(video_ids)]

        # Build samples for ALL permutations
        self.samples = []
        print(f"Scanning {len(self.metadata)} videos for mouse pairs...")

        # Scan videos for mice
        for _, row in self.metadata.iterrows():
            vid = str(row['video_id'])
            lab = row['lab_id']
            fpath = self.tracking_dir / lab / f"{vid}.parquet"
            mice = []
            if fpath.exists():
                try:
                    df_small = pd.read_parquet(fpath, columns=['mouse_id'])
                    mice = df_small['mouse_id'].unique().tolist()
                except:
                    mice = ['mouse1', 'mouse2']
            else:
                mice = []

            # Create permutations
            for agent in mice:
                for target in mice:
                    if agent != target:
                        self.samples.append({
                            'video_id': vid,
                            'lab_id': lab,
                            'agent_id': str(agent), # FORCE STRING
                            'target_id': str(target) # FORCE STRING
                        })

        # --- Filter Bad Samples ---
        def _quick_parquet_has_frames(path):
            try:
                pf = pq.ParquetFile(str(path))
                md = pf.metadata
                if md.num_rows == 0:
                    return False
                try:
                    col_names = pf.schema_arrow.names
                    if 'video_frame' in col_names:
                        return True
                    return True
                except:
                    return True
            except:
                return False

        print("Filtering invalid videos...")
        filtered = []
        for s in self.samples:
            p = self.tracking_dir / s['lab_id'] / f"{s['video_id']}.parquet"
            if p.exists() and _quick_parquet_has_frames(p):
                filtered.append(s)
        self.samples = filtered
        print(f"Retained {len(self.samples)} valid samples.")

    def _fix_teleport(self, pos):
        T, N, _ = pos.shape
        missing = (np.abs(pos).sum(axis=2) < 1e-6)
        cleaned = pos.copy()
        for n in range(N):
            m = missing[:, n]
            if np.any(m) and not np.all(m):
                valid_t = np.where(~m)[0]
                missing_t = np.where(m)[0]
                cleaned[missing_t, n, 0] = np.interp(missing_t, valid_t, pos[valid_t, n, 0])
                cleaned[missing_t, n, 1] = np.interp(missing_t, valid_t, pos[valid_t, n, 1])
        return cleaned

    def _geo_feats(self, pos, other, pix_cm):
        # 1. Normalize
        pos = pos / pix_cm
        other = other / pix_cm

        # 2. Centering (Relative to Tail Base)
        origin = pos[:, 9:10, :]
        centered = pos - origin
        other_centered = other - origin

        # 3. ROTATION (Face East) - CRITICAL FOR INFERENCE MATCHING
        spine = centered[:, 3, :] # [T, 2]
        angles = np.arctan2(spine[:, 1], spine[:, 0]) # [T]

        c = np.cos(-angles)
        s = np.sin(-angles)

        # Apply to Agent
        x = centered[..., 0]
        y = centered[..., 1]

        c_exp = c[:, None]
        s_exp = s[:, None]

        centered_rot_x = x * c_exp - y * s_exp
        centered_rot_y = x * s_exp + y * c_exp
        centered = np.stack([centered_rot_x, centered_rot_y], axis=-1)

        # Apply to Target
        x_o = other_centered[..., 0]
        y_o = other_centered[..., 1]

        other_rot_x = x_o * c_exp - y_o * s_exp
        other_rot_y = x_o * s_exp + y_o * c_exp
        other_centered = np.stack([other_rot_x, other_rot_y], axis=-1)

        # 4. Velocity (computed on Rotated & Centered coords)
        vel = np.diff(centered, axis=0, prepend=centered[0:1])
        speed = np.sqrt((vel**2).sum(axis=-1))

        # 5. Relation
        dist = np.sqrt(((centered - other_centered)**2).sum(axis=-1))

        feat = np.stack([
            centered[...,0], centered[...,1],
            vel[...,0], vel[...,1],
            speed, dist,
            np.zeros_like(speed), np.zeros_like(speed),
            np.zeros_like(speed), np.zeros_like(speed),
            np.zeros_like(speed), np.zeros_like(speed),
            np.zeros_like(speed), np.zeros_like(speed),
            np.zeros_like(speed), np.zeros_like(speed),
        ], axis=-1)

        return feat.astype(np.float32)

    def load_full_video_features_for_pair(self, idx):
        """
        Loads the FULL video features for sliding window inference for a SPECIFIC pair.
        Returns: (feats, lab_idx, agent_id, target_id, frames)
        """
        sample = self.samples[idx]
        lab = sample['lab_id']
        vid = sample['video_id']
        agent_id = sample['agent_id']
        target_id = sample['target_id']
        conf = LAB_CONFIGS.get(lab, LAB_CONFIGS['DEFAULT'])

        fpath = self.tracking_dir / lab / f"{vid}.parquet"

        if not fpath.exists():
            return None, None, None, None, None, None

        try:
            # Polars Optimization
            lf = pl.scan_parquet(fpath)

            # 1. Get Limits
            meta_df = lf.select(pl.col('video_frame').max()).collect()
            if meta_df.shape[0] == 0 or meta_df.item(0, 0) is None:
                 return None, None, None, None, None

            max_frame = meta_df.item(0, 0)
            L_alloc = max_frame + 1

            # 2. Fetch Data for this Pair
            q = (
                lf
                .filter(
                    pl.col('mouse_id').cast(pl.Utf8).is_in([str(agent_id), str(target_id)])
                )
                .collect()
            )

            if q.is_empty():
                return None, None, None, None, None

            df = q.to_pandas()

            raw_m1 = np.zeros((L_alloc, 11, 2), dtype=np.float32)
            raw_m2 = np.zeros((L_alloc, 11, 2), dtype=np.float32)

            # Validity Masks
            valid_m1 = np.zeros(L_alloc, dtype=bool)
            valid_m2 = np.zeros(L_alloc, dtype=bool)

            # Mouse 1
            d1 = df[df['mouse_id'].astype(str) == str(agent_id)]
            for i, bp in enumerate(BODY_PARTS):
                rows = d1[d1['bodypart']==bp]
                if not rows.empty:
                    indices = rows['video_frame'].values
                    valid = (indices >= 0) & (indices < L_alloc)
                    raw_m1[indices[valid], i] = rows[['x', 'y']].values[valid]
                    valid_m1[indices[valid]] = True

            # Mouse 2
            d2 = df[df['mouse_id'].astype(str) == str(target_id)]
            for i, bp in enumerate(BODY_PARTS):
                rows = d2[d2['bodypart']==bp]
                if not rows.empty:
                    indices = rows['video_frame'].values
                    valid = (indices >= 0) & (indices < L_alloc)
                    raw_m2[indices[valid], i] = rows[['x', 'y']].values[valid]
                    valid_m2[indices[valid]] = True

            # Intersection of presence
            valid_frames = valid_m1 & valid_m2

            # Fix Teleport
            raw_m1 = self._fix_teleport(raw_m1)
            raw_m2 = self._fix_teleport(raw_m2)

            # Feature Extraction
            feats = self._geo_feats(raw_m1, raw_m2, conf['pix_cm'])

            lab_idx = list(LAB_CONFIGS.keys()).index(lab) if lab in LAB_CONFIGS else 0

            # Frames: Just indices since we don't have 'frame' column
            frames = np.arange(L_alloc)

            return torch.tensor(feats), lab_idx, agent_id, target_id, frames, valid_frames, vid

        except Exception as e:
            print(f"Error loading {vid}: {e}")
            return None, None, None, None, None, None

    def __len__(self): return len(self.samples)

class ValidationDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        return self.ds.load_full_video_features_for_pair(idx)

# ==============================================================================
# 2. MODEL ARCHITECTURE (Copied exactly from train.py)
# ==============================================================================
# (Pasting the V4 Architecture here)
class CanonicalGraphAdapter(nn.Module):
    def __init__(self, input_nodes=11, canonical_nodes=11, feat_dim=16, num_labs=20):
        super().__init__()
        self.projection = nn.Parameter(torch.eye(input_nodes).unsqueeze(0).repeat(num_labs, 1, 1))
        self.projection.data += torch.randn_like(self.projection) * 0.01
        self.bias = nn.Parameter(torch.zeros(num_labs, 1, canonical_nodes, feat_dim))
        self.refine = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2), nn.LayerNorm(feat_dim * 2), nn.GELU(), nn.Linear(feat_dim * 2, feat_dim)
        )
    def forward(self, x, lab_idx):
        b, t, n, f = x.shape
        W = self.projection[lab_idx]
        B = self.bias[lab_idx]
        x_flat = x.view(-1, n, f)
        W_flat = W.unsqueeze(1).repeat(1, t, 1, 1).view(-1, n, n)
        x_t = x_flat.transpose(1, 2)
        out = torch.bmm(x_t, W_flat).transpose(1, 2).view(b, t, n, f)
        out = out + B
        return self.refine(out)

class SocialInteractionBlock(nn.Module):
    def __init__(self, node_dim=16, hidden_dim=64):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=node_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(node_dim)
        self.relational_mlp = nn.Sequential(nn.Linear(node_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 32))
    def forward(self, agent_canon, target_canon):
        b, t, n, f = agent_canon.shape
        a_flat = agent_canon.view(b*t, n, f)
        t_flat = target_canon.view(b*t, n, f)
        attn_out, _ = self.attention(query=a_flat, key=t_flat, value=t_flat)
        interact_ctx = self.norm(a_flat + attn_out)
        combined = torch.cat([a_flat, interact_ctx], dim=-1)
        interact_summ = combined.mean(dim=1).view(b, t, -1)
        return agent_canon, target_canon, self.relational_mlp(interact_summ)

class MorphologicalInteractionCore(nn.Module):
    def __init__(self, num_labs=20):
        super().__init__()
        self.adapter = CanonicalGraphAdapter(input_nodes=11, canonical_nodes=11, num_labs=num_labs)
        self.interaction = SocialInteractionBlock()
        self.frame_fusion = nn.Linear(384, 128)
    def forward(self, agent_x, target_x, lab_idx):
        a_c = self.adapter(agent_x, lab_idx)
        t_c = self.adapter(target_x, lab_idx)
        _, _, rel_embed = self.interaction(a_c, t_c)
        b, t, n, f = a_c.shape
        a_flat = a_c.view(b, t, -1)
        t_flat = t_c.view(b, t, -1)
        combined = torch.cat([a_flat, t_flat, rel_embed], dim=-1)
        out = self.frame_fusion(combined)
        return out, a_c, t_c

class SplitStreamInteractionBlock(nn.Module):
    def __init__(self, node_dim=16, hidden_dim=128):
        super().__init__()
        self.self_input_size = 11 * 4
        self.self_projector = nn.Sequential(nn.Linear(self.self_input_size, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.relational_mlp = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 32))
        full_node_dim = 11 * node_dim
        pair_input_dim = (full_node_dim * 2) + 32 + 2
        self.pair_projector = nn.Sequential(nn.Linear(pair_input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.role_embedding = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    def forward(self, agent_c, target_c, role_indices=None):
        batch, time, nodes, feat = agent_c.shape
        agent_proprioception = agent_c[..., 0:4]
        agent_flat_self = agent_proprioception.contiguous().view(batch, time, -1)
        self_feat = self.self_projector(agent_flat_self)
        agent_flat_full = agent_c.view(batch, time, -1)
        target_flat_full = target_c.view(batch, time, -1)
        rel_feats = agent_c[..., 4:7].mean(dim=2)
        rel_embed = self.relational_mlp(rel_feats)
        if role_indices is None:
            selected_role = self.role_embedding[0].view(1, 1, 2).expand(batch, time, 2)
        else:
            selected_role = self.role_embedding[role_indices].unsqueeze(1).expand(batch, time, 2)
        pair_input = torch.cat([agent_flat_full, target_flat_full, rel_embed, selected_role], dim=-1)
        pair_feat = self.pair_projector(pair_input)
        return self_feat, pair_feat

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        L = x.size(1)
        if L > self.pe.size(0):
            return x + self.pe[:self.pe.size(0), :].repeat(math.ceil(L/self.pe.size(0)), 1)[:L, :]
        return x + self.pe[:L, :]

class LocalGlobalChronosEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super().__init__()
        self.global_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=5000)
        global_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=512, batch_first=True, dropout=0.1, activation="gelu")
        self.global_transformer = nn.TransformerEncoder(global_layer, num_layers=2)
        # AUX HEAD
        self.global_classifier = nn.Linear(hidden_dim, 37)

        self.self_tcn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4), nn.BatchNorm1d(hidden_dim), nn.GELU(),
        )
        self.self_local_attn = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden_dim)

        self.pair_tcn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4), nn.BatchNorm1d(hidden_dim), nn.GELU(),
        )
        self.pair_local_attn = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.pair_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.pair_norm = nn.LayerNorm(hidden_dim)

    def forward(self, global_feat, local_self, local_pair):
        g_emb = self.global_proj(global_feat)
        g_emb = self.pos_encoder(g_emb)
        global_memory = self.global_transformer(g_emb)

        g_logits = self.global_classifier(global_memory)

        s_in = local_self.permute(0, 2, 1)
        s_tcn = self.self_tcn(s_in).permute(0, 2, 1)
        s_tcn = self.self_local_attn(s_tcn)
        s_ctx, _ = self.self_attn(query=s_tcn, key=global_memory, value=global_memory)
        self_out = self.self_norm(s_tcn + s_ctx)

        p_in = local_pair.permute(0, 2, 1)
        p_tcn = self.pair_tcn(p_in).permute(0, 2, 1)
        p_tcn = self.pair_local_attn(p_tcn)
        p_ctx, _ = self.pair_attn(query=p_tcn, key=global_memory, value=global_memory)
        pair_out = self.pair_norm(p_tcn + p_ctx)

        return self_out, pair_out, g_logits

class MultiTaskLogicHead(nn.Module):
    def __init__(self, input_dim=128, num_labs=20):
        super().__init__()
        self.lab_embedding = nn.Embedding(num_labs, 32)
        fusion_dim = input_dim + 32
        expanded_dim = 256
        self.self_classifier = nn.Sequential(nn.Linear(fusion_dim, expanded_dim), nn.LayerNorm(expanded_dim), nn.GELU(), nn.Linear(expanded_dim, 11))
        self.pair_classifier = nn.Sequential(nn.Linear(fusion_dim, expanded_dim), nn.LayerNorm(expanded_dim), nn.GELU(), nn.Linear(expanded_dim, 26))
        self.center_regressor = nn.Sequential(nn.Linear(fusion_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.gate_control = nn.Linear(1, 1)

    def forward(self, self_feat, pair_feat, lab_idx, agent_c, target_c):
        batch, time, _ = self_feat.shape
        lab_context = self.lab_embedding(lab_idx).unsqueeze(1).expand(-1, time, -1)
        self_input = torch.cat([self_feat, lab_context], dim=-1)
        pair_input = torch.cat([pair_feat, lab_context], dim=-1)
        self_logits = self.self_classifier(self_input)
        pair_logits = self.pair_classifier(pair_input)
        center_score = torch.sigmoid(self.center_regressor(pair_input))
        a_pos = agent_c[:, :, 0, :2]
        t_pos = target_c[:, :, 0, :2]
        dist = torch.norm(a_pos - t_pos, dim=-1, keepdim=True)
        gate = torch.sigmoid(self.gate_control(dist))
        self_probs = torch.sigmoid(self_logits)
        pair_probs = torch.sigmoid(pair_logits) * gate
        return self_probs, pair_probs, center_score

# BEHAVIOR DEFINITIONS
SELF_BEHAVIORS = sorted(["biteobject", "climb", "dig", "exploreobject", "freeze", "genitalgroom", "huddle", "rear", "rest", "run", "selfgroom"])
PAIR_BEHAVIORS = sorted(["allogroom", "approach", "attack", "attemptmount", "avoid", "chase", "chaseattack", "defend", "disengage", "dominance", "dominancegroom", "dominancemount", "ejaculate", "escape", "flinch", "follow", "intromit", "mount", "reciprocalsniff", "shepherd", "sniff", "sniffbody", "sniffface", "sniffgenital", "submit", "tussle"])

class EthoSwarmNet(nn.Module):
    def __init__(self, num_classes=37, input_dim=128):
        super().__init__()
        self.morph_core = MorphologicalInteractionCore(num_labs=20)
        self.split_interaction = SplitStreamInteractionBlock(hidden_dim=128)
        self.chronos = LocalGlobalChronosEncoder(input_dim=128, hidden_dim=128)
        self.logic_head = MultiTaskLogicHead(input_dim=128, num_labs=20)
        self.register_buffer('self_indices', self._get_indices(SELF_BEHAVIORS))
        self.register_buffer('pair_indices', self._get_indices(PAIR_BEHAVIORS))

    def _get_indices(self, subset_list):
        indices = []
        for beh in subset_list:
            if beh in ACTION_LIST: indices.append(ACTION_LIST.index(beh))
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, global_agent, global_target, local_agent, local_target, lab_idx, role_idx=None):
        g_out, _, _ = self.morph_core(global_agent, global_target, lab_idx)
        _, l_ac, l_tc = self.morph_core(local_agent, local_target, lab_idx)
        l_self, l_pair = self.split_interaction(l_ac, l_tc, role_indices=role_idx)
        t_self, t_pair, g_logits = self.chronos(g_out, l_self, l_pair)
        p_self, p_pair, center_score = self.logic_head(t_self, t_pair, lab_idx, l_ac, l_tc)
        batch, time, _ = p_self.shape
        final_output = torch.zeros(batch, time, 37, device=p_self.device, dtype=p_self.dtype)
        final_output.index_copy_(2, self.self_indices, p_self)
        final_output.index_copy_(2, self.pair_indices, p_pair)
        return final_output, center_score, g_logits

# ==============================================================================
# UTILS (Copied from train.py)
# ==============================================================================
def load_lab_vocabulary(vocab_path, action_to_idx, num_classes, device):
    """
    Loads a boolean mask [20, 37] where 1.0 means the lab annotates that action.
    """
    if not os.path.exists(vocab_path):
        return torch.ones(25, 37).to(device)

    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    lab_names = sorted(list(LAB_CONFIGS.keys()))
    mask = torch.zeros(len(lab_names), num_classes).to(device)

    for i, name in enumerate(lab_names):
        if name in vocab:
            for a in vocab[name]:
                if a in action_to_idx:
                    mask[i, action_to_idx[a]] = 1.0
        else:
            mask[i, :] = 1.0
    return mask

# ==============================================================================
# 3. INFERENCE ENGINE (Sliding Window & Post-Processing)
# ==============================================================================
def run_inference():
    if 'mabe_mouse_behavior_detection_path' in globals():
        DATA_PATH = globals()['mabe_mouse_behavior_detection_path']
    elif os.path.exists('/kaggle/input/MABe-mouse-behavior-detection'):
        DATA_PATH = '/kaggle/input/MABe-mouse-behavior-detection'
    else:
        print("Dataset not found."); return

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load Thresholds
    thresholds = torch.ones(37).to(DEVICE) * 0.4

    # Check paths as requested
    THRESH_PATH = "/kaggle/input/mabe-separated/thresholds.json"
    if not os.path.exists(THRESH_PATH):
        THRESH_PATH = "thresholds.json"

    if os.path.exists(THRESH_PATH):
        with open(THRESH_PATH, "r") as f:
            th_list = json.load(f)
            thresholds = torch.tensor(th_list).to(DEVICE)
        print(f"Loaded Optimized Thresholds from {THRESH_PATH}")
    else:
        print(f"Warning: thresholds not found at {THRESH_PATH}, using defaults.")

    # 2. Load Model
    model = EthoSwarmNet(num_classes=NUM_CLASSES)
    model.to(DEVICE)

    # Weights Path
    MODEL_PATH = "/kaggle/input/mabe-separated/ethoswarm_v4_ep10.pth"
    if not os.path.exists(MODEL_PATH):
        # Fallback to local
        local_weights = sorted([f for f in os.listdir(".") if f.startswith("ethoswarm_v4_ep")])
        if local_weights:
            MODEL_PATH = local_weights[-1]

    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from: {MODEL_PATH}")
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
    else:
        print("No weights found! Inference will be random.")

    model.eval()

    # Load Masks
    VOCAB_PATH = '/kaggle/input/mabe-metadata/results/lab_vocabulary.json'
    # Fallback to local if running offline/differently
    if not os.path.exists(VOCAB_PATH):
        VOCAB_PATH = "lab_vocabulary.json"

    lab_masks = load_lab_vocabulary(VOCAB_PATH, ACTION_TO_IDX, NUM_CLASSES, DEVICE)

    # 3. Data
    ds = BioPhysicsDataset(DATA_PATH, 'test')

    # 4. Processing Loop
    submission_rows = []

    WINDOW_SIZE = 256
    STRIDE = 128

    # Parallel Loader
    val_dataset_full = ValidationDataset(ds)
    val_loader_full = torch.utils.data.DataLoader(val_dataset_full, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: x[0])

    print(f"Starting Inference on {len(ds)} samples...")
    for i, batch_data in enumerate(val_loader_full):
        # New Pair-Based Loader
        feats, lab_idx, agent_id, target_id, frames, valid_mask, vid = batch_data

        if feats is None: continue

        # Prepare Tensors
        T_total = len(feats)
        prob_accum = torch.zeros((T_total, 37), device=DEVICE)
        count_accum = torch.zeros((T_total, 37), device=DEVICE)

        # REVISED STRATEGY FOR SPEED/ACCURACY
        # 1. Compute Global Embedding for entire video (subsampled)
        g_feats_full = feats.unsqueeze(0).to(DEVICE) # [1, T, 11, 16]

        # NOTE: Training uses gx=lx (local window).
        # Using subsampled global video here causes distribution shift.
        # We switch to gx=lx to match training distribution.

        # 2. Loop Windows for Local (Batched)
        windows = []
        starts = []

        for start in range(0, T_total, STRIDE):
            end = min(start + WINDOW_SIZE, T_total)

            # Skip empty windows (no tracking data)
            if not valid_mask[start:end].any():
                continue

            l_input = g_feats_full[:, start:end, :, :]
            if l_input.shape[1] < WINDOW_SIZE:
                 pad_n = WINDOW_SIZE - l_input.shape[1]
                 l_input = F.pad(l_input, (0,0,0,0,0,pad_n))

            windows.append(l_input)
            starts.append(start)

        # Process in Batches
        INF_BATCH_SIZE = 32
        for i in range(0, len(windows), INF_BATCH_SIZE):
            batch_windows = windows[i : i+INF_BATCH_SIZE]
            batch_starts = starts[i : i+INF_BATCH_SIZE]

            if not batch_windows: continue

            # Stack
            lx_batch = torch.cat(batch_windows, dim=0) # [B, 256, 11, 16]
            B = lx_batch.shape[0]

            # MATCH TRAINING: Global Input = Local Input
            gx_batch = lx_batch

            lid_batch = torch.tensor([lab_idx]*B).to(DEVICE)

            with torch.no_grad():
                probs, _, _ = model(gx_batch, gx_batch, lx_batch, lx_batch, lid_batch)

            # Accumulate
            for b in range(B):
                start = batch_starts[b]
                end = min(start + WINDOW_SIZE, T_total)
                valid_len = end - start
                prob_accum[start:end] += probs[b, :valid_len]
                count_accum[start:end] += 1.0

        # Average
        final_probs = prob_accum / (count_accum + 1e-6)

        # Post-Processing: Single Action Selection + Lab Masking

        # 1. Apply Lab Mask
        # Note: lab_masks is [20, 37]. final_probs is [T, 37].
        if lab_idx < len(lab_masks):
            mask = lab_masks[lab_idx].unsqueeze(0) # [1, 37]
            final_probs = final_probs * mask

        # 2. Multi-Label Thresholding (Matches train.py validation)
        # Using per-class thresholds loaded from JSON
        # final_probs: [T, 37], thresholds: [37]

        preds = (final_probs > thresholds.unsqueeze(0)).int().cpu().numpy() # [T, 37]

        # Generate Segments per Class
        for c in range(NUM_CLASSES):
            action_name = ACTION_LIST[c]
            binary_seq = preds[:, c]

            # Find contiguous segments
            diffs = np.diff(np.concatenate(([0], binary_seq, [0])))
            starts_seg = np.where(diffs == 1)[0]
            stops_seg = np.where(diffs == -1)[0]

            for s, e in zip(starts_seg, stops_seg):
                # Map local indices s, e to real frames
                # s is start index (inclusive), e is stop index (exclusive in local 0..T)

                # Safety check for boundaries
                if s >= len(frames): continue
                if e > len(frames): e = len(frames)

                real_start = frames[s]
                # frames[e-1] is the last included frame. +1 makes it exclusive.
                real_stop = frames[e-1] + 1

                submission_rows.append([
                        0,
                        vid,
                        str(agent_id),
                        str(target_id),
                        action_name,
                        real_start,
                        real_stop
                ])

    # Create DF
    df_sub = pd.DataFrame(submission_rows, columns=['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])

    # Sort and re-index
    df_sub = df_sub.sort_values(['video_id', 'start_frame'])
    df_sub['row_id'] = np.arange(len(df_sub))

    df_sub.to_csv("submission.csv", index=False)
    print(f"Inference Complete. Saved {len(df_sub)} rows to submission.csv")

if __name__ == '__main__':
    run_inference()
