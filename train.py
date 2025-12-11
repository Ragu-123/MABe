# Module 1: The Bio-Physics Data Engine
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from pathlib import Path
import ast
import random
import os
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import math
from collections import defaultdict
import polars as pl

# --- CONFIGURATION ---
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
PART_TO_IDX = {p: i for i, p in enumerate(BODY_PARTS)}

class BioPhysicsDataset(Dataset):
    def __init__(self, data_root, mode='train', video_ids=None):
        self.root = Path(data_root)
        self.mode = mode
        self.tracking_dir = self.root / f"{mode}_tracking"
        self.annot_dir = self.root / f"{mode}_annotation"
        
        self.metadata = pd.read_csv(self.root / f"{mode}.csv")
        if video_ids is not None:
            self.metadata = self.metadata[self.metadata['video_id'].astype(str).isin(video_ids)]
        
        # --- OPTIMIZED PAIR PERMUTATION LOGIC ---
        # Using a default heuristic to avoid slow Parquet scans in __init__.
        # We assume standard 2-mouse setup ("mouse1", "mouse2") unless we find cached info.
        # Scanning 100k files takes too long.
        self.samples = []

        # Default mice list
        default_mice = ['mouse1', 'mouse2']

        print(f"Building samples for {len(self.metadata)} videos (Optimized)...")
        for _, row in self.metadata.iterrows():
            vid = str(row['video_id'])
            lab = row['lab_id']

            # Use default mice to speed up initialization.
            # Ideally, we would use a pre-computed metadata file for groups.
            mice = default_mice
            
            for agent in mice:
                for target in mice:
                    if agent != target:
                        self.samples.append({
                            'video_id': vid,
                            'lab_id': lab,
                            'agent_id': agent,
                            'target_id': target
                        })

        self.local_window = 256
        self.action_windows = []
        if self.mode == 'train':
            self._scan_actions_safe()

    def _scan_actions_safe(self):
        # Scan subset for center sampling
        count = 0
        scan_limit = 200 # optimization
        for i, s in enumerate(self.samples):
            if i > scan_limit: break
            p = self.annot_dir / s['lab_id'] / f"{s['video_id']}.parquet"
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    # Filter for action
                    df = df[df['action'].isin(ACTION_TO_IDX)]
                    if not df.empty:
                        for c in ((df['start_frame'] + df['stop_frame']) // 2).values:
                            self.action_windows.append((i, int(c)))
                            count += 1
                except: pass

    def _fix_teleport(self, pos):
        # pos: [T, 11, 2]
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
        # Optimized 8-Channel Feature Stack
        # 1. Normalize
        pos = pos / pix_cm
        other = other / pix_cm
        
        # 2. Centering (Relative to Tail Base)
        origin = pos[:, 9:10, :]
        centered = pos - origin
        other_centered = other - origin
        
        # 3. ROTATION (Face East)
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

        # 4. Dynamics
        # Velocity
        vel = np.diff(centered, axis=0, prepend=centered[0:1])
        # Fix NaNs: maximum(0) before sqrt
        speed = np.sqrt(np.maximum((vel**2).sum(axis=-1), 0))

        # Acceleration
        acc = np.diff(vel, axis=0, prepend=vel[0:1])

        # 5. Relation
        dist = np.sqrt(np.maximum(((centered - other_centered)**2).sum(axis=-1), 0))
        
        # Pack to 8 Channels
        # [Pos X, Pos Y, Vel X, Vel Y, Speed, Dist, Acc X, Acc Y]
        feat = np.stack([
            centered[...,0], centered[...,1],
            vel[...,0], vel[...,1],
            speed, dist,
            acc[...,0], acc[...,1]
        ], axis=-1)
        
        # Final NaN Safety
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        return feat.astype(np.float32)

    def _load(self, idx, center=None):
        sample = self.samples[idx]
        lab = sample['lab_id']
        conf = LAB_CONFIGS.get(lab, LAB_CONFIGS['DEFAULT'])
        
        agent_id = sample['agent_id']
        target_id = sample['target_id']
        
        fpath = self.tracking_dir / lab / f"{sample['video_id']}.parquet"
        
        raw_m1 = np.zeros((self.local_window, 11, 2), dtype=np.float32)
        raw_m2 = np.zeros((self.local_window, 11, 2), dtype=np.float32)

        if fpath.exists():
            try:
                # Optimized Load using Polars if possible, else Pandas
                # We need to filter rows efficiently.
                # Pandas 'read_parquet' filters are okay but not great.
                # 'polars' is installed now.
                
                # Use Polars for speed
                # Query: mouse_id in [agent, target]
                q = pl.scan_parquet(fpath).filter(
                    pl.col("mouse_id").is_in([agent_id, target_id])
                )
                df = q.collect().to_pandas()
                
                if not df.empty:
                    # Optimized Vectorized Buffer Fill
                    L = df['frame'].max() + 1

                    if center is None: center = random.randint(0, L)
                    s = max(0, min(center - self.local_window//2, L - self.local_window))
                    e = min(s + self.local_window, L)

                    # Slice Frame Window
                    df = df[(df['frame'] >= s) & (df['frame'] < e)]

                    # Map bodyparts to integers
                    df['bp_idx'] = df['bodypart'].map(PART_TO_IDX)
                    df['frame_idx'] = (df['frame'] - s).astype(int)

                    # Drop invalid mappings
                    df = df.dropna(subset=['bp_idx'])
                    df['bp_idx'] = df['bp_idx'].astype(int)

                    # Split
                    df_a = df[df['mouse_id'] == agent_id]
                    df_t = df[df['mouse_id'] == target_id]

                    # Vectorized Assign
                    if not df_a.empty:
                        raw_m1[df_a['frame_idx'], df_a['bp_idx'], 0] = df_a['x'].values
                        raw_m1[df_a['frame_idx'], df_a['bp_idx'], 1] = df_a['y'].values
                        
                    if not df_t.empty:
                        raw_m2[df_t['frame_idx'], df_t['bp_idx'], 0] = df_t['x'].values
                        raw_m2[df_t['frame_idx'], df_t['bp_idx'], 1] = df_t['y'].values

            except Exception as ex:
                # Fallback to zeros on error
                pass

        # 1. Teleport Fix
        raw_m1 = self._fix_teleport(raw_m1)
        raw_m2 = self._fix_teleport(raw_m2)
        
        # 3. Features
        feats = self._geo_feats(raw_m1, raw_m2, conf['pix_cm'])
        
        # 4. Targets
        target = torch.zeros((self.local_window, NUM_CLASSES), dtype=torch.float32)
        weights = torch.zeros(self.local_window, dtype=torch.float32)
        centerness = torch.zeros((self.local_window, 1), dtype=torch.float32)
        
        valid_len = e - s if 'e' in locals() else self.local_window
        weights[:valid_len] = 1.0

        if self.mode == 'train':
            ap = self.annot_dir / lab / f"{sample['video_id']}.parquet"
            if ap.exists():
                try:
                    # Optimized annotation read
                    # We only need rows matching our pair
                    # Polars again
                    aq = pl.scan_parquet(ap).filter(
                        (pl.col("action").is_in(ACTION_LIST))
                    )
                    # Filter agent/target if columns exist?
                    # Assuming standard mabe format
                    adf = aq.collect().to_pandas()

                    # Manual filter if columns exist
                    if 'agent' in adf.columns and 'target' in adf.columns:
                         adf = adf[(adf['agent'] == agent_id) & (adf['target'] == target_id)]

                    for _, row in adf.iterrows():
                        a_idx = ACTION_TO_IDX.get(row['action'])
                        if a_idx is not None:
                            st, et = int(row['start_frame'])-s, int(row['stop_frame'])-s
                            st, et = max(0, st), min(self.local_window, et)
                            if st < et:
                                target[st:et, a_idx] = 1.0
                                # Centerness
                                c_local = (st + et) / 2.0
                                width = et - st
                                t_grid = torch.arange(st, et, dtype=torch.float32)
                                sigma = width / 6.0 + 1e-6
                                g = torch.exp( - (t_grid - c_local)**2 / (2 * sigma**2) )
                                centerness[st:et, 0] = torch.max(centerness[st:et, 0], g)
                except: pass
        
        lab_idx = list(LAB_CONFIGS.keys()).index(lab) if lab in LAB_CONFIGS else 0
        return torch.tensor(feats), torch.tensor(feats), target, weights, lab_idx, centerness

    def __getitem__(self, idx):
        if self.mode=='train' and random.random() < 0.9 and len(self.action_windows)>0:
            i, c = self.action_windows[random.randint(0, len(self.action_windows)-1)]
            return self._load(i, c)
        return self._load(idx)
    
    def __len__(self): return len(self.samples)

def pad_collate_dual(batch):
    gx, lx, t, w, lid, center = zip(*batch)
    return torch.stack(gx), torch.stack(lx), torch.stack(t), torch.stack(w), torch.tensor(lid), torch.stack(center)

# Module 2: The Morphological & Interaction Core.

# ==============================================================================
# 1. CANONICAL GRAPH ADAPTER (Signal Refinement)
# ==============================================================================
class CanonicalGraphAdapter(nn.Module):
    # INPUT: [B, T, 11, 8] (Optimized Features)
    def __init__(self, input_nodes=11, canonical_nodes=11, feat_dim=8, num_labs=20):
        super().__init__()

        # Learnable Projection Matrix: (NumLabs, 11, 11)
        # MATH CONFIRMATION:
        # We want to mix NODES. X is [Nodes x Feats].
        # We compute X^T [Feats x Nodes] @ W [Nodes x Nodes] = [Feats x Nodes]
        # This effectively creates new nodes as linear combos of old nodes.
        self.projection = nn.Parameter(torch.eye(input_nodes).unsqueeze(0).repeat(num_labs, 1, 1))
        
        self.projection.data += torch.randn_like(self.projection) * 0.01
        self.bias = nn.Parameter(torch.zeros(num_labs, 1, canonical_nodes, feat_dim))

        self.refine = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim)
        )

    def forward(self, x, lab_idx):
        b, t, n, f = x.shape
        W = self.projection[lab_idx]
        B = self.bias[lab_idx]
        
        x_flat = x.view(-1, n, f)
        W_flat = W.unsqueeze(1).repeat(1, t, 1, 1).view(-1, n, n)

        x_t = x_flat.transpose(1, 2) 
        out = torch.bmm(x_t, W_flat) 

        out = out.transpose(1, 2).view(b, t, n, f)
        out = out + B
        out = self.refine(out)

        return out

# ==============================================================================
# 2. SOCIAL INTERACTION BLOCK (Upgraded to Graph Attention)
# ==============================================================================
class SocialInteractionBlock(nn.Module):
    def __init__(self, node_dim=8, hidden_dim=64):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=node_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(node_dim)
        self.relational_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32)
        )

    def forward(self, agent_canon, target_canon):
        b, t, n, f = agent_canon.shape
        a_flat = agent_canon.view(b*t, n, f)
        t_flat = target_canon.view(b*t, n, f)
        
        attn_out, _ = self.attention(query=a_flat, key=t_flat, value=t_flat)
        interact_ctx = self.norm(a_flat + attn_out)
        
        combined = torch.cat([a_flat, interact_ctx], dim=-1)
        interact_summ = combined.mean(dim=1)
        interact_summ = interact_summ.view(b, t, -1)
        
        rel_embed = self.relational_mlp(interact_summ)
        return agent_canon, target_canon, rel_embed

# ==============================================================================
# WRAPPER: MORPHOLOGICAL INTERACTION CORE
# ==============================================================================
class MorphologicalInteractionCore(nn.Module):
    def __init__(self, num_labs=20):
        super().__init__()
        # Input dim 8
        self.adapter = CanonicalGraphAdapter(input_nodes=11, canonical_nodes=11, feat_dim=8, num_labs=num_labs)
        self.interaction = SocialInteractionBlock(node_dim=8)

        # Fusion: (11 nodes * 8 features * 2 agents) + 32 relation = 176 + 32 = 208
        self.frame_fusion = nn.Linear(208, 128)

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

# Module 3: The Split-Stream Interaction Block

class SplitStreamInteractionBlock(nn.Module):
    def __init__(self, node_dim=8, hidden_dim=128):
        super(SplitStreamInteractionBlock, self).__init__()

        # Branch A: Self (Pos+Vel = 4 feats? No, we have 8. Pos,Vel,Speed,Dist,Acc)
        # Let's take first 4 (Pos, Vel)
        self.self_input_size = 11 * 4
        
        self.self_projector = nn.Sequential(
            nn.Linear(self.self_input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.relational_mlp = nn.Sequential(
            nn.Linear(3, 32), # Fallback relation
            nn.GELU(),
            nn.Linear(32, 32)
        )

        full_node_dim = 11 * node_dim
        pair_input_dim = (full_node_dim * 2) + 32 + 2
        
        self.pair_projector = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.role_embedding = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    def forward(self, agent_c, target_c, role_indices=None):
        batch, time, nodes, feat = agent_c.shape
        # Take first 4 channels for self (Pos X/Y, Vel X/Y)
        agent_proprioception = agent_c[..., 0:4]
        agent_flat_self = agent_proprioception.contiguous().view(batch, time, -1)
        
        self_feat = self.self_projector(agent_flat_self)

        agent_flat_full = agent_c.view(batch, time, -1)
        target_flat_full = target_c.view(batch, time, -1)
        
        # Rel feats: Channels 4,5,6? (Speed, Dist, AccX) - Just take mean of Dist (5)
        # Let's simple take Dist (ch 5) and Speed (ch 4)
        rel_feats = agent_c[..., 4:7].mean(dim=2) # [B, T, 3] (approx)
        
        rel_embed = self.relational_mlp(rel_feats)

        if role_indices is None:
            selected_role = self.role_embedding[0].view(1, 1, 2).expand(batch, time, 2)
        else:
            selected_role = self.role_embedding[role_indices].unsqueeze(1).expand(batch, time, 2)

        pair_input = torch.cat([
            agent_flat_full, 
            target_flat_full, 
            rel_embed, 
            selected_role
        ], dim=-1)
        
        pair_feat = self.pair_projector(pair_input)

        return self_feat, pair_feat

# Module 4: The Local-Global Chronos Encoder

class LocalGlobalChronosEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(LocalGlobalChronosEncoder, self).__init__()

        self.global_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=5000)

        global_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=512,
            batch_first=True,
            dropout=0.1,
            activation="gelu"
        )
        self.global_transformer = nn.TransformerEncoder(global_layer, num_layers=2)

        # AUX HEAD
        self.global_classifier = nn.Linear(hidden_dim, 37)

        self.self_tcn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        
        self.self_local_attn = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden_dim)

        self.pair_tcn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
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


# Module 5: The Multi-Task Logic Head

class MultiTaskLogicHead(nn.Module):
    def __init__(self, input_dim=128, num_labs=20):
        super(MultiTaskLogicHead, self).__init__()

        self.lab_embedding = nn.Embedding(num_labs, 32)
        fusion_dim = input_dim + 32
        expanded_dim = 256 

        self.self_classifier = nn.Sequential(
            nn.Linear(fusion_dim, expanded_dim),
            nn.LayerNorm(expanded_dim),
            nn.GELU(),
            nn.Linear(expanded_dim, 11) 
        )

        self.pair_classifier = nn.Sequential(
            nn.Linear(fusion_dim, expanded_dim),
            nn.LayerNorm(expanded_dim),
            nn.GELU(),
            nn.Linear(expanded_dim, 26) 
        )

        self.center_regressor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1) 
        )

        self.gate_control = nn.Linear(1, 1)
        
        with torch.no_grad():
            self.gate_control.bias.fill_(2.0)
            nn.init.constant_(self.self_classifier[3].bias, -4.59)
            nn.init.constant_(self.pair_classifier[3].bias, -4.59)
            nn.init.constant_(self.center_regressor[2].bias, 0.0)

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

# Module 6: Final Assembly (EthoSwarmNet V4 - Enhanced)

# BEHAVIOR DEFINITIONS
SELF_BEHAVIORS = sorted(["biteobject", "climb", "dig", "exploreobject", "freeze", "genitalgroom", "huddle", "rear", "rest", "run", "selfgroom"])
PAIR_BEHAVIORS = sorted(["allogroom", "approach", "attack", "attemptmount", "avoid", "chase", "chaseattack", "defend", "disengage", "dominance", "dominancegroom", "dominancemount", "ejaculate", "escape", "flinch", "follow", "intromit", "mount", "reciprocalsniff", "shepherd", "sniff", "sniffbody", "sniffface", "sniffgenital", "submit", "tussle"])

class EthoSwarmNet(nn.Module):
    def __init__(self, num_classes=37, input_dim=128):
        super(EthoSwarmNet, self).__init__()

        self.morph_core = MorphologicalInteractionCore(num_labs=20)
        self.split_interaction = SplitStreamInteractionBlock(hidden_dim=128, node_dim=8) # 8 dim input
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
        
        g_logits_up = F.interpolate(g_logits.permute(0,2,1), size=time, mode='linear').permute(0,2,1)

        return final_output, center_score, g_logits_up

# Module 7: The Training Loop & Validation

# ==============================================================================
# UTILS & METRICS
# ==============================================================================
def load_lab_vocabulary(vocab_path, action_to_idx, num_classes, device):
    if not os.path.exists(vocab_path):
        return torch.ones(25, 37).to(device)
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    lab_names = sorted(list(LAB_CONFIGS.keys()))
    mask = torch.zeros(len(lab_names), num_classes).to(device)
    for i, name in enumerate(lab_names):
        if name in vocab:
            for a in vocab[name]:
                if a in action_to_idx: mask[i, action_to_idx[a]] = 1.0
        else:
            mask[i, :] = 1.0
    return mask

def get_batch_f1(probs_in, targets, batch_vocab_mask, temporal_weights, thresholds=0.4):
    preds = (probs_in > thresholds).float()
    valid_pixels = temporal_weights.unsqueeze(-1) * batch_vocab_mask.unsqueeze(1)
    tp = (preds * targets * valid_pixels).sum()
    fp = (preds * (1-targets) * valid_pixels).sum()
    fn = ((1-preds) * targets * valid_pixels).sum()
    f1 = 2*tp / (2*tp + fp + fn + 1e-6)
    return f1.item()

# ==============================================================================
# LOSS FUNCTION
# ==============================================================================
class DualStreamMaskedFocalLoss(nn.Module):
    def __init__(self, model_self_indices, model_pair_indices, gamma=2.0):
        super().__init__()
        self.self_idx = model_self_indices
        self.pair_idx = model_pair_indices
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction='none')
        self.bce_aux = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, model_output_probs, center_pred, aux_logits, target, center_target, weight_mask, lab_vocab_mask):
        p_self = model_output_probs[:, :, self.self_idx]
        p_pair = model_output_probs[:, :, self.pair_idx]
        t_self = target[:, :, self.self_idx]
        t_pair = target[:, :, self.pair_idx]
        
        p_self = torch.clamp(p_self, 1e-7, 1 - 1e-7)
        p_pair = torch.clamp(p_pair, 1e-7, 1 - 1e-7)
        
        m_self = lab_vocab_mask[:, self.self_idx].unsqueeze(1)
        m_pair = lab_vocab_mask[:, self.pair_idx].unsqueeze(1)
        tm = weight_mask.unsqueeze(-1)
        
        l_self_pos = -t_self * torch.pow(1 - p_self, self.gamma) * torch.log(p_self)
        l_self_neg = -(1 - t_self) * torch.pow(p_self, self.gamma) * torch.log(1 - p_self)
        l_self_raw = l_self_pos + l_self_neg

        l_pair_pos = -t_pair * torch.pow(1 - p_pair, self.gamma) * torch.log(p_pair)
        l_pair_neg = -(1 - t_pair) * torch.pow(p_pair, self.gamma) * torch.log(1 - p_pair)
        l_pair_raw = l_pair_pos + l_pair_neg

        loss_main = (l_self_raw * m_self * tm).sum() / ((m_self * tm).sum() + 1e-6) + \
                    (l_pair_raw * m_pair * tm).sum() / ((m_pair * tm).sum() + 1e-6)
        
        l_center = self.mse(center_pred, center_target)
        loss_center = (l_center * tm).sum() / (tm.sum() + 1e-6)

        vocab_mask_exp = lab_vocab_mask.unsqueeze(1)
        l_aux = self.bce_aux(aux_logits, target)
        loss_aux = (l_aux * vocab_mask_exp * tm).sum() / ((vocab_mask_exp * tm).sum() + 1e-6)

        return loss_main + 0.5 * loss_center + 0.3 * loss_aux

# ==============================================================================
# THRESHOLD TUNER
# ==============================================================================
def find_optimal_thresholds(model, val_loader, device, vocab_mask):
    print("Optimization: Tuning Per-Class Thresholds on Validation Set...")
    model.eval()
    all_preds, all_targs, all_masks, all_weights = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting Val Data"):
            gx, lx, tgt, weights, lid, c_tgt = [b.to(device) for b in batch]
            probs, _, _ = model(gx, gx, lx, lx, lid)
            all_preds.append(probs.cpu())
            all_targs.append(tgt.cpu())
            all_masks.append(vocab_mask[lid].unsqueeze(1).repeat(1, weights.shape[1], 1).cpu())
            all_weights.append(weights.cpu())

    preds = torch.cat(all_preds, dim=0)
    targs = torch.cat(all_targs, dim=0)
    masks = torch.cat(all_masks, dim=0)
    weights = torch.cat(all_weights, dim=0).unsqueeze(-1)

    preds = preds[weights.bool().repeat(1,1,37)]
    targs = targs[weights.bool().repeat(1,1,37)]
    masks = masks[weights.bool().repeat(1,1,37)]

    best_thresholds = torch.ones(37) * 0.4
    search_space = np.linspace(0.1, 0.9, 17)

    for c in range(37):
        p_c = preds.view(-1, 37)[:, c]
        t_c = targs.view(-1, 37)[:, c]
        m_c = masks.view(-1, 37)[:, c]
        valid_idx = m_c > 0.5
        if valid_idx.sum() == 0: continue
        p_c = p_c[valid_idx]
        t_c = t_c[valid_idx]

        best_f1 = -1
        best_th = 0.4
        for th in search_space:
            bin_preds = (p_c > th).float()
            tp = (bin_preds * t_c).sum()
            fp = (bin_preds * (1-t_c)).sum()
            fn = ((1-bin_preds) * t_c).sum()
            f1 = 2*tp / (2*tp + fp + fn + 1e-6)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        best_thresholds[c] = best_th
    return best_thresholds

# ==============================================================================
# TRAINING CONTROLLER
# ==============================================================================
def train_ethoswarm_v3():
    if 'mabe_mouse_behavior_detection_path' in globals():
        DATA_PATH = globals()['mabe_mouse_behavior_detection_path']
    elif os.path.exists('/kaggle/input/MABe-mouse-behavior-detection'):
        DATA_PATH = '/kaggle/input/MABe-mouse-behavior-detection'
    else: 
        DATA_PATH = "./"
        if not os.path.exists(DATA_PATH + "/train.csv"):
             print("Dataset not found."); return
    
    VOCAB_PATH = '/kaggle/input/mabe-metadata/results/lab_vocabulary.json'
    
    gpu_count = torch.cuda.device_count()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 8 * max(1, gpu_count)
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 10

    print(f"Start Training on {gpu_count} GPU(s) | Batch Size: {BATCH_SIZE}")

    meta = pd.read_csv(f"{DATA_PATH}/train.csv")
    vids = meta['video_id'].astype(str).unique()
    np.random.shuffle(vids)
    
    split = int(len(vids) * 0.90)
    train_ids = vids[:split]
    val_ids = vids[split:]
    
    train_ds = BioPhysicsDataset(DATA_PATH, 'train', video_ids=train_ids)
    val_ds = BioPhysicsDataset(DATA_PATH, 'train', video_ids=val_ids)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_dual, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_dual, num_workers=2)
    
    model = EthoSwarmNet(num_classes=NUM_CLASSES, input_dim=8) # Feature dim optimized to 8
    
    model.to(DEVICE)
    if gpu_count > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    
    lab_masks = load_lab_vocabulary(VOCAB_PATH, ACTION_TO_IDX, NUM_CLASSES, DEVICE)
    
    self_indices = [ACTION_TO_IDX[a] for a in SELF_BEHAVIORS]
    pair_indices = [ACTION_TO_IDX[a] for a in PAIR_BEHAVIORS]
         
    loss_fn = DualStreamMaskedFocalLoss(self_indices, pair_indices, gamma=2.0)

    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}")
        
        run_loss = 0.0
        run_f1 = 0.0
        
        for i, batch in enumerate(loop):
            gx, lx, tgt, weights, lid, c_tgt = [b.to(DEVICE) for b in batch]
            
            optimizer.zero_grad()
            role_idx = None
            if random.random() < 0.5:
                 role_idx = torch.ones(gx.shape[0], dtype=torch.long).to(DEVICE)
            
            with torch.cuda.amp.autocast():
                probs, center_pred, aux_logits = model(gx, gx, lx, lx, lid, role_idx)
                loss = loss_fn(probs, center_pred, aux_logits, tgt, c_tgt, weights, lab_masks[lid])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            with torch.no_grad():
                f1 = get_batch_f1(probs, tgt, lab_masks[lid], weights)
                
            run_loss = 0.9*run_loss + 0.1*loss.item() if i>0 else loss.item()
            run_f1 = 0.9*run_f1 + 0.1*f1 if i>0 else f1
            
            if i % 20 == 0:
                loop.set_postfix({'Loss': f"{run_loss:.4f}", 'F1': f"{run_f1:.3f}"})
        
        print("Validating...")
        model.eval()
        val_loss_sum = 0
        val_f1_sum = 0.0
        batches = 0
        with torch.no_grad():
            for batch in val_loader:
                gx, lx, tgt, weights, lid, c_tgt = [b.to(DEVICE) for b in batch]
                probs, center_pred, aux_logits = model(gx, gx, lx, lx, lid)
                loss = loss_fn(probs, center_pred, aux_logits, tgt, c_tgt, weights, lab_masks[lid])
                f1 = get_batch_f1(probs, tgt, lab_masks[lid], weights)
                val_loss_sum += loss.item()
                val_f1_sum += f1
                batches += 1
                
        print(f"Val Loss: {val_loss_sum/batches:.4f} | Val F1: {val_f1_sum/batches:.4f}")
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state, f"ethoswarm_v4_ep{epoch+1}.pth")

    final_thresholds = find_optimal_thresholds(model, val_loader, DEVICE, lab_masks)
    with open("thresholds.json", "w") as f:
        json.dump(final_thresholds.cpu().tolist(), f)
    print("Saved optimal thresholds to thresholds.json")

if __name__ == '__main__':
    train_ethoswarm_v3()
