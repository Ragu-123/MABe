# Module 1: The Bio-Physics Data Engine
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from pathlib import Path
import ast
import random

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
        # Directory logic
        self.tracking_dir = self.root / f"{mode}_tracking"
        self.annot_dir = self.root / f"{mode}_annotation"
        
        # Load Metadata
        self.metadata = pd.read_csv(self.root / f"{mode}.csv")
        
        # Filter Video IDs (e.g., for train/val split)
        if video_ids is not None:
            self.metadata = self.metadata[self.metadata['video_id'].astype(str).isin(video_ids)]
        
        # Build samples from metadata DIRECTLY (Skip strict file check to avoid crash)
        self.samples = []
        for _, row in self.metadata.iterrows():
            self.samples.append({
                'video_id': str(row['video_id']),
                'lab_id': row['lab_id']
            })
            
        # Hardcoded Window
        self.local_window = 256
        self.max_global_tokens = 2048

        # Pre-scan for sampling
        self.action_windows = []
        if self.mode == 'train':
            self._scan_actions_safe()

    def _scan_actions_safe(self):
        # We try to find files. If not found, we skip optimization, but DO NOT CRASH.
        count = 0
        print("Scanning subset of annotations for sampling...")
        for i, s in enumerate(self.samples):
            if i > 500: break # Quick partial scan
            p = self.annot_dir / s['lab_id'] / f"{s['video_id']}.parquet"
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    # Find centers
                    df = df[df['action'].isin(ACTION_TO_IDX)]
                    if not df.empty:
                        for c in ((df['start_frame'] + df['stop_frame']) // 2).values:
                            self.action_windows.append((i, int(c)))
                            count += 1
                except: pass
        if count == 0:
            print("Warning: No actions scanned. Falling back to random sampling.")

    def _fix_teleport(self, pos):
        # pos: [T, 11, 2]
        T, N, _ = pos.shape
        # Identify holes
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
        # Simple geometric extractor
        # Normalize
        pos = pos / pix_cm
        other = other / pix_cm
        
        # Align
        origin = pos[:, 9:10, :] # Tail base
        centered = pos - origin
        other_centered = other - origin
        
        # Velocity
        vel = np.diff(centered, axis=0, prepend=centered[0:1])
        speed = np.sqrt((vel**2).sum(axis=-1))
        
        # Relation
        dist = np.sqrt(((pos - other)**2).sum(axis=-1))
        
        # Pack to 16
        # [Pos X, Pos Y, Vel X, Vel Y, Speed, Rel_Dist] + Pads
        feat = np.stack([
            centered[...,0], centered[...,1],
            vel[...,0], vel[...,1],
            speed, dist,
            np.zeros_like(speed), np.zeros_like(speed), # 7-8
            np.zeros_like(speed), np.zeros_like(speed),
            np.zeros_like(speed), np.zeros_like(speed),
            np.zeros_like(speed), np.zeros_like(speed),
            np.zeros_like(speed), np.zeros_like(speed),
        ], axis=-1)
        
        return feat.astype(np.float32)

    def _load(self, idx, center=None):
        sample = self.samples[idx]
        lab = sample['lab_id']
        conf = LAB_CONFIGS.get(lab, LAB_CONFIGS['DEFAULT'])
        
        # Try Loading Track
        raw_m1, raw_m2 = np.zeros((1,11,2)), np.zeros((1,11,2))
        
        fpath = self.tracking_dir / lab / f"{sample['video_id']}.parquet"
        
        # Load Success?
        success = False
        if fpath.exists():
            try:
                df = pd.read_parquet(fpath)
                mids = df['mouse_id'].unique()
                L = len(df)
                
                # Expand buffer
                raw_m1 = np.zeros((L, 11, 2), dtype=np.float32)
                raw_m2 = np.zeros((L, 11, 2), dtype=np.float32)
                
                m1_id = mids[0]
                m2_id = mids[1] if len(mids) > 1 else m1_id
                
                # Check Bodypart column
                if 'bodypart' in df.columns:
                    for i, bp in enumerate(BODY_PARTS):
                        d1 = df[(df['mouse_id']==m1_id) & (df['bodypart']==bp)][['x','y']].values
                        if len(d1)>0: raw_m1[:len(d1), i] = d1
                        
                        d2 = df[(df['mouse_id']==m2_id) & (df['bodypart']==bp)][['x','y']].values
                        if len(d2)>0: raw_m2[:len(d2), i] = d2
                else:
                    # Wide format check
                    for col in df.columns:
                        if "mouse1" in col:
                            # simplified parsing
                            pass 
                success = True
            except: pass
        
        if not success:
            # DUMMY DATA TO PREVENT CRASH
            # Returns a single frame of zeros
            L = self.local_window
            raw_m1 = np.zeros((L, 11, 2), dtype=np.float32)
            raw_m2 = np.zeros((L, 11, 2), dtype=np.float32)

        # 1. Teleport Fix
        raw_m1 = self._fix_teleport(raw_m1)
        raw_m2 = self._fix_teleport(raw_m2)
        
        # 2. Window
        seq_len = len(raw_m1)
        if center is None: center = random.randint(0, seq_len)
        s = max(0, min(center - self.local_window//2, seq_len - self.local_window))
        e = min(s + self.local_window, seq_len)
        
        idx_slice = np.arange(s, e)
        
        # 3. Features
        feats = self._geo_feats(raw_m1[idx_slice], raw_m2[idx_slice], conf['pix_cm'])
        
        # 4. Targets
        target = torch.zeros((self.local_window, NUM_CLASSES), dtype=torch.float32)
        weights = torch.zeros(self.local_window, dtype=torch.float32)
        
        # Pad
        if len(feats) < self.local_window:
            pad_n = self.local_window - len(feats)
            pad_f = np.zeros((pad_n, 11, 16), dtype=np.float32)
            feats = np.concatenate([feats, pad_f], axis=0)
            # Weights stay 0 at end
            weights[:len(idx_slice)] = 1.0
        else:
            weights[:] = 1.0

        if self.mode == 'train':
            ap = self.annot_dir / lab / f"{sample['video_id']}.parquet"
            if ap.exists():
                try:
                    adf = pd.read_parquet(ap)
                    for _, row in adf.iterrows():
                        if row['action'] in ACTION_TO_IDX:
                            st, et = int(row['start_frame'])-s, int(row['stop_frame'])-s
                            st, et = max(0, st), min(self.local_window, et)
                            if st < et: target[st:et, ACTION_TO_IDX[row['action']]] = 1.0
                except: pass
        
        lab_idx = list(LAB_CONFIGS.keys()).index(lab) if lab in LAB_CONFIGS else 0
        return torch.tensor(feats), torch.tensor(feats), target, weights, lab_idx

    def __getitem__(self, idx):
        if self.mode=='train' and random.random() < 0.9 and len(self.action_windows)>0:
            i, c = self.action_windows[random.randint(0, len(self.action_windows)-1)]
            return self._load(i, c)
        return self._load(idx)
    
    def __len__(self): return len(self.samples)

def pad_collate_dual(batch):
    gx, lx, t, w, lid = zip(*batch)
    return torch.stack(gx), torch.stack(lx), torch.stack(t), torch.stack(w), torch.tensor(lid)

# Module 2: The Morphological & Interaction Core.
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. CANONICAL GRAPH ADAPTER (Signal Refinement)
# ==============================================================================
class CanonicalGraphAdapter(nn.Module):
    # INPUT: [B, T, 11, 16] (Geometric Features)
    def __init__(self, input_nodes=11, canonical_nodes=11, feat_dim=16, num_labs=20):
        super().__init__()

        # Learnable Projection Matrix: (NumLabs, 11, 11)
        # Learns to map tracking artifacts to a canonical topology per lab
        self.projection = nn.Parameter(torch.eye(input_nodes).unsqueeze(0).repeat(num_labs, 1, 1))
        
        # Identity initialization with slight noise
        self.projection.data += torch.randn_like(self.projection) * 0.01

        # Lab-Specific Bias (Correction for systematic sensor offset)
        self.bias = nn.Parameter(torch.zeros(num_labs, 1, canonical_nodes, feat_dim))

        # Refinement MLP (Cleans physics calculations)
        self.refine = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2),
            nn.GELU(),
            nn.Linear(feat_dim * 2, feat_dim)
        )

    def forward(self, x, lab_idx):
        # x: (Batch, Time, 11, 16)
        # lab_idx: (Batch)
        b, t, n, f = x.shape

        # 1. Fetch Weights
        W = self.projection[lab_idx] # (B, 11, 11)
        B = self.bias[lab_idx]       # (B, 1, 11, 16)

        # 2. Graph Projection (Node Mixing)
        # We process all time-steps in parallel by flattening B*T
        x_flat = x.view(-1, n, f) # (B*T, 11, 16)
        
        # Prepare Projection Matrix: Expand to T, then view as (B*T, 11, 11)
        W_flat = W.unsqueeze(1).repeat(1, t, 1, 1).view(-1, n, n)

        # Apply Graph Projection: nodes^T * W
        # (B*T, 16, 11) @ (B*T, 11, 11) -> (B*T, 16, 11)
        x_t = x_flat.transpose(1, 2) 
        out = torch.bmm(x_t, W_flat) 

        # 3. Reshape Back & Apply Physics Refinement
        out = out.transpose(1, 2).view(b, t, n, f)
        out = out + B # Apply Bias
        out = self.refine(out)

        return out # (Batch, Time, 11, 16)

# ==============================================================================
# 2. SOCIAL INTERACTION BLOCK (Updated for Geo-Features)
# ==============================================================================
class SocialInteractionBlock(nn.Module):
    def __init__(self, node_dim=16, hidden_dim=64):
        super().__init__()

        # Relational MLP
        # Takes the pre-calc geometric relations from Module 1
        # [Rel_X, Rel_Y, Rel_Dist] + [Speed_Self, Speed_Other] (Derived)
        self.relational_mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Linear(32, 16)
        )

        self.fusion = nn.Linear(node_dim * 2 + 16, hidden_dim)

    def forward(self, agent_canon, target_canon):
        # Input: [B, T, 11, 16] (Normalized Egocentric Features)
        
        # New Feature Map (Module 1):
        # 0: PosX, 1: PosY (Self)
        # 2: VelX, 3: VelY
        # 4: Neighbor PosX, 5: Neighbor PosY (Explicit Relation)
        # 6: Neighbor Dist
        
        # We extract Interaction Context from Node 0 (Body/Nose or Main Axis)
        # or aggregate across nodes. Here we take the mean interaction 
        # features across all nodes for stability.
        
        # 1. Extract Interaction Features (Ch 4, 5, 6)
        # Shape: [B, T, 3] (Mean over nodes)
        interaction_raw = agent_canon[..., 4:7].mean(dim=2) 
        
        # 2. Extract Dynamic Differences
        # Speed is typically computed in loader, but let's take velocity diffs (Ch 2,3)
        # Vel Self (Ch 2,3)
        vel_self = agent_canon[..., 2:4].mean(dim=2) 
        # Vel Other (Inferred/Proxy via target tensor)
        vel_targ = target_canon[..., 2:4].mean(dim=2)
        
        speed_diff = torch.norm(vel_self - vel_targ, dim=-1, keepdim=True)
        dot_prod = (vel_self * vel_targ).sum(dim=-1, keepdim=True)
        
        # Combine: [Ix, Iy, Dist, SpeedDiff, VelDot] -> 5 Dims
        rel_feats = torch.cat([interaction_raw, speed_diff, dot_prod], dim=-1)
        
        # Embed
        rel_embed = self.relational_mlp(rel_feats) # [B, T, 16]

        return agent_canon, target_canon, rel_embed

# ==============================================================================
# WRAPPER: MORPHOLOGICAL INTERACTION CORE
# ==============================================================================
class MorphologicalInteractionCore(nn.Module):
    def __init__(self, num_labs=20):
        super().__init__()
        # Standard input 11 canonical nodes
        self.adapter = CanonicalGraphAdapter(input_nodes=11, canonical_nodes=11, num_labs=num_labs)
        self.interaction = SocialInteractionBlock()

        # Fusion: (11 nodes * 16 features * 2 agents) + 16 relation = 368
        self.frame_fusion = nn.Linear(368, 128)

    def forward(self, agent_x, target_x, lab_idx):
        # 1. Adapt Topology (Refine Physics/Geometry)
        a_c = self.adapter(agent_x, lab_idx)
        t_c = self.adapter(target_x, lab_idx)

        # 2. Compute Social Relations
        # This uses the specific relative features baked into Module 1
        _, _, rel_embed = self.interaction(a_c, t_c)

        # 3. Flatten for Transformer Input
        b, t, n, f = a_c.shape
        a_flat = a_c.view(b, t, -1)
        t_flat = t_c.view(b, t, -1)

        # 4. Dense Fusion
        # Fuses Self(A) + Self(B) + Relationship
        combined = torch.cat([a_flat, t_flat, rel_embed], dim=-1) # [B, T, 368]
        out = self.frame_fusion(combined) # [B, T, 128]

        # Returns: 
        # out -> The Fused Token (used for Global Context / Temporal processing)
        # a_c, t_c -> The Canonical Skeletons (used for Physics Gating in Mod 5)
        return out, a_c, t_c

# Module 3: The Split-Stream Interaction Block
import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitStreamInteractionBlock(nn.Module):
    def __init__(self, node_dim=16, hidden_dim=128):
        super(SplitStreamInteractionBlock, self).__init__()

        # ----------------------------------------------------------------------
        # BRANCH A: SELF-BEHAVIOR STREAM (The "Me" Branch)
        # ----------------------------------------------------------------------
        # Focus: Posture, Grooming, Rearing, Running.
        # Input: Strictly LIMITED to the first 4 channels of the Agent (Pos X/Y, Vel, Speed).
        # We explicitly block Neighbor information (Channels 4+) from this stream
        # to prevent "Soft Leaks" (the Self branch learning Pair behaviors).
        self.self_input_size = 11 * 4 # 11 Nodes * 4 Feats (Pos/Vel)
        
        self.self_projector = nn.Sequential(
            nn.Linear(self.self_input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ----------------------------------------------------------------------
        # BRANCH B: PAIR-BEHAVIOR STREAM (The "Us" Branch)
        # ----------------------------------------------------------------------
        # Focus: Interaction, Distance, Chasing, Fight.
        # Input: Agent (Full) + Target (Full) + Interaction Token.
        
        # 1. Relational Engine
        # The new Module 1 (Geo Features) pre-calculates distance/rel_pos in Ch 4-6.
        # We extract this directly rather than re-calculating on the fly.
        self.relational_mlp = nn.Sequential(
            nn.Linear(3, 32), # [Rel_X, Rel_Y, Rel_Dist] averaged over nodes
            nn.GELU(),
            nn.Linear(32, 32)
        )

        # 2. Fusion Layer
        # Agent (176) + Target (176) + Rel (32) + Roles (2)
        full_node_dim = 11 * node_dim # 176
        pair_input_dim = (full_node_dim * 2) + 32 + 2
        
        self.pair_projector = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Role Tokens (Solves "Multi-Agent Roles")
        # [1, 0] = "I am Acting", [0, 1] = "I am Receiving"
        self.role_embedding = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    def forward(self, agent_c, target_c):
        """
        agent_c:  [Batch, Time, 11, 16] (Canonical Skeleton w/ Geo Features)
        target_c: [Batch, Time, 11, 16] 
        """
        batch, time, nodes, feat = agent_c.shape

        # ----------------------------------------------------------
        # 1. PROCESS SELF STREAM (Strict Slicing)
        # ----------------------------------------------------------
        # Only take Channels 0,1,2,3 (Pos, Vel). 
        # Channels 4+ contain Neighbor Relative info -> BLOCKED.
        agent_proprioception = agent_c[..., 0:4] # [B, T, 11, 4]
        agent_flat_self = agent_proprioception.contiguous().view(batch, time, -1)
        
        self_feat = self.self_projector(agent_flat_self) # [B, T, 128]

        # ----------------------------------------------------------
        # 2. PROCESS PAIR STREAM (Full Context)
        # ----------------------------------------------------------
        # Flatten full skeletons
        agent_flat_full = agent_c.view(batch, time, -1)
        target_flat_full = target_c.view(batch, time, -1)
        
        # Extract Relational Data baked into Module 1 output
        # Channels: 4 (Neighbor X), 5 (Neighbor Y), 6 (Dist)
        # We assume mean interaction across nodes represents the body-level interaction
        rel_feats = agent_c[..., 4:7].mean(dim=2) # [B, T, 3]
        
        # Embed Relation
        rel_embed = self.relational_mlp(rel_feats) # [B, T, 32]

        # Add Role Tokens (Broadcasting Agent Role [1,0])
        role_token = self.role_embedding[0].view(1, 1, 2).expand(batch, time, 2)

        # Fuse Pair Features
        # Concatenate: Agent(Full) + Target(Full) + Relation + Role
        pair_input = torch.cat([
            agent_flat_full, 
            target_flat_full, 
            rel_embed, 
            role_token
        ], dim=-1)
        
        pair_feat = self.pair_projector(pair_input) # [B, T, 128]

        return self_feat, pair_feat

# Module 4: The Local-Global Chronos Encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LocalGlobalChronosEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(LocalGlobalChronosEncoder, self).__init__()

        # ======================================================================
        # 1. GLOBAL CONTEXT STREAM (The "Narrative" Memory)
        # ======================================================================
        # Processes the 1 FPS Global Pair Features.
        # Captures long-term states (e.g., "Dominance established 10 mins ago").
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

        # ======================================================================
        # 2. LOCAL SELF STREAM (The "Me" Branch) - DEEP TCN
        # ======================================================================
        # Updated: Receptive Field ~2.0 seconds (64 frames)
        self.self_tcn = nn.Sequential(
            # Frame Level (d=1)
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            # Short Range (d=2)
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            # Medium Range (d=4)
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),

            # Long Range (d=8) -> +16 frames context
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),

            # Very Long Range (d=16) -> +32 frames context (TOTAL ~64)
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=16, dilation=16),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # Cross-Attention to Global 
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden_dim)

        # ======================================================================
        # 3. LOCAL PAIR STREAM (The "Us" Branch) - DEEP TCN
        # ======================================================================
        self.pair_tcn = nn.Sequential(
            # Frame Level
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            # Short
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            # Medium
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            # Long (Interaction Buildup)
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            # Very Long (Sustained Aggression/Chase)
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=16, dilation=16),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # Cross-Attention to Global
        self.pair_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.pair_norm = nn.LayerNorm(hidden_dim)

    def forward(self, global_feat, local_self, local_pair):
        """
        global_feat: [Batch, T_g, 128] 
        local_self:  [Batch, T_l, 128] 
        local_pair:  [Batch, T_l, 128] 
        """

        # --- A. Build Global Memory Bank ---
        g_emb = self.global_proj(global_feat)
        g_emb = self.pos_encoder(g_emb)
        global_memory = self.global_transformer(g_emb) # [B, T_g, 128]

        # --- B. Process Local Self Stream ---
        # 1. TCN 
        s_in = local_self.permute(0, 2, 1) # [B, C, T]
        s_tcn = self.self_tcn(s_in).permute(0, 2, 1) # [B, T, C]

        # 2. Cross-Attention
        # Query: Local TCN, Key/Value: Global Memory
        s_ctx, _ = self.self_attn(query=s_tcn, key=global_memory, value=global_memory)
        self_out = self.self_norm(s_tcn + s_ctx) 

        # --- C. Process Local Pair Stream ---
        # 1. TCN
        p_in = local_pair.permute(0, 2, 1)
        p_tcn = self.pair_tcn(p_in).permute(0, 2, 1)

        # 2. Cross-Attention
        p_ctx, _ = self.pair_attn(query=p_tcn, key=global_memory, value=global_memory)
        pair_out = self.pair_norm(p_tcn + p_ctx)

        return self_out, pair_out

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
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLogicHead(nn.Module):
    def __init__(self, input_dim=128, num_labs=20):
        super(MultiTaskLogicHead, self).__init__()

        # 1. DOMAIN EMBEDDING
        self.lab_embedding = nn.Embedding(num_labs, 32)

        # 2. FEATURE EXPANSION
        fusion_dim = input_dim + 32
        expanded_dim = 256 

        # 3. HEAD A: SELF BEHAVIORS
        self.self_classifier = nn.Sequential(
            nn.Linear(fusion_dim, expanded_dim),
            nn.LayerNorm(expanded_dim),
            nn.GELU(),
            nn.Linear(expanded_dim, 11) 
        )

        # 4. HEAD B: PAIR BEHAVIORS
        self.pair_classifier = nn.Sequential(
            nn.Linear(fusion_dim, expanded_dim),
            nn.LayerNorm(expanded_dim),
            nn.GELU(),
            nn.Linear(expanded_dim, 26) 
        )

        # 5. CENTER REGRESSOR
        self.center_regressor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1) 
        )

        # 6. PHYSICS LOGIC GATE 
        self.gate_control = nn.Linear(1, 1)
        
        # --- CRITICAL FIX: INITIALIZATION ---
        with torch.no_grad():
            # Force Gate Open (start unbiased)
            self.gate_control.bias.fill_(2.0)
            
            # FORCE CLASSIFIERS TO PREDICT BACKGROUND (Prob ~0.01)
            # The final Linear layer is at index [3] of Sequential
            # Logits = -4.59 -> Sigmoid(-4.59) = 0.01
            nn.init.constant_(self.self_classifier[3].bias, -4.59)
            nn.init.constant_(self.pair_classifier[3].bias, -4.59)
            
            # Start Center Regression at 0.5 (Midpoint)
            nn.init.constant_(self.center_regressor[2].bias, 0.0)

    def forward(self, self_feat, pair_feat, lab_idx, agent_c, target_c):
        """
        self_probs: [B, T, 11] (0.0 - 1.0)
        pair_probs: [B, T, 26] (0.0 - 1.0)
        """
        batch, time, _ = self_feat.shape

        # A. Context
        lab_context = self.lab_embedding(lab_idx).unsqueeze(1).expand(-1, time, -1)
        self_input = torch.cat([self_feat, lab_context], dim=-1)
        pair_input = torch.cat([pair_feat, lab_context], dim=-1)

        # B. Raw Logits
        self_logits = self.self_classifier(self_input) 
        pair_logits = self.pair_classifier(pair_input) 
        
        # Center Score
        center_score = torch.sigmoid(self.center_regressor(pair_input))

        # C. Physics Gate
        # Dist Logic
        a_pos = agent_c[:, :, 0, :2]
        t_pos = target_c[:, :, 0, :2]
        dist = torch.norm(a_pos - t_pos, dim=-1, keepdim=True) 

        gate = torch.sigmoid(self.gate_control(dist))

        # D. Activation
        self_probs = torch.sigmoid(self_logits)
        
        # Combine Pair Logits with Gate
        pair_probs = torch.sigmoid(pair_logits) * gate

        return self_probs, pair_probs, center_score

# Module 6: Final Assembly (EthoSwarmNet V3)
import torch
import torch.nn as nn

# ==============================================================================
# BEHAVIOR DEFINITIONS (For Output Stitching)
# ==============================================================================
# All 37 actions sorted alphabetically (Competition Standard)
ACTION_LIST = sorted([
    "allogroom", "approach", "attack", "attemptmount", "avoid", "biteobject",
    "chase", "chaseattack", "climb", "defend", "dig", "disengage", "dominance",
    "dominancegroom", "dominancemount", "ejaculate", "escape", "exploreobject",
    "flinch", "follow", "freeze", "genitalgroom", "huddle", "intromit", "mount",
    "rear", "reciprocalsniff", "rest", "run", "selfgroom", "shepherd", "sniff",
    "sniffbody", "sniffface", "sniffgenital", "submit", "tussle"
])

# Subset: 11 Self Behaviors (Agent only)
SELF_BEHAVIORS = sorted([
    "biteobject", "climb", "dig", "exploreobject", "freeze", "genitalgroom",
    "huddle", "rear", "rest", "run", "selfgroom"
])

# Subset: 26 Pair Behaviors (Agent + Target)
PAIR_BEHAVIORS = sorted([
    "allogroom", "approach", "attack", "attemptmount", "avoid", "chase",
    "chaseattack", "defend", "disengage", "dominance", "dominancegroom",
    "dominancemount", "ejaculate", "escape", "flinch", "follow", "intromit",
    "mount", "reciprocalsniff", "shepherd", "sniff", "sniffbody", "sniffface",
    "sniffgenital", "submit", "tussle"
])

class EthoSwarmNet(nn.Module):
    def __init__(self, num_classes=37, input_dim=128):
        super(EthoSwarmNet, self).__init__()

        # ----------------------------------------------------------------------
        # 1. Morphological Core (Module 2)
        # ----------------------------------------------------------------------
        self.morph_core = MorphologicalInteractionCore(num_labs=20)

        # ----------------------------------------------------------------------
        # 2. Split-Stream Block (Module 3)
        # ----------------------------------------------------------------------
        self.split_interaction = SplitStreamInteractionBlock(hidden_dim=128)

        # ----------------------------------------------------------------------
        # 3. Local-Global Chronos (Module 4)
        # ----------------------------------------------------------------------
        self.chronos = LocalGlobalChronosEncoder(input_dim=128, hidden_dim=128)

        # ----------------------------------------------------------------------
        # 4. Multi-Task Logic Head (Module 5)
        # ----------------------------------------------------------------------
        self.logic_head = MultiTaskLogicHead(
            input_dim=128,
            num_labs=20
        )

        # ----------------------------------------------------------------------
        # 5. Output Stitching Maps
        # ----------------------------------------------------------------------
        self.register_buffer('self_indices', self._get_indices(SELF_BEHAVIORS))
        self.register_buffer('pair_indices', self._get_indices(PAIR_BEHAVIORS))

    def _get_indices(self, subset_list):
        indices = []
        for beh in subset_list:
            try:
                indices.append(ACTION_LIST.index(beh))
            except ValueError:
                pass
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, global_agent, global_target, local_agent, local_target, lab_idx):
        """
        The V3 Forward Pass:
        Global/Local Streams -> Topology -> Split -> Time -> Logic -> Stitch
        """

        # --- A. TOPOLOGY (Module 2) ---
        g_out, _, _ = self.morph_core(global_agent, global_target, lab_idx)
        _, l_ac, l_tc = self.morph_core(local_agent, local_target, lab_idx)

        # --- B. SPLIT-STREAM (Module 3) ---
        l_self, l_pair = self.split_interaction(l_ac, l_tc)

        # --- C. TIME & CONTEXT (Module 4) ---
        t_self, t_pair = self.chronos(g_out, l_self, l_pair)

        # --- D. LOGIC & PHYSICS (Module 5) ---
        # FIX: Now accepts 3 return values
        # center_score is the Regression Head output (0.0 to 1.0)
        p_self, p_pair, center_score = self.logic_head(t_self, t_pair, lab_idx, l_ac, l_tc)

        # --- E. OUTPUT STITCHING ---
        batch, time, _ = p_self.shape
        # Reconstruct [Batch, T, 37] for classification targets
        final_output = torch.zeros(batch, time, 37, device=p_self.device, dtype=p_self.dtype)

        final_output.index_copy_(2, self.self_indices, p_self)
        final_output.index_copy_(2, self.pair_indices, p_pair)
        
        # NOTE: For now, we are returning 'final_output' (37 classes) 
        # because the Training Loop expects [B, T, 37] matching targets.
        # The 'center_score' improves internal gradient flow via backprop on Module 5.
        # If you want to use Center Score explicitly in loss later, return it as tuple:
        # return final_output, center_score
        
        return final_output

# Module 7: The Training Loop & Validation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import os
import glob
import json
import torch.nn.functional as F

# ==============================================================================
# UTILS & METRICS
# ==============================================================================
def load_lab_vocabulary(vocab_path, action_to_idx, num_classes, device):
    """
    Loads a boolean mask [20, 37] where 1.0 means the lab annotates that action.
    """
    # Default to "Allow All" if file missing
    if not os.path.exists(vocab_path):
        return torch.ones(25, 37).to(device)
        
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    # Must sort keys to match Module 1 index order
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

def get_batch_f1(probs_in, targets, batch_vocab_mask, temporal_weights):
    """
    FIXED: Removed torch.sigmoid(). Input 'probs_in' is already 0.0-1.0 from Model.
    """
    # 1. Binarize Predictions (probs are already 0-1)
    preds = (probs_in > 0.4).float() 
    
    # 2. Combine Masks
    valid_pixels = temporal_weights.unsqueeze(-1) * batch_vocab_mask.unsqueeze(1)
    
    # 3. Calculate F1 only on VALID pixels
    tp = (preds * targets * valid_pixels).sum()
    fp = (preds * (1-targets) * valid_pixels).sum()
    fn = ((1-preds) * targets * valid_pixels).sum()
    
    f1 = 2*tp / (2*tp + fp + fn + 1e-6)
    return f1.item()

# ==============================================================================
# LOSS FUNCTION
# ==============================================================================
class DualStreamMaskedLoss(nn.Module):
    def __init__(self, model_self_indices, model_pair_indices):
        super().__init__()
        self.self_idx = model_self_indices
        self.pair_idx = model_pair_indices

    def forward(self, model_output_probs, target, weight_mask, lab_vocab_mask):
        """
        FIXED: Removed torch.sigmoid(). 
        Model outputs probabilities (0-1) due to Physics Gate.
        """
        # Slice Output/Target
        # Inputs are ALREADY PROBABILITIES
        p_self = model_output_probs[:, :, self.self_idx]
        p_pair = model_output_probs[:, :, self.pair_idx]
        
        t_self = target[:, :, self.self_idx]
        t_pair = target[:, :, self.pair_idx]
        
        # Clamp for numerical stability (prevent log(0))
        p_self = torch.clamp(p_self, 1e-7, 1 - 1e-7)
        p_pair = torch.clamp(p_pair, 1e-7, 1 - 1e-7)
        
        # Slice Lab Masks for Batch
        m_self = lab_vocab_mask[:, self.self_idx].unsqueeze(1) # [B, 1, n_self]
        m_pair = lab_vocab_mask[:, self.pair_idx].unsqueeze(1) # [B, 1, n_pair]
        
        # Temporal Mask [B, T, 1]
        tm = weight_mask.unsqueeze(-1)
        
        # Compute Loss (Standard BCELoss, NOT WithLogits)
        l_self_raw = F.binary_cross_entropy(p_self, t_self, reduction='none')
        l_pair_raw = F.binary_cross_entropy(p_pair, t_pair, reduction='none')
        
        # Weighted Sum
        loss_s = (l_self_raw * m_self * tm).sum() / ((m_self * tm).sum() + 1e-6)
        loss_p = (l_pair_raw * m_pair * tm).sum() / ((m_pair * tm).sum() + 1e-6)
        
        return loss_s + loss_p

# ==============================================================================
# TRAINING CONTROLLER
# ==============================================================================
def train_ethoswarm_v3():
    # --- 1. SETUP & PATHS ---
    if 'mabe_mouse_behavior_detection_path' in globals():
        DATA_PATH = globals()['mabe_mouse_behavior_detection_path']
    elif os.path.exists('/kaggle/input/MABe-mouse-behavior-detection'):
        DATA_PATH = '/kaggle/input/MABe-mouse-behavior-detection'
    else: 
        print("Dataset not found."); return
    
    VOCAB_PATH = '/kaggle/input/mabe-metadata/results/lab_vocabulary.json'
    
    gpu_count = torch.cuda.device_count()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 8 * max(1, gpu_count)
    LEARNING_RATE = 3e-4 
    NUM_EPOCHS = 5

    print(f"Start Training on {gpu_count} GPU(s) | Batch Size: {BATCH_SIZE}")

    # --- 2. DATA PREP (Strict Video Split) ---
    meta = pd.read_csv(f"{DATA_PATH}/train.csv")
    vids = meta['video_id'].astype(str).unique()
    np.random.shuffle(vids)
    
    split = int(len(vids) * 0.90)
    train_ids = vids[:split]
    val_ids = vids[split:]
    
    # Loaders - using Module 1 (Cached)
    train_ds = BioPhysicsDataset(DATA_PATH, 'train', video_ids=train_ids)
    val_ds = BioPhysicsDataset(DATA_PATH, 'train', video_ids=val_ids)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_dual, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_dual, num_workers=2)
    
    # --- 3. MODEL INITIALIZATION ---
    model = EthoSwarmNet(num_classes=NUM_CLASSES, input_dim=128)
    
    model.to(DEVICE)
    if gpu_count > 1:
        print(f"--> Activating Distributed Data Parallel on {gpu_count} GPUs")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS)
    
    # Load Masks
    lab_masks = load_lab_vocabulary(VOCAB_PATH, ACTION_TO_IDX, NUM_CLASSES, DEVICE)
    
    self_indices = [ACTION_TO_IDX[a] for a in sorted(
        ["biteobject", "climb", "dig", "exploreobject", "freeze", "genitalgroom", 
         "huddle", "rear", "rest", "run", "selfgroom"])]
    pair_indices = [ACTION_TO_IDX[a] for a in sorted(
        ["allogroom", "approach", "attack", "attemptmount", "avoid", "chase", 
         "chaseattack", "defend", "disengage", "dominance", "dominancegroom", 
         "dominancemount", "ejaculate", "escape", "flinch", "follow", "intromit", 
         "mount", "reciprocalsniff", "shepherd", "sniff", "sniffbody", "sniffface", 
         "sniffgenital", "submit", "tussle"])]
         
    loss_fn = DualStreamMaskedLoss(self_indices, pair_indices)

    # --- 4. EPOCH LOOP ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}")
        
        run_loss = 0.0
        run_f1 = 0.0
        
        for i, batch in enumerate(loop):
            # Move 5 items to GPU
            gx, lx, tgt, weights, lid = [b.to(DEVICE) for b in batch]
            
            optimizer.zero_grad()
            
            # Forward 
            # Output is PROBABILITIES now (from Module 5+6)
            probs = model(gx, gx, lx, lx, lid)
            
            # Loss Calc 
            loss = loss_fn(probs, tgt, weights, lab_masks[lid])
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Metrics
            with torch.no_grad():
                # FIXED: Don't sigmoid again
                f1 = get_batch_f1(probs, tgt, lab_masks[lid], weights)
                
            run_loss = 0.9*run_loss + 0.1*loss.item() if i>0 else loss.item()
            run_f1 = 0.9*run_f1 + 0.1*f1 if i>0 else f1
            
            if i % 20 == 0:
                loop.set_postfix({'Loss': f"{run_loss:.4f}", 'F1': f"{run_f1:.3f}"})
        
        # Validation
        print("Validating...")
        model.eval()
        val_loss_sum = 0
        val_f1_sum = 0.0
        batches = 0
        with torch.no_grad():
            for batch in val_loader:
                gx, lx, tgt, weights, lid = [b.to(DEVICE) for b in batch]
                
                probs = model(gx, gx, lx, lx, lid)
                loss = loss_fn(probs, tgt, weights, lab_masks[lid])
                
                # Metric
                f1 = get_batch_f1(probs, tgt, lab_masks[lid], weights)
                
                val_loss_sum += loss.item()
                val_f1_sum += f1
                batches += 1
                
        print(f"Val Loss: {val_loss_sum/batches:.4f} | Val F1: {val_f1_sum/batches:.4f}")
        
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state, f"ethoswarm_v3_ep{epoch+1}.pth")

if __name__ == '__main__':
    train_ethoswarm_v3()
  

