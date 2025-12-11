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
import pyarrow.parquet as pq

# --- KAGGLE METRIC CODE (INTEGRATED) ---
class HostVisibleError(Exception): pass

def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
    label_frames: defaultdict[str, set[int]] = defaultdict(set)
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set)

    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

    for video in lab_solution['video_id'].unique():
        if video not in lab_submission['video_id']:
             continue

        active_labels_str = lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()
        if active_labels_str is None:
            active_labels = set()
        else:
            try:
                active_labels = set(json.loads(active_labels_str))
            except:
                active_labels = set()

        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set)

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts():
            if ','.join([str(row['agent_id']), str(row['target_id']), row['action']]) not in active_labels:
                continue

            new_frames = set(range(row['start_frame'], row['stop_frame']))
            if row['prediction_key'] in prediction_frames:
                new_frames = new_frames.difference(prediction_frames[row['prediction_key']])

            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])

            prediction_frames[row['prediction_key']].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    tps = defaultdict(int)
    fns = defaultdict(int)
    fps = defaultdict(int)

    for key, pred_frames in prediction_frames.items():
        action = key.split('_')[-1]
        matched_label_frames = label_frames[key]
        tps[action] += len(pred_frames.intersection(matched_label_frames))
        fns[action] += len(matched_label_frames.difference(pred_frames))
        fps[action] += len(pred_frames.difference(matched_label_frames))

    distinct_actions = set()
    for key, frames in label_frames.items():
        action = key.split('_')[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(frames)

    action_f1s = []
    for action in distinct_actions:
        if tps[action] + fns[action] + fps[action] == 0:
            action_f1s.append(0)
        else:
            action_f1s.append((1 + beta**2) * tps[action] / ((1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action]))

    if len(action_f1s) == 0: return 0.0
    return sum(action_f1s) / len(action_f1s)

def mouse_fbeta(solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1) -> float:
    if len(solution) == 0: return 0.0
    if len(submission) == 0: return 0.0

    expected_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
    for col in expected_cols:
        if col not in solution.columns or col not in submission.columns:
            return 0.0

    solution_pl = pl.DataFrame(solution)
    submission_pl = pl.DataFrame(submission)

    solution_videos = set(solution_pl['video_id'].unique())
    submission_pl = submission_pl.filter(pl.col('video_id').is_in(solution_videos))

    solution_pl = solution_pl.with_columns(
        pl.concat_str([pl.col('video_id').cast(pl.Utf8), pl.col('agent_id').cast(pl.Utf8), pl.col('target_id').cast(pl.Utf8), pl.col('action')], separator='_').alias('label_key'),
    )
    submission_pl = submission_pl.with_columns(
        pl.concat_str([pl.col('video_id').cast(pl.Utf8), pl.col('agent_id').cast(pl.Utf8), pl.col('target_id').cast(pl.Utf8), pl.col('action')], separator='_').alias('prediction_key'),
    )

    lab_scores = []
    if 'lab_id' not in solution_pl.columns:
         lab_scores.append(single_lab_f1(solution_pl, submission_pl, beta=beta))
    else:
        for lab in solution_pl['lab_id'].unique():
            lab_solution = solution_pl.filter(pl.col('lab_id') == lab).clone()
            lab_videos = set(lab_solution['video_id'].unique())
            lab_submission = submission_pl.filter(pl.col('video_id').is_in(lab_videos)).clone()
            lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    if len(lab_scores) == 0: return 0.0
    return sum(lab_scores) / len(lab_scores)

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
        
        # Verify Paths
        if not self.tracking_dir.exists():
            print(f"WARNING: Tracking directory not found at {self.tracking_dir}")
        if not self.annot_dir.exists():
            print(f"WARNING: Annotation directory not found at {self.annot_dir}")

        self.metadata = pd.read_csv(self.root / f"{mode}.csv")
        if video_ids is not None:
            self.metadata = self.metadata[self.metadata['video_id'].astype(str).isin(video_ids)]
        
        # --- PAIR PERMUTATION LOGIC ---
        self.samples = []
        print(f"Scanning {len(self.metadata)} videos for mouse pairs...")

        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            vid = str(row['video_id'])
            lab = row['lab_id']
            
            # Peek at parquet columns to find mice
            fpath = self.tracking_dir / lab / f"{vid}.parquet"
            mice = []
            if fpath.exists():
                try:
                    # Optimized: Just read 'mouse_id' column unique values
                    df_small = pd.read_parquet(fpath, columns=['mouse_id'])
                    mice = df_small['mouse_id'].unique().tolist()
                except:
                    mice = ['mouse1', 'mouse2'] # Fallback
            else:
                mice = []

            # Create permutations
            # Extract behavior labels if available (default to empty string if missing)
            b_label = row['behaviors_labeled'] if 'behaviors_labeled' in row else "[]"

            for agent in mice:
                for target in mice:
                    if agent != target:
                        self.samples.append({
                            'video_id': vid,
                            'lab_id': lab,
                            'agent_id': str(agent),  # FORCE STRING
                            'target_id': str(target), # FORCE STRING
                            'behaviors_labeled': b_label
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

        self.local_window = 256
        self.action_windows = []

        # Debug Counters
        self.print_limit = 10
        self.print_count = 0

        if self.mode == 'train':
            self._scan_actions_safe()

    def _scan_actions_safe(self):
        count = 0
        print("Scanning subset of annotations for sampling...")
        # Reduce scan size for speed
        for i, s in enumerate(self.samples):
            # if i > 200: break # REMOVED to scan all
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

        # 4. Velocity
        vel = np.diff(centered, axis=0, prepend=centered[0:1])
        speed = np.sqrt((vel**2).sum(axis=-1))
        
        # 5. Relation
        dist = np.sqrt(((centered - other_centered)**2).sum(axis=-1))
        
        # Pack
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
        Returns: (feats, lab_idx, agent_id, target_id, frames, valid_mask)
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
                 return None, None, None, None, None, None

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
                return None, None, None, None, None, None

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

            return torch.tensor(feats), lab_idx, agent_id, target_id, frames, valid_frames

        except Exception as e:
            print(f"Error loading {vid}: {e}")
            return None, None, None, None, None, None

    def _load(self, idx, center=None):
        sample = self.samples[idx]
        lab = sample['lab_id']
        conf = LAB_CONFIGS.get(lab, LAB_CONFIGS['DEFAULT'])
        
        agent_id = sample['agent_id']
        target_id = sample['target_id']
        vid = sample['video_id']
        b_label = sample['behaviors_labeled']

        fpath = self.tracking_dir / lab / f"{vid}.parquet"

        data_loaded = False
        frame_start = 0
        frame_end = 0
        debug_msg = ""

        raw_m1 = None
        raw_m2 = None

        # Targets
        target = torch.zeros((self.local_window, NUM_CLASSES), dtype=torch.float32)
        weights = torch.zeros(self.local_window, dtype=torch.float32)
        centerness = torch.zeros((self.local_window, 1), dtype=torch.float32)

        if fpath.exists():
            try:
                # OPTIMIZATION: Use Polars for lazy scanning and filtering
                lf = pl.scan_parquet(fpath)
                
                # 1. Get Limits (lazy)
                # Fetch max frame to determine L
                meta_df = lf.select(pl.col('video_frame').max()).collect()
                if meta_df.shape[0] > 0 and meta_df.item(0, 0) is not None:
                    max_frame = meta_df.item(0, 0)
                    L = max_frame + 1

                    # 2. Pick Window
                    if center is None:
                        # Smart sampling: Pick a frame where the mice are actually present
                        # This prevents empty window queries during validation/random sampling
                        try:
                            valid_frames_sample = (
                                lf
                                .filter(pl.col('mouse_id').cast(pl.Utf8).is_in([str(agent_id), str(target_id)]))
                                .select('video_frame')
                                .head(1000) # Sample a subset to avoid full scan
                                .collect()
                            )
                            if valid_frames_sample.shape[0] > 0:
                                # Pick random valid frame from the sample
                                ridx = random.randint(0, valid_frames_sample.shape[0] - 1)
                                center = valid_frames_sample.item(ridx, 0)
                            else:
                                center = random.randint(0, L)
                        except:
                            center = random.randint(0, L)

                    s = max(0, min(center - self.local_window//2, L - self.local_window))
                    e = min(s + self.local_window, L)

                    frame_start = s
                    frame_end = e

                    # 3. Filter and Fetch Window (Lazy evaluation pushes down filters)
                    # Cast mouse_id to string for robustness
                    q = (
                        lf
                        .filter(
                            (pl.col('video_frame') >= s) &
                            (pl.col('video_frame') < e) &
                            (pl.col('mouse_id').cast(pl.Utf8).is_in([str(agent_id), str(target_id)]))
                        )
                        .collect()
                    )

                    # 4. Process into Buffers
                    # If empty, data_loaded remains False
                    if not q.is_empty():
                        # Convert to pandas for easier numpy handling or handle directly
                        df = q.to_pandas()

                        # Prepare buffers - size is window length (e-s)
                        # We map video_frame to buffer index: idx = video_frame - s
                        win_len = e - s
                        raw_m1 = np.zeros((win_len, 11, 2), dtype=np.float32)
                        raw_m2 = np.zeros((win_len, 11, 2), dtype=np.float32)

                        # Mouse 1
                        d1 = df[df['mouse_id'].astype(str) == str(agent_id)]
                        for i, bp in enumerate(BODY_PARTS):
                            rows = d1[d1['bodypart']==bp]
                            if not rows.empty:
                                indices = rows['video_frame'].values - s
                                # Safety bounds
                                valid = (indices >= 0) & (indices < win_len)
                                raw_m1[indices[valid], i] = rows[['x', 'y']].values[valid]

                        # Mouse 2
                        d2 = df[df['mouse_id'].astype(str) == str(target_id)]
                        for i, bp in enumerate(BODY_PARTS):
                            rows = d2[d2['bodypart']==bp]
                            if not rows.empty:
                                indices = rows['video_frame'].values - s
                                valid = (indices >= 0) & (indices < win_len)
                                raw_m2[indices[valid], i] = rows[['x', 'y']].values[valid]

                        data_loaded = True
                    else:
                        debug_msg = f"Empty window query. Vid: {vid}, Range: {s}-{e}"
                else:
                    debug_msg = "Empty video_frame column or file."
            except Exception as e:
                debug_msg = f"Load Error: {e}"
                # import traceback
                # traceback.print_exc()
        else:
            debug_msg = f"File not found: {fpath}"

        # Fallback / Padding
        if raw_m1 is None:
            raw_m1 = np.zeros((self.local_window, 11, 2), dtype=np.float32)
            raw_m2 = np.zeros((self.local_window, 11, 2), dtype=np.float32)

        if len(raw_m1) < self.local_window:
            pad_len = self.local_window - len(raw_m1)
            pad_arr = np.zeros((pad_len, 11, 2), dtype=np.float32)
            raw_m1 = np.concatenate([raw_m1, pad_arr], axis=0)
            raw_m2 = np.concatenate([raw_m2, pad_arr], axis=0)

        # 1. Teleport Fix
        raw_m1 = self._fix_teleport(raw_m1)
        raw_m2 = self._fix_teleport(raw_m2)
        
        # 3. Features
        feats = self._geo_feats(raw_m1, raw_m2, conf['pix_cm'])
        
        # 4. Targets Setup
        if data_loaded:
             valid_len = frame_end - frame_start
             if valid_len > 0:
                 weights[:valid_len] = 1.0

        if self.mode == 'train':
            ap = self.annot_dir / lab / f"{sample['video_id']}.parquet"
            if ap.exists():
                try:
                    adf = pd.read_parquet(ap)
                    if 'agent' in adf.columns and 'target' in adf.columns:
                         # Filter
                         adf = adf[(adf['agent'] == agent_id) & (adf['target'] == target_id)]

                    for _, row in adf.iterrows():
                        if row['action'] in ACTION_TO_IDX:
                            st, et = int(row['start_frame'])-frame_start, int(row['stop_frame'])-frame_start
                            st, et = max(0, st), min(self.local_window, et)
                            if st < et:
                                target[st:et, ACTION_TO_IDX[row['action']]] = 1.0

                                # Centerness Target (Gaussian)
                                c_local = (st + et) / 2.0
                                width = et - st
                                t_grid = torch.arange(st, et, dtype=torch.float32)
                                sigma = width / 6.0 + 1e-6
                                g = torch.exp( - (t_grid - c_local)**2 / (2 * sigma**2) )
                                centerness[st:et, 0] = torch.max(centerness[st:et, 0], g)

                except: pass
        
        lab_idx = list(LAB_CONFIGS.keys()).index(lab) if lab in LAB_CONFIGS else 0

        # Debug Print
        if not data_loaded and self.print_count < self.print_limit:
             print(f"[DEBUG] Load Failed for {vid}: {debug_msg}")
             self.print_count += 1

        # DEBUG: Check signal
        if data_loaded and self.print_count < self.print_limit and weights.sum() == 0:
             print(f"[DEBUG] Weights are ZERO despite DataLoaded! Vid: {vid}, Frame range: {frame_start}-{frame_end}")
             self.print_count += 1

        # Pack Metadata
        meta_info = {
            'video_id': vid,
            'agent_id': agent_id,
            'target_id': target_id,
            'start_frame': frame_start,
            'lab_id': lab,
            'behaviors_labeled': b_label
        }

        return torch.tensor(feats), torch.tensor(feats), target, weights, lab_idx, centerness, meta_info

    def __getitem__(self, idx):
        # Force usage of action windows if available to avoid empty data
        if len(self.action_windows) > 0:
            i, c = self.action_windows[random.randint(0, len(self.action_windows)-1)]
            return self._load(i, c)
        return self._load(idx)
    
    def __len__(self): return len(self.samples)

class ValidationDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        return self.ds.load_full_video_features_for_pair(idx)

def pad_collate_dual(batch):
    gx, lx, t, w, lid, center, meta = zip(*batch)
    return torch.stack(gx), torch.stack(lx), torch.stack(t), torch.stack(w), torch.tensor(lid), torch.stack(center), meta

# Module 2: The Morphological & Interaction Core.

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
# 2. SOCIAL INTERACTION BLOCK (Upgraded to Graph Attention)
# ==============================================================================
class SocialInteractionBlock(nn.Module):
    def __init__(self, node_dim=16, hidden_dim=64):
        super().__init__()

        # Graph Attention Network (GAT)
        # Queries: Agent Nodes, Keys/Values: Target Nodes
        self.attention = nn.MultiheadAttention(embed_dim=node_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(node_dim)

        # Relational MLP (Post-Attention)
        self.relational_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32)
        )

    def forward(self, agent_canon, target_canon):
        # Input: [B, T, 11, 16] (Normalized Egocentric Features)
        b, t, n, f = agent_canon.shape
        
        # Flatten Time for Batch processing
        a_flat = agent_canon.view(b*t, n, f)
        t_flat = target_canon.view(b*t, n, f)
        
        # Attention: How much does each part of 'Agent' care about 'Target' parts?
        attn_out, _ = self.attention(query=a_flat, key=t_flat, value=t_flat)
        
        # Residual + Norm
        # This creates "Contextualized Agent" features enriched with Target info
        interact_ctx = self.norm(a_flat + attn_out) # [B*T, 11, 16]
        
        # Aggregate Interaction (Pool over nodes)
        # We concat Original + Interaction Context
        combined = torch.cat([a_flat, interact_ctx], dim=-1) # [B*T, 11, 32]
        interact_summ = combined.mean(dim=1) # [B*T, 32]
        
        # Reshape back to time
        interact_summ = interact_summ.view(b, t, -1) # [B, T, 32]
        
        # Final Embedding
        rel_embed = self.relational_mlp(interact_summ) # [B, T, 32]

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

        # Fusion: (11 nodes * 16 features * 2 agents) + 32 relation = 384
        self.frame_fusion = nn.Linear(384, 128)

    def forward(self, agent_x, target_x, lab_idx):
        # 1. Adapt Topology (Refine Physics/Geometry)
        a_c = self.adapter(agent_x, lab_idx)
        t_c = self.adapter(target_x, lab_idx)

        # 2. Compute Social Relations (Graph Attention)
        _, _, rel_embed = self.interaction(a_c, t_c)

        # 3. Flatten for Transformer Input
        b, t, n, f = a_c.shape
        a_flat = a_c.view(b, t, -1)
        t_flat = t_c.view(b, t, -1)

        # 4. Dense Fusion
        # Fuses Self(A) + Self(B) + Relationship
        combined = torch.cat([a_flat, t_flat, rel_embed], dim=-1) # [B, T, 384]
        out = self.frame_fusion(combined) # [B, T, 128]

        # Returns: 
        # out -> The Fused Token (used for Global Context / Temporal processing)
        # a_c, t_c -> The Canonical Skeletons (used for Physics Gating in Mod 5)
        return out, a_c, t_c

# Module 3: The Split-Stream Interaction Block

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
        
        # 1. Relational Engine (Simple Distance/Speed for Pair stream fallback)
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

    def forward(self, agent_c, target_c, role_indices=None):
        """
        agent_c:  [Batch, Time, 11, 16] (Canonical Skeleton w/ Geo Features)
        target_c: [Batch, Time, 11, 16] 
        role_indices: [Batch] Tensor of 0 or 1. If None, defaults to 0 (Agent).
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
        rel_feats = agent_c[..., 4:7].mean(dim=2) # [B, T, 3]
        
        # Embed Relation
        rel_embed = self.relational_mlp(rel_feats) # [B, T, 32]

        # Select Role Tokens
        # If role_indices is provided, use it. Else default to Agent (0)
        if role_indices is None:
            # Default: All are Agents [1,0]
            selected_role = self.role_embedding[0].view(1, 1, 2).expand(batch, time, 2)
        else:
            # Gather based on index [B]
            # self.role_embedding is [2, 2]
            # we want [B, 2] -> expand to [B, T, 2]
            selected_role = self.role_embedding[role_indices].unsqueeze(1).expand(batch, time, 2)

        # Fuse Pair Features
        # Concatenate: Agent(Full) + Target(Full) + Relation + Role
        pair_input = torch.cat([
            agent_flat_full, 
            target_flat_full, 
            rel_embed, 
            selected_role
        ], dim=-1)
        
        pair_feat = self.pair_projector(pair_input) # [B, T, 128]

        return self_feat, pair_feat

# Module 4: The Local-Global Chronos Encoder

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

        # AUXILIARY HEAD FOR GLOBAL STREAM
        self.global_classifier = nn.Linear(hidden_dim, 37)

        # ======================================================================
        # 2. LOCAL SELF STREAM (The "Me" Branch) - DEEP TCN + ATTENTION
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
        )
        
        # New: Bidirectional Transformer for Local Temporal Mixing
        self.self_local_attn = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)

        # Cross-Attention to Global 
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden_dim)

        # ======================================================================
        # 3. LOCAL PAIR STREAM (The "Us" Branch) - DEEP TCN + ATTENTION
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
        )

        # New: Bidirectional Transformer for Local Temporal Mixing
        self.pair_local_attn = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        
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

        # AUX OUTPUT
        global_logits = self.global_classifier(global_memory)

        # --- B. Process Local Self Stream ---
        # 1. TCN 
        s_in = local_self.permute(0, 2, 1) # [B, C, T]
        s_tcn = self.self_tcn(s_in).permute(0, 2, 1) # [B, T, C]

        # 2. Local Transformer (Bidirectional)
        s_tcn = self.self_local_attn(s_tcn)

        # 3. Cross-Attention
        # Query: Local TCN, Key/Value: Global Memory
        s_ctx, _ = self.self_attn(query=s_tcn, key=global_memory, value=global_memory)
        self_out = self.self_norm(s_tcn + s_ctx) 

        # --- C. Process Local Pair Stream ---
        # 1. TCN
        p_in = local_pair.permute(0, 2, 1)
        p_tcn = self.pair_tcn(p_in).permute(0, 2, 1)

        # 2. Local Transformer (Bidirectional)
        p_tcn = self.pair_local_attn(p_tcn)

        # 3. Cross-Attention
        p_ctx, _ = self.pair_attn(query=p_tcn, key=global_memory, value=global_memory)
        pair_out = self.pair_norm(p_tcn + p_ctx)

        return self_out, pair_out, global_logits

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
        
        # Center Score (Unactivated Logits? Or Sigmoid? Usually Sigmoid for 0-1)
        # We trained with MSE against 0-1, so Sigmoid is correct.
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

# Module 6: Final Assembly (EthoSwarmNet V4 - Enhanced)

# BEHAVIOR DEFINITIONS
SELF_BEHAVIORS = sorted(["biteobject", "climb", "dig", "exploreobject", "freeze", "genitalgroom", "huddle", "rear", "rest", "run", "selfgroom"])
PAIR_BEHAVIORS = sorted(["allogroom", "approach", "attack", "attemptmount", "avoid", "chase", "chaseattack", "defend", "disengage", "dominance", "dominancegroom", "dominancemount", "ejaculate", "escape", "flinch", "follow", "intromit", "mount", "reciprocalsniff", "shepherd", "sniff", "sniffbody", "sniffface", "sniffgenital", "submit", "tussle"])

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

    def forward(self, global_agent, global_target, local_agent, local_target, lab_idx, role_idx=None):
        """
        The V4 Forward Pass:
        Global/Local Streams -> Topology -> Split -> Time -> Logic -> Stitch
        """
        # Safety: NaNs
        # It's possible for inputs to be all-zeros (missing data) which is fine, but if NaNs appear we crash.
        # We can clip or fill NaNs.
        # Check if NaNs exist? Only if training is unstable.
        # Ideally, we trust the data loader to not produce NaNs.
        # The CUDA error might be due to very large gradients or weird values.

        # --- A. TOPOLOGY (Module 2) ---
        g_out, _, _ = self.morph_core(global_agent, global_target, lab_idx)
        _, l_ac, l_tc = self.morph_core(local_agent, local_target, lab_idx)

        # --- B. SPLIT-STREAM (Module 3) ---
        # Pass Role Index here
        l_self, l_pair = self.split_interaction(l_ac, l_tc, role_indices=role_idx)

        # --- C. TIME & CONTEXT (Module 4) ---
        # Returns global_logits now too
        t_self, t_pair, g_logits = self.chronos(g_out, l_self, l_pair)

        # --- D. LOGIC & PHYSICS (Module 5) ---
        # center_score is the Regression Head output (0.0 to 1.0)
        p_self, p_pair, center_score = self.logic_head(t_self, t_pair, lab_idx, l_ac, l_tc)

        # --- E. OUTPUT STITCHING ---
        batch, time, _ = p_self.shape
        # Reconstruct [Batch, T, 37] for classification targets
        final_output = torch.zeros(batch, time, 37, device=p_self.device, dtype=p_self.dtype)

        final_output.index_copy_(2, self.self_indices, p_self)
        final_output.index_copy_(2, self.pair_indices, p_pair)
        
        # Return all required outputs for loss
        # final_output: [B, T, 37]
        # center_score: [B, T, 1]
        # g_logits: [B, T_g, 37] (Needs resizing or matching?)
        # g_logits matches global input length. We can supervise it if we pool target?
        # Or interpolate g_logits to T?
        # Let's interpolate g_logits to T for simple supervision

        g_logits_up = F.interpolate(g_logits.permute(0,2,1), size=time, mode='linear').permute(0,2,1)

        return final_output, center_score, g_logits_up

# Module 7: The Training Loop & Validation

# ==============================================================================
# UTILS & METRICS
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

def get_batch_f1(probs_in, targets, batch_vocab_mask, temporal_weights, thresholds=0.4):
    """
    Calculates F1 Score. 'thresholds' can be a scalar or a [37] tensor.
    """
    preds = (probs_in > thresholds).float()
    
    valid_pixels = temporal_weights.unsqueeze(-1) * batch_vocab_mask.unsqueeze(1)
    
    tp = (preds * targets * valid_pixels).sum()
    fp = (preds * (1-targets) * valid_pixels).sum()
    fn = ((1-preds) * targets * valid_pixels).sum()
    
    f1 = 2*tp / (2*tp + fp + fn + 1e-6)
    return f1.item()

# ==============================================================================
# LOSS FUNCTION (CLASS-BALANCED FOCAL LOSS + AUX + CENTER)
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
        """
        Inputs:
        model_output_probs: [B, T, 37] (Sigmoid Applied)
        center_pred: [B, T, 1] (Sigmoid Applied)
        aux_logits: [B, T, 37] (Raw Logits)
        """
        # 1. MAIN CLASSIFICATION LOSS (Focal)
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

        # 2. CENTER REGRESSION LOSS (MSE)
        # Only compute on valid time pixels
        # center_target: [B, T, 1]
        l_center = self.mse(center_pred, center_target) # [B, T, 1]
        loss_center = (l_center * tm).sum() / (tm.sum() + 1e-6)
        
        # 3. AUXILIARY GLOBAL LOSS
        # aux_logits is [B, T, 37], matches target
        # We assume global stream should predict same as local ground truth?
        # Yes, "what is happening now".
        # Masked by lab vocabulary
        # Expand lab mask to [B, 1, 37]
        vocab_mask_exp = lab_vocab_mask.unsqueeze(1)
        
        l_aux = self.bce_aux(aux_logits, target) # [B, T, 37]
        loss_aux = (l_aux * vocab_mask_exp * tm).sum() / ((vocab_mask_exp * tm).sum() + 1e-6)

        # TOTAL
        # Weights: Main=1.0, Center=0.5, Aux=0.3
        return loss_main + 0.5 * loss_center + 0.3 * loss_aux

# ==============================================================================
# THRESHOLD TUNER
# ==============================================================================
def find_optimal_thresholds(model, val_loader, device, vocab_mask):
    print("Optimization: Tuning Per-Class Thresholds on Validation Set...")
    model.eval()

    all_preds = []
    all_targs = []
    all_masks = []
    all_weights = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting Val Data"):
            gx, lx, tgt, weights, lid, c_tgt, _ = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            probs, _, _ = model(gx, gx, lx, lx, lid)

            all_preds.append(probs.cpu())
            all_targs.append(tgt.cpu())
            all_masks.append(vocab_mask[lid].unsqueeze(1).repeat(1, weights.shape[1], 1).cpu())
            all_weights.append(weights.cpu())

    preds = torch.cat(all_preds, dim=0) # [N, T, 37]
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
    # --- 1. SETUP & PATHS ---
    DATA_PATH = None
    if 'mabe_mouse_behavior_detection_path' in globals():
        DATA_PATH = globals()['mabe_mouse_behavior_detection_path']
    elif os.path.exists('/kaggle/input/mabe-mouse-behavior-detection/output_dataset'):
        DATA_PATH = '/kaggle/input/mabe-mouse-behavior-detection/output_dataset'
    elif os.path.exists('/kaggle/input/MABe-mouse-behavior-detection'):
        DATA_PATH = '/kaggle/input/MABe-mouse-behavior-detection'
    else: 
        DATA_PATH = "./" # Fallback

    print(f"Data Path: {DATA_PATH}")
    if not os.path.exists(f"{DATA_PATH}/train.csv"):
         print(f"Dataset not found at {DATA_PATH}/train.csv. Aborting.")
         return
    
    VOCAB_PATH = '/kaggle/input/mabe-metadata/results/lab_vocabulary.json'
    
    gpu_count = torch.cuda.device_count()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 8 * max(1, gpu_count)
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 10

    print(f"Start Training on {gpu_count} GPU(s) | Batch Size: {BATCH_SIZE}")

    # --- 2. DATA PREP (Strict Video Split) ---
    meta = pd.read_csv(f"{DATA_PATH}/train.csv")
    vids = meta['video_id'].astype(str).unique()
    np.random.shuffle(vids)
    
    split = int(len(vids) * 0.90)
    train_ids = vids[:split]
    val_ids = vids[split:]
    
    # Loaders
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
    scaler = torch.cuda.amp.GradScaler() # Mixed Precision
    
    # Load Masks
    lab_masks = load_lab_vocabulary(VOCAB_PATH, ACTION_TO_IDX, NUM_CLASSES, DEVICE)
    
    self_indices = [ACTION_TO_IDX[a] for a in SELF_BEHAVIORS]
    pair_indices = [ACTION_TO_IDX[a] for a in PAIR_BEHAVIORS]
         
    loss_fn = DualStreamMaskedFocalLoss(self_indices, pair_indices, gamma=2.0)

    # --- 4. EPOCH LOOP ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}")
        
        run_loss = 0.0
        run_f1 = 0.0
        
        for i, batch in enumerate(loop):
            # Move items to GPU
            # New: c_tgt, meta
            batch = [b.to(DEVICE) if isinstance(b, torch.Tensor) else b for b in batch]
            gx, lx, tgt, weights, lid, c_tgt, meta = batch

            # Ensure float32/contiguous
            gx = gx.float().contiguous()
            lx = lx.float().contiguous()

            # Safety Checks
            if not torch.isfinite(gx).all() or not torch.isfinite(lx).all():
                print(f"[WARN] Non-finite inputs in batch {i}, skipping")
                continue
            if (weights.sum(dim=1) == 0).all():
                continue
            
            optimizer.zero_grad()
            
            # Role Flipping Augmentation
            role_idx = None
            if random.random() < 0.5:
                 role_idx = torch.ones(gx.shape[0], dtype=torch.long).to(DEVICE) # Role 1
            
            try:
                # Mixed Precision Forward
                with torch.cuda.amp.autocast():
                    # Forward returns 3 items
                    probs, center_pred, aux_logits = model(gx, gx, lx, lx, lid, role_idx)

                    loss = loss_fn(probs, center_pred, aux_logits, tgt, c_tgt, weights, lab_masks[lid])

                # Backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            except Exception as e:
                print(f"[ERROR] Forward/Backward failed at batch {i}: {e}")
                # print("Sample metas:", meta[:2])
                torch.cuda.empty_cache()
                continue
            
            # Metrics
            with torch.no_grad():
                f1 = get_batch_f1(probs, tgt, lab_masks[lid], weights)
                
            run_loss = 0.9*run_loss + 0.1*loss.item() if i>0 else loss.item()
            run_f1 = 0.9*run_f1 + 0.1*f1 if i>0 else f1
            
            if i % 20 == 0:
                loop.set_postfix({'Loss': f"{run_loss:.4f}", 'F1': f"{run_f1:.3f}"})
        
        # Validation
        print("Validating...")
        model.eval()
        val_loss_sum = 0
        batches = 0

        # Accumulate for Real Metric
        submission_rows = []
        solution_rows = []

        # INFERENCE-STYLE VALIDATION LOOP
        print("Running Inference-Style Validation...")

        # We iterate over the dataset directly to get full videos
        # Limit to first 100 samples if too slow? Or full validation?
        # User wants full validation logic.

        WINDOW_SIZE = 256
        STRIDE = 128
        thresholds = torch.ones(37).to(DEVICE) * 0.4

        # Parallel Loader for Validation
        val_dataset_full = ValidationDataset(val_ds)
        val_loader_full = DataLoader(val_dataset_full, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: x[0])

        for i, batch_data in tqdm(enumerate(val_loader_full), total=len(val_loader_full), desc="Validating Videos"):
            try:
                feats, lab_idx, agent_id, target_id, frames, valid_mask = batch_data

                if feats is None: continue

                # Get Ground Truth from Annotations (Slow but necessary for exact metric)
                # We need to read the annotation file for this video/pair
                # This duplicates logic from _load but for full video.
                # Optimization: Load ALL annotations for this video once.
                # Since we loop by pair, we can just read the file.

                # Load GT
                sample = val_ds.samples[i]
                lab = sample['lab_id']
                vid = sample['video_id']

                gt_actions = []
                annot_path = val_ds.annot_dir / lab / f"{vid}.parquet"
                if annot_path.exists():
                     adf = pd.read_parquet(annot_path)
                     # Filter for this pair
                     # Column names might vary, assuming 'agent', 'target' based on previous code
                     if 'agent' in adf.columns:
                         adf = adf[(adf['agent'] == agent_id) & (adf['target'] == target_id)]

                     for _, row in adf.iterrows():
                         if row['action'] in ACTION_TO_IDX:
                             gt_actions.append(row)
                             solution_rows.append({
                                'video_id': vid,
                                'agent_id': agent_id,
                                'target_id': target_id,
                                'action': row['action'],
                                'start_frame': row['start_frame'],
                                'stop_frame': row['stop_frame'],
                                'lab_id': lab,
                                'behaviors_labeled': sample.get('behaviors_labeled', "[]")
                             })

                # Run Inference (Batched for Speed & Memory Safety)
                T_total = len(feats)
                prob_accum = torch.zeros((T_total, NUM_CLASSES), device=DEVICE)
                count_accum = torch.zeros((T_total, NUM_CLASSES), device=DEVICE)

                g_feats_full = feats.unsqueeze(0).to(DEVICE).float().contiguous()
                g_input = g_feats_full[:, ::30, :, :]
                if g_input.shape[1] == 0: g_input = g_feats_full

                # Collect valid windows first
                windows = []
                starts = []

                for start in range(0, T_total, STRIDE):
                    end = min(start + WINDOW_SIZE, T_total)
                    if not valid_mask[start:end].any(): continue

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
                    gx_batch = g_input.repeat(B, 1, 1, 1)
                    lid_batch = torch.tensor([lab_idx]*B).to(DEVICE)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
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
                preds = (final_probs > thresholds.unsqueeze(0)).int().cpu().numpy()

                # Convert to Segments
                for c in range(NUM_CLASSES):
                    action_name = ACTION_LIST[c]
                    binary_seq = preds[:, c]
                    diffs = np.diff(np.concatenate(([0], binary_seq, [0])))
                    starts = np.where(diffs == 1)[0]
                    stops = np.where(diffs == -1)[0]

                    for s, e in zip(starts, stops):
                        submission_rows.append({
                             'video_id': vid,
                             'agent_id': agent_id,
                             'target_id': target_id,
                             'action': action_name,
                             'start_frame': frames[s], # Map back to video frame
                             'stop_frame': frames[e-1]
                        })

                batches += 1 # Count successful videos

            except Exception as e:
                print(f"Validation Error on {i}: {e}")
                continue

        # CALCULATE METRIC
        try:
            val_loss_avg = val_loss_sum / batches if batches > 0 else 0.0

            if len(solution_rows) > 0 and len(submission_rows) > 0:
                sol_df = pd.DataFrame(solution_rows)
                sub_df = pd.DataFrame(submission_rows)
                real_score = mouse_fbeta(sol_df, sub_df)
                print(f"Val Loss: {val_loss_avg:.4f} | REAL F1 Score: {real_score:.4f}")
            else:
                 print(f"Val Loss: {val_loss_avg:.4f} | REAL F1 Score: 0.0000 (Empty predictions/solutions)")
        except Exception as e:
            print(f"Metric Calculation Failed: {e}")
            print(f"Val Loss: {val_loss_sum/batches if batches > 0 else 0.0:.4f}")
        
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state, f"ethoswarm_v4_ep{epoch+1}.pth")

    # --- 5. POST-TRAINING THRESHOLD OPTIMIZATION ---
    final_thresholds = find_optimal_thresholds(model, val_loader, DEVICE, lab_masks)

    # Save Thresholds
    with open("thresholds.json", "w") as f:
        json.dump(final_thresholds.cpu().tolist(), f)
    print("Saved optimal thresholds to thresholds.json")

if __name__ == '__main__':
    train_ethoswarm_v3()
