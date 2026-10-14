"""
This file builds and trains the neural network that guesses the normal taxi-out time of every departure
The network does not guess the whole time from scratch - it starts from a rough first guess (the offset) and only learns how much to add or take away
It learns from the 2025 flights where the answer is known, and only from normal records (regime N) - not from the ones where the off-block time was copied from the schedule (P) or written with the previous day's date (S)

This is how it works:
-load the flights, the network input cache and any extra feature blocks, and drop the inputs it is told to leave out
-for each fold (a week fold, where every fourth week of 2025 is held out to test on, or 'full' = all of 2025 for the real predictions):
    -work out the target: the taxi time minus the offset, cut and rescaled - the offset is the NM baseline (take-off minus the off-block time known to EUROCONTROL's Network Manager, NM) or the unimpeded time (a usual quick taxi time for that stand and runway)
    -for each seed:
        1. build the network (a ResNet, TabM or RealMLP backbone, optionally with attention over the neighboring take-offs)
        2. train it on the GPU, or with --predict-only load a stored fit instead
        3. predict every flight, and retry with seed + 100, then seed + 200 if the predictions come out as NaN
    -average the seeds and score the average on the held-out weeks
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rtdl_num_embeddings import PeriodicEmbeddings, PiecewiseLinearEmbeddings, compute_bins
from tabm import EnsembleView, LinearEnsemble, make_tabm_backbone
from torch import Tensor, nn

from taxi.features.network_inputs import quantile_normal_transform
from taxi.main import FEATURE_MATRIX, MODEL_WEIGHTS, RUN_PREDICTIONS, departure_frame, held_out_week_fold_rows, rmse

warnings.filterwarnings('ignore', message='.*has just two bin edges.*')


# inputs
DEFAULT_BLOCK_CACHE = 'network_block_cache'   # the only block cache folder load_feature_block rebuilds

# feature blocks
MISSING_FLAG_MIN_SHARE = 0.005     # same 0.5 % cut-off as the network input cache
MISSING_FLAG_MAX_SHARE = 0.995     # same 99.5 % cut-off as the network input cache

# neighbouring take-offs and the neighbour attention
NEIGHBOUR_MAX_GAP_SECONDS = 3600            # one hour, a round limit picked when the neighbour attention was built, not tuned
NEIGHBOUR_TIME_OFFSET_UNIT_SECONDS = 600    # time gaps are given in units of 10 minutes, so the one-hour window becomes -6 to 6, a scale networks handle well
NEIGHBOUR_ENCODER_HIDDEN_WIDTH = 256        # the same width as the main blocks (DEFAULT_BLOCK_WIDTH)
NEIGHBOUR_ATTENTION_MODEL_DIM = 128         # a usual small attention size
NEIGHBOUR_ATTENTION_HEADS = 4               # 4 heads of 32 numbers each, a usual way to split 128
NEIGHBOUR_ATTENTION_DROPOUT = 0.1           # the same dropout as the rest of the network (DEFAULT_DROPOUT)

# network parts
RESNET_HIDDEN_FACTOR = 2                    # each block's inner layer is twice the block width, a usual choice for tabular ResNets
PBLD_FREQUENCIES_PER_INPUT = 16             # RealMLP default (Holzmueller et al. 2024; pytabkit plr_hidden_1 = 16)
PBLD_DEFAULT_OUTPUT_DIM = 4                 # RealMLP default (pytabkit plr_hidden_2 = 4); the RealMLP jobs also pass --embedding-dim 4
PBLD_FREQUENCY_INIT_SCALE = 0.1             # RealMLP default starting spread of the wave speeds (pytabkit plr_sigma = 0.1)
CATEGORICAL_EMBEDDING_MIN_DIM = 2           # even a category column with very few codes gets 2 numbers
CATEGORICAL_EMBEDDING_SIZE_FACTOR = 1.6     # fastai's rule of thumb for embedding sizes: round(1.6 * number_of_codes ** 0.56)
CATEGORICAL_EMBEDDING_SIZE_EXPONENT = 0.56  # the exponent of the same fastai rule
TWOHOT_ASINH_SCALE = 0.5                    # in standard deviations of the target: bucket centres sit close together within about half a standard deviation of 0, then spread out

# training and prediction
NM_BASELINE_INPUT_CENTRE_SECONDS = 900         # 15 minutes, roughly a typical taxi-out time, so the NM baseline input sits around 0
NM_BASELINE_INPUT_UNIT_SECONDS = 600           # measured in units of 10 minutes, so ordinary values fall within a few units of 0
NM_BASELINE_INPUT_CLIP = 5                     # cut at 5 units (50 minutes either side of the centre), the same +-5 limit as the other inputs' bell-curve scale
BIN_FIT_SAMPLE_ROWS = 300_000                  # a random sample is plenty to place 48 bin edges per input and keeps the bin fit quick
ADAMW_BETA1 = 0.9                              # the standard AdamW default (PyTorch)
REALMLP_SCALE_LR_FACTOR = 6                    # RealMLP default: its input scaling layer learns 6 times faster (pytabkit scale_lr_factor = 6.0)
GRADIENT_CLIP_NORM = 1.0                       # common practice; the TabM example training recipe also clips gradients at 1.0
PREDICTION_BATCH_ROW_MEMBER_BUDGET = 32768     # rows x TabM members per prediction batch: TabM-16 predicts 2048 rows (its training batch) - handpicked to not run OOM on 8GB GPU VRAM
PREDICTION_BATCH_MIN_ROWS = 256                # a floor that only matters for TabM k above 128
PREDICTION_BATCH_MAX_ROWS = 4096               # cap for small k: prediction batches stay small since a 65,536-row batch overflowed the 8 GB GPU and crashed the laptop
PREDICTION_BATCH_CONTEXT_ROWS = 1024           # with neighbour attention each row also brings the inputs of 16 or 32 neighbours, so batches are smaller to fit the 8 GB GPU
GPU_MEMORY_FRACTION = 0.85                     # PyTorch may use at most 85 % of the 8 GB GPU (about 6.8 GB): too big a batch fails with out-of-memory instead of spilling into shared memory
RETRY_SEED_STEP = 100                          # retry seeds (seed + 100, seed + 200) can never clash with the ordinary seeds 1 to 5

# command-line defaults
DEFAULT_TWOHOT_CENTRES = 128           # number of two-hot buckets
DEFAULT_MUON_LEARNING_RATE = 2e-3      # the same rate as AdamW (DEFAULT_LEARNING_RATE)
DEFAULT_MUON_WEIGHT_DECAY = 0.1        # PyTorch's Muon default
DEFAULT_ADAMW_BETA2 = 0.999            # PyTorch's AdamW default; the RealMLP jobs pass 0.95, the RealMLP default
DEFAULT_EPOCHS = 12                    # chosen on held-out 2025 data with a cosine schedule: 20 epochs did worse; the two-hot jobs use 24
DEFAULT_TARGET_CLIP_SECONDS = 6000.0   # 100 minutes, so a few extreme rows cannot dominate the squared loss; the clip15000 jobs use 15000 s, better on the 2025 week folds
DEFAULT_BATCH_SIZE_ROWS = 2048         # the setting of the first version; 1024 did worse on held-out 2025 data
DEFAULT_LEARNING_RATE = 2e-3           # AdamW rate of the TabM example recipe; 5e-4 did worse on held-out 2025 data (1e-3 is also trained, as a variant)
DEFAULT_WEIGHT_DECAY = 3e-4            # AdamW weight decay of the TabM example recipe
DEFAULT_TABM_K = 32                    # TabM's own default number of members (tabm package); the TabM jobs use 16 because 32 ran out of memory on the 8 GB GPU
DEFAULT_RESIDUAL_BLOCKS = 3            # a small ResNet of 3 blocks x 256 matched TabM on held-out 2025 data; bigger and deeper networks did worse
DEFAULT_BLOCK_WIDTH = 256              # see DEFAULT_RESIDUAL_BLOCKS
DEFAULT_DROPOUT = 0.1                  # TabM's default dropout (tabm package)
DEFAULT_EMBEDDING_DIM = 16             # the setting of the first version; 32 did worse on held-out 2025 data
DEFAULT_EMBEDDING_BINS = 48            # the default bin count of rtdl_num_embeddings.compute_bins; 96 bins did worse on held-out 2025 data
DEFAULT_CATEGORICAL_DIM = 16           # cap on category embedding sizes, the setting of the first version; 32 was no better on the 2025 week folds


def load_feature_block(block_name: str, fit_rows: np.ndarray, cache_name: str = DEFAULT_BLOCK_CACHE) -> tuple[np.ndarray, list[str]]:
    """
    This function turns one feature block (an extra table of inputs) into numbers the network can read easily
    """
    cache_dir = FEATURE_MATRIX / cache_name
    cache_dir.mkdir(exist_ok=True)
    cached_matrix_file = cache_dir / f'{block_name}.npy'
    cached_names_file = cache_dir / f'{block_name}.json'
    block_file = FEATURE_MATRIX / f'block_{block_name}.parquet'
    cache_is_stale = _block_cache_is_stale(block_file, cached_matrix_file, cached_names_file)
    if cache_is_stale and cache_name != DEFAULT_BLOCK_CACHE:
        raise RuntimeError(f'{cache_dir / block_name}: stale or missing, and only the default block cache is rebuilt')
    if cache_is_stale:
        _build_block_cache(block_file, fit_rows, cached_matrix_file, cached_names_file)
    return np.load(cached_matrix_file), json.loads(cached_names_file.read_text())


def _block_cache_is_stale(block_file: Path, cached_matrix_file: Path, cached_names_file: Path) -> bool:
    """
    This function checks if the cached copy of a block can still be trusted
    """
    if not cached_matrix_file.exists() or not cached_names_file.exists():
        return True
    return block_file.stat().st_mtime_ns > cached_matrix_file.stat().st_mtime_ns


def _build_block_cache(block_file: Path, fit_rows: np.ndarray, cached_matrix_file: Path, cached_names_file: Path) -> None:
    """
    This function makes the cached copy of one block: it skips columns that say nothing on the training rows (fewer than two known values, or all the same) and puts every other column on the same bell-curve scale (-5 to 5, missing values become 0)
    It also adds a 'was this missing?' flag for columns missing on between 0.5% and 99.5% of the rows - columns missing on exactly the same rows share one flag
    """
    block_table = pd.read_parquet(block_file)
    normalised_columns: list[np.ndarray] = []
    column_names: list[str] = []
    seen_pattern_hashes: set[str] = set()
    missing_flags: list[np.ndarray] = []
    missing_flag_names: list[str] = []
    for column in block_table.columns:
        values = block_table[column].to_numpy(dtype=np.float64)
        is_missing = ~np.isfinite(values)
        fit_values = values[fit_rows & ~is_missing]
        if len(fit_values) < 2 or fit_values.min() == fit_values.max():
            continue   # no information on the fit rows (all missing or constant)
        if MISSING_FLAG_MIN_SHARE < is_missing.mean() < MISSING_FLAG_MAX_SHARE:
            # Columns missing together (same pattern) share one flag: the hash identifies the pattern.
            pattern_hash = hashlib.sha1(np.packbits(is_missing).tobytes()).hexdigest()
            if pattern_hash not in seen_pattern_hashes:
                seen_pattern_hashes.add(pattern_hash)
                missing_flags.append(is_missing.astype(np.float16))
                missing_flag_names.append(f'miss:{column}')
        normalised = quantile_normal_transform(values, fit_rows)
        normalised[~np.isfinite(normalised)] = 0.0
        normalised_columns.append(normalised.astype(np.float16))
        column_names.append(column)
    np.save(cached_matrix_file, np.stack(normalised_columns + missing_flags, 1))
    cached_names_file.write_text(json.dumps(column_names + missing_flag_names))


def neighbouring_takeoffs(frame: pd.DataFrame, neighbour_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function finds, for every flight, the planes that took off just before and just after it at the same airport (e.g. 8 before and 8 after for 16 neighbors), for the neighbor attention
    A neighbor slot is left empty when it would be at another airport, more than an hour (3600 seconds) away, or past the start or end of the list
    """
    row_count = len(frame)
    # MVT_TIME_UTC_mvt has microsecond resolution, so the integer division gives whole seconds.
    takeoff_seconds = (frame.MVT_TIME_UTC_mvt.astype('int64') // 10 ** 6).to_numpy()
    airport_codes = pd.factorize(frame.ADEP_mvt)[0]
    # Sorting by airport, then take-off time puts the take-offs just before / after a row next to it.
    sorted_rows = np.lexsort((takeoff_seconds, airport_codes))
    sorted_position = np.empty(row_count, np.int64)
    sorted_position[sorted_rows] = np.arange(row_count)
    position_offsets = [offset for offset in range(-neighbour_count // 2, neighbour_count // 2 + 1) if offset != 0]
    own_rows = np.arange(row_count)
    neighbour_index = np.zeros((row_count, len(position_offsets)), np.int32)
    time_offset = np.zeros((row_count, len(position_offsets)), np.float32)
    is_padding = np.zeros((row_count, len(position_offsets)), bool)
    for slot, position_offset in enumerate(position_offsets):
        wanted_position = sorted_position + position_offset
        clipped_position = np.clip(wanted_position, 0, row_count - 1)
        neighbour_rows = sorted_rows[clipped_position]
        seconds_apart = (takeoff_seconds[neighbour_rows] - takeoff_seconds).astype(np.float64)
        at_other_airport = airport_codes[neighbour_rows] != airport_codes
        more_than_an_hour_apart = np.abs(seconds_apart) > NEIGHBOUR_MAX_GAP_SECONDS
        past_first_or_last_row = clipped_position != wanted_position
        is_invalid = at_other_airport | more_than_an_hour_apart | past_first_or_last_row
        neighbour_index[:, slot] = np.where(is_invalid, own_rows, neighbour_rows)
        time_offset[:, slot] = np.where(is_invalid, 0, seconds_apart / NEIGHBOUR_TIME_OFFSET_UNIT_SECONDS)
        is_padding[:, slot] = is_invalid
    return neighbour_index, time_offset, is_padding


@dataclass
class NetworkTrainingData:
    """
    This data structure holds everything a training run needs
    """
    frame: pd.DataFrame
    label: np.ndarray
    no_nm_link: np.ndarray
    airport: np.ndarray
    is_normal_regime: np.ndarray
    numeric_inputs: np.ndarray
    categorical_inputs: np.ndarray
    category_cardinalities: list[int]
    unimpeded_time: np.ndarray | None = None
    neighbour_table: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    runway_column: int | None = None


def load_training_data(args: argparse.Namespace) -> NetworkTrainingData:
    """
    This function loads everything a run needs
    """
    frame = departure_frame()
    label = frame.TAXITIME_SEC_mvt.to_numpy()
    is_2025 = (frame.year.to_numpy() == 2025)
    cache_dir = FEATURE_MATRIX / args.network_inputs
    cache_columns = json.loads((cache_dir / 'columns.json').read_text())   # keys: num, cat, card (see taxi.features.network_inputs)
    numeric_inputs = np.load(cache_dir / 'x_num.npy')
    categorical_inputs = np.load(cache_dir / 'x_cat.npy').astype(np.int64)
    category_cardinalities = list(cache_columns['card'])
    numeric_names = list(cache_columns['num'])
    if args.drop_column_prefixes:
        numeric_inputs, numeric_names = _drop_base_inputs_with_prefixes(numeric_inputs, numeric_names, args.drop_column_prefixes)
    if args.drop_columns:
        numeric_inputs, numeric_names = _drop_named_base_inputs(numeric_inputs, numeric_names, args.drop_columns)
    block_cache = getattr(args, 'block_cache', DEFAULT_BLOCK_CACHE)   # a Namespace not built by build_argument_parser may lack it
    for block_spec in [spec for spec in args.feature_blocks.split(',') if spec]:
        block_matrix, block_names = _selected_block_columns(block_spec, is_2025, block_cache)
        numeric_inputs = np.concatenate([numeric_inputs, block_matrix], 1)
        numeric_names += block_names
        print(f'block {block_spec}: {block_matrix.shape[1]} columns')
    data = NetworkTrainingData(frame=frame, label=label, no_nm_link=frame.no_nm_link.to_numpy(), airport=frame.ADEP_mvt.to_numpy(), is_normal_regime=(~frame.is_schedule_copy & ~frame.is_previous_date).to_numpy(), numeric_inputs=numeric_inputs, categorical_inputs=categorical_inputs, category_cardinalities=category_cardinalities)
    if args.target == 'unimpeded':
        block_name, column = args.unimpeded_column.split(':')
        unimpeded_block = pd.read_parquet(FEATURE_MATRIX / f'block_{block_name}.parquet', columns=[column])
        data.unimpeded_time = unimpeded_block[column].to_numpy(dtype=float)
    if args.context_neighbours > 0:
        data.neighbour_table = neighbouring_takeoffs(frame, args.context_neighbours)
        data.runway_column = cache_columns['cat'].index('airport_runway')
    return data


def _kept_columns(matrix: np.ndarray, names: list[str], kept: list[int]) -> tuple[np.ndarray, list[str]]:
    """
    This function keeps only the chosen columns of a table, in the order given, together with their names
    """
    return matrix[:, kept], [names[index] for index in kept]


def _drop_base_inputs_with_prefixes(numeric_inputs: np.ndarray, numeric_names: list[str], prefix_list: str) -> tuple[np.ndarray, list[str]]:
    """
    This function removes every input whose name starts with one of the given prefixes
    """
    prefixes = tuple(prefix_list.split(','))
    kept = [index for index, name in enumerate(numeric_names) if not name.removeprefix('miss:').startswith(prefixes)]
    print(f'dropped {len(numeric_names) - len(kept)} base inputs with prefixes {prefixes}')
    return _kept_columns(numeric_inputs, numeric_names, kept)


def _drop_named_base_inputs(numeric_inputs: np.ndarray, numeric_names: list[str], name_list: str) -> tuple[np.ndarray, list[str]]:
    """
    This function removes inputs by their exact names
    """
    dropped_names = set(name_list.split(','))
    kept = [index for index, name in enumerate(numeric_names) if name.removeprefix('miss:') not in dropped_names]
    print(f'dropped base inputs {sorted(set(numeric_names) - {numeric_names[index] for index in kept})}')
    return _kept_columns(numeric_inputs, numeric_names, kept)


def _selected_block_columns(block_spec: str, fit_rows: np.ndarray, block_cache: str) -> tuple[np.ndarray, list[str]]:
    """
    This function loads one feature block named in --feature-blocks and picks its columns
    """
    block_name, _, column_selector = block_spec.partition(':')
    block_matrix, block_names = load_feature_block(block_name, fit_rows, block_cache)
    if not column_selector:
        return block_matrix, block_names
    prefixes = tuple(column_selector.lstrip('!').split('+'))
    drop_matching = column_selector.startswith('!')
    kept = [index for index, name in enumerate(block_names) if name.removeprefix('miss:').startswith(prefixes) != drop_matching]
    return _kept_columns(block_matrix, block_names, kept)


# models
class ResNetBackbone(nn.Module):
    """
    This is a class that holds the main part (the backbone) of the standard network, a ResNet: an input layer, then a few residual blocks that each keep the numbers coming in and only add a correction on top, which makes deep networks easier to train
    """

    def __init__(self, input_dim: int, block_count: int, block_width: int, dropout: float) -> None:
        """
        This function builds the input layer, the residual blocks and the final rescaling, always in this order - the order decides the random starting values of a seed
        """
        super().__init__()
        self.inp = nn.Linear(input_dim, block_width)
        self.blocks = nn.ModuleList([_residual_block(block_width, dropout) for _ in range(block_count)])
        self.out_norm = nn.Sequential(nn.BatchNorm1d(block_width), nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        """
        This function runs a batch of flights through the backbone - each block adds its correction to the numbers coming in
        """
        x = self.inp(x)
        for residual_block in self.blocks:
            x = x + residual_block(x)
        return self.out_norm(x)


def _residual_block(width: int, dropout: float) -> nn.Sequential:
    """
    This function builds the correction branch of one ResNet block: rescale, widen to twice the width, cut negative numbers to 0, randomly switch some numbers off during training (dropout), narrow back and dropout again
    """
    return nn.Sequential(nn.BatchNorm1d(width), nn.Linear(width, RESNET_HIDDEN_FACTOR * width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(RESNET_HIDDEN_FACTOR * width, width), nn.Dropout(dropout))


class PBLDEmbedding(nn.Module):
    """
    This is a class that turns every number input into a few numbers the network can bend more easily - it keeps the input itself and adds a mix of wavy cosine curves of it, learned separately for every input
    """

    def __init__(self, feature_count: int, hidden_dim: int = PBLD_FREQUENCIES_PER_INPUT, output_dim: int = PBLD_DEFAULT_OUTPUT_DIM) -> None:
        """
        This function sets up random wave speeds and shifts for every input, and a learned table that mixes the waves into the output numbers
        """
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(feature_count, hidden_dim) * PBLD_FREQUENCY_INIT_SCALE)
        self.b1 = nn.Parameter(torch.rand(feature_count, hidden_dim) * 2 * math.pi - math.pi)
        self.w2 = nn.Parameter(torch.randn(feature_count, hidden_dim, output_dim - 1) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(feature_count, output_dim - 1))

    def forward(self, x: Tensor) -> Tensor:
        """
        This function computes the cosine waves of each input, mixes them with the learned table and puts the input itself in front
        """
        periodic = torch.cos(2 * math.pi * self.w1 * x.unsqueeze(-1) + self.b1)
        linear_map = torch.einsum('bnh,nho->bno', periodic, self.w2) + self.b2
        return torch.cat([x.unsqueeze(-1), linear_map], -1)


class ParametricMish(nn.Module):
    """
    This is a class that bends every number so the network can learn curved relationships, not just straight lines - it blends the number with the smooth Mish curve using one learned weight
    """

    def __init__(self) -> None:
        """
        This function creates the single learned weight, starting at 1
        """
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        """
        This function applies the bend x + a (mish(x) - x) activation function
        """
        return x + self.a * (F.mish(x) - x)


class RealMLPBackbone(nn.Module):
    """
    This is a class that holds the backbone of the RealMLP networks
    """

    def __init__(self, input_dim: int, block_count: int, block_width: int, dropout: float) -> None:
        """
        This function creates the "volume knobs" and then the layers
        """
        super().__init__()
        self.scale = nn.Parameter(torch.ones(input_dim))   # trained with a 6x learning rate (see _build_optimizers)
        layers: list[nn.Module] = []
        layer_input_dim = input_dim
        for _ in range(block_count):
            layers += [nn.Linear(layer_input_dim, block_width), ParametricMish(), nn.Dropout(dropout)]
            layer_input_dim = block_width
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        This function multiplies the inputs by their weights and runs them through the layers
        """
        return self.net(x * self.scale)


class NeighbourAttention(nn.Module):
    """
    This is a class that lets a flight look at the planes that took off near it at the same airport and pick out the most useful ones (attention), even if it is "nobody"
    """

    def __init__(self, token_dim: int, query_dim: int, model_dim: int = NEIGHBOUR_ATTENTION_MODEL_DIM, head_count: int = NEIGHBOUR_ATTENTION_HEADS, dropout: float = NEIGHBOUR_ATTENTION_DROPOUT) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(token_dim, NEIGHBOUR_ENCODER_HIDDEN_WIDTH), nn.ReLU(), nn.Linear(NEIGHBOUR_ENCODER_HIDDEN_WIDTH, model_dim))
        self.q = nn.Linear(query_dim, model_dim)
        self.attn = nn.MultiheadAttention(model_dim, head_count, batch_first=True, dropout=dropout)
        self.null = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.norm = nn.LayerNorm(2 * model_dim)
        self.output_dim = 2 * model_dim

    def forward(self, tokens: Tensor, is_padding: Tensor, query: Tensor) -> Tensor:
        """
        This function encodes the neighbors, adds the 'nobody' token and hides the empty slots, then lets the flight look at its neighbors - the result is put next to the neighbors' average and rescaled
        """
        null_tokens = self.null.expand(len(tokens), -1, -1)
        encoded = torch.cat([self.enc(tokens), null_tokens], 1)
        null_is_padding = torch.zeros(len(is_padding), 1, dtype=torch.bool, device=is_padding.device)
        is_padding = torch.cat([is_padding, null_is_padding], 1)
        attended, _ = self.attn(self.q(query).unsqueeze(1), encoded, encoded, key_padding_mask=is_padding, need_weights=False)
        is_real_token = (~is_padding).unsqueeze(-1).to(encoded.dtype)
        token_mean = (encoded * is_real_token).sum(1) / is_real_token.sum(1)
        return self.norm(torch.cat([attended.squeeze(1), token_mean], 1))


def _categorical_embedding_sizes(category_cardinalities: list[int], max_dim: int) -> list[int]:
    """
    This function decides how many learned numbers each category input (like the runway) gets - it grows slowly with the number of codes, and is at least 2 and at most 16 by default
    """
    return [min(max_dim, max(CATEGORICAL_EMBEDDING_MIN_DIM, round(CATEGORICAL_EMBEDDING_SIZE_FACTOR * cardinality ** CATEGORICAL_EMBEDDING_SIZE_EXPONENT))) for cardinality in category_cardinalities]


def _numeric_embedding(args: argparse.Namespace, numeric_count: int, piecewise_bins: list[Tensor] | None) -> nn.Module:
    """
    This function picks how each number input is turned into several numbers, as chosen with --embedding: piecewise-linear bins (the default), periodic waves, or the PBLD embedding
    """
    if args.embedding == 'piecewise_linear':
        return PiecewiseLinearEmbeddings(piecewise_bins, args.embedding_dim, activation=False, version='B')
    if args.embedding == 'periodic':
        return PeriodicEmbeddings(numeric_count, args.embedding_dim, lite=False)
    return PBLDEmbedding(numeric_count, PBLD_FREQUENCIES_PER_INPUT, args.embedding_dim)


class TaxiTimeNetwork(nn.Module):
    """
    This is a class that holds the whole network: it turns the number and category inputs into embeddings, runs them through a backbone (ResNet, TabM or RealMLP), can let a flight look at its neighboring take-offs, and ends with an output head that gives the answer
    TabM works like 32 slightly different networks (by default) sharing most of their parts, so the output has one slot per TabM member
    """

    def __init__(self, args: argparse.Namespace, numeric_count: int, category_cardinalities: list[int], piecewise_bins: list[Tensor] | None, output_count: int) -> None:
        """
        This function builds the parts in a fixed order (embeddings, backbone, head, neighbor attention)
        With neighbor attention the simple head is still built first, to keep the random numbers the same, and then swapped for a bigger head that also reads the attention's output
        """
        super().__init__()
        # Plain (non-parameter) attributes, not in the state_dict: the parsed arguments and the TabM ensemble size.
        self.args = args
        self.ensemble_size = args.tabm_k if args.architecture == 'tabm' else 1
        self.num_emb: nn.Module = _numeric_embedding(args, numeric_count, piecewise_bins)
        embedding_dims = _categorical_embedding_sizes(category_cardinalities, args.categorical_dim)
        self.cat_emb = nn.ModuleList([nn.Embedding(cardinality, dim) for cardinality, dim in zip(category_cardinalities, embedding_dims)])
        backbone_input_dim = numeric_count * args.embedding_dim + sum(embedding_dims)
        backbone_output_dim = args.block_width
        if args.architecture == 'tabm':
            # TabM: k ensemble members share the weight matrices and have their own scaling vectors and biases; the view
            # repeats the input for the k members.
            self.view = EnsembleView(k=self.ensemble_size)
            self.backbone = make_tabm_backbone(d_in=backbone_input_dim, n_blocks=args.residual_blocks, d_block=args.block_width, dropout=args.dropout, k=self.ensemble_size, arch_type='tabm', start_scaling_init='normal', start_scaling_init_chunks=[args.embedding_dim] * numeric_count + embedding_dims)
            self.head: nn.Module = LinearEnsemble(args.block_width, output_count, k=self.ensemble_size)
        else:
            backbone_class = ResNetBackbone if args.architecture == 'resnet' else RealMLPBackbone
            self.backbone = backbone_class(backbone_input_dim, args.residual_blocks, args.block_width, args.dropout)
            self.head = nn.Linear(backbone_output_dim, output_count)
        self.ctx: NeighbourAttention | None = None
        if args.context_neighbours > 0:
            # Neighbour tokens: numerical inputs, embedded categoricals and 3 extras (time offset, same runway, before / after).
            # The linear head above was already created (its initialisation draws are part of the seed's random stream) and is
            # replaced by a two-layer head over the backbone output plus the attention output.
            self.ctx = NeighbourAttention(numeric_count + sum(embedding_dims) + 3, backbone_output_dim)
            self.head = nn.Sequential(nn.Linear(backbone_output_dim + self.ctx.output_dim, args.block_width), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(args.block_width, output_count))

    def embed_categoricals(self, categorical_codes: Tensor) -> Tensor:
        """
        This function looks up the learned numbers of every category code and puts them side by side
        """
        return torch.cat([embedding(categorical_codes[..., column]) for column, embedding in enumerate(self.cat_emb)], -1)

    def forward(self, numeric_inputs: Tensor, categorical_codes: Tensor, neighbour_context: tuple[Tensor, Tensor, Tensor, Tensor] | None = None) -> Tensor:
        """
        This function runs one batch through the network: embeddings, backbone, the neighbor attention when it is switched on, then the head
        """
        embedded_numeric = self.num_emb(numeric_inputs).flatten(1)
        embedded_categoricals = [embedding(categorical_codes[:, column]) for column, embedding in enumerate(self.cat_emb)]
        x = torch.cat([embedded_numeric] + embedded_categoricals, 1)
        if self.args.architecture == 'tabm':
            return self.head(self.backbone(self.view(x)))
        hidden = self.backbone(x)
        if self.ctx is not None:
            neighbour_numeric, neighbour_categorical, neighbour_extras, is_padding = neighbour_context
            tokens = torch.cat([neighbour_numeric, self.embed_categoricals(neighbour_categorical), neighbour_extras], -1)
            hidden = torch.cat([hidden, self.ctx(tokens, is_padding, hidden)], 1)
        return self.head(hidden).unsqueeze(1)


def twohot_bin_centres(lowest: float, highest: float, centre_count: int, scale: float = TWOHOT_ASINH_SCALE) -> Tensor:
    """
    This function places the bucket centers of the two-hot head from the lowest to the highest possible answer - close together near 0 (the usual answers) and far apart at the ends (the rare very long taxi times)
    """
    asinh_grid = torch.linspace(math.asinh(lowest / scale), math.asinh(highest / scale), centre_count)
    return scale * torch.sinh(asinh_grid)


def twohot_target(values: Tensor, centres: Tensor) -> Tensor:
    """
    This function turns each right answer into a two-hot target: its weight of 1 is split between the two bucket centers on either side, so the weighted average of the buckets is exactly the answer
    """
    values = values.clamp(centres[0], centres[-1])
    upper_index = torch.searchsorted(centres, values, right=True).clamp(1, len(centres) - 1)
    lower_index = upper_index - 1
    upper_weight = (values - centres[lower_index]) / (centres[upper_index] - centres[lower_index])
    distribution = torch.zeros(len(values), len(centres), device=values.device)
    distribution.scatter_(1, lower_index.unsqueeze(1), (1 - upper_weight).unsqueeze(1))
    distribution.scatter_(1, upper_index.unsqueeze(1), upper_weight.unsqueeze(1))
    return distribution


# ---------------------------------------------------------------- one fit
def train_one_network(args: argparse.Namespace, data: NetworkTrainingData, fold: str, seed: int, print_summary: bool, saved_weights: Path | None = None) -> tuple[np.ndarray, float | None, float]:
    """
    This is the actual function that trains one network for one fold and one seed on the GPU: it works out the target, builds and trains the network, scores it on the held-out normal flights (RMSE) and predicts every flight
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    started = time.time()
    device = torch.device('cuda')
    target = _fit_target(args, data, fold)
    extra_inputs = _extra_numeric_inputs(target.nm_baseline, data.no_nm_link)
    numeric_inputs = torch.from_numpy(np.concatenate([data.numeric_inputs, extra_inputs], 1)).to(device)
    categorical_inputs = torch.from_numpy(data.categorical_inputs).to(device)
    numeric_count = numeric_inputs.shape[1]
    fit_rows_on_device = torch.from_numpy(target.fit_rows).to(device)
    fit_targets = torch.from_numpy(target.standardised_label[target.fit_rows].astype(np.float32)).to(device)
    piecewise_bins = None
    if args.embedding == 'piecewise_linear':
        piecewise_bins = _piecewise_linear_bins(numeric_inputs, fit_rows_on_device, args.embedding_bins, rng, device)
    bin_centres, output_count = _output_head_centres(args, target.label_mean, target.label_sd, device)
    label_mean = target.label_mean
    label_sd = target.label_sd
    saved = None
    if saved_weights is not None:
        # predict-only: the stored fit's label scaling, two-hot centres and embedding bins replace the ones computed above
        saved = torch.load(saved_weights, map_location='cpu', weights_only=False)
        label_mean = float(saved['mu'])
        label_sd = float(saved['sd'])
        bin_centres = None if saved['centres'] is None else saved['centres'].to(device)
        if 'num_emb.impl.mask' in saved['state_dict']:
            piecewise_bins = _placeholder_bins_of_stored_embedding(saved['state_dict']['num_emb.impl.mask'])
    model = TaxiTimeNetwork(args, numeric_count, data.category_cardinalities, piecewise_bins, output_count).to(device)
    if print_summary:
        parameter_millions = sum(parameter.numel() for parameter in model.parameters()) / 1e6
        print(f'rows train {len(target.fit_rows)} valid {len(target.evaluation_rows)}, inputs {numeric_count} num + {len(data.category_cardinalities)} cat, parameters {parameter_millions:.1f}M', flush=True)
    inputs = _device_inputs(args, data, numeric_inputs, categorical_inputs, device)
    optimizers = _build_optimizers(args, model)
    # Cosine decay of the learning rate from its initial value to 0 over all training steps.
    total_steps = args.epochs * math.ceil(len(target.fit_rows) / args.batch_size)
    schedulers = _cosine_schedulers(optimizers, total_steps)
    every_row = np.arange(len(data.label))
    if saved is not None:
        model.load_state_dict(saved['state_dict'])
        value = target.offset + _predict_residual(model, inputs, every_row, args, bin_centres, label_mean, label_sd, device)
        del model, inputs, numeric_inputs, categorical_inputs, optimizers
        torch.cuda.empty_cache()
        return value, None, (time.time() - started) / 60
    _train_epochs(args, model, inputs, fit_rows_on_device, fit_targets, bin_centres, optimizers, schedulers, device)
    last_epoch_rmse = None
    if len(target.evaluation_rows) and args.epochs > 0:   # scored after the last epoch
        held_out_residual = _predict_residual(model, inputs, target.evaluation_rows, args, bin_centres, label_mean, label_sd, device)
        held_out_value = target.offset[target.evaluation_rows] + held_out_residual
        last_epoch_rmse = rmse(held_out_value - data.label[target.evaluation_rows])
    value = target.offset + _predict_residual(model, inputs, every_row, args, bin_centres, label_mean, label_sd, device)
    if np.isfinite(value).all():     # only the weights of a kept seed (a diverged seed is retrained under another seed)
        _save_fit(args, fold, seed, model, label_mean, label_sd, bin_centres)
    del model, inputs, numeric_inputs, categorical_inputs, optimizers
    torch.cuda.empty_cache()
    return value, last_epoch_rmse, (time.time() - started) / 60


@dataclass
class _FitTarget:
    """
    This data structure holds what one fit learns
    """

    nm_baseline: np.ndarray          # NM baseline (s); the airport's median label for departures without an NM link
    offset: np.ndarray               # subtracted from the label: the NM baseline, or the unimpeded time where known (s)
    fit_rows: np.ndarray             # row numbers trained on: labelled 2025 normal-regime rows outside the held-out fold whose offset is known (a NaN offset makes the residual label NaN)
    evaluation_rows: np.ndarray      # row numbers scored: labelled normal-regime rows of the held-out fold (none for 'full')
    label_mean: float                # mean of the clipped residual label (label - offset) over the fit rows (s)
    label_sd: float                  # its standard deviation (s)
    standardised_label: np.ndarray   # (clipped residual label - label_mean) / label_sd


def _fit_target(args: argparse.Namespace, data: NetworkTrainingData, fold: str) -> _FitTarget:
    """
    This function works out what the network learns in one fold: the taxi time minus the offset, cut at plus or minus 6000 seconds, then standardized
    It trains on the normal 2025 flights outside the held-out weeks whose answer and offset are known, and tests on the normal flights of the held-out weeks
    """
    label = data.label
    frame = data.frame
    is_held_out = held_out_week_fold_rows(frame, fold) if fold != 'full' else np.zeros(len(label), bool)
    is_training = np.isfinite(label) & ~is_held_out & (frame.year.to_numpy() == 2025)
    nm_baseline = _nm_baseline_with_airport_fallback(data, is_training)
    if args.target == 'nm_baseline':
        offset = nm_baseline
    else:   # unimpeded time where known, the NM baseline otherwise
        assert data.unimpeded_time is not None
        offset = np.where(np.isfinite(data.unimpeded_time), data.unimpeded_time, nm_baseline)
    residual_label = np.clip(label - offset, -args.target_clip, args.target_clip)
    is_fit_row = is_training & data.is_normal_regime & np.isfinite(residual_label)
    is_evaluation_row = is_held_out & data.is_normal_regime & np.isfinite(label)
    label_mean = float(residual_label[is_fit_row].mean())
    label_sd = float(residual_label[is_fit_row].std())
    return _FitTarget(nm_baseline=nm_baseline, offset=offset, fit_rows=np.flatnonzero(is_fit_row), evaluation_rows=np.flatnonzero(is_evaluation_row), label_mean=label_mean, label_sd=label_sd, standardised_label=(residual_label - label_mean) / label_sd)


def _output_head_centres(args: argparse.Namespace, label_mean: float, label_sd: float, device: torch.device) -> tuple[Tensor | None, int]:
    """
    This function decides what the last layer gives out: one number for the regression head, or 128 bucket centers covering the whole clipped target range for the two-hot head
    """
    if args.output_head != 'twohot':
        return None, 1
    lowest_centre = (-args.target_clip - label_mean) / label_sd
    highest_centre = (args.target_clip - label_mean) / label_sd
    return twohot_bin_centres(lowest_centre, highest_centre, args.twohot_centres).to(device), args.twohot_centres


def _nm_baseline_with_airport_fallback(data: NetworkTrainingData, is_training: np.ndarray) -> np.ndarray:
    """
    This function gives every flight its NM baseline
    Flights without an NM link get the median taxi time of the normal, NM-linked training flights at their airport instead
    """
    is_linked_training_row = is_training & data.is_normal_regime & ~data.no_nm_link
    airport_median_label = pd.Series(data.label[is_linked_training_row]).groupby(data.airport[is_linked_training_row]).median()
    airport_fallback = data.frame.ADEP_mvt.map(airport_median_label).to_numpy(dtype=float)
    return np.where(data.no_nm_link, airport_fallback, data.frame.baseline.to_numpy())


def _extra_numeric_inputs(nm_baseline: np.ndarray, no_nm_link: np.ndarray) -> np.ndarray:
    """
    This function makes the two extra inputs every network gets: the NM baseline and a flag that is 1 when the flight has no NM link
    This way the network sees the rough guess it is correcting, and whether that guess is a real NM time or an airport median
    """
    return np.stack([np.clip((nm_baseline - NM_BASELINE_INPUT_CENTRE_SECONDS) / NM_BASELINE_INPUT_UNIT_SECONDS, -NM_BASELINE_INPUT_CLIP, NM_BASELINE_INPUT_CLIP), no_nm_link.astype(float)], 1).astype(np.float16)


def _piecewise_linear_bins(numeric_inputs: Tensor, fit_rows_on_device: Tensor, bin_count: int, rng: np.random.Generator, device: torch.device) -> list[Tensor]:
    """
    This function works out where to cut each number input into bins for the piecewise-linear embedding - quantile cuts on a random sample of at most 300,000 training rows
    """
    sample_size = min(len(fit_rows_on_device), BIN_FIT_SAMPLE_ROWS)
    bin_sample = torch.from_numpy(rng.choice(len(fit_rows_on_device), sample_size, replace=False)).to(device)
    return compute_bins(numeric_inputs[fit_rows_on_device[bin_sample]].float().cpu(), n_bins=bin_count)


def _placeholder_bins_of_stored_embedding(mask: Tensor) -> list[Tensor]:
    """
    This function makes dummy bin edges with the stored number of bins per input, so a stored fit can be rebuilt with exactly the right shape before its weights are loaded
    """
    return [torch.linspace(0, 1, int(mask[index].sum()) + 1) for index in range(mask.shape[0])]


@dataclass
class _DeviceInputs:
    """
    This data structure keeps all the network inputs on the GPU
    """
    numeric: Tensor                                          # (rows, numerical inputs) float16, the two extra inputs included
    categorical: Tensor                                      # (rows, categorical inputs) int64 codes
    neighbour_table: tuple[Tensor, Tensor, Tensor] | None    # neighbouring_takeoffs(): row index, time offset, padding mask
    runway_codes: Tensor | None                              # airport_runway code of every row

    def batch(self, rows: Tensor) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor, Tensor] | None]:
        """
        This function hands out the inputs for a batch of flights, with the neighbor information when it is switched on
        """
        numeric = self.numeric[rows].float()
        categorical = self.categorical[rows]
        context = self.neighbour_context(rows) if self.neighbour_table is not None else None
        return numeric, categorical, context

    def neighbour_context(self, rows: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        This function gathers the inputs of each flight's neighbors, plus 3 extras: the time gap, whether the neighbor used the same runway, and whether it took off before
        """
        neighbour_rows = self.neighbour_table[0][rows].long()
        time_offset = self.neighbour_table[1][rows]
        is_padding = self.neighbour_table[2][rows]
        same_runway = (self.runway_codes[neighbour_rows] == self.runway_codes[rows].unsqueeze(1)).float()
        is_before = (time_offset < 0).float()
        neighbour_extras = torch.stack([time_offset, same_runway, is_before], -1)
        return self.numeric[neighbour_rows].float(), self.categorical[neighbour_rows], neighbour_extras, is_padding


def _device_inputs(args: argparse.Namespace, data: NetworkTrainingData, numeric_inputs: Tensor, categorical_inputs: Tensor, device: torch.device) -> _DeviceInputs:
    """
    This function bundles the inputs already on the GPU - when neighbor attention is on, it also moves the neighbor tables there and picks out every flight's runway code
    """
    neighbour_table = None
    runway_codes = None
    if args.context_neighbours > 0:
        assert data.neighbour_table is not None and data.runway_column is not None
        neighbour_table = tuple(torch.from_numpy(table).to(device) for table in data.neighbour_table)
        runway_codes = categorical_inputs[:, data.runway_column]
    return _DeviceInputs(numeric=numeric_inputs, categorical=categorical_inputs, neighbour_table=neighbour_table, runway_codes=runway_codes)


def _build_optimizers(args: argparse.Namespace, model: TaxiTimeNetwork) -> list[torch.optim.Optimizer]:
    """
    This function builds the optimizers that adjust the network's weights after every batch
    """
    adamw_settings = dict(lr=args.learning_rate, weight_decay=args.weight_decay, betas=(ADAMW_BETA1, args.beta2))
    if args.optimizer == 'muon':
        backbone_matrices = [parameter for parameter_name, parameter in model.named_parameters() if parameter_name.startswith('backbone') and parameter.ndim == 2]
        backbone_matrix_ids = {id(parameter) for parameter in backbone_matrices}
        other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_matrix_ids]
        return [torch.optim.Muon(backbone_matrices, lr=args.muon_learning_rate, weight_decay=args.muon_weight_decay), torch.optim.AdamW(other_parameters, **adamw_settings)]
    if args.architecture == 'realmlp':
        scale_parameters = [model.backbone.scale]
        other_parameters = [parameter for parameter in model.parameters() if parameter is not model.backbone.scale]
        parameter_groups = [{'params': other_parameters}, {'params': scale_parameters, 'lr': REALMLP_SCALE_LR_FACTOR * args.learning_rate, 'weight_decay': 0.0}]
        return [torch.optim.AdamW(parameter_groups, **adamw_settings)]
    return [torch.optim.AdamW(model.parameters(), **adamw_settings)]


def _cosine_schedulers(optimizers: list[torch.optim.Optimizer], total_steps: int) -> list[torch.optim.lr_scheduler.LambdaLR]:
    """
    This function makes the learning rate start at full size and shrink smoothly to 0 by the last training step (half a cosine wave), one schedule per optimizer
    """
    return [torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps))) for optimizer in optimizers]


def _train_epochs(args: argparse.Namespace, model: TaxiTimeNetwork, inputs: _DeviceInputs, fit_rows_on_device: Tensor, fit_targets: Tensor, bin_centres: Tensor | None, optimizers: list[torch.optim.Optimizer], schedulers: list[torch.optim.lr_scheduler.LambdaLR], device: torch.device) -> None:
    """
    This function is the training loop
    """
    for _ in range(args.epochs):
        permutation = torch.randperm(len(fit_rows_on_device), device=device)
        for start in range(0, len(permutation), args.batch_size):
            batch_positions = permutation[start:start + args.batch_size]
            batch_rows = fit_rows_on_device[batch_positions]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(*inputs.batch(batch_rows)).float()
            loss = _batch_loss(prediction, fit_targets[batch_positions], bin_centres)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            for optimizer in optimizers:
                optimizer.step()
            for scheduler in schedulers:
                scheduler.step()


def _batch_loss(prediction: Tensor, targets: Tensor, bin_centres: Tensor | None) -> Tensor:
    """
    This function measures how wrong one batch of guesses is
    """
    if bin_centres is not None:
        target_distribution = twohot_target(targets, bin_centres).unsqueeze(1).expand(-1, prediction.shape[1], -1)
        row_loss = -(target_distribution * torch.log_softmax(prediction, -1)).sum(-1).mean(1)
    else:
        row_loss = ((prediction.squeeze(-1) - targets.unsqueeze(1)) ** 2).mean(1)
    return row_loss.mean()


@torch.no_grad()
def _predict_residual(model: TaxiTimeNetwork, inputs: _DeviceInputs, rows: np.ndarray, args: argparse.Namespace, bin_centres: Tensor | None, label_mean: float, label_sd: float, device: torch.device) -> np.ndarray:
    """
    This function uses a trained network to predict the correction (taxi time minus offset, in seconds) for the given rows
    """
    model.eval()
    batch_outputs = []
    batch_size = max(PREDICTION_BATCH_MIN_ROWS, min(PREDICTION_BATCH_MAX_ROWS, PREDICTION_BATCH_ROW_MEMBER_BUDGET // model.ensemble_size)) if args.context_neighbours == 0 else PREDICTION_BATCH_CONTEXT_ROWS
    rows_on_device = torch.from_numpy(rows).to(device)
    for start in range(0, len(rows_on_device), batch_size):
        batch_rows = rows_on_device[start:start + batch_size]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(*inputs.batch(batch_rows)).float()
        if bin_centres is not None:
            output = (torch.softmax(output, -1) * bin_centres).sum(-1)
        else:
            output = output.squeeze(-1)
        batch_outputs.append(output.mean(1))
    model.train()
    torch.cuda.empty_cache()
    return torch.cat(batch_outputs).cpu().numpy() * label_sd + label_mean


def _save_fit(args: argparse.Namespace, fold: str, seed: int, model: TaxiTimeNetwork, label_mean: float, label_sd: float, bin_centres: Tensor | None) -> None:
    """
    This function saves a trained network
    """
    weights_dir = MODEL_WEIGHTS / f'{args.name}_{fold}'
    weights_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
    centres = None if bin_centres is None else bin_centres.cpu()
    torch.save({'state_dict': state_dict, 'mu': label_mean, 'sd': label_sd, 'centres': centres}, weights_dir / f'seed{seed}.pt')


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train a neural network.')
    # rows and inputs
    parser.add_argument('--name', required=True)
    parser.add_argument('--folds', required=True)
    parser.add_argument('--seeds', default='1,2,3')
    parser.add_argument('--network-inputs', default='network_inputs_pass1')
    parser.add_argument('--feature-blocks', default='')
    parser.add_argument('--drop-column-prefixes', default='')
    parser.add_argument('--drop-columns', default='')
    # target and model
    parser.add_argument('--target', default='nm_baseline', choices=['nm_baseline', 'unimpeded'])
    parser.add_argument('--unimpeded-column', default='unimpeded_times:te_unimpeded')
    parser.add_argument('--architecture', default='resnet', choices=['resnet', 'tabm', 'realmlp'])
    parser.add_argument('--embedding', default='piecewise_linear', choices=['piecewise_linear', 'periodic', 'pbld'])
    parser.add_argument('--context-neighbours', type=int, default=0)
    parser.add_argument('--output-head', default='regression', choices=['regression', 'twohot'])
    parser.add_argument('--twohot-centres', type=int, default=DEFAULT_TWOHOT_CENTRES)
    # optimisation
    parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'muon'])
    parser.add_argument('--muon-learning-rate', type=float, default=DEFAULT_MUON_LEARNING_RATE)
    parser.add_argument('--muon-weight-decay', type=float, default=DEFAULT_MUON_WEIGHT_DECAY)
    parser.add_argument('--beta2', type=float, default=DEFAULT_ADAMW_BETA2)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--target-clip', type=float, default=DEFAULT_TARGET_CLIP_SECONDS)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE_ROWS)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--weight-decay', type=float, default=DEFAULT_WEIGHT_DECAY)
    # sizes
    parser.add_argument('--tabm-k', type=int, default=DEFAULT_TABM_K)
    parser.add_argument('--residual-blocks', type=int, default=DEFAULT_RESIDUAL_BLOCKS)
    parser.add_argument('--block-width', type=int, default=DEFAULT_BLOCK_WIDTH)
    parser.add_argument('--dropout', type=float, default=DEFAULT_DROPOUT)
    parser.add_argument('--embedding-dim', type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument('--embedding-bins', type=int, default=DEFAULT_EMBEDDING_BINS)
    parser.add_argument('--categorical-dim', type=int, default=DEFAULT_CATEGORICAL_DIM)
    # predicting with stored fits, other locations
    parser.add_argument('--predict-only', default='', metavar='WEIGHTS_ROOT')
    parser.add_argument('--block-cache', default=DEFAULT_BLOCK_CACHE)
    parser.add_argument('--output-root', default='')
    return parser


ATTEMPTS_PER_SEED = 3                                   # a diverged seed is retried as seed + 100, then seed + 200
SCORED_AIRPORTS = ('LIRF', 'LTFM', 'EGLL', 'LFPG')      # scored separately: Rome, Istanbul, London Heathrow, Paris CDG


def main() -> None:
    args = build_argument_parser().parse_args()
    fold_list = args.folds.split(',')
    seed_list = [int(seed) for seed in args.seeds.split(',')]
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    data = load_training_data(args)
    for fold in fold_list:
        seed_values, seed_rmses, minutes = _fit_fold_seeds(args, data, fold, seed_list, is_first_fold=fold == fold_list[0])
        _write_fold_run(args, data, fold, seed_list, seed_values, seed_rmses, minutes)


def _fit_fold_seeds(args: argparse.Namespace, data: NetworkTrainingData, fold: str, seed_list: list[int], is_first_fold: bool) -> tuple[list[np.ndarray], list[float | None], float]:
    """
    This function trains every seed of one fold and collects their predictions, printing one line per attempt
    """
    seed_values: list[np.ndarray] = []
    seed_rmses: list[float | None] = []
    minutes = 0.0
    for seed in seed_list:
        for attempt in range(ATTEMPTS_PER_SEED):
            attempt_seed = seed + RETRY_SEED_STEP * attempt
            is_last_attempt = attempt == ATTEMPTS_PER_SEED - 1
            saved_weights = None
            if args.predict_only:            # the stored fit of this seed (or of the seed that replaced a diverged one)
                saved_weights = Path(args.predict_only) / f'{args.name}_{fold}' / f'seed{attempt_seed}.pt'
                if not saved_weights.exists():
                    if not is_last_attempt:
                        continue
                    raise FileNotFoundError(f'no stored fit for {args.name}_{fold} seed {seed} under {args.predict_only}')
            print_summary = attempt_seed == seed_list[0] and is_first_fold
            value, last_rmse, fit_minutes = train_one_network(args, data, fold, attempt_seed, print_summary=print_summary, saved_weights=saved_weights)
            minutes += fit_minutes
            rounded_rmse = last_rmse if last_rmse is None else round(last_rmse, 2)
            print(f'  {args.name} {fold} seed {attempt_seed}: last-epoch RMSE {rounded_rmse} ({fit_minutes:.1f} min)', flush=True)
            if np.isfinite(value).all():
                seed_values.append(value)
                seed_rmses.append(last_rmse)
                break
            what_next = ', dropped' if is_last_attempt else ', retrying'
            print(f'  {args.name} {fold} seed {attempt_seed} diverged (NaN predictions)' + what_next, flush=True)
    if not seed_values:
        seed_values = [np.full(len(data.frame), np.nan)]
        seed_rmses = [float('nan')]
    return seed_values, seed_rmses, minutes


def _write_fold_run(args: argparse.Namespace, data: NetworkTrainingData, fold: str, seed_list: list[int], seed_values: list[np.ndarray], seed_rmses: list[float | None], minutes: float) -> None:
    """
    This function writes the run folder of one fold
    """
    value = np.mean(seed_values, 0)
    output_root = Path(args.output_root) if args.output_root else RUN_PREDICTIONS
    run_dir = output_root / f'{args.name}_{fold}'
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'MVT_ID_mvt': data.frame.MVT_ID_mvt, 'v_nn': value}).to_parquet(run_dir / 'pred.parquet', index=False)
    # metrics.json keeps the key names of the stored runs; 'month' holds the fold (wk0..wk3, or 'full'; 'none' in older runs).
    result: dict[str, object] = dict(name=args.name, month=fold, seeds=seed_list, seed_rmse=seed_rmses, minutes=round(minutes, 1), args={key: setting for key, setting in vars(args).items() if key != 'name'})
    if fold != 'full':
        is_scored = held_out_week_fold_rows(data.frame, fold) & data.is_normal_regime & np.isfinite(data.label)
        error = value - data.label
        result['ens_rmse'] = rmse(error[is_scored])
        result['seed_mean_rmse'] = float(np.mean(seed_rmses))
        result['groups'] = _group_rmses(error, is_scored, data.airport, data.no_nm_link)
        rounded_seed_rmses = [round(seed_rmse, 2) for seed_rmse in seed_rmses]
        print(f'== {args.name} {fold}: seeds {rounded_seed_rmses} mean {result["seed_mean_rmse"]:.2f} | mean of seeds {result["ens_rmse"]:.2f} | {result["groups"]}', flush=True)
    (run_dir / 'metrics.json').write_text(json.dumps(result, indent=1))


def _group_rmses(error: np.ndarray, is_scored: np.ndarray, airport: np.ndarray, no_nm_link: np.ndarray) -> dict[str, float]:
    """
    This function works out the RMSE of the scored flights separately for Rome, Istanbul, London Heathrow and Paris CDG, for all other airports together, and for the flights without an NM link
    """
    group_rows = {airport_code: airport == airport_code for airport_code in SCORED_AIRPORTS}
    group_rows['other'] = ~np.isin(airport, list(SCORED_AIRPORTS))
    group_rows['no-link'] = no_nm_link
    return {group: round(rmse(error[is_scored & in_group]), 2) for group, in_group in group_rows.items()}


if __name__ == '__main__':
    main()
