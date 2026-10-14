"""
This file is the long to-do list that says which neural networks to train
Each job is one call of neural_network.py with its own name, folds, seeds, feature blocks and extra settings - 160 jobs in total, which make 240 runs

This is how it works:
-76 jobs are full fits trained on all of 2025 - their predictions are used on the ranking rows (the flights the challenge asks to predict)
-the other 84 jobs are week-fold practice runs of 41 recipes:
    -the weeks of 2025 are split into four folds, wk0 to wk3, by what is left over when the ISO week number is divided by 4
    -each run hides one fold, learns from the rest and predicts the hidden weeks
    -most recipes are split into two jobs with the same name (folds wk1,wk3 first, then wk0,wk2)
    -these predictions on hidden weeks train the second stages, which learn how to correct the mix of networks
-every job leaves out the calendar inputs (month, day, day of year)
"""
from __future__ import annotations

from typing import NamedTuple

BASE_BLOCKS = 'neighbours,takeoff_sequence:!sq_rwy_since_arr+sq_rwy_to_arr+sq_rwy_since_any'
NM_NEIGHBOURS_AND_SEQUENCE = ('neighbours_adsb:nb_rwy_nmb+nb_apt_nmb+nb_rwy_delay+nb_apt_delay,' 'takeoff_sequence:!sq_rwy_since_arr+sq_rwy_to_arr+sq_rwy_since_any')
UNIMPEDED_TIMES = 'unimpeded_times:te_unimpeded+te_stand_rwy'
WINTER_WEATHER = 'metar_weather:wx_sn+wx_fz+wx_icing+wx_icing_3h+wx_sn_3h+wx_icing_6h+wx_sn_6h+wx_icing_day+wx_sn_day+wx_temp'

# combinations used by the jobs below
BASE_AND_UNIMPEDED = BASE_BLOCKS + ',' + UNIMPEDED_TIMES
BASE_UNIMPEDED_AND_WINTER = BASE_BLOCKS + ',' + UNIMPEDED_TIMES + ',' + WINTER_WEATHER
BASE_AND_WINTER = BASE_BLOCKS + ',' + WINTER_WEATHER
BASE_AND_ORDERING = BASE_BLOCKS + ',' + 'ordering'
BASE_AND_WIDE_WINDOWS = BASE_BLOCKS + ',' + 'neighbours_wide'
NM_NEIGHBOURS_SEQUENCE_AND_UNIMPEDED = NM_NEIGHBOURS_AND_SEQUENCE + ',' + UNIMPEDED_TIMES


class NetworkJob(NamedTuple):
    """
    This data structure is one line of the to-do list
    """
    name: str            # run base name: the runs <name>_<fold>
    folds: str           # comma-separated held-out week folds (wk0..wk3), or full
    seeds: str           # comma-separated seeds; the saved prediction is their mean
    feature_blocks: str  # --feature-blocks (see above)
    options: str = ''    # the other arguments, as on the command line

    def arguments(self) -> list[str]:
        return ['--name', self.name, '--folds', self.folds, '--seeds', self.seeds, '--feature-blocks', self.feature_blocks, '--drop-columns', 'month,day,dayofyear', *self.options.split()]


def week_fold_pair(name: str, seeds: str, feature_blocks: str, options: str) -> list[NetworkJob]:
    """
    This function makes the two jobs of one week-fold recipe - first hiding folds wk1 and wk3, then wk0 and wk2 - which together give the four runs <name>_wk0 to <name>_wk3
    """
    return [NetworkJob(name, 'wk1,wk3', seeds, feature_blocks, options), NetworkJob(name, 'wk0,wk2', seeds, feature_blocks, options)]


# where the numbers in the options come from:
#   --target-clip 15000.0: did better than the default 6000 s on both 2025 week folds it was checked on
#   --context-neighbours 16 / 32: 8 or 16 take-offs before and after the flight; both sizes are kept as variants
#   --epochs 24: the two-hot networks trained twice as long; the 24-epoch versions were picked by the greedy selection on the 2025 week folds
#   --tabm-k 16 --residual-blocks 2 --block-width 512: TabM's default shape with embeddings (2 blocks of 512), with 16 members because 32 ran out of memory on the 8 GB GPU
#   --beta2 0.95 --dropout 0.15 --embedding-dim 4: RealMLP defaults (pytabkit)
#   --learning-rate 0.001: half the default rate, trained as a variant
NETWORK_JOBS = [
    # full fits (all of 2025): their predictions are used on the ranking rows
    NetworkJob('network_pass3_context16', 'full', '1,2', BASE_BLOCKS, '--context-neighbours 16 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_muon', 'full', '1,2', BASE_BLOCKS, '--optimizer muon --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_nm_neighbours', 'full', '1,2', NM_NEIGHBOURS_AND_SEQUENCE, '--network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_periodic', 'full', '1,2', BASE_BLOCKS, '--embedding periodic --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3', 'full', '1,2', BASE_BLOCKS, '--network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_twohot_epochs24', 'full', '1,2', BASE_BLOCKS, '--output-head twohot --epochs 24 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_unimpeded_context16', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --context-neighbours 16 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_unimpeded_context32', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --context-neighbours 32 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_unimpeded_muon', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --optimizer muon --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_unimpeded_periodic_muon', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --optimizer muon --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_unimpeded_periodic_context16', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --context-neighbours 16 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_unimpeded_periodic', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_unimpeded_twohot_epochs24', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --output-head twohot --epochs 24 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_unimpeded_winter_weather', 'full', '1,2', BASE_UNIMPEDED_AND_WINTER, '--target unimpeded --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_context16_seeds34', 'full', '3,4', BASE_BLOCKS, '--context-neighbours 16 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_context16', 'full', '1,2', BASE_BLOCKS, '--context-neighbours 16 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_muon', 'full', '1,2', BASE_BLOCKS, '--optimizer muon --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_muon_seeds34', 'full', '3,4', BASE_BLOCKS, '--optimizer muon --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_nm_neighbours', 'full', '1,2', NM_NEIGHBOURS_AND_SEQUENCE, '--target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_nm_neighbours_seeds34', 'full', '3,4', NM_NEIGHBOURS_AND_SEQUENCE, '--target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_periodic', 'full', '1,2', BASE_BLOCKS, '--embedding periodic --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_periodic_seeds34', 'full', '3,4', BASE_BLOCKS, '--embedding periodic --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000', 'full', '1,2', BASE_BLOCKS, '--target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_twohot_epochs24', 'full', '1,2', BASE_BLOCKS, '--output-head twohot --epochs 24 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_twohot_epochs24_seeds34', 'full', '3,4', BASE_BLOCKS, '--output-head twohot --epochs 24 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_context16', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --context-neighbours 16 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_context16_seeds34', 'full', '3,4', BASE_AND_UNIMPEDED, '--target unimpeded --context-neighbours 16 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_context32', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --context-neighbours 32 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_context32_seeds34', 'full', '3,4', BASE_AND_UNIMPEDED, '--target unimpeded --context-neighbours 32 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_muon', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --optimizer muon --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_periodic_muon', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --optimizer muon --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_periodic_muon_seeds34', 'full', '3,4', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --optimizer muon --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_muon_seeds34', 'full', '3,4', BASE_AND_UNIMPEDED, '--target unimpeded --optimizer muon --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_periodic_context16_seeds34', 'full', '3,4', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --context-neighbours 16 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_periodic_context16', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --context-neighbours 16 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_periodic', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_periodic_seeds34', 'full', '3,4', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_twohot_epochs24', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --output-head twohot --epochs 24 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_twohot_epochs24_seeds34', 'full', '3,4', BASE_AND_UNIMPEDED, '--target unimpeded --output-head twohot --epochs 24 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_winter_weather', 'full', '1,2', BASE_UNIMPEDED_AND_WINTER, '--target unimpeded --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass3_clip15000_unimpeded_winter_weather_seeds34', 'full', '3,4', BASE_UNIMPEDED_AND_WINTER, '--target unimpeded --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    NetworkJob('network_pass2_context16_seed1', 'full', '1', BASE_BLOCKS, '--context-neighbours 16 --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_context16_seed2', 'full', '2', BASE_BLOCKS, '--context-neighbours 16 --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_context16_seed3', 'full', '3', BASE_BLOCKS, '--context-neighbours 16 --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_lr0.001', 'full', '1,2', BASE_BLOCKS, '--learning-rate 0.001 --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_muon_seeds123', 'full', '1,2,3', BASE_BLOCKS, '--optimizer muon --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_nm_neighbours_seeds123', 'full', '1,2,3', NM_NEIGHBOURS_AND_SEQUENCE, '--network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_without_adsb', 'full', '1,2', BASE_BLOCKS, '--drop-column-prefixes ads_ --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_ordering', 'full', '1,2', BASE_AND_ORDERING, '--network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_realmlp_pbld', 'full', '1,2', BASE_BLOCKS, '--architecture realmlp --embedding pbld --beta2 0.95 --block-width 512 --dropout 0.15 --embedding-dim 4 --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_periodic', 'full', '1,2', BASE_BLOCKS, '--embedding periodic --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_periodic_seeds34', 'full', '3,4', BASE_BLOCKS, '--embedding periodic --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_seeds123', 'full', '1,2,3', BASE_BLOCKS, '--network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_tabm16_seed1', 'full', '1', BASE_BLOCKS, '--architecture tabm --tabm-k 16 --residual-blocks 2 --block-width 512 --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_tabm16_seed2', 'full', '2', BASE_BLOCKS, '--architecture tabm --tabm-k 16 --residual-blocks 2 --block-width 512 --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_twohot_epochs24', 'full', '1,2', BASE_BLOCKS, '--output-head twohot --epochs 24 --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_twohot', 'full', '1,2', BASE_BLOCKS, '--output-head twohot --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_context16_seed1', 'full', '1', BASE_AND_UNIMPEDED, '--target unimpeded --context-neighbours 16 --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_context16_seed2', 'full', '2', BASE_AND_UNIMPEDED, '--target unimpeded --context-neighbours 16 --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_context16_seed3', 'full', '3', BASE_AND_UNIMPEDED, '--target unimpeded --context-neighbours 16 --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_context32', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --context-neighbours 32 --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_muon', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --optimizer muon --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_periodic_muon', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --optimizer muon --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_muon_seeds34', 'full', '3,4', BASE_AND_UNIMPEDED, '--target unimpeded --optimizer muon --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_nm_neighbours', 'full', '1,2', NM_NEIGHBOURS_SEQUENCE_AND_UNIMPEDED, '--target unimpeded --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_without_adsb', 'full', '1,2', BASE_AND_UNIMPEDED, '--drop-column-prefixes ads_ --target unimpeded --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_periodic_context16', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --context-neighbours 16 --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_periodic', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_periodic_seeds34', 'full', '3,4', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_twohot_epochs24', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --output-head twohot --epochs 24 --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_twohot', 'full', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --output-head twohot --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_winter_weather', 'full', '1,2', BASE_UNIMPEDED_AND_WINTER, '--target unimpeded --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_seeds123', 'full', '1,2,3', BASE_AND_UNIMPEDED, '--target unimpeded --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_unimpeded_seeds45', 'full', '4,5', BASE_AND_UNIMPEDED, '--target unimpeded --network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_wide_windows', 'full', '1,2', BASE_AND_WIDE_WINDOWS, '--network-inputs network_inputs_pass2'),
    NetworkJob('network_pass2_winter_weather', 'full', '1,2', BASE_AND_WINTER, '--network-inputs network_inputs_pass2'),
    # week-fold runs (four held-out folds): their held-out predictions train the second stages
    *week_fold_pair('network_pass3_clip15000_context16', '1,2', BASE_BLOCKS, '--context-neighbours 16 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    *week_fold_pair('network_pass3_clip15000_muon', '1,2', BASE_BLOCKS, '--optimizer muon --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    *week_fold_pair('network_pass3_clip15000_nm_neighbours', '1,2', NM_NEIGHBOURS_AND_SEQUENCE, '--target-clip 15000.0 --network-inputs network_inputs_pass3'),
    *week_fold_pair('network_pass3_clip15000_periodic', '1,2', BASE_BLOCKS, '--embedding periodic --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    *week_fold_pair('network_pass3_clip15000_twohot_epochs24', '1,2', BASE_BLOCKS, '--output-head twohot --epochs 24 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    *week_fold_pair('network_pass3_clip15000_unimpeded_context16_seeds34', '3,4', BASE_AND_UNIMPEDED, '--target unimpeded --context-neighbours 16 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    *week_fold_pair('network_pass3_clip15000_unimpeded_context32', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --context-neighbours 32 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    *week_fold_pair('network_pass3_clip15000_unimpeded_periodic_muon', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --optimizer muon --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    *week_fold_pair('network_pass3_clip15000_unimpeded_muon', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --optimizer muon --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    *week_fold_pair('network_pass3_clip15000_unimpeded_periodic_context16', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --context-neighbours 16 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    *week_fold_pair('network_pass3_clip15000_unimpeded_periodic', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    *week_fold_pair('network_pass3_clip15000_unimpeded_twohot_epochs24', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --output-head twohot --epochs 24 --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    *week_fold_pair('network_pass3_clip15000_unimpeded_winter_weather', '1,2', BASE_UNIMPEDED_AND_WINTER, '--target unimpeded --target-clip 15000.0 --network-inputs network_inputs_pass3'),
    *week_fold_pair('network_pass1_context16_seed1', '1', BASE_BLOCKS, '--context-neighbours 16 --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_context16_seed2', '2', BASE_BLOCKS, '--context-neighbours 16 --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_lr0.001', '1,2', BASE_BLOCKS, '--learning-rate 0.001 --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_muon', '1,2', BASE_BLOCKS, '--optimizer muon --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_nm_neighbours', '1,2', NM_NEIGHBOURS_AND_SEQUENCE, '--network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_without_adsb', '1,2', BASE_BLOCKS, '--drop-column-prefixes ads_ --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_ordering', '1,2', BASE_AND_ORDERING, '--network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_periodic', '1,2', BASE_BLOCKS, '--embedding periodic --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_realmlp_pbld', '1,2', BASE_BLOCKS, '--architecture realmlp --embedding pbld --beta2 0.95 --block-width 512 --dropout 0.15 --embedding-dim 4 --network-inputs network_inputs_pass1'),
    NetworkJob('network_pass1_tabm16_seed1', 'wk0', '1', BASE_BLOCKS, '--architecture tabm --tabm-k 16 --residual-blocks 2 --block-width 512 --network-inputs network_inputs_pass1'),
    NetworkJob('network_pass1_tabm16_seed1', 'wk1', '1', BASE_BLOCKS, '--architecture tabm --tabm-k 16 --residual-blocks 2 --block-width 512 --network-inputs network_inputs_pass1'),
    NetworkJob('network_pass1_tabm16_seed1', 'wk2', '1', BASE_BLOCKS, '--architecture tabm --tabm-k 16 --residual-blocks 2 --block-width 512 --network-inputs network_inputs_pass1'),
    NetworkJob('network_pass1_tabm16_seed1', 'wk3', '1', BASE_BLOCKS, '--architecture tabm --tabm-k 16 --residual-blocks 2 --block-width 512 --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_twohot_epochs24', '1,2', BASE_BLOCKS, '--output-head twohot --epochs 24 --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_twohot', '1,2', BASE_BLOCKS, '--output-head twohot --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_unimpeded_context16_seed1', '1', BASE_AND_UNIMPEDED, '--target unimpeded --context-neighbours 16 --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_unimpeded_context32', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --context-neighbours 32 --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_unimpeded_periodic_muon', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --optimizer muon --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_unimpeded_muon', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --optimizer muon --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_unimpeded_nm_neighbours', '1,2', NM_NEIGHBOURS_SEQUENCE_AND_UNIMPEDED, '--target unimpeded --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_unimpeded_without_adsb', '1,2', BASE_AND_UNIMPEDED, '--drop-column-prefixes ads_ --target unimpeded --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_unimpeded_periodic_context16_seed1', '1', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --context-neighbours 16 --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_unimpeded_periodic', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --embedding periodic --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_unimpeded_twohot_epochs24', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --output-head twohot --epochs 24 --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_unimpeded_twohot', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --output-head twohot --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_unimpeded_winter_weather', '1,2', BASE_UNIMPEDED_AND_WINTER, '--target unimpeded --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_unimpeded', '1,2', BASE_AND_UNIMPEDED, '--target unimpeded --network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass3_clip15000', '1,2', BASE_BLOCKS, '--target-clip 15000.0 --network-inputs network_inputs_pass3'),
    *week_fold_pair('network_pass1', '1,2', BASE_BLOCKS, '--network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_wide_windows', '1,2', BASE_AND_WIDE_WINDOWS, '--network-inputs network_inputs_pass1'),
    *week_fold_pair('network_pass1_winter_weather', '1,2', BASE_AND_WINTER, '--network-inputs network_inputs_pass1'),
]
