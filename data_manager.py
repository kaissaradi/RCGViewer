import numpy as np
import pandas as pd
from pathlib import Path
from qtpy.QtCore import QObject

# It's good practice to place utility imports at the top.
# The original gui.py had a try-except block for this import, 
# which is better handled at the application's entry point.
import cleaning_utils_cpu 

class DataManager(QObject):
    """
    Manages all data loading, processing, and caching.
    """
    is_dirty = False

    def __init__(self, kilosort_dir):
        super().__init__()
        self.kilosort_dir = Path(kilosort_dir)
        self.ei_cache = {}
        self.heavyweight_cache = {}
        self.dat_path = None
        self.cluster_df = pd.DataFrame()
        self.original_cluster_df = pd.DataFrame()
        self.info_path = None
        self.uV_per_bit = 0.195

    def load_kilosort_data(self):
        try:
            self.spike_times = np.load(self.kilosort_dir / 'spike_times.npy').flatten()
            self.spike_clusters = np.load(self.kilosort_dir / 'spike_clusters.npy').flatten()
            self.channel_positions = np.load(self.kilosort_dir / 'channel_positions.npy')
            info_path = self.kilosort_dir / 'cluster_info.tsv'
            group_path = self.kilosort_dir / 'cluster_group.tsv'
            if info_path.exists():
                self.info_path = info_path
                self.cluster_info = pd.read_csv(info_path, sep='\t')
            elif group_path.exists():
                self.info_path = group_path
                self.cluster_info = pd.read_csv(group_path, sep='\t')
            else:
                raise FileNotFoundError("'cluster_info.tsv' or 'cluster_group.tsv' not found.")
            self._load_kilosort_params()
            return True, "Successfully loaded Kilosort data."
        except Exception as e:
            return False, f"Error during Kilosort data loading: {e}"

    def _load_kilosort_params(self):
        params_path = self.kilosort_dir / 'params.py'
        if not params_path.exists(): raise FileNotFoundError("params.py not found.")
        params = {}
        with open(params_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, val = map(str.strip, line.split('=', 1))
                    try: params[key] = eval(val)
                    except (NameError, SyntaxError): params[key] = val.strip("'\"")
        self.sampling_rate = params.get('fs', 30000)
        self.n_channels = params.get('n_channels_dat', 512)
        dat_path_str = params.get('dat_path', '')
        if isinstance(dat_path_str, (list, tuple)) and dat_path_str: dat_path_str = dat_path_str[0]
        suggested_path = Path(dat_path_str)
        if not suggested_path.is_absolute():
            self.dat_path_suggestion = self.kilosort_dir.parent / suggested_path
        else:
            self.dat_path_suggestion = suggested_path

    def build_cluster_dataframe(self):
        cluster_ids, n_spikes = np.unique(self.spike_clusters, return_counts=True)
        df = pd.DataFrame({'cluster_id': cluster_ids, 'n_spikes': n_spikes})
        isi_rates_dict = self._calculate_all_isi_violations_vectorized()
        df['isi_violations_pct'] = df['cluster_id'].map(isi_rates_dict).fillna(0)
        col = 'KSLabel' if 'KSLabel' in self.cluster_info.columns else 'group'
        if col not in self.cluster_info.columns: self.cluster_info[col] = 'unsorted'
        info_subset = self.cluster_info[['cluster_id', col]].rename(columns={col: 'KSLabel'})
        df = pd.merge(df, info_subset, on='cluster_id', how='left')
        df['status'] = 'Original'
        self.cluster_df = df[['cluster_id', 'KSLabel', 'n_spikes', 'isi_violations_pct', 'status']]
        self.original_cluster_df = self.cluster_df.copy()

    def _calculate_all_isi_violations_vectorized(self, refractory_period_ms=2.0):
        rates_dict = {}
        refractory_period_samples = (refractory_period_ms / 1000.0) * self.sampling_rate
        for cid in np.unique(self.spike_clusters):
            cluster_spikes = np.sort(self.spike_times[self.spike_clusters == cid])
            if len(cluster_spikes) < 2:
                rates_dict[cid] = 0.0
                continue
            isis = np.diff(cluster_spikes)
            violations = np.sum(isis < refractory_period_samples)
            rates_dict[cid] = (violations / (len(cluster_spikes) - 1)) * 100
        return rates_dict

    def get_cluster_spikes(self, cluster_id):
        return self.spike_times[self.spike_clusters == cluster_id]

    def get_lightweight_features(self, cluster_id, max_spikes_for_vis=100, n_raw_snippets=30):
        if cluster_id in self.ei_cache: return self.ei_cache[cluster_id]
        spike_times_cluster = self.get_cluster_spikes(cluster_id)
        if len(spike_times_cluster) == 0: return None
        sample_size = min(len(spike_times_cluster), max_spikes_for_vis)
        spike_sample = np.random.choice(spike_times_cluster, size=sample_size, replace=False)
        snippets_raw = cleaning_utils_cpu.extract_snippets(str(self.dat_path), spike_sample, n_channels=self.n_channels)
        snippets_uV = snippets_raw.astype(np.float32) * self.uV_per_bit
        if snippets_uV.shape[2] == 0: return None
        pre_samples = 20
        snippets_bc = cleaning_utils_cpu.baseline_correct(snippets_uV, pre_samples=pre_samples)
        mean_ei = cleaning_utils_cpu.compute_ei(snippets_bc, pre_samples=pre_samples)
        features = {'mean_ei': mean_ei, 'raw_snippets': snippets_bc[:, :, :min(n_raw_snippets, snippets_bc.shape[2])]}
        self.ei_cache[cluster_id] = features
        return features

    def get_heavyweight_features(self, cluster_id):
        if cluster_id in self.heavyweight_cache: return self.heavyweight_cache[cluster_id]
        lightweight_data = self.get_lightweight_features(cluster_id)
        if not lightweight_data: return None
        features = cleaning_utils_cpu.compute_spatial_features(
            lightweight_data['mean_ei'], self.channel_positions, self.sampling_rate)
        self.heavyweight_cache[cluster_id] = features
        return features
        
    def update_after_refinement(self, parent_id, new_clusters_data):
        self.is_dirty = True
        parent_indices = np.where(self.spike_clusters == parent_id)[0]
        self.cluster_df.loc[self.cluster_df['cluster_id'] == parent_id, 'status'] = 'Refined (Parent)'
        max_id = self.spike_clusters.max()
        new_rows = []
        for i, new_cluster in enumerate(new_clusters_data):
            new_id = max_id + 1 + i
            sub_indices = parent_indices[new_cluster['inds']]
            self.spike_clusters[sub_indices] = new_id
            isi_violations = self._calculate_isi_violations(new_id)
            new_row = {
                'cluster_id': new_id, 'KSLabel': 'good', 'n_spikes': len(sub_indices),
                'isi_violations_pct': isi_violations, 'status': f'Refined (from C{parent_id})'
            }
            new_rows.append(new_row)
        self.cluster_df = pd.concat([self.cluster_df, pd.DataFrame(new_rows)], ignore_index=True)

    def _calculate_isi_violations(self, cluster_id, refractory_period_ms=2.0):
        spike_times_cluster = self.get_cluster_spikes(cluster_id)
        if len(spike_times_cluster) < 2: return 0.0
        isis = np.diff(np.sort(spike_times_cluster))
        refractory_period_samples = (refractory_period_ms / 1000.0) * self.sampling_rate
        violations = np.sum(isis < refractory_period_samples)
        return (violations / (len(spike_times_cluster) - 1)) * 100
