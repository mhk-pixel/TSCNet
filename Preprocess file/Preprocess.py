# Author: Majid Hameed Khalaf
# Date: 2026 
#=============================
import os
import mne
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel import label_components
from asrpy import ASR
from pyprep.prep_pipeline import PrepPipeline
from scipy.stats import kurtosis, rankdata

class EEGPreprocessor:
    def __init__(
        self,
        epoch_duration=5.0,
        epoch_overlap=0.0,
        resample_freq=128,
        l_freq=0.5,
        h_freq=45.0,
        notch_freq=50.0,
        asr_cutoff=20,
        iclabel_threshold=0.90,
        random_state=42,
        use_pyprep=False,
        flat_std_thresh=1.5e-6,  
        verbose=False,
    ):
        self.epoch_duration = epoch_duration
        self.epoch_overlap = epoch_overlap
        self.resample_freq = resample_freq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.asr_cutoff = asr_cutoff
        self.iclabel_threshold = iclabel_threshold
        self.random_state = random_state
        self.use_pyprep = use_pyprep
        self.flat_std_thresh = flat_std_thresh
        self.verbose = verbose

        self.standard_19 = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3',
            'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'
        ]

        self.qc_report = {}
        mne.set_log_level("INFO" if verbose else "ERROR")
   
    ######################################################
    def _find_clean_baseline(self, raw_obj, duration=30.0):
      
        sfreq = raw_obj.info['sfreq']
        window_size = int(duration * sfreq)
        step_size = int(2 * sfreq) 
        
        data = raw_obj.get_data(picks="eeg")
        n_channels, n_times = data.shape
        
        if n_times < window_size:
            print("[Warning] Data is shorter than the requested duration.")
            return 0.0, n_times / sfreq

        valid_windows = []
        var_vals = []
        amp_vals = []
        kurt_vals = []

        for start_idx in range(0, n_times - window_size, step_size):
            window_data = data[:, start_idx:start_idx + window_size]
            
            stds = np.std(window_data, axis=1)
            if np.any(stds < self.flat_std_thresh): 
                continue
                
            window_var = np.mean(np.var(window_data, axis=1))
        
            window_amp = np.mean(np.max(np.abs(window_data), axis=1))
            
            kurt_val = np.mean(np.abs(kurtosis(window_data, axis=1, fisher=False) - 3.0))
            
            valid_windows.append(start_idx)
            var_vals.append(window_var)
            amp_vals.append(window_amp)
            kurt_vals.append(kurt_val)

        if not valid_windows:
            print("[Warning] No perfectly clean window found. Falling back to the beginning.")
            return 0.0, min(duration, raw_obj.times[-1])
            
        rank_var = rankdata(var_vals)
        rank_amp = rankdata(amp_vals)
        rank_kurt = rankdata(kurt_vals)
        
        total_rank = rank_var + rank_amp + rank_kurt
        
        target_rank = np.percentile(total_rank, 25)
        best_idx = np.argmin(np.abs(total_rank - target_rank))
        
        best_start_idx = valid_windows[best_idx]
        best_tmin = best_start_idx / sfreq
            
        print(f"[Success] Optimal ASR baseline selected at 25th percentile: {best_tmin:.2f}s to {best_tmin + duration:.2f}s")
        
        return best_tmin-1, best_tmin + duration

    ######################################################################
    def _run_pyprep(self, raw):
        """
        Perform robust referencing and bad-channel detection using PyPREP.
        """
        if not self.use_pyprep:
            print("-> PyPREP disabled.")
            self.qc_report["bad_channels"] = []
            return raw

        print("-> Running PyPREP...")
        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": [self.notch_freq],
        }

        try:
            prep = PrepPipeline(
                raw,
                prep_params=prep_params,
                montage=raw.get_montage(),
                random_state=self.random_state,
                ransac=True,
            )
            prep.fit()
            raw = prep.raw
            bads = prep.interpolated_channels
            
            self.qc_report["bad_channels"] = bads
            self.qc_report["n_bad_channels"] = len(bads)

            if len(bads):
                print(f"-> Bad channels detected ({len(bads)}): {bads}")
            else:
                print("-> No bad channels detected.")
        except Exception as e:
            print(f"PyPrep failed: {e}. Proceeding without PyPrep.")
            self.qc_report["bad_channels"] = []

        return raw
    #######################################################
    def _run_asr(self, raw):
        """
        Artifact Subspace Reconstruction (ASR)
        """
        print("-> Selecting optimal ASR calibration window...")
        tmin, tmax = self._find_clean_baseline(raw, duration=30.0)
       
        baseline = raw.copy().crop(
        tmin=tmin,
        tmax=tmax #tmin + 30 - 1/raw.info["sfreq"]
        )
        print(baseline.get_data().shape)
        variance_before = np.var(raw.get_data())

        X = baseline.get_data()

        print("---------------------")
        print("Shape:", X.shape)
        print("dtype:", X.dtype)
        print("NaN:", np.isnan(X).sum())
        print("Inf:", np.isinf(X).sum())
        print("Samples:", baseline.n_times)
        print("---------------------")
        
        asr = ASR(sfreq=raw.info["sfreq"], cutoff=self.asr_cutoff)
        asr.fit(baseline)
        raw_clean = asr.transform(raw)

        variance_after = np.var(raw_clean.get_data())

        self.qc_report["variance_before_asr"] = variance_before
        self.qc_report["variance_after_asr"] = variance_after
        self.qc_report["variance_ratio"] = variance_after / variance_before
        ratio = variance_after / max(variance_before,1e-12)
        print(f"-> Variance ratio : {ratio :.3f}")
        return raw_clean

    ##############################################################
    def _run_ica(self, raw):
        """
        ICA + ICLabel
        """
        print("-> Fitting ICA...")
        rank = mne.compute_rank(raw)
       
        n_components = min(
        rank.get("eeg", len(raw.ch_names)),
        len(raw.ch_names)
        )
        ica = ICA(
        n_components,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=self.random_state
        )
        ica.fit(raw, verbose=False)

        print("-> Running ICLabel...")
        labels = label_components(raw, ica, method="iclabel")
        classes = labels["labels"]
        prob = labels["y_pred_proba"]
        exclude = []
        artifact_classes = {"eye blink", "muscle artifact", "heart beat" ,"channel noise"}

        print("\n" + "="*60)
        print("ICLabel Classification")
        print("="*60)
        
        for i, (c, p) in enumerate(zip(classes, prob)):
            print(f"IC {i:02d} | {c:18s} | {p:.3f}")
            if c in artifact_classes and p >= self.iclabel_threshold:
                exclude.append(i)
                
        print("="*60)
        print(f"Removing {len(exclude)} ICs")

        raw = ica.apply(raw.copy(), exclude=exclude, verbose=False)
        self.qc_report["removed_ic"] = len(exclude)
        return raw
        
    #######################################################################   
    def _harmonize_channels(self, raw):
        common = [ch for ch in self.standard_19 if ch in raw.ch_names]
        print(f"Harmonized channels count: {len(common)}/19") 
        
        raw.pick(common)
        raw.reorder_channels(common)
        return raw

    ##########################################################################
    def harmonize_nearest_neighbor(self, raw, target_montage):
        
        if raw.get_montage() is None:
            raw = raw.copy().set_montage("standard_1020", on_missing="ignore")
    
        raw_pos = raw.get_montage().get_positions()["ch_pos"]
        target_pos = target_montage.get_positions()["ch_pos"]
    
        target_chs = [ch for ch in target_montage.ch_names if ch in target_pos]
    
        src_data = raw.get_data()
        new_data = []
    
        for tgt_ch in target_chs:
            t_coord = target_pos[tgt_ch]
            closest_ch = min(
                raw.ch_names,
                key=lambda src_ch: np.linalg.norm(raw_pos[src_ch] - t_coord)
                if src_ch in raw_pos
                else float("inf"),
            )
            closest_idx = raw.ch_names.index(closest_ch)
            new_data.append(src_data[closest_idx])
    
        new_data = np.array(new_data)
    
        info = mne.create_info(
            ch_names=target_chs, sfreq=raw.info["sfreq"], ch_types="eeg"
        )
        raw_nearest = mne.io.RawArray(new_data, info)
        raw_nearest.set_montage(target_montage)
    
        return raw_nearest
    ##########################################################################
    def preprocess(self, file_path,method='nearest'):
        print(f"\n[Pipeline Start] -> {file_path}")
        try:
            # 1. Load EEG
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
            original_sfreq = raw.info["sfreq"]

            # 2. Montage
            if len(raw.ch_names) >= 120:
                print("-> High-density EEG detected")
                orig_montage = mne.channels.make_standard_montage('biosemi128')
                raw.set_montage(orig_montage, on_missing='ignore')
        
                print("Applying spatial layout harmonization...")
                target_montage = mne.channels.make_standard_montage("standard_1020")
        
                try:
                    if method == "nearest":
                        raw = self.harmonize_nearest_neighbor(raw, target_montage)
                    elif hasattr(raw, 'interpolate_to'):
                        raw = raw.interpolate_to(target_montage, method="spline")
                        
                except Exception as e:
                    print(f"Spatial interpolation failed: {e}")
            
                raw = self._harmonize_channels(raw)
        
            else:
                print("-> Standard 10-20 EEG detected")
                montage = mne.channels.make_standard_montage("standard_1020")
                raw.set_montage(montage, on_missing="ignore")

            # 3. Band-pass
            print("-> Band-pass filtering")

            raw.filter(
            l_freq=self.l_freq,  
            h_freq=self.h_freq,
            method="fir",
            fir_design="firwin",
            phase="zero"
            )
            # 4. Notch
            print("-> Removing line noise")
            raw.notch_filter(freqs=self.notch_freq, verbose=False)

            # 5. Resample
            if self.resample_freq is not None and original_sfreq != self.resample_freq:
                print(f"-> Resampling to {self.resample_freq} Hz")
                raw.resample(self.resample_freq, verbose=False)
                
            # 6. Robust referencing + Bad channel detection (PyPREP)
            raw = self._run_pyprep(raw) 
            
            # 7. ASR
            raw = self._run_asr(raw)
            
            # 8. reference average   
            print(f"-> eeg_reference average ")
            raw = raw.set_eeg_reference('average', projection=False, verbose=False)
            
            # 9. ICA
            raw = self._run_ica(raw)

            # 10. Harmonize Channels
            raw = self._harmonize_channels(raw)
            
            # 11. Epoching
            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=self.epoch_duration,
                overlap=self.epoch_overlap,
                preload=True,
                verbose=False
            )
           
            # 12. get_data and convert to micro volt
            data = epochs.get_data() * 1e6

            print("[Pipeline Success]")
            print("\n========== QC REPORT ==========")
            print(f"Channels           : {len(raw.ch_names)}")
            print(f"Sampling Rate      : {raw.info['sfreq']} Hz")
            print(f"ASR Variance Ratio : {self.qc_report['variance_ratio']:.3f}")
            print(f"Removed ICs        : {self.qc_report['removed_ic']}")
            print(f"Epochs             : {len(epochs)}")
            print(f"Output Shape       : {data.shape}")
            print("================================")
            return data  

        except Exception as e:
            print(f"[Error] Pipeline failure: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])
