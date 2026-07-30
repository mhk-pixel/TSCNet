import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, train_test_split
from src.EEGNormalizer import EEGNormalizer


class ManageDataset:
    def __init__(self, config):
        self.config = config
        
        if config.get('CrossDataset') == "on":
            self.traindataset = self.config['traindataset']
            self.train_subjects_path = os.path.join(self.traindataset, "Subjects/")
            self.train_labels_path = os.path.join(self.traindataset, "Labels/labels.csv")
            self.train_labels_file = self._load_labels(self.train_labels_path)
            
            self.testdataset = self.config['testdataset']
            self.test_subjects_path = os.path.join(self.testdataset, "Subjects/")
            self.test_labels_path = os.path.join(self.testdataset, "Labels/labels.csv")
            self.test_labels_file = self._load_labels(self.test_labels_path)
        else:
            self.dataset_path = self.config['dataset_path']
            self.subjects_path = os.path.join(self.dataset_path, "Subjects/")
            self.labels_path = os.path.join(self.dataset_path, "Labels/labels.csv")
            self.labels_file = self._load_labels(self.labels_path)

    # ======================================================================================
    def _load_labels(self, path):
        print(f"Loading information from: {path}")
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Contains {len(df)} Subjects")
            return df
        else:
            print(f"Warning: File not found at {path}")
            return pd.DataFrame()

    # ======================================================================================
    def load_subject_data(self, subject_id, label, name, max_segments=None):
       
        if self.config.get('CrossDataset') == "on":
            if name in ["Train", "Validation"]:
                current_subjects_path = self.train_subjects_path
            elif name == "Test":
                current_subjects_path = self.test_subjects_path
            else:
                current_subjects_path = self.subjects_path
        else:
            current_subjects_path = self.subjects_path

        npy_file = os.path.join(current_subjects_path, f"{subject_id}.npy")
        if not os.path.exists(npy_file):
            print(f"Missing file: {npy_file}")
            return None, None, None

        try:
            data = np.load(npy_file).astype(np.float32)  # (N, C, T)
            if max_segments and len(data) > max_segments:
                data = data[:max_segments]
            labels = np.full(len(data), label, dtype=np.int64)
            subjects = np.full(len(data), subject_id, dtype=object)

            return data, labels, subjects
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
            return None, None, None

    # ======================================================================================
    def _print_split_info(self, fold_config):
        print(f"\nFold {fold_config['fold_num']}:")
        print(f"  Train: {len(fold_config['train_ids'])} subjects")
        if len(fold_config['val_ids']) > 0:
            print(f"  Val:   {len(fold_config['val_ids'])} subjects")
        else:
            print(f"  Val:   None")
        print(f"  Test:  {len(fold_config['test_ids'])} subjects")

    # ======================================================================================
    def prepare_CrossDataset(self):
        mapping_dict = {'HC': 0, 'AD': 1}
        random_state = self.config.get('seed', 42)

        train_ids = self.train_labels_file['subject_id'].values
        train_labels = self.train_labels_file['Group'].map(mapping_dict).values

        test_ids = self.test_labels_file['subject_id'].values
        test_labels = self.test_labels_file['Group'].map(mapping_dict).values

        if self.config.get('val_split', 0) > 0:
            train_ids, val_ids, train_labels, val_labels = train_test_split(
                train_ids, train_labels, 
                test_size=self.config['val_split'], 
                stratify=train_labels, 
                random_state=random_state
            )
        else:
            val_ids, val_labels = np.array([]), np.array([])

        split_config = [{
            'fold_num': 1,
            'train_ids': train_ids.tolist(),
            'val_ids': val_ids.tolist(),
            'test_ids': test_ids.tolist(),
            'train_labels': train_labels,
            'val_labels': val_labels,
            'test_labels': test_labels
        }]

        self._print_split_info(split_config[-1])
        return split_config

    # ======================================================================================
    def data_split(self, k_folds, groups, random_state, fold_type, strategy, val_split):
        print("Preparing split data..")
        print("Loading ids and labels..")
        
        selected_df = self.labels_file[self.labels_file['Group'].isin(groups)].copy()
        if len(selected_df) == 0:
            raise ValueError(f"No subjects found for groups: {groups}")

        subject_ids = selected_df['subject_id'].values
        label_map = {'HC': 0, 'AD': 1}
        subject_labels = selected_df['Group'].map(label_map).values

        if strategy == "crossvalidation":
            print(f"Preparing {fold_type} cross-validation with {k_folds} folds")

            if fold_type == 'stratified':
                cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
                splits = list(cv.split(subject_ids, subject_labels))
            elif fold_type == 'kfold':
                cv = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
                splits = list(cv.split(subject_ids, subject_labels))
            elif fold_type == 'loso':
                cv = LeaveOneOut()
                splits = list(cv.split(subject_ids, subject_labels))
            elif fold_type in ["manual65", "manual67", "manual132"]:
                if fold_type == "manual65":
                    from f_assignment_file65 import folds
                elif fold_type == "manual67":
                    from f_assignment_file67 import folds
                elif fold_type == "manual132":
                    from f_assignment_file132 import folds

                manual_folds = folds()
                id_to_fold = {
                    subj_id: fold_name 
                    for fold_name, fold_subj_ids in manual_folds.items() 
                    for subj_id in fold_subj_ids
                }

                selected_df['Fold'] = selected_df['subject_id'].map(id_to_fold)
                subject_folds = selected_df['Fold'].values
                unique_folds = sorted(selected_df['Fold'].dropna().unique())

                splits = [
                    (np.where(subject_folds != fold)[0], np.where(subject_folds == fold)[0])
                    for fold in unique_folds
                ]
            else:
                raise ValueError(f"Unsupported fold_type: {fold_type}")

        fold_config = []
        for idx, (train_val_idx, test_idx) in enumerate(splits):
            train_val_ids, test_ids = subject_ids[train_val_idx], subject_ids[test_idx]
            train_val_labels, test_labels = subject_labels[train_val_idx], subject_labels[test_idx]

            if val_split > 0:
                train_ids, val_ids, train_labels, val_labels = train_test_split(
                    train_val_ids, train_val_labels, 
                    test_size=val_split,
                    stratify=train_val_labels, 
                    random_state=random_state
                )
            else:
                train_ids, val_ids = train_val_ids, np.array([])
                train_labels, val_labels = train_val_labels, np.array([])

            fold_config.append({
                'fold_num': idx + 1,
                'train_ids': train_ids.tolist(),
                'val_ids': val_ids.tolist(),
                'test_ids': test_ids.tolist(),
                'train_labels': train_labels,
                'val_labels': val_labels,
                'test_labels': test_labels
            })
            self._print_split_info(fold_config[-1])

        return fold_config

    # ======================================================================================
    def prepare_fold_data(self, fold_config, segments_per_subject, normalization, norm_level):
        fold_num = fold_config['fold_num']

        print(f"\n{'='*80}\nPreparing Data for Fold {fold_num}\n{'='*80}")
        print(f"  Segments per subject: {segments_per_subject if segments_per_subject else 'All'}")
        print(f"  Normalization:        {normalization}")
        print(f"  Norm level:           {norm_level}")

        def load_data(ids, labels, name):
            all_data, all_labels, all_subjects = [], [], []
            print(f"\nLoading {name} data:\n{'─'*60}")
            
            # Shuffle loading order
            permutation = np.random.permutation(len(ids))
            ids = [ids[i] for i in permutation]
            labels = [labels[i] for i in permutation]

            for sid, lbl in zip(ids, labels):
                data, lab, subs = self.load_subject_data(sid, lbl, name, segments_per_subject)
                if data is not None:
                    all_data.append(data)
                    all_labels.append(lab)
                    all_subjects.append(subs)
                    print(f"{sid}: {len(subs):4d} segments (class {lbl})")
                else:
                    print(f"{sid}: Failed to load")

            if not all_data:
                return None, None, None

            X = np.vstack(all_data)
            y = np.hstack(all_labels)
            Z = np.hstack(all_subjects)

            if self.config.get('CrossDataset') == "on":
                print(f"--> Shuffling {name} segments randomly...")
                perm = np.random.permutation(len(X))
                X, y, Z = X[perm], y[perm], Z[perm]

            return X, y, Z

        # Load datasets
        train_data, train_labels, train_subjects = load_data(
            fold_config['train_ids'], fold_config['train_labels'], "Train"
        )
        
        val_data, val_labels, val_subjects = (None, None, None)
        if len(fold_config['val_ids']) > 0:
            val_data, val_labels, val_subjects = load_data(
                fold_config['val_ids'], fold_config['val_labels'], "Validation"
            )

        test_data, test_labels, test_subjects = load_data(
            fold_config['test_ids'], fold_config['test_labels'], "Test"
        )

        if train_data is None or test_data is None:
            raise ValueError("Failed to load data for train/test sets")

        # Normalization
        if normalization != 'none':
            normalizer = EEGNormalizer(
                method=normalization,
                level=norm_level,
                clip_sigma=5.0
            )
            train_data, val_data, test_data = normalizer.normalize(
                train_data, val_data, test_data
            )
            print("Normalization applied successfully")

        print(f"\n{'='*80}\nData Preparation Complete for Fold {fold_num}\n{'='*80}")
        print(f"  Train:       {train_data.shape}")
        print(f"  Validation:  {val_data.shape if val_data is not None else 'None'}")
        print(f"  Test:        {test_data.shape}\n{'='*80}\n")

        gc.collect()

        return (
            (train_data, train_labels, train_subjects),
            (val_data, val_labels, val_subjects) if val_data is not None else (None, None, None),
            (test_data, test_labels, test_subjects)
        )

    # ======================================================================================
    def show_data_visualization(self, train_data, val_data, test_data):
        if val_data is not None:
            datasets = [train_data, val_data, test_data]
            names = ['Train', 'Validation', 'Test']
        else:
            datasets = [train_data, test_data]
            names = ['Train', 'Test']

        n_plots = len(datasets)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        for i, (data, name) in enumerate(zip(datasets, names)):
            sample = data[0]
            if sample.ndim == 2:
                im = axes[i].imshow(sample, cmap='viridis', aspect='auto')
                axes[i].set_title(f'{name}\nShape: {sample.shape}')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('Channels')
                plt.colorbar(im, ax=axes[i])

        plt.tight_layout()
        plt.show()
