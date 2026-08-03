# Author: Majid Hameed Khalaf
# Date: 2026 
# ============================================================================
import os
import sys
import gc
import random
import pickle
import traceback
from datetime import datetime
from pathlib import Path
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import re
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix, 
    roc_auc_score
)
from src.visualize import EEGVisualizer   
from src.dataset import ManageDataset
from src.train import train_model_with_early_stopping
from src.evaluate import comprehensive_evaluate_model, _save_records_to_csv
from config import config


# ============================================================================
# MAIN EEG PIPELINE CLASS
# ============================================================================
class tools:
    def __init__(self):
        self.config = config()
        self.EEGV = EEGVisualizer()
        self.dataset = ManageDataset(self.config)
        
        if self.config['models_path'] not in sys.path:
            sys.path.append(self.config['models_path'])
            
        self.paths = None
        self.train_model_with_early_stopping = train_model_with_early_stopping
        self.comprehensive_evaluate_model = comprehensive_evaluate_model

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def define_model(self, choice: int, n_channels: int, n_time_points: int, n_classes: int, dropout_rate: float):
        if choice == 0:
            from TSCNET import EEGModel 
        elif choice == 1:
            from TSCNET1 import EEGModel
        elif choice == 2:
            from TSCNET2 import EEGModel
        elif choice == 3:
            from TSCNET3 import EEGModel
        else:
            raise ValueError(f"Invalid model choice: {choice}")
            
        model = EEGModel(
            channels=n_channels,
            num_classes=n_classes,
            dropout_rate=dropout_rate
        )
        return model 

    def print_config(self):
        print("\n" + "=" * 80)
        print("EEG CLASSIFICATION PIPELINE")
        if self.config['val_split'] > 0:
            print("Mode: Train-Validation-Test Split with Early Stopping")
        else:
            print("Mode: Train-Test Split (No Validation)")
        print("=" * 80)
        
        print("Configuration:")
        print("-" * 80)
        print(f"  Dataset path:      {self.config['dataset_path']}")
        print(f"  Device:            {self.config['device']}")
        print(f"  Model choice:      {self.config['model_choice']}")
        print(f"  Fold strategy:     {self.config['fold_strategy']}")
        print(f"  Number of folds:   {self.config['k_folds']}")
        
        if self.config['val_split'] > 0:
            print(f"  Validation split:  {self.config['val_split'] * 100:.0f}%")
            print(f"  Early stopping:    Enabled (patience={self.config['params']['patience']})")
        else:
            print("  Validation split:  None")
            print("  Early stopping:    Disabled")
        
        print(f"\nStratification: on ({self.config['stratify_columns']})")
        
        print("\nData Settings:")
        print(f"  Class names:       {self.config['class_names']}")
        print(f"  Normalization:     {self.config['normalization']}")
        print(f"  Norm level:        {self.config['norm_level']}")
        
        print("\nTraining Settings:")
        print(f"  Batch size:        {self.config['params']['batch_size']}")
        print(f"  Learning rate:     {self.config['params']['learning_rate']}")
        print(f"  Max epochs:        {self.config['params']['epochs']}")
        
        print("\nEvaluation Settings:")
        print(f"  Aggregation mode:  {self.config['evaluation']['aggregation_mode']}")
        print(f"  Confidence filter: {self.config['evaluation']['confidence_filter']}")
        print(f"  Save models:       {self.config['save_models']}")
        print("=" * 80) 


    #=========================================================================================
    def setup_output_directory(self, N = True):
        exp_name = self.config['experiment_name'] or datetime.now().strftime('%Y%m%d_%H%M%S')
    
        base_dir = Path(self.config['output_dir']) / exp_name
        if N:
            if base_dir.exists() and base_dir.is_dir():
                shutil.rmtree(base_dir)
                print(f'Existing experiment directory deleted: {base_dir}')
    
        self.paths = {
            'base': base_dir,
            'models': base_dir / 'models',
            'plots': base_dir / 'plots',
            'tables': base_dir / 'tables',
        }
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
        print(f'\nOutput directory setup complete: {base_dir}')
    #=========================================================================================
    
    def clear_memory(self):
        plt.close('all')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_single_fold(self, fold_config: dict):
        fold_num = fold_config['fold_num']
        print(f"\n{'='*80}\nProcessing Fold {fold_num}\n{'='*80}")
        
        try:
            self.clear_memory()
         
            fold_data = self.dataset.prepare_fold_data(
                fold_config, 
                segments_per_subject=self.config['segments_per_subject'],
                normalization=self.config['normalization'],
                norm_level=self.config['norm_level']
            )
            
            train_set = fold_data[0]       # (train_data, train_labels)
            val_set   = fold_data[1]       # (val_data, val_labels) or (None, None)
            test_fold = fold_data[2]       # (test_data, test_labels, [test_subject_ids])
            
            if len(test_fold) == 3:
                test_data, test_labels, test_subject_ids = test_fold
            else:
                test_data, test_labels = test_fold
                test_subject_ids = None
            
            has_validation = val_set[0] is not None
            
            print("\nDataset split:")
            print(f"  Train: {train_set[0].shape[0]} samples")
            print(f"  Val:   {val_set[0].shape[0] if has_validation else 'None (val_split=0)'} samples")
            print(f"  Test:  {test_data.shape[0]} samples")
            
            
            model = self.define_model(
                choice=self.config['model_choice'], 
                n_channels=self.config['model_params']['n_channels'], 
                n_time_points=self.config['model_params']['n_time_points'],
                n_classes=self.config['model_params']['n_classes'],
                dropout_rate=self.config['model_params']['dropout_rate']
            ).to(self.config['device'])

            print(f"\nModel: {model.__class__.__name__} | Device: {self.config['device']}")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")
            
           
            if has_validation:
                print("\nTraining with validation set and early stopping...")
                trained_model, history = self.train_model_with_early_stopping(
                    model, 
                    train_set, 
                    val_set, 
                    self.config['params']['batch_size'], 
                    self.config['params']['epochs'], 
                    self.config['params']['learning_rate'], 
                    self.config['params']['patience'],
                    device=self.config['device']
                )
            else:
                print("\nTraining without validation set (using all epochs)...")
                trained_model, history = self.train_model_with_early_stopping(
                    model, 
                    train_set, 
                    None, 
                    self.config['params']['batch_size'], 
                    self.config['params']['epochs'], 
                    self.config['params']['learning_rate'], 
                    patience=None,
                    device=self.config['device']
                )
            
          
            subject_ids = test_subject_ids
            if subject_ids is None and hasattr(self.dataset, "test_subject_ids"):
                subject_ids = self.dataset.test_subject_ids
            
            test_metrics = self.comprehensive_evaluate_model(
                trained_model,
                (test_data, test_labels),
                self.config.get('class_names', ['HC', 'AD']),
                subject_ids=subject_ids, 
                aggregation_mode=self.config['evaluation']['aggregation_mode'],  
                confidence_filter=self.config['evaluation']['confidence_filter'],      
                save_segment_csv=None,
                device=self.config['device']
            )
            
          
            print(f"\n{'='*80}\nFold {fold_num} Test Results\n{'='*80}")
            print(f"  Accuracy:    {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
            print(f"  Sensitivity: {test_metrics['sensitivity']:.4f} ({test_metrics['sensitivity']*100:.2f}%)")
            print(f"  Specificity: {test_metrics['specificity']:.4f} ({test_metrics['specificity']*100:.2f}%)")
            print(f"  ROC AUC:     {test_metrics['roc_auc']:.4f}")
            print(f"  Precision:   {np.mean(test_metrics['precision']):.4f}")
            print(f"  Recall:      {np.mean(test_metrics['recall']):.4f}")
            print(f"  F1-Score:    {np.mean(test_metrics['f1_score']):.4f}")
            print(f"{'='*80}\n")
            
            
            torch.save(trained_model.state_dict(), self.paths['models'] / f"fold_{fold_num}.pth")

            del model, trained_model, train_set, val_set, test_fold, test_data, test_labels
            self.clear_memory()
            
            return {
                'fold': fold_num,
                'test_metrics': test_metrics,
                'history': history
            }
            
        except Exception as e:
            print(f"\n{'!'*80}\nERROR in fold {fold_num}: {e}\n{'!'*80}")
            traceback.print_exc()
            self.clear_memory()
            return None
    #=============================================================================================
    
    def save_segment_csvs_from_results(self):
     
        
        pkl_path = self.paths['tables'] / 'raw_all_fold_results.pkl'
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            print(f"\n Pickle file not found: {pkl_path}")
            return
        with open(pkl_path, 'rb') as f:
            all_fold_results = pickle.load(f)

        print("\n" + "=" * 80)
        print("Generating Segment CSV Files for All Folds...")
        print("=" * 80)

        class_names = self.config.get('class_names', ['HC', 'AD'])

        for result in all_fold_results:
            self.save_segment_details(result)
        print("=" * 80)
        
    #================================================================================================    
    def save_segment_details(self, fold_results=None):
        if not fold_results:
            print("No fold results.")
            return
    
        print("\n" + "=" * 80)
        print("Segment CSV Files ...")
        print("=" * 80)
    
        class_names = self.config.get('class_names', ['HC', 'AD'])
    
        fold_num = fold_results.get('fold', 'unknown')
        segment_records = fold_results.get('test_metrics', {}).get('segment_records', {})
        
        if not segment_records:
            print(f" No segment records found for Fold {fold_num}")
            return  

        base_path = self.paths['tables']
    
        # 1. raw_prob.csv
        if 'raw_prob' in segment_records and segment_records['raw_prob']:
            _save_records_to_csv(
                segment_records['raw_prob'],
                base_path / f"fold_{fold_num}_raw_prob.csv",  
                class_names,
                is_prob=True)
    
        # 2. prob.csv
        if 'prob' in segment_records and segment_records['prob']:
            _save_records_to_csv(
                segment_records['prob'],
                base_path / f"fold_{fold_num}_prob.csv",
                class_names,
                is_prob=True)
    
        # 3. predict.csv
        if 'predict' in segment_records and segment_records['predict']:
            _save_records_to_csv(
                segment_records['predict'],
                base_path / f"fold_{fold_num}_predict.csv",
                class_names,
                is_prob=False)
    
        print(f" Fold {fold_num}: files saved successfully.")
        print("=" * 80)
    #================================================================================================
 
    def save_subject_voting_proportions(self, all_fold_results):
      
        class_names = self.config.get('class_names', ['HC', 'AD'])
        summary_rows = []
    
        for fold_result in all_fold_results:
            fold_num = fold_result['fold']
            segment_records = (
                fold_result.get('test_metrics', {})
                .get('segment_records', {})
                .get('predict', [])
            )
    
            if not segment_records:
                continue
    
            for row in segment_records:
                sub_id = row[0]

                preds = [
                    int(p)
                    for p in row[1:]
                    if pd.notna(p) and str(p).strip() != ''
                ]
    
                total_segments = len(preds)
                if total_segments == 0:
                    continue
    
                row_dict = {'Fold': f'Fold {fold_num}', 'sub': sub_id}
    
                for cls_idx, cls_name in enumerate(class_names):
                    cls_count = sum(1 for p in preds if p == cls_idx)
                    row_dict[cls_name] = cls_count / total_segments
    
                summary_rows.append(row_dict)
    
        if not summary_rows:
            print('[WARNING] No prediction records found to aggregate.')
            return None
    
        df_summary = pd.DataFrame(summary_rows)
    
        preferred_order = ['Fold', 'sub'] + [
            c for c in reversed(class_names)
        ] 
        df_summary = df_summary[preferred_order]
    
        output_path = (
            self.paths['tables'] / 'subject_level_voting_proportions.csv'
        )
        df_summary.to_csv(output_path, index=False, float_format='%.2f')
    
        print(
            f'\n Subject-level voting proportions successfully saved to: {output_path}'
        )
        return df_summary
    #================================================================================================ 
    def report_performance_separately(self, all_fold_results, output_path=None):
    
        cols_order = ["Fold", "Dataset", "sub", "sub_id", "true", "pred"]
        all_dfs = []
    
        def _extract_id(subj):
            nums = re.findall(r"\d+", str(subj))
            return int(nums[0]) if nums else 0
    
        def label_dataset(df):
            unique_subs_count = df["sub_id"].nunique()
            max_id = df["sub_id"].max()
    
            if unique_subs_count == 65 or max_id == 65:
                df["Dataset"] = "AHEPA"
            elif unique_subs_count == 67 or max_id == 67:
                df["Dataset"] = "BrainLat"
            else:
                df["Dataset"] = df["sub_id"].apply(
                    lambda x: "AHEPA" if x <= 65 else "BrainLat"
                )
            return df
    
        for fold_result in all_fold_results:
            fold_num = fold_result.get("fold", "N/A")
            df_fold = (
                fold_result.get("test_metrics", {}).get("dataset_metrics", {})
            )
    
            if isinstance(df_fold, dict):
                df_fold = pd.DataFrame(df_fold)
    
            df_fold = df_fold.copy()
            df_fold["Fold"] = f"Fold {fold_num}"
            df_fold["sub_id"] = df_fold["sub"].apply(_extract_id)
            all_dfs.append(df_fold)
    
        df = pd.concat(all_dfs, ignore_index=True)
        df = label_dataset(df)
        df = df[cols_order]
    
        def _evaluate_subset(sub_df, dataset_name):
            if sub_df.empty:
                return {
                    "Dataset": dataset_name,
                    "AD": 0,
                    "HC": 0,
                    "Correct/All": 0,
                    "Acc": 0.0,
                    "Sens": 0.0,
                    "Spec": 0.0,
                }
    
            correct_mask = sub_df["true"] == sub_df["pred"]
            ad_correct = int((correct_mask & (sub_df["true"] == 1)).sum())
            hc_correct = int((correct_mask & (sub_df["true"] == 0)).sum())
            
           
            
            acc = accuracy_score(sub_df["true"], sub_df["pred"])
            _, recall, _, _ = precision_recall_fscore_support(
                sub_df["true"],
                sub_df["pred"],
                average=None,
                labels=[0, 1],
                zero_division=0,
            )
    
            return {
                "Dataset": dataset_name,
                "AD": ad_correct,
                "HC": hc_correct,
                "Correct/All": ad_correct + hc_correct,
                "Acc": round(float(acc), 4),
                "Sens": round(float(recall[1]), 4),
                "Spec": round(float(recall[0]), 4),
            }
    
       
        dataset_report = [
            _evaluate_subset(df[df["Dataset"] == "AHEPA"], "AHEPA"),
            _evaluate_subset(df[df["Dataset"] == "BrainLat"], "BrainLat"),
            _evaluate_subset(df, "combined"),  
        ]
    
        df_dataset_report = pd.DataFrame(dataset_report)
    
        if (
            output_path is None
            and hasattr(self, "paths")
            and isinstance(self.paths, dict)
        ):
            output_path = self.paths.get("tables")
    
        if output_path:
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            df_dataset_report.to_csv(
                out_dir / "report_performance_separately.csv", index=False
            )
            df.to_csv(out_dir / "report.csv", index=False)
            print(
                f"Stratified performance report saved successfully to: {out_dir}"
            )
    
        return df
        
    #=================================================================================================
    def save_all_fold_details(self, all_fold_results: list):
       
        raw_results_pkl = self.paths['tables'] / 'raw_all_fold_results.pkl'
        with open(raw_results_pkl, 'wb') as f:
            pickle.dump(all_fold_results, f)
        print(f"\nComplete raw results object saved to: {raw_results_pkl}")
        
    #================================================================================================     
    def print_final_results(self, all_fold_results: list):

        print("\n" + "=" * 80)
        print("FINAL CROSS-VALIDATION RESULTS")
        print("=" * 80)
        
        metrics = {}
        for metric_name in ['accuracy', 'roc_auc', 'sensitivity', 'specificity']:
            metrics[metric_name] = [r['test_metrics'][metric_name] for r in all_fold_results]
        
        for metric_name in ['precision', 'recall', 'f1_score']:
            metrics[metric_name] = [np.mean(r['test_metrics'][metric_name]) for r in all_fold_results]
        
        print("\nPerformance Summary (Mean ± Std):")
        print("-" * 80)
        for metric_name, values in metrics.items():
            mean_val, std_val = np.mean(values), np.std(values, ddof=1)
            min_val, max_val = np.min(values), np.max(values)
            print(f"  {metric_name.replace('_', ' ').title():15s}: "
                  f"{mean_val:.4f} ± {std_val:.4f} [{min_val:.4f}, {max_val:.4f}]")
        
        print("\n" + "=" * 80)
        print("Individual Fold Results")
        print("=" * 80)
        print(f"{'Fold':>6s} | {'Accuracy':>8s} | {'F1':>8s} | {'AUC':>8s} | {'Sens':>8s} | {'Spec':>8s}")
        print("-" * 80)
        
        for i, result in enumerate(all_fold_results):
            fold_num = result['fold']
            print(f"{fold_num:6d} | {metrics['accuracy'][i]:8.4f} | {metrics['f1_score'][i]:8.4f} | "
                  f"{metrics['roc_auc'][i]:8.4f} | {metrics['sensitivity'][i]:8.4f} | {metrics['specificity'][i]:8.4f}")
        print("=" * 80)
        
        results_df = pd.DataFrame({
            'Fold': [r['fold'] for r in all_fold_results],
            **{m.replace('_', ' ').title(): values for m, values in metrics.items()}
        })
        
        summary_mean = {'Fold': 'Mean', **{m.replace('_', ' ').title(): np.mean(v) for m, v in metrics.items()}}
        summary_std = {'Fold': 'Std', **{m.replace('_', ' ').title(): np.std(v, ddof=1) for m, v in metrics.items()}}
        results_df = pd.concat([results_df, pd.DataFrame([summary_mean, summary_std])], ignore_index=True)
        
        filename = self.paths['tables'] / f'cv_results_{self.config["fold_strategy"]}_{self.config["k_folds"]}folds.csv'
        results_df.to_csv(filename, index=False, float_format='%.4f')
        print(f"\nResults saved to: {filename}")
        
        print("\n95% Confidence Intervals:")
        print("-" * 80)
        ci_data = []
        for metric_name, values in metrics.items():
            mean_val, std_val, n = np.mean(values), np.std(values, ddof=1), len(values)
            ci_lower = max(0.0, mean_val - 1.96 * std_val / np.sqrt(n))
            ci_upper = min(1.0, mean_val + 1.96 * std_val / np.sqrt(n))
            print(f"  {metric_name.replace('_', ' ').title():15s}: {mean_val:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
            ci_data.append({
                'Metric': metric_name.replace('_', ' ').title(),
                'Mean': mean_val,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper
            })
        print("=" * 80)
        
        ci_filename = self.paths['tables'] / 'confidence_intervals.csv'
        pd.DataFrame(ci_data).to_csv(ci_filename, index=False, float_format='%.4f')
        print(f"Confidence intervals saved to: {ci_filename}")
        
       
       
        if self.config.get('save_models', True):  
            best_fold_idx = np.argmax(metrics['accuracy'])
            best_fold = all_fold_results[best_fold_idx]
        
            best_model_performance = {
                'best_fold_number': int(best_fold['fold']),
                'accuracy': float(metrics['accuracy'][best_fold_idx]),
                'f1_score': float(metrics['f1_score'][best_fold_idx]),
                'roc_auc': float(metrics['roc_auc'][best_fold_idx]),
                'sensitivity': float(metrics['sensitivity'][best_fold_idx]),
                'specificity': float(metrics['specificity'][best_fold_idx]),
                'precision': float(metrics['precision'][best_fold_idx]),
            }
        
            full_experiment_summary = {
                'experiment_info': {
                    'experiment_name': self.config.get('experiment_name'),
                    'execution_timestamp': datetime.now().strftime(
                        '%Y-%m-%d %H:%M:%S'
                    ),
                    'device_used': str(self.config.get('device')),
                },
                'best_fold_performance': best_model_performance,
                'configuration_and_hyperparameters': self.config,  
            }
        
            output_json_path = self.paths['base'] / 'model_info.json'
        
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(
                    full_experiment_summary,
                    f,
                    indent=4,
                    ensure_ascii=False,
                    default=str)
            print(f'\nFull model config, parameters & best fold metrics successfully saved to: {output_json_path}')
        
        return results_df
    #============================================================================================================
    def generate_plots_from_saved_results(self, pkl_path=None):
       
        if pkl_path is None:
            pkl_path = self.paths['tables'] / 'raw_all_fold_results.pkl'
        
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            print(f"\n[ERROR] Results file not found: {pkl_path}")
            return
            
        print(f"\n{'='*80}\nGenerating Plots From Saved Results: {pkl_path}\n{'='*80}")
        with open(pkl_path, 'rb') as f:
            all_fold_results = pickle.load(f)
            
        for result in all_fold_results:
            fold_num = result['fold']
            history = result.get('history', {})
            test_metrics = result.get('test_metrics', {})
            
           
            if history:
                self.EEGV.plot_training_curves(history, fold_num, self.paths['plots'])
                if 'val_loss' in history:
                    self.EEGV.plot_overfitting_analysis(history, fold_num, self.paths['plots'])
            
          
            if 'confusion_matrix' in test_metrics:
                self.EEGV.plot_confusion_matrix(
                    test_metrics['confusion_matrix'], ['HC', 'AD'], fold_num, self.paths['plots']
                )
        
  
        plt.figure()
        self.EEGV.plot_cross_validation_summary(all_fold_results, self.paths['plots'])
        plt.close()
        
        plt.figure()
        self.EEGV.plot_roc_from_folds(all_fold_results, self.paths['plots'])
        plt.close()
        self.EEGV.plot_combined_confusion_matrices(all_fold_results, ['HC', 'AD'],self.paths['plots'])
        print(f"\nAll plots generated successfully and saved in: {self.paths['plots']}")
