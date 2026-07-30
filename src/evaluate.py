import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix, 
    roc_auc_score
)

from src.train import MemoryEfficientDataLoader


def _save_records_to_csv(records, file_path, class_names, is_prob=False, float_format=None):
  
    if file_path is None or len(records) == 0:
        return

    file_path = str(file_path)

    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    padded_records = []

   
    if is_prob:
        num_classes = len(class_names)
        max_segment_cols = max(len(r) - 1 for r in records)
        max_segments = (
            max_segment_cols // num_classes if max_segment_cols > 0 else 0
        )

        target_len = 1 + (max_segments * num_classes)
        for row in records:
            new_row = list(row)  
            while len(new_row) < target_len:
                new_row.append(np.nan)
            padded_records.append(new_row)

        cols = ['sub']
        for i in range(max_segments):
            for c_name in class_names:
                cols.append(f'seg{i+1}_{c_name}')

        if float_format is None:
            float_format = '%.4f'

    else:
        max_segments = max(len(r) - 1 for r in records)
        target_len = 1 + max_segments

        for row in records:
            new_row = list(row)  
            while len(new_row) < target_len:
                new_row.append(np.nan)
            padded_records.append(new_row)

        cols = ['sub'] + [f'seg{i+1}_pred' for i in range(max_segments)]

        if float_format is None:
            float_format = '%g'

    # DataFrame
    df_segments = pd.DataFrame(padded_records, columns=cols)

    #=========================
    if not is_prob:
        pred_cols = [c for c in cols if c != 'sub']
        for col in pred_cols:
            df_segments[col] = df_segments[col].astype('Int64')

    # save
    df_segments.to_csv(file_path, index=False, float_format=float_format)
    
def comprehensive_evaluate_model(
    trained_model,
    test_set,
    class_names=['HC', 'AD'],
    subject_ids=None,
    confidence_filter=0.8,
    aggregation_mode='average',
    save_segment_csv=None,  
    device='cuda',
    verbose=True
):
  
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    trained_model.to(device)
    trained_model.eval()
    
    test_data, test_labels = test_set[0], test_set[1]

    HC_index = class_names.index('HC') if 'HC' in class_names else 0
    AD_index = class_names.index('AD') if 'AD' in class_names else 1

    # ==========================================================
    # 1. SEGMENT-LEVEL EVALUATION
    # ==========================================================
    if subject_ids is None:
        if verbose:
            print("\n=== Segment-level Evaluation ===")
        
        loader = MemoryEfficientDataLoader(test_data, test_labels, batch_size=32, shuffle=False, device=device)

        all_preds, all_labels, all_probs, all_conf = [], [], [], []

        with torch.no_grad():
            for batch_data, batch_labels in loader:
                outputs = trained_model(batch_data)
                probs = F.softmax(outputs, dim=1)
                max_p, preds = torch.max(probs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_conf.extend(max_p.cpu().numpy())

                del outputs, probs, preds
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_conf = np.array(all_conf)

        # 
        if confidence_filter is not None:
            mask = all_conf >= confidence_filter
            if mask.sum() > 0:
                all_preds = all_preds[mask]
                all_labels = all_labels[mask]
                all_probs = all_probs[mask]

        acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=[0, 1], zero_division=0
        )
        
        sensitivity = recall[AD_index]
        specificity = recall[HC_index]
        f1_ad = f1[AD_index]
        precision_ad = precision[AD_index]

        roc_auc = 0.0
        if len(np.unique(all_labels)) > 1 and all_probs.shape[1] > 1:
            try:
                roc_auc = roc_auc_score(all_labels, all_probs[:, AD_index])
            except Exception:
                pass

        metrics = {
            'mode': 'segment',
            'accuracy': acc,
            'precision': precision_ad,
            'recall': sensitivity, 
            'f1_score': f1_ad,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'n_samples': len(all_labels),
            'segment_records': {}
        }

        if verbose:
            print(f"Samples: {metrics['n_samples']} | Accuracy: {acc:.4f}")
            print(f"Sensitivity (AD): {sensitivity:.4f} | Specificity (HC): {specificity:.4f}")
            print(f"Precision (AD):   {precision_ad:.4f} | F1-Score (AD):  {f1_ad:.4f}")
            print(f"ROC-AUC:         {roc_auc:.4f}")

        return metrics

    # ==========================================================
    # 2. SUBJECT-LEVEL EVALUATION
    # ==========================================================
    subject_ids = np.array(subject_ids)
    unique_subjects = np.unique(subject_ids)

    if verbose:
        print(f"\n=== Subject-level Evaluation ({len(unique_subjects)} subjects) ===")

    subj_true, subj_pred, subj_prob = [], [], []
    segment_raw_prob_records = [] # 1. الاحتماليات الخام قبل الماسك
    segment_prob_records = []     # 2. الاحتماليات بعد الماسك
    segment_pred_records = []     # 3. القرارات بعد الماسك

    for subj in unique_subjects:
        mask = subject_ids == subj
        subj_data = test_data[mask]
        subj_labels = test_labels[mask]

        if len(subj_data) == 0:
            continue

        subj_tensor = torch.tensor(subj_data, dtype=torch.float32, device=device)

        with torch.no_grad():
            outputs = trained_model(subj_tensor)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            max_conf = torch.max(probs, dim=1)[0]

        segment_preds = preds.cpu().numpy()
        segment_probs = probs.cpu().numpy()
        segment_conf = max_conf.cpu().numpy()

        # ------------------------------------------------------
        #
        # ------------------------------------------------------
        raw_prob_row = [subj]
        for p in segment_probs:
            for c_idx in range(len(class_names)):
                raw_prob_row.append(round(float(p[c_idx]), 4))
        segment_raw_prob_records.append(raw_prob_row)

        # ------------------------------------------------------
        # 
        # ------------------------------------------------------
        if confidence_filter is not None:
            conf_mask = segment_conf >= confidence_filter
            if conf_mask.sum() > 0:
                segment_preds = segment_preds[conf_mask]
                segment_probs = segment_probs[conf_mask]
            else:
                segment_preds = np.array([])
                segment_probs = np.empty((0, len(class_names)))

        if len(segment_preds) == 0:
            continue

        # ------------------------------------------------------
        # 
        # ------------------------------------------------------
        if aggregation_mode == 'average':
            mean_p = np.mean(segment_probs, axis=0)
            pred_class = np.argmax(mean_p)
        elif aggregation_mode == 'majority':
            pred_class = np.bincount(segment_preds).argmax()
            mean_p = np.mean(segment_probs, axis=0)
        else:
            raise ValueError("aggregation_mode must be 'average' or 'majority'.")
        
        true_class = subj_labels[0]
        subj_true.append(true_class)
        subj_pred.append(pred_class)
        subj_prob.append(mean_p)

        # ------------------------------------------------------
        # 
        # ------------------------------------------------------
        # 
        pred_row = [subj] + list(segment_preds)
        segment_pred_records.append(pred_row)

        #
        prob_row = [subj]
        for p in segment_probs:
            for c_idx in range(len(class_names)):
                prob_row.append(round(float(p[c_idx]), 4))
        segment_prob_records.append(prob_row)

        del subj_tensor, outputs, probs, preds
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ==========================================================
    # 
    # ==========================================================
    if save_segment_csv is not None:
        save_segment_csv = str(save_segment_csv)
        base_name, ext = os.path.splitext(save_segment_csv)
        if not ext:
            ext = '.csv'

        raw_probs_csv_path = f"{base_name}_raw_prob{ext}"
        probs_csv_path     = f"{base_name}_prob{ext}"
        preds_csv_path     = f"{base_name}_predict{ext}"

        _save_records_to_csv(segment_raw_prob_records, raw_probs_csv_path, class_names, is_prob=True)
        _save_records_to_csv(segment_prob_records, probs_csv_path, class_names, is_prob=True)
        _save_records_to_csv(segment_pred_records, preds_csv_path, class_names, is_prob=False)

    #
    all_labels = np.array(subj_true)
    all_preds = np.array(subj_pred)
    all_probs = np.vstack(subj_prob) if len(subj_prob) > 0 else np.zeros((0, len(class_names)))

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1], zero_division=0
    )
    
    sensitivity = recall[AD_index]
    specificity = recall[HC_index]
    f1_ad = f1[AD_index]
    precision_ad = precision[AD_index]

    roc_auc = 0.0
    if len(np.unique(all_labels)) > 1 and all_probs.shape[0] > 0 and all_probs.shape[1] > 1:
        try:
            roc_auc = roc_auc_score(all_labels, all_probs[:, AD_index])
        except Exception:
            pass

    metrics = {
        'mode': 'subject',
        'accuracy': acc,
        'precision': precision_ad,
        'recall': sensitivity,
        'f1_score': f1_ad,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'n_subjects': len(unique_subjects),
        'n_evaluated': len(all_labels),
        'y_true': all_labels,
        'y_proba': all_probs,
        
        'segment_records': {
            'raw_prob': segment_raw_prob_records,
            'prob': segment_prob_records,
            'predict': segment_pred_records
        }
    }

    if verbose:
        print("="*50)
        print("=== Final Subject-Level Metrics ===")
        print("="*50)
        print(f"Evaluated Subjects: {metrics['n_evaluated']}/{metrics['n_subjects']}")
        print(f"Accuracy:             {acc:.4f}")
        print(f"Sensitivity (AD):    {sensitivity:.4f}")
        print(f"Specificity (HC):    {specificity:.4f}")
        print(f"Precision (AD):      {precision_ad:.4f}")
        print(f"F1-Score (AD):       {f1_ad:.4f}")
        print(f"ROC-AUC:             {roc_auc:.4f}")
        print("="*50)

    return metrics