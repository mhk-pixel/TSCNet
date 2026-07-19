#Training and Evaluation class for EEG Classification
#=====================================================
import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix, 
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


# ============================================================================
# MEMORY-EFFICIENT DATALOADER
# ============================================================================

class MemoryEfficientDataLoader:
    
    def __init__(self, data, labels, batch_size=32, shuffle=True, drop_last=False, device='cpu'):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = device
        self.indices = np.arange(len(data))
        if shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration

        end_idx = min(self.current_idx + self.batch_size, len(self.indices))
        batch_indices = self.indices[self.current_idx:end_idx]

        if self.drop_last and len(batch_indices) < self.batch_size:
            raise StopIteration

        batch_data = torch.FloatTensor(self.data[batch_indices]).to(self.device)
        batch_labels = torch.LongTensor(self.labels[batch_indices]).to(self.device)
        self.current_idx = end_idx
        return batch_data, batch_labels

    def __len__(self):
        if self.drop_last:
            return len(self.indices) // self.batch_size
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def get_class_weights(train_labels, device='cpu'):
 
    classes = np.unique(train_labels)
    weights = compute_class_weight('balanced', classes=classes, y=train_labels)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
 
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def save_checkpoint(model, optimizer, epoch, best_acc, history, filepath):

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'history': history
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
   
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['best_acc'], checkpoint['history']


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model_with_early_stopping(
    model,
    train_set,
    val_set=None,
    batch_size=32,
    epochs=50,
    learning_rate=1e-3,
    patience=10,
    validate_like_test=False,
    val_subject_ids=None,
    confidence_filter=None,
    aggregation_mode='average',
   
    device='cuda',
    use_amp=True,
    gradient_accumulation_steps=1,
    warmup_epochs=0,
    scheduler_type='cosine',
    checkpoint_dir=None,
    verbose=True
):
    
    # ========================================================================
    # 1. SETUP
    # ========================================================================
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    has_validation = val_set is not None and val_set[0] is not None
    train_data, train_labels,_ = train_set

    # Compute class weights for balanced loss
    class_weights = get_class_weights(train_labels, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate scheduler
    total_steps = len(train_data) // batch_size * epochs
    warmup_steps = len(train_data) // batch_size * warmup_epochs
    
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
            min_lr=1e-6, 
            threshold=0.001, 
            cooldown=2
            
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6
        )
    elif scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    

    # Create data loaders
    train_loader = MemoryEfficientDataLoader(
        train_data, train_labels, batch_size, shuffle=True, drop_last=True, device=device
    )
    
    if has_validation:
        val_data, val_labels = val_set
        val_loader = MemoryEfficientDataLoader(
            val_data, val_labels, batch_size, shuffle=False, device=device
        )

    # Initialize history
    history = {
        'train_loss': [], 
        'train_acc': [],
        'learning_rate': []
    }
    
    if has_validation:
        history['val_loss'] = []
        history['val_acc'] = []
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    if verbose:
        print(f"\n{'='*80}")
        print("<<TRAINING STARTED>>")
        print(f"{'='*80}")
        print(f"Device: {device}")
        print(f"Train samples: {len(train_data):,}")
        if has_validation:
            print(f"Val samples: {len(val_data):,}")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"Max epochs: {epochs}")
        if warmup_epochs > 0:
            print(f"Warmup epochs: {warmup_epochs}")
        if patience:
            print(f"Early stopping patience: {patience}")
        if scheduler_type:
            print(f"LR scheduler: {scheduler_type}")
        print(f"{'='*80}\n")

    # ========================================================================
    # 2. TRAINING LOOP
    # ========================================================================
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_correct = 0
        total_samples = 0
        
      
        optimizer.zero_grad()
        
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            # ================================================================
          
            # ================================================================
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
            else:
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

            # ================================================================
            # BACKWARD PASS
            # ================================================================
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # ================================================================
            # OPTIMIZER STEP (with gradient accumulation)
            # ================================================================
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Step linear scheduler per batch
                if scheduler_type == 'linear' and scheduler is not None:
                    scheduler.step()

            # ================================================================
            # METRICS
            # ================================================================
            total_loss += loss.item() * gradient_accumulation_steps
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

            # Cleanup
            del outputs, loss, preds
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # ====================================================================
        # EPOCH METRICS
        # ====================================================================
        epoch_train_loss = total_loss / len(train_loader)
        epoch_train_acc = 100 * total_correct / total_samples
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['learning_rate'].append(current_lr)
        

        # ====================================================================
        # VALIDATION
        # ====================================================================
        if has_validation:
            model.eval()
            
            # ----------------------------------------------------------------
            # Subject-level validation (like test)
            # ----------------------------------------------------------------
            if validate_like_test and val_subject_ids is not None:
                val_metrics = comprehensive_evaluate_model(
                    trained_model=model,
                    test_set=val_set,
                    class_names=['HC', 'AD'],
                    subject_ids=val_subject_ids,
                    confidence_filter=confidence_filter,
                    aggregation_mode=aggregation_mode,
                    save_segment_csv=None,
                    device=device,
                    verbose=False
                )
                val_acc = val_metrics['accuracy'] * 100
                val_loss = 1 - val_metrics['accuracy']
                
                if verbose:
                    print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                          f"Train: {epoch_train_acc:6.2f}% | "
                          f"Val: {val_acc:6.2f}% | "
                          f"AUC: {val_metrics['roc_auc']:.3f} | "
                          f"Sens: {val_metrics['sensitivity']*100:.1f}% | "
                          f"Spec: {val_metrics['specificity']*100:.1f}% | "
                          f"LR: {current_lr:.2e}", end='')
                
            # ----------------------------------------------------------------
            # Standard segment-level validation
            # ----------------------------------------------------------------
            else:
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_data, batch_labels in val_loader:
                        if scaler is not None:
                            with torch.cuda.amp.autocast():
                                outputs = model(batch_data)
                                loss = criterion(outputs, batch_labels)
                        else:
                            outputs = model(batch_data)
                            loss = criterion(outputs, batch_labels)
                        
                        val_loss += loss.item()
                        
                        _, preds = torch.max(outputs, 1)
                        val_correct += (preds == batch_labels).sum().item()
                        val_total += batch_labels.size(0)

                        del outputs, loss, preds
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                
                val_loss /= len(val_loader)
                val_acc = 100 * val_correct / val_total
                
                if verbose:
                    print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                          f"Train: {epoch_train_acc:6.2f}% | "
                          f"Val: {val_acc:6.2f}% | "
                          f"LR: {current_lr:.2e}", end='')

            # ----------------------------------------------------------------
            # Log validation metrics
            # ----------------------------------------------------------------
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Step scheduler
            if scheduler_type == 'plateau' and scheduler is not None:
                scheduler.step(val_loss)
            elif scheduler_type == 'cosine' and scheduler is not None:
                scheduler.step()

            
            if verbose:
                print()  # New line

            # ----------------------------------------------------------------
            # Early stopping and checkpointing
            # ----------------------------------------------------------------
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                if verbose:
                    print(f"  New best model (val acc={best_val_acc:.2f}%)")
                
                # Save checkpoint
                if checkpoint_dir is not None:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                    save_checkpoint(model, optimizer, epoch, best_val_acc, history, checkpoint_path)
                    if verbose:
                        print(f"  Checkpoint saved: {checkpoint_path}")
            else:
                patience_counter += 1
                if patience and patience_counter >= patience:
                    if verbose:
                        print(f"\n Early stopping at epoch {epoch+1}. Best acc={best_val_acc:.2f}%\n")
                    break
        
        # ====================================================================
        # NO VALIDATION
        # ====================================================================
        else:
            if verbose:
                print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                      f"Train acc={epoch_train_acc:.2f}% | "
                      f"LR: {current_lr:.2e}", end='')
              
                print()
            
            # Step scheduler
            if scheduler_type == 'cosine' and scheduler is not None:
                scheduler.step()

        # Cleanup
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ========================================================================
    # 3. LOAD BEST MODEL
    # ========================================================================
    if best_model_state:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"\nLoaded best model (val acc={best_val_acc:.2f}%)")

    if verbose:
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print(f"{'='*80}\n")
    
    return model, history


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def comprehensive_evaluate_model(
    trained_model,
    test_set,
    class_names=['HC', 'AD'],
    subject_ids=None,
    confidence_filter=0.8,
    aggregation_mode='average',
    save_segment_csv=None,
    device='cuda',
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.eval()
    test_data, test_labels = test_set

 
    HC_index = class_names.index('HC')  
    AD_index = class_names.index('AD')  

    
    # 1) segment-level
    # ==========================================================
    if subject_ids is None:
        print("\n=== Segment-level Evaluation ===")
       
        loader = MemoryEfficientDataLoader(test_data, test_labels, batch_size=32, shuffle=False)

        all_preds, all_labels, all_probs, all_conf = [], [], [], []

        with torch.no_grad():
            for batch_data, batch_labels in loader:
                batch_data = batch_data.to(device)
                outputs = trained_model(batch_data)
                probs = F.softmax(outputs, dim=1)
                max_p, preds = torch.max(probs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                all_conf.extend(max_p.cpu().numpy())

                del outputs, probs, preds, batch_data, batch_labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_conf = np.array(all_conf)

        if confidence_filter is not None:
            mask = all_conf >= confidence_filter
            if mask.sum() == 0:
                print(f"No segments above confidence threshold {confidence_filter}, using all segments.")
            else:
                all_preds = all_preds[mask]
                all_labels = all_labels[mask]
                all_probs = all_probs[mask]
                all_conf = all_conf[mask]

     
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
            'n_samples': len(all_labels)
        }

        print("\n=== Segment-level Metrics ===")
        print(f"Samples:              {metrics['n_samples']}")
        print(f"Accuracy:             {acc:.4f}")
        print(f"Sensitivity (AD Rec): {sensitivity:.4f}")
        print(f"Specificity (HC Rec): {specificity:.4f}")
        print(f"Precision (AD):       {precision_ad:.4f}")
        print(f"F1-Score (AD):        {f1_ad:.4f}")
        print(f"ROC-AUC:              {roc_auc:.4f}")
        print("\nConfusion Matrix (rows=True, cols=Pred):")
        print(f"           {class_names[0]}    {class_names[1]}")
        print(f" {class_names[0]}(0):   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f" {class_names[1]}(1):   {cm[1,0]:4d}  {cm[1,1]:4d}")

        return metrics

    
    #  (Subject-level evaluation)
    # ==========================================================
    subject_ids = np.array(subject_ids)
    unique_subjects = np.unique(subject_ids)

    print(f"\n=== Subject-level Evaluation ({len(unique_subjects)} subjects) ===")

    subj_true, subj_pred, subj_prob = [], [], []
    segment_records = []  

    for subj in unique_subjects:
        mask = subject_ids == subj
        subj_data = test_data[mask]
        subj_labels = test_labels[mask]

        if len(subj_data) == 0:
            continue

        subj_tensor = torch.tensor(subj_data).float().to(device)

        with torch.no_grad():
            outputs = trained_model(subj_tensor)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            max_conf = torch.max(probs, dim=1)[0]

        segment_preds = preds.cpu().numpy()
        segment_probs = probs.cpu().numpy()
        segment_conf = max_conf.cpu().numpy()

        if confidence_filter is not None:
            conf_mask = segment_conf >= confidence_filter
            if conf_mask.sum() == 0:
                print(f"Subject {subj}: No segments above confidence threshold {confidence_filter}, skipping subject.")
                continue
            segment_preds = segment_preds[conf_mask]
            segment_probs = segment_probs[conf_mask]
            segment_conf = segment_conf[conf_mask]

        if len(segment_preds) == 0:
            print(f"Subject {subj}: no segments left after filtering, skipping.")
            continue

        AD_n = np.sum(segment_preds == AD_index)
        HC_n = np.sum(segment_preds == HC_index)
        total_n = AD_n + HC_n
        AD_ratio = AD_n / total_n if total_n > 0 else 0.0
        HC_ratio = HC_n / total_n if total_n > 0 else 0.0

        print(f"Subject {subj}: AD={AD_n}, HC={HC_n} | Ratios → AD={AD_ratio:.2f}, HC={HC_ratio:.2f}")

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
        
        print(f"   Final: Pred={pred_class} ({class_names[pred_class]}), True={true_class} ({class_names[true_class]})")

        row = [subj] + list(segment_preds)
        segment_records.append(row)

        del subj_tensor, outputs, probs, preds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==============================
    if save_segment_csv is not None and len(segment_records) > 0:
        max_segments = max(len(r) - 1 for r in segment_records)
        for row in segment_records:
            while (len(row) - 1) < max_segments:
                row.append(np.nan)
        cols = ['sub'] + [f'seg{i+1}_pred' for i in range(max_segments)]
        df_segments = pd.DataFrame(segment_records, columns=cols)

        if os.path.exists(save_segment_csv):
            df_segments.to_csv(save_segment_csv, mode='a', index=False, header=False)
            print(f"Appended segment predictions to → {save_segment_csv}")
        else:
            df_segments.to_csv(save_segment_csv, index=False)
            print(f"Saved segment predictions to → {save_segment_csv}")

    # =============================================
    all_labels = np.array(subj_true)
    all_preds = np.array(subj_pred)
    all_probs = np.vstack(subj_prob) if len(subj_prob) > 0 else np.zeros((0, len(class_names)))

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1], zero_division=0
    )
    
    # ربط المقاييس ديناميكياً بالمؤشرات الصحيحة للفئات
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
        'y_proba': all_probs
    }

    print("\n" + "="*50)
    print("=== Final Subject-Level Metrics ===")
    print("="*50)
    print(f"Total Subjects:      {metrics['n_subjects']}")
    print(f"Evaluated Subjects:  {metrics['n_evaluated']}")
    print(f"Accuracy:            {acc:.4f}")
    print(f"Sensitivity (AD):    {sensitivity:.4f}")
    print(f"Specificity (HC):    {specificity:.4f}")
    print(f"Precision (AD):      {precision_ad:.4f}")
    print(f"F1-Score (AD):       {f1_ad:.4f}")
    print(f"ROC-AUC:             {roc_auc:.4f}")
    print("\nConfusion Matrix (rows=True, cols=Pred):")
    print(f"           {class_names[0]}    {class_names[1]}")
    print(f" {class_names[0]}(0):   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f" {class_names[1]}(1):   {cm[1,0]:4d}  {cm[1,1]:4d}")
    print("="*50)

    return metrics
