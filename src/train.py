# Author: Majid Hameed Khalaf
# Date: 2026 
#=============================
import os
import gc
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

try:
    from src.evaluate import comprehensive_evaluate_model
except ImportError:
    comprehensive_evaluate_model = None


# ============================================================================
# DATALOADER & HELPER UTILITIES
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

        batch_data = torch.tensor(self.data[batch_indices], dtype=torch.float32, device=self.device)
        batch_labels = torch.tensor(self.labels[batch_indices], dtype=torch.long, device=self.device)
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
  
    
    # 1. SETUP
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    has_validation = val_set is not None and val_set[0] is not None
    train_data, train_labels, _ = train_set if len(train_set) == 3 else (*train_set, None)

  
    class_weights = get_class_weights(train_labels, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # (Optimizer)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    #  (LR Scheduler)
    total_steps = (len(train_data) // batch_size) * epochs
    warmup_steps = (len(train_data) // batch_size) * warmup_epochs
    
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, threshold=0.001, cooldown=2
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    elif scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = None

    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == 'cuda') else None

    #  (Data Loaders)
    train_loader = MemoryEfficientDataLoader(
        train_data, train_labels, batch_size, shuffle=True, drop_last=True, device=device
    )
    
    if has_validation:
        val_data, val_labels = val_set[0], val_set[1]
        val_loader = MemoryEfficientDataLoader(
            val_data, val_labels, batch_size, shuffle=False, device=device
        )

    #  (History)
    history = {'train_loss': [], 'train_acc': [], 'learning_rate': []}
    if has_validation:
        history['val_loss'] = []
        history['val_acc'] = []
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    if verbose:
        print(f"\n{'='*80}\n<< TRAINING STARTED >>\n{'='*80}")
        print(f"Device: {device} | Train samples: {len(train_data):,}")
        if has_validation:
            print(f"Val samples: {len(val_data):,}")
        print(f"Batch size: {batch_size} | Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"Max epochs: {epochs} | Early stopping patience: {patience}")
        print(f"{'='*80}\n")

    # 2. TRAINING LOOP
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        optimizer.zero_grad()
        
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            # Forward pass
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels) / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels) / gradient_accumulation_steps
                loss.backward()

            # Optimizer Step (مع Gradient Accumulation)
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                if scheduler_type == 'linear' and scheduler is not None:
                    scheduler.step()

            # 
            total_loss += loss.item() * gradient_accumulation_steps
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

            del outputs, loss, preds
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # 
        epoch_train_loss = total_loss / len(train_loader)
        epoch_train_acc = 100 * total_correct / total_samples
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['learning_rate'].append(current_lr)

        # 3. VALIDATION
        if has_validation:
            model.eval()
            
            # (Subject-level)
            if validate_like_test and val_subject_ids is not None and comprehensive_evaluate_model is not None:
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
            
            #  (Segment-level)
            else:
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_data, batch_labels in val_loader:
                        if scaler is not None:
                            with torch.amp.autocast('cuda'):
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
                
                val_loss /= len(val_loader)
                val_acc = 100 * val_correct / val_total

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # update
            if scheduler_type == 'plateau' and scheduler is not None:
                scheduler.step(val_loss)
            elif scheduler_type == 'cosine' and scheduler is not None:
                scheduler.step()

            if verbose:
                print(f"Epoch [{epoch+1:3d}/{epochs}] | Train Acc: {epoch_train_acc:6.2f}% | Val Acc: {val_acc:6.2f}% | LR: {current_lr:.2e}")

            # Early Stopping & Checkpointing
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                if checkpoint_dir is not None:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                    save_checkpoint(model, optimizer, epoch, best_val_acc, history, checkpoint_path)
            else:
                patience_counter += 1
                if patience and patience_counter >= patience:
                    if verbose:
                        print(f"\n Early stopping triggered at epoch {epoch+1}. Best Val Acc: {best_val_acc:.2f}%\n")
                    break
        else:
            if verbose:
                print(f"Epoch [{epoch+1:3d}/{epochs}] | Train Acc: {epoch_train_acc:6.2f}% | LR: {current_lr:.2e}")
            if scheduler_type == 'cosine' and scheduler is not None:
                scheduler.step()

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 
    if best_model_state:
        model.load_state_dict(best_model_state)

    if verbose:
        print(f"\n{'='*80}\nTRAINING COMPLETED | Best Val Acc: {best_val_acc:.2f}%\n{'='*80}\n")
    
    return model, history