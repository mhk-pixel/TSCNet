import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import auc, roc_curve
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')


class EEGVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        self.COLORS = {
            'train': '#1f77b4',       # Blue
            'val': '#ff7f0e',         # Orange
            'test': '#2ca02c',        # Green
            'best': '#d62728',        # Red
            'entropy': '#9467bd',     # Purple
            'gap': '#8c564b',         # Brown
            'overfitting': '#e377c2', # Pink
            'lr': '#17becf',          # Cyan
            'ce_loss': '#bcbd22'      # Yellow-green
        }

        self.FONT_TITLE = {'fontweight': 'bold', 'fontsize': 14}
        self.FONT_LABEL = {'fontweight': 'bold', 'fontsize': 12}
        self.FONT_LEGEND = {'fontsize': 10}

    # ============================================================================
    # TRAINING CURVES
    # ============================================================================

    def plot_training_curves(self, history, fold_num, save_dir='plots'):
        os.makedirs(save_dir, exist_ok=True)

        has_validation = 'val_loss' in history and bool(history['val_loss'])
        has_lr = 'learning_rate' in history and bool(history['learning_rate'])

        n_plots = 2
        if has_lr:
            n_plots += 1

        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        epochs = range(1, len(history['train_loss']) + 1)
        plot_idx = 0

        # Loss Curves
        ax = axes[plot_idx]
        ax.plot(
            epochs, history['train_loss'],
            color=self.COLORS['train'],
            label='Training Loss',
            linewidth=2.5,
            marker='o',
            markersize=4,
            alpha=0.8
        )

        if has_validation:
            ax.plot(
                epochs, history['val_loss'],
                color=self.COLORS['val'],
                label='Validation Loss',
                linewidth=2.5,
                marker='s',
                markersize=4,
                alpha=0.8
            )

            best_epoch = int(np.argmin(history['val_loss'])) + 1
            best_val_loss = min(history['val_loss'])

            ax.axvline(
                best_epoch,
                color=self.COLORS['best'],
                linestyle='--',
                linewidth=2,
                label=f'Best Epoch ({best_epoch})',
                alpha=0.7
            )

            ax.scatter(
                [best_epoch], [best_val_loss],
                color=self.COLORS['best'],
                s=150,
                marker='*',
                zorder=5,
                edgecolors='black',
                linewidths=1.5
            )

        ax.set_title(f'Fold {fold_num} - Loss Curves', **self.FONT_TITLE)
        ax.set_xlabel('Epoch', **self.FONT_LABEL)
        ax.set_ylabel('Loss', **self.FONT_LABEL)
        ax.legend(loc='best', framealpha=0.9, **self.FONT_LEGEND)
        ax.grid(alpha=0.3, linestyle='--')

        plot_idx += 1

        # Accuracy Curves
        ax = axes[plot_idx]
        ax.plot(
            epochs, history['train_acc'],
            color=self.COLORS['train'],
            label='Training Accuracy',
            linewidth=2.5,
            marker='o',
            markersize=4,
            alpha=0.8
        )

        if has_validation:
            ax.plot(
                epochs, history['val_acc'],
                color=self.COLORS['val'],
                label='Validation Accuracy',
                linewidth=2.5,
                marker='s',
                markersize=4,
                alpha=0.8
            )

            best_val_acc = history['val_acc'][best_epoch - 1]

            ax.axvline(
                best_epoch,
                color=self.COLORS['best'],
                linestyle='--',
                linewidth=2,
                label=f'Best Epoch ({best_epoch})',
                alpha=0.7
            )

            ax.scatter(
                [best_epoch], [best_val_acc],
                color=self.COLORS['best'],
                s=150,
                marker='*',
                zorder=5,
                edgecolors='black',
                linewidths=1.5
            )

        ax.set_title(f'Fold {fold_num} - Accuracy Curves', **self.FONT_TITLE)
        ax.set_xlabel('Epoch', **self.FONT_LABEL)
        ax.set_ylabel('Accuracy (%)', **self.FONT_LABEL)
        ax.legend(loc='best', framealpha=0.9, **self.FONT_LEGEND)
        ax.grid(alpha=0.3, linestyle='--')

        plot_idx += 1

        # Learning Rate
        if has_lr:
            ax = axes[plot_idx]
            ax.plot(
                epochs, history['learning_rate'],
                color=self.COLORS['lr'],
                label='Learning Rate',
                linewidth=2.5,
                marker='v',
                markersize=4,
                alpha=0.8
            )

            min_lr = max(min(history['learning_rate']), 1e-12)
            lr_range = max(history['learning_rate']) / min_lr
            if lr_range > 10:
                ax.set_yscale('log')

            ax.set_title(f'Fold {fold_num} - Learning Rate Schedule', **self.FONT_TITLE)
            ax.set_xlabel('Epoch', **self.FONT_LABEL)
            ax.set_ylabel('Learning Rate', **self.FONT_LABEL)
            ax.legend(loc='best', framealpha=0.9, **self.FONT_LEGEND)
            ax.grid(alpha=0.3, linestyle='--', which='both')

        plt.tight_layout()

        save_path = os.path.join(save_dir, f'fold_{fold_num}_training_curves.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"Training curves saved: {save_path}")
        return fig

    # ============================================================================
    # OVERFITTING ANALYSIS
    # ============================================================================

    def plot_overfitting_analysis(self, history, fold_num, save_dir='plots'):
        if 'val_loss' not in history or not history['val_loss']:
            print(f"Skipping overfitting analysis for Fold {fold_num} (no validation data)")
            return

        os.makedirs(save_dir, exist_ok=True)
        epochs = range(1, len(history['train_loss']) + 1)

        loss_gap = np.array(history['val_loss']) - np.array(history['train_loss'])
        acc_gap = np.array(history['train_acc']) - np.array(history['val_acc'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss Gap
        axes[0].plot(
            epochs, loss_gap,
            color=self.COLORS['gap'],
            linewidth=2.5,
            label='Val - Train Loss',
            marker='o',
            markersize=4
        )
        axes[0].axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        axes[0].fill_between(
            epochs, 0, loss_gap,
            where=(loss_gap > 0),
            color=self.COLORS['overfitting'],
            alpha=0.3,
            label='Overfitting Zone'
        )
        axes[0].fill_between(
            epochs, 0, loss_gap,
            where=(loss_gap < 0),
            color='lightblue',
            alpha=0.3,
            label='Underfitting Zone'
        )

        stats_text = (
            f'Mean Gap: {np.mean(loss_gap):.4f}\n'
            f'Max Gap: {np.max(loss_gap):.4f}\n'
            f'Min Gap: {np.min(loss_gap):.4f}\n'
            f'Final Gap: {loss_gap[-1]:.4f}'
        )

        axes[0].text(
            0.02, 0.98, stats_text,
            transform=axes[0].transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9
        )

        axes[0].set_title(f'Fold {fold_num} - Loss Gap Analysis', **self.FONT_TITLE)
        axes[0].set_xlabel('Epoch', **self.FONT_LABEL)
        axes[0].set_ylabel('Loss Gap (Val - Train)', **self.FONT_LABEL)
        axes[0].legend(loc='best', framealpha=0.9, **self.FONT_LEGEND)
        axes[0].grid(alpha=0.3, linestyle='--')

        # Accuracy Gap
        axes[1].plot(
            epochs, acc_gap,
            color='darkorange',
            linewidth=2.5,
            label='Train - Val Accuracy',
            marker='s',
            markersize=4
        )
        axes[1].axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        axes[1].fill_between(
            epochs, 0, acc_gap,
            where=(acc_gap > 0),
            color=self.COLORS['overfitting'],
            alpha=0.3,
            label='Overfitting Zone'
        )
        axes[1].fill_between(
            epochs, 0, acc_gap,
            where=(acc_gap < 0),
            color='lightblue',
            alpha=0.3,
            label='Underfitting Zone'
        )

        stats_text = (
            f'Mean Gap: {np.mean(acc_gap):.2f}%\n'
            f'Max Gap: {np.max(acc_gap):.2f}%\n'
            f'Min Gap: {np.min(acc_gap):.2f}%\n'
            f'Final Gap: {acc_gap[-1]:.2f}%'
        )

        axes[1].text(
            0.02, 0.98, stats_text,
            transform=axes[1].transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9
        )

        axes[1].set_title(f'Fold {fold_num} - Accuracy Gap Analysis', **self.FONT_TITLE)
        axes[1].set_xlabel('Epoch', **self.FONT_LABEL)
        axes[1].set_ylabel('Accuracy Gap (Train - Val) %', **self.FONT_LABEL)
        axes[1].legend(loc='best', framealpha=0.9, **self.FONT_LEGEND)
        axes[1].grid(alpha=0.3, linestyle='--')

        plt.tight_layout()

        save_path = os.path.join(save_dir, f'fold_{fold_num}_overfitting_analysis.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"Overfitting analysis saved: {save_path}")
        return fig

# ============================================================================
    # CONFUSION MATRIX
    # ============================================================================

    def plot_confusion_matrix(self, cm, class_names, fold_num, save_dir='plots', normalize=False):
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))

        sums = np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)
        cm_percent = cm.astype('float') / sums * 100

        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.2f}%)'

        sns.heatmap(
            cm_percent if normalize else cm,
            annot=annotations,
            fmt='',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Percentage (%)' if normalize else 'Number of Samples'},
            linewidths=2,
            linecolor='white',
            ax=ax,
            vmin=0,
            square=True
        )

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        ax.set_xlabel('Predicted Label', **self.FONT_LABEL)
        ax.set_ylabel('True Label', **self.FONT_LABEL)
        
      
        ax.set_title(f'Confusion Matrix - Fold {fold_num}', pad=20, **self.FONT_TITLE)

        plt.tight_layout()

        save_path = os.path.join(save_dir, f'fold_{fold_num}_confusion_matrix.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"Confusion matrix saved: {save_path}")
        return fig

    # ============================================================================
    # MEAN CONFUSION MATRIX
    # ============================================================================

    def plot_mean_confusion_matrix(self, all_fold_results, class_names, save_dir='plots'):
        os.makedirs(save_dir, exist_ok=True)

        cms = [r['test_metrics']['confusion_matrix'] for r in all_fold_results]
        cms = np.stack(cms, axis=0)
        sum_cm = np.sum(cms, axis=0)

        cms_normalized = np.array([
            cm.astype(float) / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)
            for cm in cms
        ])
        std_cm_norm = np.std(cms_normalized, axis=0) * 100
        cm_percent = sum_cm.astype(float) / np.maximum(sum_cm.sum(axis=1)[:, np.newaxis], 1e-12) * 100

        annotations = np.empty_like(sum_cm).astype(str)
        for i in range(sum_cm.shape[0]):
            for j in range(sum_cm.shape[1]):
                annotations[i, j] = (
                    f'{sum_cm[i, j]}\n'
                    f'({cm_percent[i, j]:.2f}%)\n'
                    f'±{std_cm_norm[i, j]:.2f}%'
                )

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            sum_cm,
            annot=annotations,
            fmt='',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Number of Samples'},
            linewidths=2,
            linecolor='white',
            ax=ax,
            vmin=0,
            square=True
        )

   
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        ax.set_xlabel('Predicted Label', **self.FONT_LABEL)
        ax.set_ylabel('True Label', **self.FONT_LABEL)
        
     
        ax.set_title('Mean Confusion Matrix (All Folds)', pad=20, **self.FONT_TITLE)

        plt.tight_layout()

        save_path = os.path.join(save_dir, 'mean_confusion_matrix.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"Mean confusion matrix saved: {save_path}")
        return fig
    # ============================================================================
    # CROSS-VALIDATION SUMMARY
    # ============================================================================

    def plot_cross_validation_summary(self, all_fold_results, save_dir='plots'):
        os.makedirs(save_dir, exist_ok=True)

        folds = [r['fold'] for r in all_fold_results]
        accs = [r['test_metrics']['accuracy'] for r in all_fold_results]
        aucs = [r['test_metrics']['roc_auc'] for r in all_fold_results]
        sens = [r['test_metrics']['sensitivity'] for r in all_fold_results]
        spec = [r['test_metrics']['specificity'] for r in all_fold_results]
        f1s = [
            np.mean(r['test_metrics']['f1_score'])
            if isinstance(r['test_metrics']['f1_score'], (list, np.ndarray))
            else r['test_metrics']['f1_score']
            for r in all_fold_results
        ]

        x = np.arange(len(folds))

        # 1. Bar Chart
        fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
        width = 0.15
        colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        bars = [
            ax_bar.bar(x - 2*width, accs, width, label='Accuracy', color=colors_bar[0], alpha=0.8),
            ax_bar.bar(x - width, aucs, width, label='AUC', color=colors_bar[1], alpha=0.8),
            ax_bar.bar(x, sens, width, label='Sensitivity', color=colors_bar[2], alpha=0.8),
            ax_bar.bar(x + width, spec, width, label='Specificity', color=colors_bar[3], alpha=0.8),
            ax_bar.bar(x + 2*width, f1s, width, label='F1-Score', color=colors_bar[4], alpha=0.8)
        ]

        for bar_group in bars:
            for bar in bar_group:
                height = bar.get_height()
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=7
                )

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f'Fold {f}' for f in folds])
        ax_bar.set_ylim(0, 1.1)
        ax_bar.set_title('Performance Metrics per Fold', **self.FONT_TITLE)
        ax_bar.set_xlabel('Fold', **self.FONT_LABEL)
        ax_bar.set_ylabel('Score', **self.FONT_LABEL)
        ax_bar.grid(alpha=0.3, axis='y', linestyle='--')
        ax_bar.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., framealpha=0.9)
        ax_bar.axhline(y=np.mean(accs), color='red', linestyle='--', linewidth=1, alpha=0.5, label='Mean Accuracy')

        fig_bar.tight_layout()
        save_path = os.path.join(save_dir, 'cv_bar_per_fold.png')
        fig_bar.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig_bar)

        # 2. Boxplot
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        metrics = [accs, aucs, sens, spec, f1s]
        labels = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'F1-Score']
        colors_box = ['skyblue', 'plum', 'lightgreen', 'orange', 'gold']

        bp = ax_box.boxplot(
            metrics,
            patch_artist=True,
            labels=labels,
            widths=0.6,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=8, markeredgecolor='black'),
            medianprops=dict(color='darkblue', linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5)
        )

        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax_box.set_title('Metric Distributions Across Folds', **self.FONT_TITLE)
        ax_box.set_ylim(0, 1.05)
        ax_box.set_ylabel('Score', **self.FONT_LABEL)
        ax_box.grid(alpha=0.3, axis='y', linestyle='--')

        for i, (metric, label) in enumerate(zip(metrics, labels)):
            ax_box.text(
                i + 1, 1.02,
                f'μ={np.mean(metric):.3f}±{np.std(metric):.3f}\nM={np.median(metric):.3f}',
                ha='center',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        fig_box.tight_layout()
        save_path = os.path.join(save_dir, 'cv_box_summary.png')
        fig_box.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig_box)

        # 3. Accuracy Trend
        fig_trend, ax_trend = plt.subplots(figsize=(10, 6))
        mean_acc, std_acc = np.mean(accs), np.std(accs)
        min_acc, max_acc = min(accs), max(accs)
        xs = np.arange(1, len(folds) + 1)

        ax_trend.plot(
            xs, accs,
            marker='o', linestyle='-', color='blue',
            linewidth=2.5, markersize=8, label='Accuracy', alpha=0.8
        )
        ax_trend.axhline(
            mean_acc, linestyle='--', color='red', linewidth=2,
            label=f'Mean = {mean_acc:.3f} ± {std_acc:.3f}', alpha=0.7
        )
        ax_trend.fill_between(
            xs, mean_acc - std_acc, mean_acc + std_acc,
            alpha=0.2, color='red', label='±1 Std Dev'
        )

        for i, (fold, acc) in enumerate(zip(folds, accs)):
            ax_trend.text(
                xs[i], acc + 0.01,
                f'{acc:.3f}',
                ha='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.3)
            )

        ax_trend.set_xticks(xs)
        ax_trend.set_xticklabels([str(f) for f in folds])
        ax_trend.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax_trend.set_title('Accuracy Trend Across Folds', **self.FONT_TITLE)
        ax_trend.set_xlabel('Fold', **self.FONT_LABEL)
        ax_trend.set_ylabel('Accuracy', **self.FONT_LABEL)
        ax_trend.set_ylim(max(0.5, min_acc - 0.05), min(1.0, max_acc + 0.05))
        ax_trend.grid(alpha=0.3, linestyle='--')
        ax_trend.legend(loc='best', framealpha=0.9)

        fig_trend.tight_layout()
        save_path = os.path.join(save_dir, 'cv_accuracy_trend.png')
        fig_trend.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig_trend)

        # 4. Radar Chart
        fig_radar = plt.figure(figsize=(8, 8))
        ax_radar = fig_radar.add_subplot(111, projection='polar')

        mean_metrics = [np.mean(accs), np.mean(aucs), np.mean(sens), np.mean(spec), np.mean(f1s)]
        metric_labels = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'F1-Score']

        num_vars = len(metric_labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        mean_metrics += mean_metrics[:1]
        angles += angles[:1]

        ax_radar.plot(angles, mean_metrics, 'o-', linewidth=2, color='blue', label='Mean Performance')
        ax_radar.fill(angles, mean_metrics, alpha=0.25, color='blue')

        ax_radar.set_theta_offset(np.pi / 2)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metric_labels, fontsize=11)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_radar.grid(True, linestyle='--', alpha=0.7)
        ax_radar.set_title('Mean Performance Radar Chart', **self.FONT_TITLE, pad=20)

        fig_radar.tight_layout()
        save_path = os.path.join(save_dir, 'cv_radar_chart.png')
        fig_radar.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig_radar)

        # Print Summary
        print(f"\n{'='*80}\nCross-Validation Summary\n{'='*80}")
        print(f"  Accuracy:     {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  AUC:          {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  Sensitivity:  {np.mean(sens):.4f} ± {np.std(sens):.4f}")
        print(f"  Specificity:  {np.mean(spec):.4f} ± {np.std(spec):.4f}")
        print(f"  F1-Score:     {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n{'='*80}\n")

        self.plot_all_training_loss(all_fold_results, save_dir=save_dir)
        self.plot_mean_confusion_matrix(all_fold_results, class_names=['HC', 'AD'], save_dir=save_dir)
   
        
    # ============================================================================
    # COMBINED CONFUSION MATRICES (3x3 GRID LAYOUT)
    # ============================================================================
    
    def plot_combined_confusion_matrices(self, all_fold_results, class_names, save_dir='plots'):
      
        os.makedirs(save_dir, exist_ok=True)
    
       
        fig = plt.figure(figsize=(16, 16))
        
       
        gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.35, hspace=0.35)
    
       
        fold_positions = [
            (0, 0),  # Fold 1
            (1, 0),  # Fold 2
            (2, 0),  # Fold 3
            (2, 1),  # Fold 4
            (2, 2)   # Fold 5
        ]
    
        # ------------------------------------------------------------------------
        # 1. (F1, F2, F3, F4, F5)
        # ------------------------------------------------------------------------
        for idx, result in enumerate(all_fold_results[:5]):
            row, col = fold_positions[idx]
            ax = fig.add_subplot(gs[row, col])
    
            cm = result['test_metrics']['confusion_matrix']
            fold_num = result.get('fold', idx + 1)
    
            sums = np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)
            cm_percent = cm.astype('float') / sums * 100
    
            annotations = np.empty_like(cm, dtype=object)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.2f}%)'
    
            sns.heatmap(
                cm,
                annot=annotations,
                fmt='',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar=True, 
                cbar_kws={'label': 'Samples'}, 
               
                linewidths=1.5,
                linecolor='white',
                ax=ax,
                vmin=0,
                square=True,
                annot_kws={'size': 10}
            )
    
            
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_xlabel('Predicted Label', **self.FONT_LABEL)
            ax.set_ylabel('True Label', **self.FONT_LABEL)
            ax.set_title(f'Fold {fold_num}', pad=15, **self.FONT_TITLE)
    
        # ------------------------------------------------------------------------
        # 
        # ------------------------------------------------------------------------
        ax_mean = fig.add_subplot(gs[0:2, 1:3])
    
        cms = [r['test_metrics']['confusion_matrix'] for r in all_fold_results]
        cms = np.stack(cms, axis=0)
        sum_cm = np.sum(cms, axis=0)
    
        cms_normalized = np.array([
            cm.astype(float) / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)
            for cm in cms
        ])
        std_cm_norm = np.std(cms_normalized, axis=0) * 100
        cm_percent_mean = sum_cm.astype(float) / np.maximum(sum_cm.sum(axis=1)[:, np.newaxis], 1e-12) * 100
    
        annotations_mean = np.empty_like(sum_cm, dtype=object)
        for i in range(sum_cm.shape[0]):
            for j in range(sum_cm.shape[1]):
                annotations_mean[i, j] = (
                    f'{sum_cm[i, j]}\n'
                    f'({cm_percent_mean[i, j]:.2f}%)\n'
                    f'±{std_cm_norm[i, j]:.2f}%'
                )
    
        sns.heatmap(
            sum_cm,
            annot=annotations_mean,
            fmt='',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Total Samples'},
            linewidths=2,
            linecolor='white',
            ax=ax_mean,
            vmin=0,
            square=True,
            annot_kws={'size': 12}
        )
    
        
        ax_mean.xaxis.tick_top()
        ax_mean.xaxis.set_label_position('top')
        ax_mean.set_xlabel('Predicted Label', **self.FONT_LABEL)
        ax_mean.set_ylabel('True Label', **self.FONT_LABEL)
        ax_mean.set_title('Mean Confusion Matrix (All Folds)', pad=20, **self.FONT_TITLE)
    
        # save
        save_path = os.path.join(save_dir, 'combined_confusion_matrices.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
        print(f"Combined confusion matrices saved successfully: {save_path}")
        return fig
    # ============================================================================
    # ROC CURVE
    # ============================================================================

    def plot_roc_from_folds(self, all_fold_results, save_dir='plots'):
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))

        tprs, aucs_fold = [], []
        mean_fpr = np.linspace(0, 1, 500)
        fold_colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_fold_results), 1)))

        for i, result in enumerate(all_fold_results):
            y_true = result['test_metrics']['y_true']
            y_proba = result['test_metrics']['y_proba']

            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                scores = y_proba[:, 1]
            else:
                scores = y_proba.ravel()

            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            aucs_fold.append(roc_auc)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

            ax.plot(
                fpr, tpr,
                lw=1.2, alpha=0.6,
                color=fold_colors[i],
                label=f'ROC Fold {result["fold"]} (AUC = {roc_auc:.2f})'
            )

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        mean_auc, std_auc = np.mean(aucs_fold), np.std(aucs_fold)

        ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1.3, label='Chance', zorder=1)

        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=0.3, label='± 1 std. dev.', zorder=2)

        ax.plot(
            mean_fpr, mean_tpr,
            color='blue', lw=2.2,
            label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
            zorder=6
        )

        ax.set_xlabel('False Positive Rate', **self.FONT_LABEL)
        ax.set_ylabel('True Positive Rate', **self.FONT_LABEL)
        ax.set_title('Receiver Operating Characteristic (ROC)', **self.FONT_TITLE)

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc='lower right', framealpha=0.95, **self.FONT_LEGEND)
        ax.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()

        save_path = os.path.join(save_dir, 'roc_publication.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"ROC curve saved: {save_path}")
        return fig

    # ============================================================================
    # ALL TRAINING LOSS
    # ============================================================================

    def plot_all_training_loss(self, all_histories, save_dir='plots'):
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.tab10.colors

        for i, result in enumerate(all_histories):
            history = result.get('history', {})
            if 'train_loss' in history:
                epochs = range(1, len(history['train_loss']) + 1)
                ax.plot(
                    epochs, history['train_loss'],
                    linewidth=2,
                    color=colors[i % len(colors)],
                    label=f'Fold {result.get("fold", i+1)}'
                )

        ax.set_title("Training Loss - All Folds", **self.FONT_TITLE)
        ax.set_xlabel("Epoch", **self.FONT_LABEL)
        ax.set_ylabel("Loss", **self.FONT_LABEL)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=10, framealpha=0.9)

        plt.tight_layout()
        save_path = os.path.join(save_dir, "training_loss_all_folds.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ============================================================================
    # METRICS TABLE
    # ============================================================================

    def save_metrics_table(self, all_fold_results, save_dir='plots'):
        os.makedirs(save_dir, exist_ok=True)

        data = []
        for result in all_fold_results:
            metrics = result['test_metrics']
            data.append({
                'Fold': result['fold'],
                'Accuracy': metrics['accuracy'],
                'AUC': metrics['roc_auc'],
                'Sensitivity': metrics['sensitivity'],
                'Specificity': metrics['specificity'],
                'Precision': metrics['precision'][1] if isinstance(metrics['precision'], (list, np.ndarray)) and len(metrics['precision']) > 1 else metrics['precision'],
                'Recall': metrics['recall'][1] if isinstance(metrics['recall'], (list, np.ndarray)) and len(metrics['recall']) > 1 else metrics['recall'],
                'F1-Score': metrics['f1_score'][1] if isinstance(metrics['f1_score'], (list, np.ndarray)) and len(metrics['f1_score']) > 1 else metrics['f1_score'],
                'N_Samples': metrics.get('n_subjects', metrics.get('n_samples', 0))
            })

        df = pd.DataFrame(data)

        mean_row = df.select_dtypes(include=[np.number]).mean()
        mean_row['Fold'] = 'Mean'
        std_row = df.select_dtypes(include=[np.number]).std()
        std_row['Fold'] = 'Std'

        df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

        csv_path = os.path.join(save_dir, 'metrics_table.csv')
        df.to_csv(csv_path, index=False, float_format='%.4f')

        fig, ax = plt.subplots(figsize=(14, len(df) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')

        df_display = df.copy()
        for col in df_display.columns:
            if col not in ['Fold', 'N_Samples']:
                df_display[col] = df_display[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '')
            elif col == 'N_Samples':
                df_display[col] = df_display[col].apply(lambda x: f'{int(x)}' if pd.notna(x) and x > 0 else '')

        table = ax.table(
            cellText=df_display.values,
            colLabels=df_display.columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        for i in range(len(df_display.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        for i in range(len(df_display.columns)):
            table[(len(df) - 1, i)].set_facecolor('#FFF9C4')
            table[(len(df), i)].set_facecolor('#FFE0B2')

        plt.title('Cross-Validation Metrics Summary', **self.FONT_TITLE, pad=20)

        png_path = os.path.join(save_dir, 'metrics_table.png')
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"Metrics table saved: {csv_path} & {png_path}")

    # ============================================================================
    # PLOT ALL FOLD RESULTS (FIXED CALLS)
    # ============================================================================

    def plot_all_fold_results(self, all_fold_results, save_dir='plots'):
        print(f"\n{'='*80}\nGenerating All Plots\n{'='*80}\n")

        for result in all_fold_results:
            fold_num = result['fold']

            if 'history' in result:
                self.plot_training_curves(result['history'], fold_num, save_dir)
                self.plot_overfitting_analysis(result['history'], fold_num, save_dir)

            if 'test_metrics' in result and 'confusion_matrix' in result['test_metrics']:
                cm = result['test_metrics']['confusion_matrix']
                self.plot_confusion_matrix(cm, ['HC', 'AD'], fold_num, save_dir)

        self.plot_cross_validation_summary(all_fold_results, save_dir)
        self.plot_roc_from_folds(all_fold_results, save_dir)
        self.save_metrics_table(all_fold_results, save_dir)

        print(f"\n{'='*80}\nAll plots saved to: {save_dir}\n{'='*80}\n")