import os
import argparse
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from scipy.special import expit
import random

# Import your actual model and preprocessing
from model.pretrained.dual_head_cnn14 import DualHeadCnn14Simple
from predict import preprocess_audio


class ModelEvaluator:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        """Load the trained model with correct architecture"""
        print(f"🔄 Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = DualHeadCnn14Simple(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        model.eval()
        model.to(self.device)
        
        print(f"Model loaded successfully on {self.device}")
        return model
    
    def predict_batch(self, file_paths: list) -> tuple:
        """Get predictions for a batch of files"""
        logits = []
        labels = []
        failed_files = []
        
        print(f"Predicting on {len(file_paths)} files...")
        
        for path in tqdm(file_paths, desc="Predicting"):
            try:
                # Determine label from path (same logic as training)
                label = 1 if "ai" in str(path).lower() else 0
                
                # Use same preprocessing as training
                waveform = preprocess_audio(str(path), sample_rate=16000, duration=10.0)
                
                # Ensure correct tensor shape for model
                if waveform.ndim == 2:
                    waveform = waveform.unsqueeze(0)  # [1, 1, samples]
                
                input_tensor = waveform.to(self.device)
                
                with torch.no_grad():
                    binary_logit, _ = self.model(input_tensor.squeeze(1))
                    logits.append(binary_logit.squeeze().item())
                    labels.append(label)
                    
            except Exception as e:
                print(f"Error processing {Path(path).name}: {e}")
                failed_files.append(str(path))
                continue
        
        if failed_files:
            print(f"{len(failed_files)} files failed to process")
            
        print(f"Successfully processed {len(logits)} files")
        return np.array(logits), np.array(labels), failed_files
    
    def threshold_analysis(self, logits: np.ndarray, labels: np.ndarray, 
                          thresholds: np.ndarray = None) -> pd.DataFrame:
        """Analyze performance across different thresholds"""
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 41)
        
        probs = expit(logits)  # Convert logits to probabilities
        
        results = []
        for threshold in thresholds:
            preds = (probs >= threshold).astype(int)
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy_score(labels, preds),
                'precision': precision_score(labels, preds, zero_division=0),
                'recall': recall_score(labels, preds, zero_division=0),
                'f1': f1_score(labels, preds, zero_division=0),
                'true_positives': np.sum((labels == 1) & (preds == 1)),
                'false_positives': np.sum((labels == 0) & (preds == 1)),
                'true_negatives': np.sum((labels == 0) & (preds == 0)),
                'false_negatives': np.sum((labels == 1) & (preds == 0))
            })
        
        return pd.DataFrame(results)
    
    def plot_performance_curves(self, logits: np.ndarray, labels: np.ndarray, 
                               save_path: str = None):
        """Plot ROC curve and threshold analysis"""
        probs = expit(logits)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        ax1.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True)
        
        # Logit Distribution
        ai_logits = logits[labels == 1]
        real_logits = logits[labels == 0]
        
        ax2.hist(real_logits, bins=30, alpha=0.7, label='Real', density=True, color='blue')
        ax2.hist(ai_logits, bins=30, alpha=0.7, label='AI', density=True, color='red')
        ax2.axvline(0, color='gray', linestyle='--', label='Decision Boundary (logit=0)')
        ax2.set_xlabel('Logit Value')
        ax2.set_ylabel('Density')
        ax2.set_title('Logit Distribution')
        ax2.legend()
        ax2.grid(True)
        
        # Threshold Analysis
        threshold_df = self.threshold_analysis(logits, labels)
        ax3.plot(threshold_df['threshold'], threshold_df['accuracy'], label='Accuracy', linewidth=2)
        ax3.plot(threshold_df['threshold'], threshold_df['precision'], label='Precision', linewidth=2)
        ax3.plot(threshold_df['threshold'], threshold_df['recall'], label='Recall', linewidth=2)
        ax3.plot(threshold_df['threshold'], threshold_df['f1'], label='F1-Score', linewidth=2)
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Score')
        ax3.set_title('Performance vs Threshold')
        ax3.legend()
        ax3.grid(True)
        
        # Confusion Matrix at optimal threshold (best F1)
        best_threshold = threshold_df.loc[threshold_df['f1'].idxmax(), 'threshold']
        best_preds = (probs >= best_threshold).astype(int)
        cm = confusion_matrix(labels, best_preds)
        
        im = ax4.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax4.figure.colorbar(im, ax=ax4)
        ax4.set(xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'],
                title=f'Confusion Matrix (Threshold = {best_threshold:.2f})',
                ylabel='True Label', xlabel='Predicted Label')
        
        # Add text annotations to confusion matrix
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax4.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {save_path}")
        plt.show()
        
        return threshold_df
    
    def comprehensive_evaluation(self, test_files: list, output_dir: str = None):
        """Run complete evaluation and generate report"""
        if output_dir is None:
            output_dir = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("Running comprehensive evaluation...")
        
        # Get predictions
        logits, labels, failed_files = self.predict_batch(test_files)
        
        if len(logits) == 0:
            print("No successful predictions. Check your files and model.")
            return None, None
        
        probs = expit(logits)
        
        # Overall metrics
        auc = roc_auc_score(labels, probs)
        
        # Find optimal threshold (best F1)
        threshold_df = self.threshold_analysis(logits, labels)
        best_row = threshold_df.loc[threshold_df['f1'].idxmax()]
        optimal_threshold = best_row['threshold']
        
        # Predictions at optimal threshold
        optimal_preds = (probs >= optimal_threshold).astype(int)
        
        # Generate comprehensive report
        report = {
            'model_performance': {
                'auc_score': float(auc),
                'optimal_threshold': float(optimal_threshold),
                'accuracy_at_optimal': float(best_row['accuracy']),
                'precision_at_optimal': float(best_row['precision']),
                'recall_at_optimal': float(best_row['recall']),
                'f1_at_optimal': float(best_row['f1'])
            },
            'data_summary': {
                'total_samples': int(len(labels)),
                'real_samples': int(np.sum(labels == 0)),
                'ai_samples': int(np.sum(labels == 1)),
                'class_balance': float(np.sum(labels == 1) / len(labels)),
                'failed_files': len(failed_files)
            },
            'prediction_distribution': {
                'mean_logit_real': float(np.mean(logits[labels == 0])) if np.sum(labels == 0) > 0 else 0,
                'mean_logit_ai': float(np.mean(logits[labels == 1])) if np.sum(labels == 1) > 0 else 0,
                'std_logit_real': float(np.std(logits[labels == 0])) if np.sum(labels == 0) > 0 else 0,
                'std_logit_ai': float(np.std(logits[labels == 1])) if np.sum(labels == 1) > 0 else 0
            }
        }
        
        # Print summary to console
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"AUC Score: {auc:.3f}")
        print(f"Optimal Threshold: {optimal_threshold:.3f}")
        print(f"Best F1-Score: {best_row['f1']:.3f}")
        print(f"Accuracy: {best_row['accuracy']:.3f}")
        print(f"Precision: {best_row['precision']:.3f}")
        print(f"Recall: {best_row['recall']:.3f}")
        
        print(f"\nData Summary:")
        print(f"   Total samples: {len(labels)}")
        print(f"   Real samples: {np.sum(labels == 0)}")
        print(f"   AI samples: {np.sum(labels == 1)}")
        if failed_files:
            print(f"   Failed files: {len(failed_files)}")
        
        print("\nClassification Report:")
        print(classification_report(labels, optimal_preds, 
                                  target_names=['Real', 'AI'], digits=3))
        
        # Save all results
        base_name = os.path.join(output_dir, "evaluation")
        
        # Save threshold analysis
        threshold_path = f"{base_name}_thresholds.csv"
        threshold_df.to_csv(threshold_path, index=False)
        print(f"Threshold analysis saved to: {threshold_path}")
        
        # Save plots
        plot_path = f"{base_name}_plots.png"
        self.plot_performance_curves(logits, labels, plot_path)
        
        # Save detailed text report
        report_path = f"{base_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Test Files: {len(test_files)}\n")
            f.write(f"Successfully Processed: {len(logits)}\n")
            f.write(f"Failed Files: {len(failed_files)}\n\n")
            
            for section, metrics in report.items():
                f.write(f"{section.upper().replace('_', ' ')}:\n")
                f.write("-" * 40 + "\n")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            f.write("CLASSIFICATION REPORT:\n")
            f.write("-" * 40 + "\n")
            f.write(classification_report(labels, optimal_preds, 
                                        target_names=['Real', 'AI'], digits=3))
            
            if failed_files:
                f.write("\n\nFAILED FILES:\n")
                f.write("-" * 40 + "\n")
                for file_path in failed_files:
                    f.write(f"{file_path}\n")
        
        print(f"Detailed report saved to: {report_path}")
        print(f"All results saved in: {output_dir}")
        
        return report, threshold_df


def get_test_files(data_split_folder: str) -> list:
    """Get test files from the split data folder"""
    split_path = Path(data_split_folder)
    
    if not split_path.exists():
        raise FileNotFoundError(f"Split data folder not found: {data_split_folder}")
    
    test_folder = split_path / "test"
    if not test_folder.exists():
        raise FileNotFoundError(f"Test folder not found: {test_folder}")
    
    # Get all test files
    test_files = []
    
    for class_folder in ["real", "ai"]:
        class_path = test_folder / class_folder
        if class_path.exists():
            # Get all audio files
            patterns = ["*.wav", "*.mp3", "*.flac", "*.m4a"]
            for pattern in patterns:
                test_files.extend(class_path.glob(pattern))
    
    return sorted(test_files)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained AI audio detection model"
    )
    parser.add_argument(
        "--model", 
        required=True,
        help="Path to trained model (.pth file)"
    )
    parser.add_argument(
        "--data-split", 
        required=True,
        help="Path to split data folder (containing train/val/test subdirs)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results (default: auto-generated)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    try:
        # Get test files
        print(f"Looking for test files in: {args.data_split}")
        test_files = get_test_files(args.data_split)
        
        if len(test_files) == 0:
            print("No test files found. Please check your data split.")
            return
        
        print(f"Found {len(test_files)} test files")
        
        # Count by class
        real_count = sum(1 for f in test_files if "real" in str(f))
        ai_count = sum(1 for f in test_files if "ai" in str(f))
        print(f"   - {real_count} real files")
        print(f"   - {ai_count} AI files")
        
        # Run evaluation
        evaluator = ModelEvaluator(args.model)
        report, _ = evaluator.comprehensive_evaluation(
            test_files, args.output_dir
        )
        
        if report is not None:
            print("\nEvaluation completed successfully!")
        else:
            print("\nEvaluation failed!")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())