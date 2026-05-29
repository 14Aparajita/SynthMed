import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from .metrics import ExperimentMetrics
import logging

logger = logging.getLogger("synthmed.evaluation")

class ResultsReporter:
    """Generate experiment reports and visualizations."""
    
    def __init__(self, output_dir: str = "outputs/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: List[ExperimentMetrics] = []
    
    def add_experiment(self, metrics: ExperimentMetrics):
        """Add experiment results."""
        self.experiments.append(metrics)
    
    def generate_report(self):
        """Generate comprehensive results report."""
        if not self.experiments:
            logger.warning("No experiments to report.")
            return
        
        # Save individual results
        for exp in self.experiments:
            exp_path = self.output_dir / f"{exp.experiment_name}.json"
            with open(exp_path, 'w') as f:
                json.dump(exp.to_dict(), f, indent=2)
        
        # Create comparison table
        self._create_comparison_table()
        
        # Create visualizations
        self._plot_classification_comparison()
        self._plot_schema_metrics()
        self._plot_confusion_matrices()
        
        logger.info(f"Report generated in {self.output_dir}")
    
    def _create_comparison_table(self):
        """Create CSV comparison of all experiments."""
        data = []
        for exp in self.experiments:
            data.append(exp.to_dict())
        
        df = pd.DataFrame(data)
        df.to_csv(self.output_dir / "experiment_comparison.csv", index=False)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPARISON")
        logger.info("="*80)
        
        columns = [
            'experiment_name', 'accuracy', 'f1_score', 'roc_auc',
            'schema_validity_rate', 'repair_success_rate', 'mean_grounding_score'
        ]
        available_cols = [c for c in columns if c in df.columns]
        logger.info("\n" + df[available_cols].to_string())
        logger.info("="*80)
    
    def _plot_classification_comparison(self):
        """Plot classification metrics comparison."""
        if not self.experiments:
            return
        
        names = [e.experiment_name for e in self.experiments]
        accuracies = [e.accuracy for e in self.experiments]
        f1_scores = [e.f1_score for e in self.experiments]
        roc_aucs = [e.roc_auc for e in self.experiments]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(names))
        width = 0.25
        
        ax.bar([i - width for i in x], accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x, f1_scores, width, label='F1 Score', alpha=0.8)
        ax.bar([i + width for i in x], roc_aucs, width, label='ROC-AUC', alpha=0.8)
        
        ax.set_ylabel('Score')
        ax.set_title('Classification Performance by Experiment')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "classification_comparison.png", dpi=150)
        plt.close()
    
    def _plot_schema_metrics(self):
        """Plot schema-related metrics."""
        if not self.experiments:
            return
        
        exp_with_schema = [e for e in self.experiments 
                          if e.num_synthetic_metadata > 0]
        
        if not exp_with_schema:
            return
        
        names = [e.experiment_name for e in exp_with_schema]
        validity = [e.schema_validity_rate for e in exp_with_schema]
        repair = [e.repair_success_rate for e in exp_with_schema]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = range(len(names))
        ax.bar([i - 0.15 for i in x], validity, 0.3, label='Schema Validity Rate', alpha=0.8)
        ax.bar([i + 0.15 for i in x], repair, 0.3, label='Repair Success Rate', alpha=0.8)
        
        ax.set_ylabel('Rate')
        ax.set_title('Schema Validation & Repair Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "schema_metrics.png", dpi=150)
        plt.close()
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for each experiment."""
        for exp in self.experiments:
            if exp.confusion_matrix.size == 0:
                continue
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                exp.confusion_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=[f'Grade {i}' for i in range(5)],
                yticklabels=[f'Grade {i}' for i in range(5)],
                ax=ax
            )
            ax.set_title(f'Confusion Matrix - {exp.experiment_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
            plt.tight_layout()
            plt.savefig(
                self.output_dir / f"confusion_{exp.experiment_name}.png",
                dpi=150
            )
            plt.close()