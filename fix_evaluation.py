# save as: fix_evaluation.py
# This patches the evaluation metrics

import sys
sys.path.insert(0, '.')

# Fix the evaluation metrics
eval_file = "src/evaluation/metrics.py"
with open(eval_file, 'r') as f:
    content = f.read()

# Replace the ROC-AUC section
old_roc = """    # ROC-AUC (one-vs-rest for multi-class)
    try:
        metrics['roc_auc'] = roc_auc_score(
            all_labels, all_probs, multi_class='ovr', average='weighted'
        )
    except ValueError:
        metrics['roc_auc'] = 0.0"""

new_roc = """    # ROC-AUC (one-vs-rest for multi-class)
    try:
        # Handle case where not all classes present
        n_classes = all_probs.shape[1]
        if len(np.unique(all_labels)) > 1:
            metrics['roc_auc'] = roc_auc_score(
                all_labels, all_probs, multi_class='ovr', average='weighted'
            )
        else:
            metrics['roc_auc'] = 0.5  # Random baseline for single class
    except Exception as e:
        logger.warning(f"ROC-AUC calculation failed: {e}")
        metrics['roc_auc'] = 0.0"""

content = content.replace(old_roc, new_roc)

with open(eval_file, 'w') as f:
    f.write(content)

print("✅ Fixed ROC-AUC calculation")