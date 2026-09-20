import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

logger = logging.getLogger("synthmed.classifier")

class ClassifierTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 0.0001,
        weight_decay: float = 0.001
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_acc = 0.0
        self.patience = 10
        self.patience_counter = 0
    
    def _unpack_batch(self, batch):
        """Helper to unpack batch, handling 2 or 3 elements."""
        if len(batch) == 3:
            images, labels, metadata = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            metadata = metadata.to(self.device)
            return images, labels, metadata
        else:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            return images, labels, None
    
    def _forward(self, images, metadata):
        """Forward pass with optional metadata."""
        if self.model.use_metadata and metadata is not None:
            return self.model(images, metadata)
        return self.model(images)
    
    def _mixup(self, images, labels):
        lam = np.random.beta(0.5, 0.5)
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(self.device)
        mixed_images = lam * images + (1 - lam) * images[index]
        return mixed_images, labels, labels[index], lam
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            images, labels, metadata = self._unpack_batch(batch)
            
            if torch.rand(1).item() > 0.5 and len(images) > 1:
                images, labels_a, labels_b, lam = self._mixup(images, labels)
                self.optimizer.zero_grad()
                outputs = self._forward(images, metadata)
                loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
            else:
                self.optimizer.zero_grad()
                outputs = self._forward(images, metadata)
                loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / max(num_batches, 1)
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images, labels, metadata = self._unpack_batch(batch)
                outputs = self._forward(images, metadata)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        metrics = {
            'loss': total_loss / max(len(val_loader), 1),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        }
        try:
            if len(np.unique(all_labels)) > 1:
                metrics['roc_auc'] = roc_auc_score(
                    all_labels, all_probs, multi_class='ovr', average='weighted'
                )
            else:
                metrics['roc_auc'] = 0.5
        except Exception as e:
            logger.warning(f"ROC-AUC error: {e}")
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    def train(self, train_loader, val_loader, epochs, save_dir):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        history = {
            'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_roc_auc': []
        }
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()
            
            logger.info(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}"
            )
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_roc_auc'].append(val_metrics['roc_auc'])
            
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                self.save_model(save_path / "best_model.pt")
                logger.info(f"New best model! Accuracy: {self.best_val_acc:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        self.save_model(save_path / "final_model.pt")
        return history
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        logger.info(f"Model loaded from {path}")