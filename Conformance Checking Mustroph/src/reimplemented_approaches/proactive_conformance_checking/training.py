import math
import copy
from pathlib import Path
from typing import Optional, List, Dict, Any

import os

# performance imports for torch: torch kernel uses one core only.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1" 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


class Training:
    def __init__(self,
                 model: torch.nn.Module,
                 train_set: TensorDataset,
                 val_set: TensorDataset,
                 optimizer_values: Dict[str, Any],
                 # choose mode: either separate or collective training
                 loss_mode: str = 'collective',
                 device: torch.device = torch.device("cuda"),
                 saving_path: str = "./model.pkl",
                 label_weights: Optional[torch.Tensor] = None,
                 alpha_dc: float = 16.0, 
                 beta_scale: float = 16.0):
        
        self.model = model
        self.train_set = train_set
        self.val_set = val_set

        self.batch_size = optimizer_values["mini_batches"]
        self.shuffle = optimizer_values["shuffle"]
        self.num_epochs = optimizer_values["epochs"]
        self.optimizer = optimizer_values["optimizer"]

        self.loss_mode = loss_mode
        if self.loss_mode not in {"collective", "separate"}:
            raise ValueError(f"Unsupported loss_mode '{self.loss_mode}'.")
        
        # weights for WCEL according to paper
        self.alpha_dc = alpha_dc # separate
        self.beta_scale = beta_scale # part for collective

        self.device = torch.device(device)
        self.saving_path = Path(saving_path)

        self.pos_weight = (label_weights if label_weights is not None else self._compute_pos_weight(train_set))
        self.pos_weight = self.pos_weight.to(self.device)
        
        # Early stopping always used with the hyperparams described in the paper
        self.early_stopper = EarlyStopping()

    def _compute_pos_weight(self, data_source) -> torch.Tensor:
        """
        Compute the pos_weight tensor for WCEL according to the training mode.
        - Collective: a vector-based on the label imbalance of the deviation type
        - Separate: fixed weight alpha_dc.
        """
        targets = self._extract_targets(data_source)  # shape: [num_samples, num_labels]
        num_labels = targets.shape[1]

        if self.loss_mode == "collective":
            num_samples = targets.shape[0]
            deviating = targets.sum(dim=0).clamp(min=1e-8)
            conforming = (num_samples - deviating).clamp(min=1e-8)
            lir = conforming / deviating
            denom = (2 * math.e) + torch.log(lir)
            denom = torch.where(denom.abs() < 1e-8, torch.full_like(denom, 1e-8), denom)
            beta = self.beta_scale * (1.0 / denom)          # shape: [num_labels]
            return torch.clamp(beta, min=1e-6).to(self.device)

        elif self.loss_mode == "separate":
            # Single weight per label, scalar if single-label
            alpha_tensor = torch.full((num_labels,), self.alpha_dc, device=self.device)
            return alpha_tensor

        else:
            raise ValueError(f"Unsupported loss_mode '{self.loss_mode}'.")

    def _weighted_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_mode == "separate":
            if preds.ndim != 2 or preds.size(-1) != 2:
                raise ValueError("Separate mode expects logits with shape [batch_size, 2].")
            targets = targets.view(-1).long()
            class_weights = torch.tensor([1.0, self.alpha_dc], device=preds.device, dtype=preds.dtype)
            return F.cross_entropy(preds, targets, weight=class_weights)

        targets = targets.float()
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        pw = self.pos_weight
        if pw.ndim == 1 and pw.numel() != preds.shape[1]:
            pw = pw.expand(preds.shape[1]).to(preds.device)

        return F.binary_cross_entropy_with_logits(preds, targets, pos_weight=pw)

    def _extract_targets(self, data_source) -> torch.Tensor:
        if hasattr(data_source, "tensors"):
            dataset = data_source
        elif hasattr(data_source, "dataset") and hasattr(data_source.dataset, "tensors"):
            dataset = data_source.dataset
        else:
            raise ValueError("Training data must expose a TensorDataset to compute label weights.")
        targets = dataset.tensors[-1].detach().float()
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)
        return targets

    def _weighted_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Handling of seperate mode, output predicts for a label two logit values (positions: 0: False, 1: true) for binary class, target only contains the class as integer.
        if self.loss_mode == "separate":
            
            #print("shape preds", preds.shape)
            
            targets = targets.view(-1).long()
            
            #print("shape tgts", targets.shape)
            
            class_weights = torch.tensor([1.0, self.alpha_dc], device=preds.device, dtype=preds.dtype)
            return F.cross_entropy(preds, targets, weight=class_weights)

        targets = targets.float()
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        pw = self.pos_weight
        if pw.ndim == 1 and pw.numel() != preds.shape[1]:
            pw = pw.expand(preds.shape[1]).to(preds.device)

        return F.binary_cross_entropy_with_logits(preds, targets, pos_weight=pw)

    def train(self, mode:Optional[str]='lstm') -> List[Dict[str, Any]]:
        self.model.to(self.device)
        history: List[Dict[str, Any]] = []

        # Depending on the python version, error is thrown:
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0, pin_memory=True)
        val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)

            for batch in progress:
                # ffn
                if mode == 'ffn':
                    x, y = batch
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    
                    self.optimizer.zero_grad()
                    
                    logits = self.model(x)
                
                # lstm
                else:
                    x_act, x_res, x_month, x_trace, y = batch
                    x_act = x_act.to(self.device, non_blocking=True)
                    x_res = x_res.to(self.device, non_blocking=True)
                    x_month = x_month.to(self.device, non_blocking=True)
                    x_trace = x_trace.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    self.optimizer.zero_grad()
                    
                    logits = self.model(x_act, x_res, x_month, x_trace)

                # Weighted BCE
                loss = self._weighted_loss(logits, y)
                loss.backward()
                self.optimizer.step()

                if mode == 'ffn':
                    running_loss += loss.item() * x.size(0)
                else:
                    running_loss += loss.item() * x_act.size(0)

            train_loss = running_loss / len(self.train_set)

            # Validation
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        # ffn
                        if mode == 'ffn':
                            x, y = batch
                            x = x.to(self.device, non_blocking=True)
                            y = y.to(self.device, non_blocking=True)
                            
                            self.optimizer.zero_grad()
                            
                            logits = self.model(x)
                        # lstm
                        else:
                            x_act, x_res, x_month, x_trace, y = batch
                            x_act = x_act.to(self.device, non_blocking=True)
                            x_res = x_res.to(self.device, non_blocking=True)
                            x_month = x_month.to(self.device, non_blocking=True)
                            x_trace = x_trace.to(self.device, non_blocking=True)
                            y = y.to(self.device, non_blocking=True)

                            self.optimizer.zero_grad()
                            
                            logits = self.model(x_act, x_res, x_month, x_trace)
                        
                        loss = self._weighted_loss(logits, y)
                        
                        if mode == 'ffn':
                            total_val_loss += loss.item() * x.size(0)
                        else:
                            total_val_loss += loss.item() * x_act.size(0)
                
                val_loss = total_val_loss / len(self.val_set)

            # Record
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            tqdm.write(
                f"Epoch {epoch+1}/{self.num_epochs} — train loss: {train_loss:.4f}" +
                (f", val loss: {val_loss:.4f}" if val_loss is not None else "")
            )

            # Early stopping
            if self.early_stopper and val_loss is not None:
                if self.early_stopper(self.model, val_loss):
                    tqdm.write("Early stopping triggered.")
                    break

        # Save model
        if hasattr(self.model, "save"):
            self.model.save(path=self.saving_path)
        else:
            torch.save(self.model.state_dict(), self.saving_path)
        return history
    

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss: Optional[float] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.counter = 0

    def __call__(self, model: torch.nn.Module, val_loss: float) -> bool:
        if self.best_loss is None or (self.best_loss - val_loss) > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_state = copy.deepcopy(model.state_dict())
            return False

        self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_state is not None:
                model.load_state_dict(self.best_state)
            return True
        return False