"""
Loss functions incorporating combined epistemic and aleatoric uncertainty estimations.

Based on:
- Kendall, Alex, and Yarin Gal. "What uncertainties do we need in bayesian deep learning for computer vision?." Advances in neural information processing systems 30 (2017).
"""

# performance imports for torch: torch kernel uses one core only.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1" 

import torch

class Loss:
    def __init__(self):
        pass
    
    def _reduce_loss(self, loss_matrix, eos_paddings):
        # normal loss reduction
        if eos_paddings is None:
            # Old: sequence averaging:
            # loss over events
            # L = torch.mean(loss_matrix, dim=1)
            # loss over batches
            # return torch.mean(L)
            
            # Token averaging:
            # Global mean over all batch and sequence elements (Token Averaging implicit)
            return torch.mean(loss_matrix)
        
        # eos padd masking loss reduction
        else:
            if loss_matrix.shape == eos_paddings.shape:
                # Mask the loss matrix: Use torch.where to avoid NaN propagation from padded regions
                L = torch.where(eos_paddings.bool(), loss_matrix, torch.tensor(0.0, device=loss_matrix.device))
                
                # Normalize by number of valid tokens per batch
                # valid_tokens = torch.sum(eos_paddings, dim=1)  # [batch]
                # L = torch.sum(L, dim=1) / (valid_tokens + 1e-8)
                # Mean over batches
                #return torch.mean(L)
            
                # Normalize loss per active timestep
                # Count total valid tokens in the batch
                total_valid_tokens = torch.sum(eos_paddings)
                
                # Sum loss over all tokens and divide by total count
                return torch.sum(L) / (total_valid_tokens + 1e-8)
            else:
                return ValueError("loss and eos paddings have wrong shape!")
        
    def standard_mse(self, preds, targets, eos_paddings):
        """
        Standard MSE loss.
        
        INPUTS:
        - preds: Predicted (time) values for N events: dim: seq len x batch x features (1)
        - targets: Target (time) values for N events: dim: batch x seq len
        - eos_paddings: Optional EOS mask (batch x seq len). Required when EOS masking is enabled.
        
        OUTPUTS:
        - L: Global Loss value for numerical values of events of different batches: Tensor (float)
        """
        
        # Bring into structure: batch x seq len x output features (1)
        preds = preds.permute(1,0,2)
        targets = targets.unsqueeze(2)
        # Loss value
        L = torch.sum((targets - preds) ** 2, dim=2)
        
        L = self._reduce_loss(L, eos_paddings)
        
        return L
        
    def loss_attenuation_mse(self, pred_means, pred_logvars, targets, eos_paddings):
        """
        Loss attenuation MSE: Combined Epistemic and Aleatoric Uncertainty.
    
        INPUTS:
        - pred_means: Predicted values for N events: dim: seq len x batch x output values (1)
        - pred_logvars: Predicted log variance values for predicted mean for N events: dim: seq len x batch x output values (1)
        - targets: Target (time) values for N events: dim: batch x sequence length
        - eos_paddings: Optional EOS mask (batch x seq len). Required when EOS masking is enabled.
        
        OUTPUTS:
        - L: Global Loss value for numerical values of events of different batches: Tensor (float)
        """
        
        # Clamp the predicted variance to avoid extreme values
        min_logvariance = (torch.tensor(-6).to(pred_logvars.device))
        max_logvariance = (torch.tensor(6).to(pred_logvars.device))
        pred_logvars = torch.clamp(pred_logvars, min=min_logvariance, max=max_logvariance)
        
        # Bring into structure: batch x seq len x output features (1)
        pred_means = pred_means.permute(1,0,2)
        pred_logvars = pred_logvars.permute(1,0,2)
        targets = targets.unsqueeze(2)
        # Stable inverse variance: exp(-log(var)) = 1/sig^2 (1/var)
        inv_variances = torch.exp(-pred_logvars)
        L = torch.sum(0.5 * (inv_variances * ((targets - pred_means) ** 2) + pred_logvars), dim=2)
        
        L = self._reduce_loss(L, eos_paddings)

        return L
    
    def loss_attenuation_mse_log_normal(self, pred_means, pred_logvars, log_targets, eos_paddings):
        """
        Loss attenuation MSE: Combined Epistemic and Aleatoric Uncertainty of an assumed Log normal probability density function for our time input.
    
        INPUTS:
        - pred_logmeans: Predicted log time values for N events: dim: seq len x batch x output values (1)
        - pred_logvars: Predicted log variance values for predicted mean for N events: dim: seq len x batch x output values (1)
        - targets: Target (time) values for N events: dim: batch x sequence length
        - eos_paddings: Optional EOS mask (batch x seq len). Required when EOS masking is enabled.
        
        OUTPUTS:
        - L: Global Loss value for numerical values of events of different batches: Tensor (float)
        """
        
        # Clamp the predicted variance to avoid extreme values
        min_logvariance = (torch.tensor(-6).to(pred_logvars.device))
        max_logvariance = (torch.tensor(6).to(pred_logvars.device))
        pred_logvars = torch.clamp(pred_logvars, min=min_logvariance, max=max_logvariance)
                
        # Bring into structure: batch x seq len x output features (1)
        pred_means = pred_means.permute(1,0,2) # t := log(x)
        pred_logvars = pred_logvars.permute(1,0,2) # s := log(sigma^2)
        # Stable inverse variance: exp(-log(var)) = 1/sig^2 (1/var)
        inv_variances = torch.exp(-pred_logvars)
        # log the observed targets according to NLL log-normal PDF
        log_targets = log_targets.unsqueeze(2)
        L = torch.sum(log_targets + 0.5 * (pred_logvars + (inv_variances * (log_targets - pred_means)**2)), dim=2)
        
        L = self._reduce_loss(L, eos_paddings)

        return L
    
    def standard_cross_entropy(self, pred_logits, targets, eos_paddings):
        """
        Standard Cross Entropy loss.
      
        INPUTS:
        - pred_logits: Predicted logit values for N events: dim: seq len x batch x labels (logit value for each label)
        - targets: Target class indices for N events: dim: batch x seq len
        - eos_paddings: Optional EOS mask (batch x seq len). Required when EOS masking is enabled.
        
        OUTPUTS:
        - L: Global Loss value for numerical values of events of different batches: Tensor (float)
        """
        # Cross Entropy Loss
        CEL = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Change the shape of the prediction to: shape: batch_size x num_classes x seq len
        pred_logits = pred_logits.permute(1,2,0)
        L = CEL(input=pred_logits, target=targets)
        
        L = self._reduce_loss(L, eos_paddings)
        
        return L
    
    def loss_attenuation_cross_entropy(self, pred_logits, pred_logvars, T, targets, eos_paddings):
        """
        Loss attenuation cross entropy: Combined Epistemic and Aleatoric Uncertainty.
          
        INPUTS:
        - pred_logits: Predicted logit values for N events: dim: seq_len x batch x classes
        - pred_logvars: Predicted log variances per logit value for N events: dim: seq len x batch x classes
        - T: T gaussian distributed random epsilon value generations.
        - targets: Target class indices for N events: dim: batch x  seq len
        - eos_paddings: Optional EOS mask (batch x seq len). Required when EOS masking is enabled.
        
        OUTPUTS:
        - L: Global Loss value for numerical values of events of different batches: Tensor (float)
        """
            
        # Clamp the predicted log-variance to avoid collapse/instability.
        # Keeps std in [exp(-3)=0.05, exp(3)=20] since std=exp(0.5*logvar).
        min_logvariance = torch.tensor(-6.0, device=pred_logvars.device)
        max_logvariance = torch.tensor(6.0, device=pred_logvars.device)
        pred_logvars = torch.clamp(pred_logvars, min=min_logvariance, max=max_logvariance)

        # Cross Entropy Loss
        CEL = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Get standard deviation
        variance = torch.exp(pred_logvars)
        std = torch.sqrt(variance)
        
        L = 0
        # T monte carlo iterations for approx. gaussian distribution
        for _ in range(T):
            # epsilon_t: Generate a random matrix to distribute the standard deviations
            noise = torch.randn_like(pred_logits)    
            pred_logits_std_noise = pred_logits + std * noise
            # Change the shape of the prediction to: shape: batch_size x num_classes x seq len
            pred_logits_std_noise = pred_logits_std_noise.permute(1,2,0)
            # CEL of gaussian distributed unaries and target
            ce_loss = CEL(input=pred_logits_std_noise, target=targets)
            L += ce_loss
        L = (1/T) * L
        
        L = self._reduce_loss(L, eos_paddings)
          
        return L
        
