"""
Train process following the same experimental setup as for the probabilistic suffix predictor (slightly adopted for prediction goal).

To predict only case_elapsed_time as weytjens LSTM adoptions has been made:
- For each train sample the EOS token is removed and all num values in tensor are removed on same index as the EOS tokens, to ensure same tensor length.
- When a train sample does not contain EOS, the last (target) case elapsed time is not given and therefore the sample is removed from training.
"""

import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

class Training:
    def __init__(self,
                 model,
                 device,
                 data_train,
                 data_val,
                 concept_name_id,
                 eos_id,
                 loss_obj,
                 optimize_values,
                 writer,
                 save_model_n_th_epoch: int = 0,
                 saving_path: str = 'model.pkl',):
        
        
        self.device = device
        print("Device: ", device)
        self.data_train = data_train
        self.data_val = data_val
        
        self.concept_name_id=concept_name_id
        self.eos_id=eos_id
        
        self.model = model.to(self.device)

        self.loss_obj = loss_obj
        
        # Standard Optimization parameters
        self.optimize_values = optimize_values
        self.optimizer = optimize_values["optimizer"]
        print("Optimizer: ", self.optimizer)
        self.scheduler = optimize_values["scheduler"]
        print("Scheduler: ", self.scheduler)
        self.epochs = optimize_values["epochs"]
        print("Epochs: ", self.epochs)
        self.mini_batches = optimize_values["mini_batches"]
        print("Mini baches: ", self.mini_batches)
        self.shuffle = optimize_values["shuffle"]
        print("Shuffle batched dataset: ", self.shuffle)
        
        # TensorBoard
        self.writer = writer
        
        # Model saving
        self.save_model_n_th_epoch = save_model_n_th_epoch
        self.saving_path = saving_path
        
    def _preprocess_batch(self, cats, nums):
        """
        Create the prefixes that fits for the case_elapsed_time (last) only prediction model of weytjens.
        
        Input:
        cats:   list with one tensor of shape [1, B, T]
        nums:   list with one tensor of shape [1, B, T]
        returns:
          prefixes: ([cat_prefixes],[num_prefixes]) each list of 1 tensor [V, T-1]
          targets:  tensor [V, 1]
          valid_count: number of rows V with at least one EOS
        """
        # squeeze leading 1
        cat = cats[self.concept_name_id].squeeze(0)  # [B, T]
        num = nums[0].squeeze(0)                     # [B, T]
        B, T = cat.shape

        filtered_cat = []
        filtered_num = []
        for b in range(B):
            row = cat[b]
            mask = (row != self.eos_id)  # True where not EOS
            
            # no EOS: skip sample entirely
            # if mask.all():
            #    continue

            kept_cat = row[mask]         # [T_b]
            kept_num = num[b][mask]      # [T_b]
            pad = T - kept_cat.size(0)

            # pad front to length T
            filtered_cat.append(torch.cat([row.new_zeros(pad), kept_cat], dim=0))
            filtered_num.append(torch.cat([kept_num.new_zeros(pad), kept_num], dim=0))

        if not filtered_cat:
            # no valid rows in this batch
            return None, None, 0

        batch_cat = torch.stack(filtered_cat, dim=0)  # [V, T]
        batch_num = torch.stack(filtered_num, dim=0)  # [V, T]
        
        V = batch_cat.size(0)

        assert(batch_cat.size(0) == batch_num.size(0))

        # build prefixes & targets
        cat_prefix = batch_cat[:, :-1].to(self.device)  # [V, T-1]
        num_prefix = batch_num[:, :-1].to(self.device)  # [V, T-1]
        prefix = [[cat_prefix], [num_prefix]]
        
        case_elapsed_time = batch_num[:, -1:].to(self.device)   # [V, 1]

        return prefix, case_elapsed_time, V

    def train(self):
        self.model.train()
        
        train_losses_unc = []

        val_losses_std = []
        val_losses_unc = []

        # Validation dataloader
        # val_dataloader = DataLoader(dataset=self.data_val, batch_size=self.mini_batches, shuffle=self.shuffle, num_workers=4, pin_memory=True)
        val_dataloader = DataLoader(dataset=self.data_val, batch_size=self.mini_batches, shuffle=self.shuffle, pin_memory=True)
        
        for epoch in range(self.epochs):
        # for epoch in tqdm(range(self.epochs)):
            self.model.train()

            # Train dataloader
            # train_dataloader = DataLoader(dataset=self.data_train, batch_size=self.mini_batches, shuffle=self.shuffle, num_workers=4, pin_memory=True)
            train_dataloader = DataLoader(dataset=self.data_train, batch_size=self.mini_batches, shuffle=self.shuffle, pin_memory=True)

            total_unc = 0
            num_batches = 0

            for i, train_cases in enumerate(train_dataloader):
                cats, nums, _ = train_cases
                
                # Get the prefixes to process, the target case elapsed time, and the new batch size as zero tensors are skipped:
                prefixes, target, V = self._preprocess_batch(cats, nums)
                
                if V == 0:
                    continue

                # Prediction:
                means, logvars = self.model(input=prefixes)
                
                # per‑sample losses, shape [V]
                loss_unc = self.loss_obj.regression_heteroscedastic_loss(true=target, mean=means, log_var=logvars)
                
                weight_reg, bias_reg = self.model.regularizer()
                reg_term = weight_reg + bias_reg
                
                loss_unc = loss_unc + reg_term.to(self.device)
                
                # backward on the total hetero loss
                loss_unc.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                self.optimizer.zero_grad()

                # Mean loss over all samples in batch of size V 
                total_unc += loss_unc.item()
                num_batches += 1

            # epoch averages train loss:
            epoch_unc = total_unc / num_batches
            
            # Current learning rate
            current_lr = self.scheduler.optimizer.param_groups[0]['lr']
            
            # Prints per Epoch:
            tqdm.write(f"Epoch [{epoch+1}/{self.epochs}], Learning Rate: {current_lr}")
            
            tqdm.write(f"Training: Avg Attenuated Training Loss: {epoch_unc:.4f}")
            
            train_losses_unc.append(epoch_unc)
            
            epoch_loss_val_std, epoch_loss_val_unc = self._validate(loader=val_dataloader)
            
            tqdm.write(f"Validation: Avg Standard Validation Loss: {epoch_loss_val_std:.4f}")
            tqdm.write(f"Validation: Avg Attenuated Validation Loss: {epoch_loss_val_unc:.4f}")
            
            val_losses_std.append(epoch_loss_val_std)
            val_losses_unc.append(epoch_loss_val_unc)
                
            # Tensorboard writer:
            # Hyperparameters
            self.writer.add_scalars(
                "Hyperparameter:", 
                {
                    'Learning Rate': current_lr,
                },
                epoch+1)
            
            # Total losses
            self.writer.add_scalars(
                "Total Losses", 
                {
                    'Training Total': epoch_unc,
                    'Stdandard Validation Total': epoch_loss_val_std,
                    'Uncertainty Validation Total': epoch_loss_val_unc
                    },
                epoch+1)
            
            # Adjust the learning rate if necessary
            tqdm.write(f"Validation Loss for Scheduler: {epoch_loss_val_std:.4f}")
            
            # Adjust learning rate
            self.scheduler.step(epoch_loss_val_std)

            if (i+1) % self.save_model_n_th_epoch == 0:
                 tqdm.write("saving model")
                 self.model.save(self.saving_path)
                                 
        print("Training complete.")

        self.model.save(self.saving_path)
        tqdm.write(f'Model saved to path: {self.saving_path}')

    def _validate(self, loader):
        self.model.eval()
        
        total_std = total_unc = 0
        num_batches = 0
        with torch.no_grad():
            for cats, nums, _ in loader:
                prefixes, target, V = self._preprocess_batch(cats, nums)
                
                if V == 0:
                    continue

                means, logvars = self.model(input=prefixes) 
                loss_unc = self.loss_obj.regression_heteroscedastic_loss(true=target, mean=means, log_var=logvars)
                
                loss_std = self.loss_obj.regression_homoscedastic_loss(true=target, mean=means)

                total_unc += loss_unc.item()
                total_std += loss_std.item()
                num_batches +=1
                
        return total_std / num_batches, total_unc / num_batches
                
                
            
            