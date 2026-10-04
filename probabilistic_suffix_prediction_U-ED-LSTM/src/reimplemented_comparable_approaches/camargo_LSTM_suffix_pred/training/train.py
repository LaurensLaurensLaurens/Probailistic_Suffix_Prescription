import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torch.nn.functional as F

"""
Same trainings process as for the Probabilistic Suffix Prediction Approach.
"""

class Training:
    def __init__(self,
                 device,
                 model,
                 data_train,
                 data_val,
                 optimize_values,
                 concept_name_id,
                 eos_id, 
                 writer=None,
                 save_model_n_th_epoch: int = 0,
                 saving_path: str = 'reimpl_model.pkl'
                 ):
        
        self.device = device
        print("Device: ", device)
        self.data_train = data_train
        self.data_val = data_val
        
        self.concept_name_id=concept_name_id
        self.eos_id = eos_id
        
        self.model = model.to(self.device)
        
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
        Filters each sample so only those with max one EOS in the prefix and in the suffix remain.
        """
        
        B, T = cats[0].shape          # cats[i].shape = [1, B, T]
        
        # squeeze out the leading dim:
        cats = [c.squeeze(0) for c in cats]  # now each: [B, T]
        nums = [n.squeeze(0) for n in nums]  # each: [B, T]

        filtered_cats = [[] for _ in cats]  # one list per categorical feature
        filtered_nums = [[] for _ in nums]

        # Check rows that contain more than one EOS:
        for b in range(B):
            act_row = cats[self.concept_name_id][b]
            eos_count = (act_row == self.eos_id).sum().item()
            
            # Skip train samples that contain more than one EOS
            if eos_count > 2:
                continue

            # KEEP the full sequence (with 0 or 1 EOS) for *every* feature
            for i, c in enumerate(cats):
                filtered_cats[i].append(c[b])  # keep full [T] row
            
            for i, n in enumerate(nums):
                filtered_nums[i].append(n[b])

            V = len(filtered_cats[0])
            if V == 0:
                return None, None, 0

            # Stack into [V, T]
            batch_cats = [torch.stack(lst, dim=0) for lst in filtered_cats]
            batch_nums = [torch.stack(lst, dim=0) for lst in filtered_nums]

            # Build prefixes exactly as your model expects:
            prefixes_cat = [c[:, :-1].to(self.device) for c in batch_cats]  # each [V, T-1]
            prefixes_num = [n[:, :-1].to(self.device) for n in batch_nums]  # each [V, T-1]
            prefixes     = [prefixes_cat, prefixes_num]

            # Next-event activity (suffix):
            act_cats = batch_cats[self.concept_name_id]   # [V, T]
            acts     = [row[-1:].to(self.device) for row in act_cats]

        return prefixes, acts, V

    def train(self):
        """
        Run full training and validation loops.
        Returns:
            train_losses: list of training epoch losses
            val_losses: list of validation epoch losses
        """
        self.model.train()
        
        train_losses = []
        val_losses = []
        
        # Validation dataloader
        val_dataloader = DataLoader(dataset=self.data_val, batch_size=self.mini_batches, shuffle=self.shuffle, num_workers=4, pin_memory=True)

        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            # Train dataloader
            train_dataloader = DataLoader(dataset=self.data_train, batch_size=self.mini_batches, shuffle=self.shuffle, num_workers=4, pin_memory=True)
            
            total = 0
            num_batches = 0

            for i, train_cases in enumerate(train_dataloader):
                
                # list of list with tensor per feature. Tensor dim: batch x window size (equal case length)
                cats, nums, _ = train_cases
                
                # Get the prefixes to process, the target case elapsed time, and the new batch size as zero tensors are skipped:
                prefixes, acts, V = self._preprocess_batch(cats, nums)
                
                if V == 0:
                    continue
                
                # Forward pass: Output dim:  a_probs: batch x activity classes
                a_probs = self.model(prefixes)
                
                # Compute losses
                # Activity: cross-entropy
                target_act = torch.stack(acts, dim=0).squeeze(1).long() # (B,)
                act_loss = F.cross_entropy(a_probs, target_act)

                loss = act_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
                # Mean loss over all samples in batch of size V 
                total += loss.item()
                num_batches += 1

            # epoch averages train loss:
            epoch_loss = total / num_batches
            
            # Current learning rate
            current_lr = self.scheduler.optimizer.param_groups[0]['lr']
            
            # Prints per Epoch:
            tqdm.write(f"Epoch [{epoch+1}/{self.epochs}], Learning Rate: {current_lr}")
            
            tqdm.write(f"Training: Avg Attenuated Training Loss: {epoch_loss:.4f}")
            
            train_losses.append(epoch_loss)
            
            val_loss = self._validate(loader=val_dataloader)
            
            tqdm.write(f"Validation: Avg Validation Loss: {val_loss:.4f}")
            
            val_losses.append(val_loss)
                
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
                    'Training Total': epoch_loss,
                    'Validation Total': val_loss,
                    },
                epoch+1)
            
            # Adjust the learning rate if necessary
            tqdm.write(f"Validation Loss for Scheduler: {val_loss:.4f}")
            
            # Adjust learning rate
            self.scheduler.step(val_loss)

            if (i+1) % self.save_model_n_th_epoch == 0:
                 tqdm.write("saving model")
                 self.model.save(self.saving_path)
                                 
        print("Training complete.")

        self.model.save(self.saving_path)
        tqdm.write(f'Model saved to path: {self.saving_path}')


    def _validate(self, loader):
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for cats, nums, _ in loader:
                
                prefixes, acts, V = self._preprocess_batch(cats=cats, nums=nums)
                
                if V == 0:
                    continue

                a_probs = self.model(input=prefixes)

                target_act = torch.stack(acts, dim=0).squeeze(1).long() # (B,)
                act_loss = F.cross_entropy(a_probs, target_act)

                total_loss += act_loss.item()
                num_batches += 1

        return total_loss / num_batches
