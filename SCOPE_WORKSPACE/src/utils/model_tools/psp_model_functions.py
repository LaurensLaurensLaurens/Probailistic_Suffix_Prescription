# Temprär als eigene Datei, um besser daran arbeiten zu können
# Eigentlich class in ModelFunctions
#
# Aufbau (analog zu LSTM + DLCausalRegressor in model_functions.py):
#   PSP                -> das Netz (nn.Module). Encoder und Decoder kommen unverändert aus
#                         probabilistic_suffix_prediction_U-ED-LSTM (MC-Dropout-LSTM-Zellen).
#   PSPCausalRegressor -> die Model Functions, die SCOPE erwartet:
#                         .model, .get_loss(...) fürs Training und .forward(...) für die Q-Werte.
#
# Training: Der Decoder sagt per Teacher Forcing das Suffix voraus (X_suffix aus tensor_prep.py):
# pro Schritt die Aktivität (inkl. EOS) und die numerischen Event-Attribute, am EOS-Event das Outcome.
# Das Outcome-Ziel ist das von SCOPEFunctions.prepare() angepasste Y aus der Backward Induction.
#
# Inferenz: PSP.rollout() zieht pro Fall mehrere Suffixe autoregressiv bis EOS und liest dort das Outcome ab.
# ACHTUNG: forward()/predict() nutzen den Rollout noch nicht und lesen das Outcome nach EINEM Decoder-Schritt ab.
# Seit das Outcome nur am EOS-Event trainiert wird, passen die Q-Werte erst, wenn predict() auf rollout() umgestellt ist.
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

# Das PSP-Repo nennt sein Top-Level-Package ebenfalls "src" und kollidiert so mit SCOPE.
# Die Modell-Dateien nutzen untereinander nur relative Imports, daher reicht es, den Ordner
# .../src/model in den Suchpfad zu legen und das Unterpackage direkt zu importieren.
PSP_MODEL_DIR = Path(__file__).resolve().parents[4] / "probabilistic_suffix_prediction_U-ED-LSTM" / "src" / "model"
if str(PSP_MODEL_DIR) not in sys.path:
    sys.path.append(str(PSP_MODEL_DIR))

from dropout_uncertainty_enc_dec_LSTM.dropout_uncertainty_encoder import DropoutUncertaintyLSTMEncoder
from dropout_uncertainty_enc_dec_LSTM.dropout_uncertainty_decoder import DropoutUncertaintyLSTMDecoder

# Wie in losses.py des PSP: Log-Varianz begrenzen, damit die Loss Attenuation stabil bleibt
MIN_LOGVAR, MAX_LOGVAR = -6.0, 6.0


class PSP(nn.Module):
    def __init__(self, model_params, classification=False, n_classes=2):
        super().__init__()

        # Lazy Import, damit model_functions.py diese Datei importieren kann (kein Zirkelimport)
        from src.utils.model_tools.model_functions import set_seed
        set_seed(model_params["seed"])

        self.model_params = model_params
        self.dim_x_case = self.model_params["dim_x_case"]
        self.dim_x_event = self.model_params["dim_x_event"]
        self.dim_output = self.model_params["dim_output"]

        # Gleiche Regel wie im LSTM: Das Treatment ist nur beim Outcome-S-Learner Teil des Inputs
        to_add = self.model_params["dim_t"] - 1 if self.model_params["dim_t"] > 1 else self.model_params["dim_t"]
        if self.model_params["target"] != "outcome" or self.model_params["action_recomm_method"] == "class" or "T" in self.model_params["method"]:
            self.dim_t_input = 0
        else:
            self.dim_t_input = to_add

        self.hidden_size = self.model_params["dim_lstm"]
        self.num_layers = self.model_params["n_lstm_layers"]
        # Muss ein float sein, sonst behandelt die LSTM-Zelle p als lernbaren Parameter
        self.p = float(self.model_params.get("psp_dropout", self.model_params["dropout"]))

        # Das PSP braucht mindestens ein kategoriales Feature (Embedding). X_event enthält die Aktivität
        # One-Hot-kodiert; mit den Spaltennamen wird daraus wieder ein Index (0 = Padding).
        # Ohne Namen bleiben alle Kanäle numerisch und das kategoriale Feature ist nur "Event / Padding".
        event_cols = self.model_params.get("event_cols_encoded")
        # Spaltennamen passen zu X_event (sonst ist keine Zuordnung über Namen möglich)
        has_event_names = event_cols is not None and len(event_cols) == self.dim_x_event
        if has_event_names:
            # Kann leer sein, z.B. wenn alle Aktivitätsspalten im Prefix konstant waren und gefiltert wurden
            self.activity_idx = [i for i, col in enumerate(event_cols) if col.startswith("activity_")]
        else:
            self.activity_idx = []
            print("PSP: keine 'event_cols_encoded' in model_params, Aktivität wird nicht eingebettet.")
        self.num_idx = [i for i in range(self.dim_x_event) if i not in self.activity_idx]
        n_cat = len(self.activity_idx) + 1 if self.activity_idx else 2
        n_num = len(self.num_idx) + self.dim_t_input

        # Embedding-Größe wie im PSP
        dim_emb = min(600, round(1.6 * n_cat**0.56))
        data_indices = [[0], list(range(n_num))]

        self.encoder = DropoutUncertaintyLSTMEncoder(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            embeddings=nn.ModuleList([nn.Embedding(n_cat, dim_emb)]),
            data_indices_enc=data_indices,
            input_size=dim_emb + n_num,
            # X_case sind die statischen (numerischen) Fallattribute
            static_input_size=self.dim_x_case,
            dropout=self.p,
        )

        # Der Decoder arbeitet im Feature-Raum des Suffixes: alle Event-Spalten (ungefiltert) + activity_EOS.
        # Die letzte Spalte (outcome) ist nur Ziel und nie Input.
        suffix_cols = self.model_params.get("suffix_cols_encoded")
        if suffix_cols is None:
            raise ValueError("PSP needs 'suffix_cols_encoded' in model_params (see create_prefix_tensors in tensor_prep.py).")
        suffix_event_cols = list(suffix_cols[:-1])
        self.dim_x_suffix = len(suffix_cols)
        self.suffix_activity_idx = [i for i, col in enumerate(suffix_event_cols) if col.startswith("activity_")]
        self.suffix_num_idx = [i for i in range(len(suffix_event_cols)) if i not in self.suffix_activity_idx]
        # Klassen des Aktivitäts-Kopfes: 0 = Padding, 1..n = Aktivitäten (inkl. EOS)
        self.n_activity_classes = len(self.suffix_activity_idx) + 1
        # Zusätzlicher Index nur für den Input: Prefix-Aktivität, die es im Suffix-Vokabular nicht gibt
        sos_token = self.n_activity_classes
        n_cat_dec = self.n_activity_classes + 1
        n_num_dec = len(self.suffix_num_idx) + self.dim_t_input
        dim_emb_dec = min(600, round(1.6 * n_cat_dec**0.56))

        # Das SOS-Event ist das letzte Prefix-Event, über die Spaltennamen in den Suffix-Raum übersetzt.
        # Index 0 (kein Aktivitäts-Kanal aktiv, z.B. weil die Spalte herausgefiltert wurde) -> sos_token
        suffix_activity_cols = [suffix_event_cols[i] for i in self.suffix_activity_idx]
        # Klasse des EOS-Events, beendet ein Suffix im Rollout
        self.eos_class = suffix_activity_cols.index("activity_EOS") + 1
        prefix_to_suffix_activity = [sos_token]
        for i in self.activity_idx:
            col = event_cols[i]
            prefix_to_suffix_activity.append(suffix_activity_cols.index(col) + 1 if col in suffix_activity_cols else sos_token)
        self.register_buffer("prefix_to_suffix_activity", torch.tensor(prefix_to_suffix_activity, dtype=torch.long))
        # Numerische Suffix-Spalte dst bekommt den Wert des Prefix-Kanals src, Spalten ohne Gegenstück bleiben 0
        self.sos_num_dst, self.sos_num_src = [], []
        if has_event_names:
            for dst, i in enumerate(self.suffix_num_idx):
                if suffix_event_cols[i] in event_cols:
                    self.sos_num_dst.append(dst)
                    self.sos_num_src.append(event_cols.index(suffix_event_cols[i]))

        # Ein Kopf für alle numerischen Event-Attribute, einer für das Outcome (jeweils mean und log-var)
        num_output_sizes = {"outcome": self.dim_output}
        if self.suffix_num_idx:
            num_output_sizes["event_attributes"] = len(self.suffix_num_idx)
        self.decoder = DropoutUncertaintyLSTMDecoder(
            input_size=dim_emb_dec + n_num_dec,
            hidden_size=self.hidden_size,
            output_sizes=[{"activity": self.n_activity_classes}, num_output_sizes],
            embeddings=nn.ModuleList([nn.Embedding(n_cat_dec, dim_emb_dec)]),
            data_indices_dec=[[0], list(range(n_num_dec))],
            num_layers=self.num_layers,
            dropout=self.p,
        )

    def regularizer(self):
        weight_reg_enc, bias_reg_enc = self.encoder.regularizer()
        weight_reg_dec, bias_reg_dec = self.decoder.regularizer()
        return weight_reg_enc + weight_reg_dec + bias_reg_enc + bias_reg_dec

    def encode(self, x_case, x_process, prefix_len=None, t=None):
        # Liefert den Encoder-Zustand und das SOS-Event des Decoders (Aktivität [B], Numerik [B, n_num_dec])
        # x_process: [B, dim_x_event, max_process_len], rechts mit 0 gepaddet (siehe tensor_prep.py)
        # t: [B, dim_t_input] bzw. None, wenn das Treatment nicht Teil des Inputs ist
        B = x_process.shape[0]
        device = x_process.device
        prefix_len = prefix_len.long().clamp(min=1).to(device)
        seq_len = int(prefix_len.max())
        last = prefix_len - 1
        rows = torch.arange(B, device=device)

        # [B, C, L] -> [B, L, C], nur so lang wie der längste Prefix im Batch
        x = x_process[:, :, :seq_len].permute(0, 2, 1)
        # 1 = echtes Event, 0 = Padding. Die LSTM-Zelle behält bei 0 ihren alten Zustand.
        mask = (torch.arange(seq_len, device=device).unsqueeze(0) < prefix_len.unsqueeze(1)).to(x.dtype)

        if self.activity_idx:
            activity = (x[:, :, self.activity_idx].argmax(dim=2) + 1) * mask.long()
        else:
            activity = mask.long()
        cats = [activity]
        nums = [x[:, :, i] for i in self.num_idx]

        # Das Treatment steht wie im SCOPE-LSTM nur am letzten Event des Prefixes
        for k in range(self.dim_t_input):
            t_seq = torch.zeros(B, seq_len, dtype=x.dtype, device=device)
            t_seq[rows, last] = t[:, k].to(x.dtype)
            nums.append(t_seq)

        static_inputs = (None, x_case) if self.dim_x_case > 0 else None
        (h, c) = self.encoder(input=[cats, nums], static_inputs=static_inputs, mask=mask)

        # SOS-Event des Decoders ist das letzte echte Prefix-Event (inkl. Treatment), im Suffix-Raum
        x_last = x[rows, last]
        if self.activity_idx:
            last_activity = x_last[:, self.activity_idx]
            prefix_activity = (last_activity.argmax(dim=1) + 1) * (last_activity.sum(dim=1) > 0).long()
        else:
            prefix_activity = torch.zeros(B, dtype=torch.long, device=device)
        sos_activity = self.prefix_to_suffix_activity[prefix_activity]
        sos_nums = torch.zeros(B, len(self.suffix_num_idx), dtype=x.dtype, device=device)
        if self.sos_num_dst:
            sos_nums[:, self.sos_num_dst] = x_last[:, self.sos_num_src]
        if self.dim_t_input > 0:
            sos_nums = torch.cat([sos_nums, t.to(x.dtype)], dim=1)

        return (h, c), sos_activity, sos_nums

    def decode_step(self, activity, nums, hx, z=None):
        # Ein Decoder-Schritt. activity: [B] (Index), nums: [B, n_num_dec]. z = Dropout-Masken des ersten Schritts.
        event = [[activity.unsqueeze(1)], [nums[:, i].unsqueeze(1) for i in range(nums.shape[1])]]
        (pred_means, pred_vars), hx, z = self.decoder(input=event, hx=hx, z=z, pred=False)
        return pred_means, pred_vars, hx, z

    def forward(self, x_case, x_process, prefix_len=None, t=None):
        hx, activity, nums = self.encode(x_case=x_case, x_process=x_process, prefix_len=prefix_len, t=t)
        pred_means, pred_vars, _, _ = self.decode_step(activity, nums, hx)

        # Beide [B, dim_output]; der "var"-Kopf des PSP liefert die Log-Varianz
        return pred_means[1]["outcome_mean"], pred_vars[1]["outcome_var"]

    def forward_suffix(self, x_case, x_process, prefix_len, t, x_suffix, suffix_len):
        # Teacher Forcing: Schritt k bekommt das echte Event k-1 (Schritt 0 das SOS-Event) und sagt Event k voraus.
        # Event suffix_len ist das EOS-Event, daher suffix_len + 1 Schritte.
        hx, activity, nums = self.encode(x_case=x_case, x_process=x_process, prefix_len=prefix_len, t=t)
        B = x_suffix.shape[0]
        n_steps = int(suffix_len.max()) + 1

        # [B, C, L] -> [B, n_steps, C], ohne die Outcome-Spalte
        # Batch
        # Channels. Zeilen in Tabllen eines Falls also, event Spalten plus EOS + outcomes
        # Length Anzahl der Positionen im Suffix max_process_len + 1 = 27
        suffix = x_suffix[:, :-1, :n_steps].permute(0, 2, 1)

        # Alle Suffixe, Alle Positionen und nur bestimmte Spalten_In diesem Falle alles Aktivitäten
        suffix_activity = suffix[:, :, self.suffix_activity_idx]
        # 0 = Padding (keine Aktivität aktiv). Padding muss mit 0,0,0 eingehen und zum Bespiel repair mit [0,1,0] EOS mit [0,0,1] etc.
        target_activity = (suffix_activity.argmax(dim=2) + 1) * (suffix_activity.sum(dim=2) > 0).long()
        # Schneidet die numerischen Spalten aus target raus
        target_nums = suffix[:, :, self.suffix_num_idx]
        # Das Treatment steht nur im SOS-Event
        t_pad = torch.zeros(B, self.dim_t_input, dtype=suffix.dtype, device=suffix.device)

        outputs = {"activity_logits": [], "outcome_mean": [], "outcome_logvar": [], "event_mean": [], "event_logvar": []}
        z = None
        for k in range(n_steps):
            pred_means, pred_vars, hx, z = self.decode_step(activity, nums, hx, z)
            outputs["activity_logits"].append(pred_means[0]["activity_mean"])
            outputs["outcome_mean"].append(pred_means[1]["outcome_mean"])
            outputs["outcome_logvar"].append(pred_vars[1]["outcome_var"])
            if self.suffix_num_idx:
                outputs["event_mean"].append(pred_means[1]["event_attributes_mean"])
                outputs["event_logvar"].append(pred_vars[1]["event_attributes_var"])
            activity = target_activity[:, k]
            nums = torch.cat([target_nums[:, k], t_pad], dim=1)

        # Alle [B, n_steps, ...]
        outputs = {key: torch.stack(values, dim=1) for key, values in outputs.items() if values}
        outputs["target_activity"] = target_activity
        outputs["target_nums"] = target_nums
        return outputs

    @torch.no_grad()
    def rollout(self, x_case, x_process, prefix_len, t, n_samples, max_steps):
        # Zieht n_samples Suffixe pro Fall autoregressiv bis EOS: das gezogene Event ist der Input des nächsten Schritts.
        # Jede Zeile bekommt eigene Dropout-Masken (Encoder und Decoder), die über alle Schritte gleich bleiben,
        # also ein eigenes Netz aus dem Posterior. Der Zufall kommt aus dem globalen torch-Seed.
        # max_steps: höchstens so viele Decoder-Schritte (inkl. EOS), z.B. max_process_len + 1 wie in X_suffix
        B = x_process.shape[0]

        # Fall i steht in den Zeilen i * n_samples ... (i + 1) * n_samples - 1
        def repeat(v):
            return v.repeat_interleave(n_samples, dim=0) if v is not None else None

        hx, activity, nums = self.encode(x_case=repeat(x_case), x_process=repeat(x_process), prefix_len=repeat(prefix_len), t=repeat(t))
        R = activity.shape[0]
        device, dtype = nums.device, nums.dtype
        n_num = len(self.suffix_num_idx)
        # Das Treatment steht nur im SOS-Event
        t_pad = torch.zeros(R, self.dim_t_input, dtype=dtype, device=device)

        outcome_mean = torch.zeros(R, self.dim_output, dtype=dtype, device=device)
        outcome_var = torch.zeros(R, self.dim_output, dtype=dtype, device=device)
        suffix_len = torch.zeros(R, dtype=torch.long, device=device)
        finished = torch.zeros(R, dtype=torch.bool, device=device)
        truncated = torch.zeros(R, dtype=torch.bool, device=device)
        # Gezogene Suffixe wie in X_suffix: Aktivität (0 = Padding) und numerische Attribute je Position
        activities = torch.zeros(R, max_steps, dtype=torch.long, device=device)
        event_attributes = torch.zeros(R, max_steps, n_num, dtype=dtype, device=device)

        z = None
        for k in range(max_steps):
            pred_means, pred_vars, hx, z = self.decode_step(activity, nums, hx, z)

            # Aktivität aus der Softmax ziehen, Padding (Klasse 0) ist ausgeschlossen
            logits = pred_means[0]["activity_mean"].clone()
            logits[:, 0] = float("-inf")
            sampled_activity = torch.multinomial(F.softmax(logits, dim=1), num_samples=1).squeeze(1)
            is_eos = sampled_activity == self.eos_class
            if k == max_steps - 1:
                # Kein EOS bis zum letzten Schritt: Suffix hier abschneiden und als abgeschnitten markieren
                truncated = ~finished & ~is_eos
                sampled_activity[truncated] = self.eos_class
                is_eos = torch.ones_like(is_eos)

            # Numerische Attribute aus N(mean, exp(logvar)) ziehen
            if n_num:
                logvar = torch.clamp(pred_vars[1]["event_attributes_var"], min=MIN_LOGVAR, max=MAX_LOGVAR)
                sampled_nums = pred_means[1]["event_attributes_mean"] + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            else:
                sampled_nums = torch.zeros(R, 0, dtype=dtype, device=device)

            active = ~finished
            activities[active, k] = sampled_activity[active]
            # Am EOS-Event gibt es keine Attribute (wie in X_suffix)
            event_attributes[active & ~is_eos, k] = sampled_nums[active & ~is_eos]

            # Outcome am EOS-Event ablesen; suffix_len = Anzahl echter Events davor
            newly_finished = active & is_eos
            outcome_mean[newly_finished] = pred_means[1]["outcome_mean"][newly_finished]
            outcome_logvar = torch.clamp(pred_vars[1]["outcome_var"], min=MIN_LOGVAR, max=MAX_LOGVAR)
            outcome_var[newly_finished] = torch.exp(outcome_logvar)[newly_finished]
            suffix_len[newly_finished] = k
            finished |= is_eos
            if finished.all():
                break

            activity = sampled_activity
            nums = torch.cat([sampled_nums, t_pad], dim=1)

        # Alle mit Form [B, n_samples, ...]
        return {
            "outcome_mean": outcome_mean.view(B, n_samples, -1),
            "outcome_var": outcome_var.view(B, n_samples, -1),
            "suffix_len": suffix_len.view(B, n_samples),
            "truncated": truncated.view(B, n_samples),
            "activities": activities.view(B, n_samples, max_steps),
            "event_attributes": event_attributes.view(B, n_samples, max_steps, n_num),
        }


class PSPCausalRegressor:
    def __init__(self, model_params):
        """
        PSP-based causal regressor compatible with external train_dl().
        Only the S-learner outcome model is supported (one network, treatment as input).
        """
        self.model_params = model_params
        self.single_model = (
            "S" in model_params["method"] and model_params["target"] == "outcome"
        )
        if not self.single_model:
            raise NotImplementedError("PSP is only implemented as S-learner outcome model.")
        self.target = model_params["target"]
        self.n_classes = model_params["dim_t"] if model_params["dim_t"] > 1 else 2

        self.model = PSP(model_params=model_params)

        # Gewicht des L2-Regularisierers (MC-Dropout als Variational Inference), wie im PSP-Trainer
        self.regularization_term = model_params.get("psp_regularization_term", 1e-4)
        # Anzahl der MC-Dropout-Durchläufe für die Q-Werte bzw. für den Validierungs-Loss
        self.mc_samples = model_params.get("psp_mc_samples", 20)
        self.mc_samples_val = model_params.get("psp_mc_samples_val", 5)
        self.forward_batch_size = model_params.get("psp_forward_batch_size", 1024)

    def get_loss(self, x_case, x_event, t, prefix_len, y, weights=None, set_eval=True, x_suffix=None, suffix_len=None, return_parts=False):
        """
        Compute batch loss for training.
        With set_eval and return_parts, also returns the loss parts and readable metrics as a dict of floats.
        """
        # x_suffix: [B, dim_x_suffix, max_process_len + 1], suffix_len: [B] (siehe create_suffix_tensors in tensor_prep.py)
        if x_suffix is None or x_suffix.ndim != 3 or x_suffix.shape[1] != self.model.dim_x_suffix:
            raise ValueError("PSP needs X_suffix of shape [n, dim_x_suffix, max_process_len + 1] for training.")
        if set_eval:
            self.model.eval()
        else:
            self.model.train()

        # Debug: Nachsehen, ob Suffixe korrekt

        # Outcome in Trainingsgerforderte Struktur bringen
        y = y.view(y.shape[0], -1)

        # Bringt das Treament in die richtige Form
        # z.b: zum beispiel für 2 Aktonen reicht für machen 1, machen 2, lassen statt [1, 0, 0] -> [0, 0]
        t = t[:, 1:] if self.n_classes > 2 else t

        # Wir brauchen die Suffic length als float32, weil vorher gannzahlig mit punkt
        suffix_len = suffix_len.long()
        out = self.model.forward_suffix(x_case=x_case, x_process=x_event, prefix_len=prefix_len, t=t, x_suffix=x_suffix, suffix_len=suffix_len)

        # Masken [B, n_steps]: echte Events stehen an Position < suffix_len, das EOS-Event an Position suffix_len
        steps = torch.arange(out["target_activity"].shape[1], device=suffix_len.device).unsqueeze(0)
        event_mask = steps < suffix_len.unsqueeze(1)
        eos_mask = steps == suffix_len.unsqueeze(1)

        # Aktivität (inkl. EOS): Cross-Entropy über alle Schritte bis einschließlich EOS
        activity_mask = (event_mask | eos_mask) & (out["target_activity"] > 0)
        activity_logits = out["activity_logits"][activity_mask]
        target_activity = out["target_activity"][activity_mask]
        parts = {"activity_ce": F.cross_entropy(activity_logits, target_activity)}

        # Numerische Event-Attribute: Loss Attenuation (Kendall & Gal), nur an echten Events
        if "event_mean" in out and event_mask.any():
            parts["event_nll"] = self.loss_attenuation(out["event_mean"][event_mask], out["event_logvar"][event_mask], out["target_nums"][event_mask])

        # Outcome: Loss Attenuation, nur am EOS-Event (genau eines pro Fall, daher bleibt die Reihenfolge von y erhalten)
        outcome_mean = out["outcome_mean"][eos_mask]
        parts["outcome_nll"] = self.loss_attenuation(outcome_mean, out["outcome_logvar"][eos_mask], y)
        loss = sum(parts.values())

        if set_eval:
            # Validierung: derselbe Loss ohne Regularisierer
            if return_parts:
                # Zusätzlich lesbare Kennzahlen unter Teacher Forcing (Outcome in der skalierten Einheit von Y)
                # Padding (Klasse 0) ist nie ein Ziel und zählt nicht als Vorhersage
                predicted_activity = activity_logits[:, 1:].argmax(dim=1) + 1
                parts["activity_acc"] = (predicted_activity == target_activity).float().mean()
                parts["outcome_mae"] = torch.mean(torch.abs(outcome_mean - y))
                return loss, {key: value.item() for key, value in parts.items()}
            return loss
        return loss + self.regularization_term * self.model.regularizer()

    def debug_print_batch(self, x_case, x_event, t, prefix_len, y, weights, x_suffix, suffix_len, return_parts=False, n_cases=2):
        """
        Debug only: shapes of the batch and the first n_cases as readable tables.
        Tables: rows = encoded columns, columns = event positions (prefix p0.., suffix s0.. up to EOS).
        """
        import pandas as pd

        event_cols = self.model_params.get("event_cols_encoded")
        suffix_cols = self.model_params.get("suffix_cols_encoded")
        event_cols = event_cols if event_cols is not None and len(event_cols) == x_event.shape[1] else None
        suffix_cols = suffix_cols if suffix_cols is not None and len(suffix_cols) == x_suffix.shape[1] else None

        def to_numpy(value):
            return value.detach().cpu().numpy()

        def activity_sequence(matrix, cols):
            # Aktivitätsnamen je Position aus den One-Hot-Zeilen, "-" wenn keine Aktivität aktiv ist
            if cols is None:
                return "(keine Spaltennamen)"
            idx = [k for k, col in enumerate(cols) if col.startswith("activity_")]
            if not idx:
                return "(keine Aktivitätsspalten, z.B. weil sie konstant waren und gefiltert wurden)"
            names = []
            for position in range(matrix.shape[1]):
                values = matrix[idx, position]
                names.append(cols[idx[int(values.argmax())]][len("activity_"):] if values.max() > 0 else "-")
            return " -> ".join(names)

        print("\n=== PSP-Debug: Batch ===")
        for name, value in [("x_case", x_case), ("x_event", x_event), ("t", t), ("prefix_len", prefix_len), ("y", y),
                            ("weights", weights), ("x_suffix", x_suffix), ("suffix_len", suffix_len)]:
            print(f"  {name:<11} {tuple(value.shape) if value is not None else None}")
        print(f"  return_parts {return_parts}")

        with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_rows", None,
                               "display.float_format", "{:.4f}".format):
            for i in range(min(n_cases, x_event.shape[0])):
                n_prefix = int(prefix_len[i])
                n_suffix = int(suffix_len[i])
                def rounded(value):
                    return [round(v, 4) for v in value[i].view(-1).tolist()] if value is not None else None

                print(f"\n--- Fall {i} im Batch: prefix_len {n_prefix}, suffix_len {n_suffix}, T {rounded(t)}, Y {rounded(y)}, Gewicht {rounded(weights)}")
                print("X_case:", rounded(x_case))

                prefix = to_numpy(x_event[i, :, :n_prefix])
                print("Prefix-Aktivitäten:", activity_sequence(prefix, event_cols))
                print(pd.DataFrame(prefix, index=event_cols, columns=[f"p{k}" for k in range(n_prefix)]))

                # Bis einschließlich EOS an Position suffix_len
                suffix = to_numpy(x_suffix[i, :, :n_suffix + 1])
                print("Suffix-Aktivitäten:", activity_sequence(suffix, suffix_cols))
                print(pd.DataFrame(suffix, index=suffix_cols, columns=[f"s{k}" for k in range(n_suffix)] + ["EOS"]))

    @staticmethod
    def loss_attenuation(mean, logvar, target):
        # Das Netz lernt Mittelwert und Log-Varianz, gemittelt über alle Einträge
        logvar = torch.clamp(logvar, min=MIN_LOGVAR, max=MAX_LOGVAR)
        return torch.mean(0.5 * (torch.exp(-logvar) * (target - mean) ** 2 + logvar))

    def predict(self, x_case, x_event, t, prefix_len, n_samples):
        """
        Monte Carlo dropout prediction for a fixed treatment input t.
        Returns mean, epistemic variance and aleatoric variance, each of shape [n, dim_output].
        """
        means, epistemic, aleatoric = [], [], []
        for start in range(0, x_event.shape[0], self.forward_batch_size):
            batch = slice(start, start + self.forward_batch_size)
            sum_mean, sum_mean_sq, sum_var = 0.0, 0.0, 0.0
            # Jeder Durchlauf zieht neue Dropout-Masken, also ein anderes Netz aus dem Posterior
            for _ in range(n_samples):
                y_, logvar = self.model.forward(x_case=x_case[batch], x_process=x_event[batch], prefix_len=prefix_len[batch], t=t[batch])
                sum_mean = sum_mean + y_
                sum_mean_sq = sum_mean_sq + y_ ** 2
                sum_var = sum_var + torch.exp(torch.clamp(logvar, min=MIN_LOGVAR, max=MAX_LOGVAR))
            mean = sum_mean / n_samples
            means.append(mean)
            # Streuung der Mittelwerte = Modellunsicherheit, mittlere Varianz = Rauschen in den Daten
            epistemic.append(torch.clamp(sum_mean_sq / n_samples - mean ** 2, min=0.0))
            aleatoric.append(sum_var / n_samples)
        return torch.cat(means), torch.cat(epistemic), torch.cat(aleatoric)

    def forward_with_uncertainty(self, x_case, x_event, t, prefix_len, y=None, set_eval=True):
        """
        Predict the outcome distribution under every action.
        Returns means, epistemic variances and aleatoric variances, each of shape [n_classes, n].
        """
        if x_event is None or x_event.ndim != 3:
            raise ValueError("PSP needs the tensor encoding (X_event of shape [n, dim_x_event, max_process_len]).")

        if set_eval:
            self.model.eval()
        else:
            self.model.train()

        results = []
        with torch.no_grad():
            for treatment in range(self.n_classes):
                # Künstliches Treatment: alle Fälle bekommen dieselbe Aktion
                t_artificial = torch.zeros((x_event.shape[0], self.model.dim_t_input), dtype=x_event.dtype, device=x_event.device)
                if self.n_classes > 2:
                    if treatment != 0:
                        t_artificial[:, treatment - 1] = 1
                else:
                    t_artificial[:] = treatment
                results.append(self.predict(x_case, x_event, t_artificial, prefix_len, n_samples=self.mc_samples))

        # Bei mehreren Outputs (dim_output > 1) bleibt wie im LSTM die letzte Dimension erhalten
        means, epistemic, aleatoric = (
            torch.stack(values).squeeze(-1) if self.model.dim_output == 1 else torch.stack(values)
            for values in zip(*results)
        )
        return means, epistemic, aleatoric

    def forward(self, x_case, x_event, t, prefix_len, y=None, ret_counterfactuals=False, set_eval=True):
        """
        Forward pass for inference or counterfactual prediction.
        """
        # preds = tensor([
        #     [0.62, 0.40, 0.71],   # erwartetes Outcome bei Aktion 0
        #     [0.70, 0.35, 0.75]    # erwartetes Outcome bei Aktion 1
        # ])
        preds, _, _ = self.forward_with_uncertainty(x_case=x_case, x_event=x_event, t=t, prefix_len=prefix_len, y=y, set_eval=set_eval)
        return preds if ret_counterfactuals else preds.mean(0)
