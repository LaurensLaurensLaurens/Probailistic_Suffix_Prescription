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
# Die Q-Werte (PSPCausalRegressor.predict) sind der Mittelwert dieser Outcomes je Aktion.
# PSP.forward() liest das Outcome dagegen nach EINEM Decoder-Schritt ab und passt nicht zum Training (Outcome nur am EOS-Event).
import sys
from contextlib import contextmanager
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


# Wir müssen das Datenformat anpassen and die Logik von Scope. Encoder und Decoder bleiben gleich
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
        # Platzhalter für fehlende numerische Werte (missing_value in main_prep.py), unskaliert neben Werten in [0, 1].
        # Das PSP zerlegt jedes numerische Event-Attribut in Wert (0, wo er fehlt) und Fehlt-Kennzeichen, siehe split_missing.
        self.missing_value = self.model_params.get("psp_missing_value", -100)

        # Das PSP braucht mindestens ein kategoriales Feature (Embedding). X_event enthält die Aktivität
        # One-Hot-kodiert; mit den Spaltennamen wird daraus wieder ein Index (0 = Padding).
        # Ohne Namen bleiben alle Kanäle numerisch und das kategoriale Feature ist nur "Event / Padding".
        event_cols = self.model_params.get("event_cols_encoded")
        # Weitere kategoriale Event-Attribute neben der Aktivität (z.B. process_type), ebenfalls One-Hot in X_event.
        # Wie im PSP bekommt jedes einen eigenen Index (0 = kein Wert), ein eigenes Embedding und einen eigenen Kopf.
        cat_names = [col for col in self.model_params.get("event_cat_cols", ["activity"]) if col != "activity"]

        def one_hot_groups(cols):
            # Ordnet jede One-Hot-Spalte ihrem kategorialen Attribut zu (längster passender Präfix "<attribut>_")
            groups = {name: [] for name in ["activity"] + cat_names}
            for i, col in enumerate(cols):
                matches = [name for name in groups if col.startswith(name + "_")]
                if matches:
                    groups[max(matches, key=len)].append(i)
            return groups

        # Embedding-Größe wie im PSP
        def embedding(n_classes):
            return nn.Embedding(n_classes, min(600, round(1.6 * n_classes**0.56)))

        # Spaltennamen passen zu X_event (sonst ist keine Zuordnung über Namen möglich)
        has_event_names = event_cols is not None and len(event_cols) == self.dim_x_event
        if has_event_names:
            groups = one_hot_groups(event_cols)
        else:
            groups = {"activity": []}
            print("PSP: keine 'event_cols_encoded' in model_params, Aktivität wird nicht eingebettet.")
        # Kann leer sein, z.B. wenn alle Aktivitätsspalten im Prefix konstant waren und gefiltert wurden
        self.activity_idx = groups["activity"]
        # Nur Attribute, von denen im Prefix Spalten übrig sind
        self.cat_idx = {name: groups[name] for name in cat_names if groups.get(name)}
        cat_channels = self.activity_idx + [i for idx in self.cat_idx.values() for i in idx]
        self.num_idx = [i for i in range(self.dim_x_event) if i not in cat_channels]
        n_cat = len(self.activity_idx) + 1 if self.activity_idx else 2
        # Je numerischem Attribut zwei Kanäle: Wert und Fehlt-Kennzeichen
        n_num = 2 * len(self.num_idx) + self.dim_t_input

        enc_embeddings = [embedding(n_cat)] + [embedding(len(idx) + 1) for idx in self.cat_idx.values()]
        data_indices = [list(range(len(enc_embeddings))), list(range(n_num))]

        self.encoder = DropoutUncertaintyLSTMEncoder(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            embeddings=nn.ModuleList(enc_embeddings),
            data_indices_enc=data_indices,
            input_size=sum(emb.embedding_dim for emb in enc_embeddings) + n_num,
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
        suffix_groups = one_hot_groups(suffix_event_cols)
        self.suffix_activity_idx = suffix_groups["activity"]
        self.suffix_cat_idx = {name: suffix_groups[name] for name in cat_names if suffix_groups[name]}
        # Die weiteren kategorialen Attribute des Decoders, in fester Reihenfolge
        self.cat_names = list(self.suffix_cat_idx)
        # Namen der Decoder-Köpfe (unabhängig von Sonderzeichen in den Spaltennamen)
        self.cat_keys = {name: f"cat_{j}" for j, name in enumerate(self.cat_names)}
        suffix_cat_channels = self.suffix_activity_idx + [i for idx in self.suffix_cat_idx.values() for i in idx]
        self.suffix_num_idx = [i for i in range(len(suffix_event_cols)) if i not in suffix_cat_channels]
        # Klassen des Aktivitäts-Kopfes: 0 = Padding, 1..n = Aktivitäten (inkl. EOS)
        self.n_activity_classes = len(self.suffix_activity_idx) + 1
        # Zusätzlicher Index nur für den Input: Prefix-Aktivität, die es im Suffix-Vokabular nicht gibt
        sos_token = self.n_activity_classes
        n_cat_dec = self.n_activity_classes + 1
        n_num_dec = 2 * len(self.suffix_num_idx) + self.dim_t_input

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
        # Dasselbe für die weiteren kategorialen Attribute: Klassen 0 = kein Wert, 1..n = Werte, n + 1 = nur Input (unbekannt)
        dec_embeddings = [embedding(n_cat_dec)]
        cat_output_sizes = {"activity": self.n_activity_classes}
        for j, name in enumerate(self.cat_names):
            suffix_value_cols = [suffix_event_cols[i] for i in self.suffix_cat_idx[name]]
            n_classes = len(suffix_value_cols) + 1
            prefix_to_suffix = [n_classes]
            for i in self.cat_idx.get(name, []):
                col = event_cols[i]
                prefix_to_suffix.append(suffix_value_cols.index(col) + 1 if col in suffix_value_cols else n_classes)
            self.register_buffer(f"prefix_to_suffix_cat_{j}", torch.tensor(prefix_to_suffix, dtype=torch.long))
            dec_embeddings.append(embedding(n_classes + 1))
            cat_output_sizes[self.cat_keys[name]] = n_classes
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
            # Dazu ein Logit je numerischem Attribut: fehlt der Wert am nächsten Event? (Sigmoid, kein Softmax)
            cat_output_sizes["missing"] = len(self.suffix_num_idx)
        self.decoder = DropoutUncertaintyLSTMDecoder(
            input_size=sum(emb.embedding_dim for emb in dec_embeddings) + n_num_dec,
            hidden_size=self.hidden_size,
            output_sizes=[cat_output_sizes, num_output_sizes],
            embeddings=nn.ModuleList(dec_embeddings),
            data_indices_dec=[list(range(len(dec_embeddings))), list(range(n_num_dec))],
            num_layers=self.num_layers,
            dropout=self.p,
        )

    def regularizer(self):
        weight_reg_enc, bias_reg_enc = self.encoder.regularizer()
        weight_reg_dec, bias_reg_dec = self.decoder.regularizer()
        return weight_reg_enc + weight_reg_dec + bias_reg_enc + bias_reg_dec

    def split_missing(self, nums):
        # Trennt numerische Attribute in Wert (0, wo er fehlt) und Fehlt-Kennzeichen (1 = fehlt), beide in der Form von nums
        missing = nums == self.missing_value
        return nums.masked_fill(missing, 0.0), missing.to(nums.dtype)

    def encode(self, x_case, x_process, prefix_len=None, t=None):
        # Liefert den Encoder-Zustand und das SOS-Event des Decoders:
        # kategorial [B, 1 + len(cat_names)] (Aktivität, dann die weiteren Attribute),
        # Numerik [B, n_num_dec] (Werte, Fehlt-Kennzeichen, Treatment)
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
        for idx in self.cat_idx.values():
            # 0 = kein Wert aktiv (Padding oder unbekannter Wert)
            cats.append((x[:, :, idx].argmax(dim=2) + 1) * (x[:, :, idx].sum(dim=2) > 0).long())
        values, missing = self.split_missing(x[:, :, self.num_idx])
        nums = [values[:, :, j] for j in range(len(self.num_idx))] + [missing[:, :, j] for j in range(len(self.num_idx))]

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
        sos_cats = [self.prefix_to_suffix_activity[prefix_activity]]
        for j, name in enumerate(self.cat_names):
            if name in self.cat_idx:
                last_cat = x_last[:, self.cat_idx[name]]
                prefix_cat = (last_cat.argmax(dim=1) + 1) * (last_cat.sum(dim=1) > 0).long()
            else:
                prefix_cat = torch.zeros(B, dtype=torch.long, device=device)
            sos_cats.append(getattr(self, f"prefix_to_suffix_cat_{j}")[prefix_cat])
        sos_nums = torch.zeros(B, len(self.suffix_num_idx), dtype=x.dtype, device=device)
        sos_missing = torch.zeros_like(sos_nums)
        if self.sos_num_dst:
            sos_nums[:, self.sos_num_dst], sos_missing[:, self.sos_num_dst] = self.split_missing(x_last[:, self.sos_num_src])
        sos_nums = torch.cat([sos_nums, sos_missing], dim=1)
        if self.dim_t_input > 0:
            sos_nums = torch.cat([sos_nums, t.to(x.dtype)], dim=1)

        return (h, c), torch.stack(sos_cats, dim=1), sos_nums

    def decode_step(self, cats, nums, hx, z=None):
        # Ein Decoder-Schritt. cats: [B, 1 + len(cat_names)] (Indizes), nums: [B, n_num_dec]. z = Dropout-Masken des ersten Schritts.
        event = [[cats[:, i].unsqueeze(1) for i in range(cats.shape[1])], [nums[:, i].unsqueeze(1) for i in range(nums.shape[1])]]
        (pred_means, pred_vars), hx, z = self.decoder(input=event, hx=hx, z=z, pred=False)
        return pred_means, pred_vars, hx, z

    def forward(self, x_case, x_process, prefix_len=None, t=None):
        hx, cats, nums = self.encode(x_case=x_case, x_process=x_process, prefix_len=prefix_len, t=t)
        pred_means, pred_vars, _, _ = self.decode_step(cats, nums, hx)

        # Beide [B, dim_output]; der "var"-Kopf des PSP liefert die Log-Varianz
        return pred_means[1]["outcome_mean"], pred_vars[1]["outcome_var"]

    @contextmanager
    def dropout_disabled(self):
        # Schaltet das MC-Dropout aller LSTM-Zellen ab, für eine deterministische Vorhersage
        # (wie _disable_model_dropout in der Auswertung des PSP)
        cells = [self.encoder.first_layer, *self.encoder.hidden_layers, self.decoder.first_layer, *self.decoder.hidden_layers]
        dropout_rates = [cell.p_logit for cell in cells]
        for cell in cells:
            cell.p_logit = 0.0
        try:
            yield
        finally:
            for cell, p_logit in zip(cells, dropout_rates):
                cell.p_logit = p_logit

    def sample_event(self, pred_means, pred_vars, sample=True):
        # Zieht ein Event aus der Vorhersage eines Decoder-Schritts (für den Rollout und für Scheduled Sampling).
        # Mit sample=False das wahrscheinlichste Event: Klasse mit dem höchsten Logit und vorhergesagter Mittelwert.
        def draw(logits):
            # Padding bzw. "kein Wert" (Klasse 0) ist ausgeschlossen
            logits = logits.clone()
            logits[:, 0] = float("-inf")
            return torch.multinomial(F.softmax(logits, dim=1), num_samples=1).squeeze(1) if sample else logits.argmax(dim=1)

        # Aktivität aus der Softmax ziehen
        logits = pred_means[0]["activity_mean"]
        sampled_activity = draw(logits)

        # Numerische Attribute aus N(mean, exp(logvar)) ziehen
        if self.suffix_num_idx and not sample:
            sampled_nums = pred_means[1]["event_attributes_mean"].clone()
        elif self.suffix_num_idx:
            logvar = torch.clamp(pred_vars[1]["event_attributes_var"], min=MIN_LOGVAR, max=MAX_LOGVAR)
            sampled_nums = pred_means[1]["event_attributes_mean"] + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        else:
            sampled_nums = torch.zeros(logits.shape[0], 0, dtype=logits.dtype, device=logits.device)

        # Fehlt-Kennzeichen je numerischem Attribut aus der Sigmoid ziehen (1 = der Wert fehlt, sampled_nums zählt dort nicht)
        if self.suffix_num_idx:
            missing_logits = pred_means[0]["missing_mean"]
            sampled_missing = torch.bernoulli(torch.sigmoid(missing_logits)) if sample else (missing_logits > 0).to(sampled_nums.dtype)
        else:
            sampled_missing = torch.zeros_like(sampled_nums)

        # Weitere kategoriale Attribute aus ihrer Softmax ziehen, ein echtes Event hat immer einen Wert (Klasse 0 ausgeschlossen)
        sampled_cats = {name: draw(pred_means[0][f"{self.cat_keys[name]}_mean"]) for name in self.cat_names}
        return sampled_activity, sampled_cats, sampled_nums, sampled_missing

    def forward_suffix(self, x_case, x_process, prefix_len, t, x_suffix, suffix_len, teacher_forcing_ratio=1.0):
        # Teacher Forcing: Schritt k bekommt das echte Event k-1 (Schritt 0 das SOS-Event) und sagt Event k voraus.
        # Event suffix_len ist das EOS-Event, daher suffix_len + 1 Schritte.
        # teacher_forcing_ratio < 1 (Scheduled Sampling wie im PSP): pro Schritt wird für den ganzen Batch gewürfelt,
        # ob das echte Event k-1 oder das vom Modell gezogene Event k-1 die Eingabe ist.
        hx, cats, nums = self.encode(x_case=x_case, x_process=x_process, prefix_len=prefix_len, t=t)
        # Encoder-Zustand, von dem alle Vorhersagen abhängen (GradNorm misst daran die Gradientennormen)
        h_enc = hx[0]
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
        # Weitere kategoriale Attribute: 0 = kein Wert (Padding, EOS-Event oder unbekannter Wert)
        target_cats = {}
        for name, idx in self.suffix_cat_idx.items():
            target_cats[name] = (suffix[:, :, idx].argmax(dim=2) + 1) * (suffix[:, :, idx].sum(dim=2) > 0).long()
        # Schneidet die numerischen Spalten aus target raus, getrennt in Wert und Fehlt-Kennzeichen
        target_nums, target_missing = self.split_missing(suffix[:, :, self.suffix_num_idx])
        # Das Treatment steht nur im SOS-Event
        t_pad = torch.zeros(B, self.dim_t_input, dtype=suffix.dtype, device=suffix.device)

        outputs = {"activity_logits": [], "outcome_mean": [], "outcome_logvar": [], "event_mean": [], "event_logvar": [], "missing_logits": []}
        cat_logits = {name: [] for name in self.cat_names}
        z = None
        for k in range(n_steps):
            pred_means, pred_vars, hx, z = self.decode_step(cats, nums, hx, z)
            outputs["activity_logits"].append(pred_means[0]["activity_mean"])
            for name in self.cat_names:
                cat_logits[name].append(pred_means[0][f"{self.cat_keys[name]}_mean"])
            outputs["outcome_mean"].append(pred_means[1]["outcome_mean"])
            outputs["outcome_logvar"].append(pred_vars[1]["outcome_var"])
            if self.suffix_num_idx:
                outputs["event_mean"].append(pred_means[1]["event_attributes_mean"])
                outputs["event_logvar"].append(pred_vars[1]["event_attributes_var"])
                outputs["missing_logits"].append(pred_means[0]["missing_mean"])
            # Decide per timestep whether to use teacher forcing
            if teacher_forcing_ratio >= 1.0 or torch.rand(1).item() < teacher_forcing_ratio:
                # Use ground-truth previous event
                next_activity, next_cats, next_nums = target_activity[:, k], [target_cats[name][:, k] for name in self.cat_names], target_nums[:, k]
                next_missing = target_missing[:, k]
            else:
                # Use model prediction: ein gezogenes Event wie im Rollout, ohne Gradient durch das Ziehen
                with torch.no_grad():
                    sampled_activity, sampled_cats, sampled_nums, sampled_missing = self.sample_event(pred_means, pred_vars)
                # Zieht das Modell EOS, gäbe es im Rollout keinen nächsten Schritt: diese Fälle bekommen das echte Event
                use_target = sampled_activity == self.eos_class
                next_activity = torch.where(use_target, target_activity[:, k], sampled_activity)
                next_cats = [torch.where(use_target, target_cats[name][:, k], sampled_cats[name]) for name in self.cat_names]
                next_nums = torch.where(use_target.unsqueeze(1), target_nums[:, k], sampled_nums * (1 - sampled_missing))
                next_missing = torch.where(use_target.unsqueeze(1), target_missing[:, k], sampled_missing)
            cats = torch.stack([next_activity] + next_cats, dim=1)
            nums = torch.cat([next_nums, next_missing, t_pad], dim=1)

        # Alle [B, n_steps, ...]
        outputs = {key: torch.stack(values, dim=1) for key, values in outputs.items() if values}
        outputs["target_activity"] = target_activity
        outputs["target_nums"] = target_nums
        outputs["target_missing"] = target_missing
        # Je weiterem kategorialen Attribut: Logits [B, n_steps, n_classes] und Ziel [B, n_steps]
        outputs["cat_logits"] = {name: torch.stack(values, dim=1) for name, values in cat_logits.items()}
        outputs["target_cats"] = target_cats
        outputs["h_enc"] = h_enc
        return outputs

    # Rollout zihet die Droputmaske derzet einmal pro Suffix. Im Suffix wird also die selbe droptoumaske verwendet
    @torch.no_grad()
    def rollout(self, x_case, x_process, prefix_len, t, n_samples, max_steps, sample=True, growing_num_values=None):
        # Zieht n_samples Suffixe pro Fall autoregressiv bis EOS: das gezogene Event ist der Input des nächsten Schritts.
        # Jede Zeile bekommt eigene Dropout-Masken (Encoder und Decoder), die über alle Schritte gleich bleiben,
        # also ein eigenes Netz aus dem Posterior. Der Zufall kommt aus dem globalen torch-Seed.
        # max_steps: höchstens so viele Decoder-Schritte (inkl. EOS), z.B. max_process_len + 1 wie in X_suffix
        # sample=False: statt zu ziehen das wahrscheinlichste Event je Schritt (siehe sample_event)
        # growing_num_values: Namen numerischer Attribute, die nie fallen (z.B. elapsed_time), wie im PSP
        B = x_process.shape[0]

        # Fall i steht in den Zeilen i * n_samples ... (i + 1) * n_samples - 1
        def repeat(v):
            return v.repeat_interleave(n_samples, dim=0) if v is not None else None

        hx, cats, nums = self.encode(x_case=repeat(x_case), x_process=repeat(x_process), prefix_len=repeat(prefix_len), t=repeat(t))
        R = cats.shape[0]
        device, dtype = nums.device, nums.dtype
        n_num = len(self.suffix_num_idx)
        # Das Treatment steht nur im SOS-Event
        t_pad = torch.zeros(R, self.dim_t_input, dtype=dtype, device=device)
        # Werte des letzten Prefix-Events, Untergrenze für die wachsenden Attribute
        last_nums = nums[:, :n_num]
        last_prefix_attributes = last_nums.clone()
        suffix_cols = self.model_params["suffix_cols_encoded"]
        growing_idx = [j for j, i in enumerate(self.suffix_num_idx) if suffix_cols[i] in (growing_num_values or [])]

        outcome_mean = torch.zeros(R, self.dim_output, dtype=dtype, device=device)
        outcome_var = torch.zeros(R, self.dim_output, dtype=dtype, device=device)
        suffix_len = torch.zeros(R, dtype=torch.long, device=device)
        finished = torch.zeros(R, dtype=torch.bool, device=device)
        truncated = torch.zeros(R, dtype=torch.bool, device=device)
        # Gezogene Suffixe wie in X_suffix: Aktivität (0 = Padding) und numerische Attribute je Position
        activities = torch.zeros(R, max_steps, dtype=torch.long, device=device)
        event_attributes = torch.zeros(R, max_steps, n_num, dtype=dtype, device=device)
        # Weitere kategoriale Attribute je Position (0 = kein Wert, wie am EOS-Event und im Padding)
        event_categoricals = {name: torch.zeros(R, max_steps, dtype=torch.long, device=device) for name in self.cat_names}

        z = None
        for k in range(max_steps):
            pred_means, pred_vars, hx, z = self.decode_step(cats, nums, hx, z)

            # Aktivität, numerische und weitere kategoriale Attribute ziehen
            sampled_activity, sampled_cats, sampled_nums, sampled_missing = self.sample_event(pred_means, pred_vars, sample=sample)
            if growing_idx:
                sampled_nums[:, growing_idx] = torch.max(sampled_nums[:, growing_idx], last_nums[:, growing_idx])
            last_nums = sampled_nums
            is_eos = sampled_activity == self.eos_class
            if k == max_steps - 1:
                # Kein EOS bis zum letzten Schritt: Suffix hier abschneiden und als abgeschnitten markieren
                truncated = ~finished & ~is_eos
                sampled_activity[truncated] = self.eos_class
                is_eos = torch.ones_like(is_eos)

            active = ~finished
            activities[active, k] = sampled_activity[active]
            # Am EOS-Event gibt es keine Attribute (wie in X_suffix), fehlende Werte stehen wie dort als missing_value
            event_attributes[active & ~is_eos, k] = sampled_nums.masked_fill(sampled_missing.bool(), self.missing_value)[active & ~is_eos]
            for name in self.cat_names:
                event_categoricals[name][active & ~is_eos, k] = sampled_cats[name][active & ~is_eos]

            # Outcome am EOS-Event ablesen; suffix_len = Anzahl echter Events davor
            newly_finished = active & is_eos
            outcome_mean[newly_finished] = pred_means[1]["outcome_mean"][newly_finished]
            outcome_logvar = torch.clamp(pred_vars[1]["outcome_var"], min=MIN_LOGVAR, max=MAX_LOGVAR)
            outcome_var[newly_finished] = torch.exp(outcome_logvar)[newly_finished]
            suffix_len[newly_finished] = k
            finished |= is_eos
            if finished.all():
                break

            cats = torch.stack([sampled_activity] + [sampled_cats[name] for name in self.cat_names], dim=1)
            nums = torch.cat([sampled_nums * (1 - sampled_missing), sampled_missing, t_pad], dim=1)

        # Alle mit Form [B, n_samples, ...]
        return {
            "outcome_mean": outcome_mean.view(B, n_samples, -1),
            "outcome_var": outcome_var.view(B, n_samples, -1),
            "suffix_len": suffix_len.view(B, n_samples),
            "truncated": truncated.view(B, n_samples),
            "activities": activities.view(B, n_samples, max_steps),
            "event_attributes": event_attributes.view(B, n_samples, max_steps, n_num),
            "event_categoricals": {name: values.view(B, n_samples, max_steps) for name, values in event_categoricals.items()},
            "last_prefix_attributes": last_prefix_attributes.view(B, n_samples, n_num),
        }

# Das PSP kann DLCausal regressor nicht übernehnem, weil es eine CustomLogik braucht
# Wir bauen also eine egene Hülle für das PSP für Scope
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
        # Anzahl der gezogenen Suffixe je Fall und Aktion für die Q-Werte bzw. MC-Dropout-Durchläufe für den Validierungs-Loss.
        # Die Q-Werte sind Mittelwerte über Suffixe: zu wenige Suffixe und das Rauschen übersteigt den Effekt der Aktion.
        self.mc_samples = model_params.get("psp_mc_samples", 200)
        self.mc_samples_val = model_params.get("psp_mc_samples_val", 5)
        self.forward_batch_size = model_params.get("psp_forward_batch_size", 1024)
        # Anteil der Decoder-Schritte im Training, die das echte vorherige Event bekommen (1.0 = volles Teacher Forcing).
        # Mit psp_use_scheduled_sampling senkt train_dl() den Wert über die Epochen, wie im PSP-Trainer.
        self.teacher_forcing_ratio = 1.0

        # GradNorm (Chen et al., 2018) wie im PSP-Trainer: ein lernbares Loss-Gewicht pro Aufgabe.
        # Aufgaben = Aktivität, jedes weitere kategoriale und jedes numerische Event-Attribut einzeln, Outcome.
        # Die Gewichte liegen bewusst nicht im Modell, damit der Optimizer aus train_dl() sie nicht anfasst.
        self.use_gradnorm = model_params.get("psp_use_gradnorm", False)
        if self.use_gradnorm:
            self.gn_alpha = model_params.get("psp_gn_alpha", 1.5)
            self.gn_learning_rate = model_params.get("psp_gn_learning_rate", 1e-4)
            suffix_cols = model_params["suffix_cols_encoded"]
            cat_task_names = ["activity"] + self.model.cat_names + (["missing"] if self.model.suffix_num_idx else [])
            self.gn_task_names = cat_task_names + [suffix_cols[i] for i in self.model.suffix_num_idx] + ["outcome"]
            self.number_tasks = len(self.gn_task_names)
            self.gn_weights = torch.nn.Parameter(torch.ones(self.number_tasks, dtype=torch.float))
            # Losses des ersten Batches, Bezugspunkt für die Trainingsrate je Aufgabe
            self.l0 = None
            self.gn_optimizer = torch.optim.AdamW(params=[self.gn_weights], lr=self.gn_learning_rate)
            # Kleinster erreichbarer Loss je Aufgabe: 0 bei der Cross-Entropy, 0.5 * MIN_LOGVAR bei der Loss Attenuation.
            # Die Loss Attenuation kann negativ werden, daher wird das Verhältnis L / L0 ab dieser Untergrenze gemessen.
            self.gn_loss_floor = torch.tensor([0.0] * len(cat_task_names) + [0.5 * MIN_LOGVAR] * (self.number_tasks - len(cat_task_names)))

    # Wird aufgerufen von train_dl()
    def get_loss(self, x_case, x_event, t, prefix_len, y, weights=None, set_eval=True, x_suffix=None, suffix_len=None, return_parts=False):
        """
        Compute batch loss for training.
        With set_eval and return_parts, also returns the loss parts and readable metrics as a dict of floats
        (keys with "/" are the standard / attenuated losses per attribute, e.g. "num_std/cost").
        With return_parts in training, also returns [cat_losses_dict, num_losses_dict] (loss per attribute, as tensors).
        With psp_use_gradnorm, the training loss is the GradNorm-weighted sum of the losses per attribute.
        """
        # x_suffix: [B, dim_x_suffix, max_process_len + 1], suffix_len: [B] (siehe create_suffix_tensors in tensor_prep.py)
        if x_suffix is None or x_suffix.ndim != 3 or x_suffix.shape[1] != self.model.dim_x_suffix:
            raise ValueError("PSP needs X_suffix of shape [n, dim_x_suffix, max_process_len + 1] for training.")
        if set_eval:
            self.model.eval()
        else:
            self.model.train()

        # Debug: Nachsehen, ob Suffixe korrekt
#        sys.exit("Debug")

        # Outcome in Trainingsgerforderte Struktur bringen
        y = y.view(y.shape[0], -1)

        # Bringt das Treament in die richtige Form
        # z.b: zum beispiel für 2 Aktonen reicht für machen 1, machen 2, lassen statt [1, 0, 0] -> [0, 0]
        t = t[:, 1:] if self.n_classes > 2 else t

        # Wir brauchen die Suffic length als float32, weil vorher gannzahlig mit punkt
        suffix_len = suffix_len.long()
        # Die Validierung läuft immer mit vollem Teacher Forcing, damit der Val-Loss zwischen den Epochen vergleichbar bleibt
        teacher_forcing_ratio = 1.0 if set_eval else self.teacher_forcing_ratio
        out = self.model.forward_suffix(x_case=x_case, x_process=x_event, prefix_len=prefix_len, t=t, x_suffix=x_suffix, suffix_len=suffix_len,
                                        teacher_forcing_ratio=teacher_forcing_ratio)

        # Masken [B, n_steps]: echte Events stehen an Position < suffix_len, das EOS-Event an Position suffix_len
        steps = torch.arange(out["target_activity"].shape[1], device=suffix_len.device).unsqueeze(0)
        event_mask = steps < suffix_len.unsqueeze(1)
        eos_mask = steps == suffix_len.unsqueeze(1)

        # Aktivität (inkl. EOS): Cross-Entropy über alle Schritte bis einschließlich EOS
        activity_mask = (event_mask | eos_mask) & (out["target_activity"] > 0)
        activity_logits = out["activity_logits"][activity_mask]
        target_activity = out["target_activity"][activity_mask]
        parts = {"activity_ce": F.cross_entropy(activity_logits, target_activity)}

        # Weitere kategoriale Event-Attribute: Cross-Entropy nur an echten Events mit einem Wert (Klasse 0 = kein Wert zählt nicht)
        cat_logits, cat_targets = {}, {}
        for name in self.model.cat_names:
            cat_mask = event_mask & (out["target_cats"][name] > 0)
            if cat_mask.any():
                cat_logits[name], cat_targets[name] = out["cat_logits"][name][cat_mask], out["target_cats"][name][cat_mask]
                parts[f"{name}_ce"] = F.cross_entropy(cat_logits[name], cat_targets[name])

        # Numerische Event-Attribute: Loss Attenuation (Kendall & Gal), nur an echten Events und nur für vorhandene Werte.
        # Ob ein Wert fehlt, lernt ein eigener Kopf (binäre Cross-Entropy je Attribut).
        has_events = "event_mean" in out and event_mask.any()
        if has_events:
            event_mean, event_logvar, target_nums = out["event_mean"][event_mask], out["event_logvar"][event_mask], out["target_nums"][event_mask]
            missing_logits, target_missing = out["missing_logits"][event_mask], out["target_missing"][event_mask]
            present = 1 - target_missing
            parts["missing_ce"] = F.binary_cross_entropy_with_logits(missing_logits, target_missing)
            parts["event_nll"] = self.loss_attenuation(event_mean, event_logvar, target_nums, present)

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
                # Losses je Attribut wie in validation_epoch des PSP-Trainers, Schlüssel "<gruppe>/<attribut>":
                # "std" = Cross-Entropy bzw. MSE (ohne Unsicherheit), "unc" = Loss Attenuation.
                # Die Aktivität wird hier mit normaler Cross-Entropy trainiert, daher gibt es für sie kein "unc".
                parts["cat_std/activity"] = parts["activity_ce"]
                parts["total/std"] = parts["activity_ce"] + F.mse_loss(outcome_mean, y)
                for name in cat_logits:
                    parts[f"{name}_acc"] = (cat_logits[name][:, 1:].argmax(dim=1) + 1 == cat_targets[name]).float().mean()
                    parts[f"cat_std/{name}"] = parts[f"{name}_ce"]
                    parts["total/std"] = parts["total/std"] + parts[f"{name}_ce"]
                if has_events:
                    suffix_cols = self.model_params["suffix_cols_encoded"]
                    parts["missing_acc"] = ((missing_logits > 0) == (target_missing > 0)).float().mean()
                    parts["cat_std/missing"] = parts["missing_ce"]
                    for j, i in enumerate(self.model.suffix_num_idx):
                        parts[f"num_std/{suffix_cols[i]}"] = self.masked_mse(event_mean[:, j], target_nums[:, j], present[:, j])
                        parts[f"num_unc/{suffix_cols[i]}"] = self.loss_attenuation(event_mean[:, j], event_logvar[:, j], target_nums[:, j], present[:, j])
                    # Gesamt wie der Val-Loss aufgebaut: die Event-Attribute zählen zusammen als ein Mittelwert
                    parts["total/std"] = parts["total/std"] + parts["missing_ce"] + self.masked_mse(event_mean, target_nums, present)
                parts["num_std/outcome"] = F.mse_loss(outcome_mean, y)
                parts["num_unc/outcome"] = parts["outcome_nll"]
                return loss, {key: value.item() for key, value in parts.items()}
            return loss
        if return_parts or self.use_gradnorm:
            # Training: Losses je Attribut wie im PSP-Trainer (cat_losses_dict / num_losses_dict)
            cat_losses_dict = {"activity": parts["activity_ce"]}
            cat_losses_dict.update({name: parts[f"{name}_ce"] for name in cat_logits})
            num_losses_dict = {}
            if has_events:
                cat_losses_dict["missing"] = parts["missing_ce"]
                suffix_cols = self.model_params["suffix_cols_encoded"]
                for j, i in enumerate(self.model.suffix_num_idx):
                    num_losses_dict[suffix_cols[i]] = self.loss_attenuation(event_mean[:, j], event_logvar[:, j], target_nums[:, j], present[:, j])
            num_losses_dict["outcome"] = parts["outcome_nll"]
        if self.use_gradnorm:
            # Statt der einfachen Summe: jede Aufgabe mit ihrem GradNorm-Gewicht
            loss = self.gradnorm_step(task_losses={**cat_losses_dict, **num_losses_dict}, h=out["h_enc"])
        loss = loss + self.regularization_term * self.model.regularizer()
        if return_parts:
            # Nur zum Mitschreiben, immer ungewichtet
            cat_losses_dict = {key: value.detach() for key, value in cat_losses_dict.items()}
            num_losses_dict = {key: value.detach() for key, value in num_losses_dict.items()}
            return loss, [cat_losses_dict, num_losses_dict]
        return loss

    # aufgerufen von get_loss
    def gradnorm_step(self, task_losses, h):
        """
        One GradNorm step as in the PSP trainer (train_epoch_gradnorm): updates the loss weights
        and returns the weighted loss for the model update (backward and step happen in train_dl()).
        task_losses: dict task name -> loss tensor, h: encoder state [B, hidden_size].
        """
        idx = [self.gn_task_names.index(name) for name in task_losses]
        gn_seq_loss = torch.stack(list(task_losses.values()))
        # Für das Modell sind die Gewichte Konstanten (Stand vor dem Update dieses Batches)
        weighted_loss = (self.gn_weights[idx].detach() * gn_seq_loss).sum()
        if len(idx) < self.number_tasks:
            # Batch ohne echte Suffix-Events: Gewichte nur anwenden, nicht anpassen
            return weighted_loss

        # Initial loss for each task
        if self.l0 is None:
            self.l0 = gn_seq_loss.detach()

        # Gradientennorm jeder Aufgabe bezüglich h. Die Norm von w_i * L_i ist w_i * Norm von L_i,
        # so hängt der GradNorm-Loss nur von den Gewichten ab und verändert die Modell-Gradienten nicht.
        grad_norms = torch.stack([torch.norm(torch.autograd.grad(loss_i, h, retain_graph=True)[0]) for loss_i in gn_seq_loss])
        gw_all = torch.abs(self.gn_weights) * grad_norms

        # Compute the average gradient norm
        gw_all_avg = gw_all.mean().detach()
        # Compute loss ratio per task (gemessen ab der Untergrenze, siehe gn_loss_floor)
        loss_ratio = ((gn_seq_loss.detach() - self.gn_loss_floor) / (self.l0 - self.gn_loss_floor)).clamp(min=1e-8)
        # Compute the relative inverse training rate per task
        rt = loss_ratio / loss_ratio.mean()

        # Compute the GradNorm loss
        constant = gw_all_avg * rt ** self.gn_alpha
        gradnorm_loss = torch.abs(gw_all - constant).sum()

        # Optimize loss weights
        self.gn_optimizer.zero_grad()
        gradnorm_loss.backward()
        self.gn_optimizer.step()

        # Renormalization of weights
        self.gn_weights.data = (self.gn_weights.data / self.gn_weights.data.sum() * self.number_tasks).detach()

        return weighted_loss

    # Nur Debug ausgabe
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

    # Wie unsicher ist sich das Netz bei der Vorhersage?
    @staticmethod
    def loss_attenuation(mean, logvar, target, mask=None):
        # Das Netz lernt Mittelwert und Log-Varianz, gemittelt über alle Einträge (mit mask nur über die Einträge mit 1)
        logvar = torch.clamp(logvar, min=MIN_LOGVAR, max=MAX_LOGVAR)
        loss = 0.5 * (torch.exp(-logvar) * (target - mean) ** 2 + logvar)
        return torch.mean(loss) if mask is None else (loss * mask).sum() / mask.sum().clamp(min=1)

    @staticmethod
    def masked_mse(mean, target, mask):
        # MSE nur über die Einträge mit mask = 1
        return ((target - mean) ** 2 * mask).sum() / mask.sum().clamp(min=1)

    # Hier wird für eine Aktion, ein Treatment X suffixe gesampled. Dann wird für diese X Suffixe der outcome gemittelt. Kosten gehen negativ ein. Wir nehmen die Aktion mit höherem Wert.
    # Wird aufgerufen von forward_with_uncertainty
    def predict(self, x_case, x_event, t, prefix_len, n_samples):
        """
        Monte Carlo prediction for a fixed treatment input t: n_samples suffixes per case are rolled out
        until EOS (PSP.rollout), the outcome is read at the EOS event.
        Returns mean, variance of the outcome means over the suffixes and aleatoric variance, each of shape [n, dim_output].
        """
        # Wie X_suffix: höchstens max_process_len Events plus EOS
        max_steps = x_event.shape[2] + 1
        # Ein Rollout hat Fälle * n_samples Zeilen, höchstens 8 * forward_batch_size
        batch_size = max(1, 8 * self.forward_batch_size // n_samples)
        means, epistemic, aleatoric = [], [], []
        for start in range(0, x_event.shape[0], batch_size):
            batch = slice(start, start + batch_size)
            # Jede Zeile zieht eigene Dropout-Masken und ein eigenes Suffix
            roll = self.model.rollout(x_case=x_case[batch], x_process=x_event[batch], prefix_len=prefix_len[batch], t=t[batch],
                                      n_samples=n_samples, max_steps=max_steps,
                                      growing_num_values=self.model_params.get("psp_growing_num_values", ["elapsed_time"]))
            means.append(roll["outcome_mean"].mean(dim=1))
            # Streuung der Mittelwerte: Modellunsicherheit und Zufall im Prozessverlauf (nicht trennbar, da jedes Suffix
            # mit eigenen Dropout-Masken gezogen wird). Mittlere Varianz = Rauschen des Outcomes bei gegebenem Suffix.
            epistemic.append(roll["outcome_mean"].var(dim=1, unbiased=False))
            aleatoric.append(roll["outcome_var"].mean(dim=1))
        return torch.cat(means), torch.cat(epistemic), torch.cat(aleatoric)

    # Predcited die outomce distribution under each action
    def forward_with_uncertainty(self, x_case, x_event, t, prefix_len, y=None, set_eval=True):
        """
        Predict the outcome distribution under every action.
        Returns means, variances of the outcome means over the suffixes (see predict) and aleatoric variances,
        each of shape [n_classes, n].
        """
        if x_event is None or x_event.ndim != 3:
            raise ValueError("PSP needs the tensor encoding (X_event of shape [n, dim_x_event, max_process_len]).")

        if set_eval:
            self.model.eval()
        else:
            self.model.train()

        results = []
        with torch.no_grad():
            # Für jedes Treatment gucken, wie der outcome aussieht
            for treatment in range(self.n_classes):
                # Künstliches Treatment: alle Fälle bekommen dieselbe Aktion
                t_artificial = torch.zeros((x_event.shape[0], self.model.dim_t_input), dtype=x_event.dtype, device=x_event.device)
                if self.n_classes > 2:
                    if treatment != 0:
                        t_artificial[:, treatment - 1] = 1
                else:
                    t_artificial[:] = treatment
                # WICHTIG
                results.append(self.predict(x_case, x_event, t_artificial, prefix_len, n_samples=self.mc_samples))

        # Bei mehreren Outputs (dim_output > 1) bleibt wie im LSTM die letzte Dimension erhalten
        means, epistemic, aleatoric = (
            torch.stack(values).squeeze(-1) if self.model.dim_output == 1 else torch.stack(values)
            for values in zip(*results)
        )
        return means, epistemic, aleatoric

    # Forward ist beim PSP forward with uncertainty
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
