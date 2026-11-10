"""
PSP-Modell (probabilistic_suffix_prediction_U-ED-LSTM, Code unverändert wie auf GitHub: EventLogLoader,
DropoutUncertaintyEncoderDecoderLSTM, Trainer, ProbabilisticEvaluation) mit derselben Schnittstelle wie
estimators/xgb_model.py und lstm_model.py (samples, fit, predict, values, knows, save, load); gespeichert in
models/psp_model<outcome>.pkl. Das PSP sagt Suffixe vorher (gezogen wie der MC-SA des Papers); den Outcome je Option
rechnet dieses Modell aus Prefix und gezogenen Suffixen.

Originalgetreu wie die BPIC17-Notebooks des PSP (loader_notebooks/version_2/BPIC17_full_loader,
training_variational_dropout_v2/BPIC17/full_enc_dec_lstm_gn):
- Eingänge je Event: Aktivität, Lifecycle und Mitarbeiter (je ein eigener Eingang) und die vier Zeiten des Loaders (seit
  Fallbeginn, seit dem vorigen Event, Wochentag, Sekunden am Tag); je Fall STATIC_CATEGORICAL und STATIC_NUMERIC (der
  Loader nimmt je Fall den ersten Wert, im Training aus dem ganzen Fall, also auch aus späteren Angeboten)
- Decoder: Aktivität und die beiden Zeiten (DECODER), im Training MIN_SUFFIX_SIZE Schritte (einstellbar: TRAIN suffix_size)
- Proben: alle Prefixe jeder Länge
- Training (TRAIN, MODEL_PARAMS): AdamW 1e-5, ReduceLROnPlateau, Batch 256, L2 1e-4, Teacher Forcing von 1 auf 0 bis
  zur Hälfte der Epochen, GradNorm, 100 Epochen; behalten wird das Modell der letzten Epoche
Abweichend vom Original: nur train_cases Fälle der Trainingsdaten, dazu Validierungsfälle im Verhältnis des Originals
(65 : 15); getestet wird auf den Testfällen der Pipeline (helpers/training), nicht auf einem Teil des Loaders. Mit TRAIN
point_weight k > 1 (Standard 1 wie im Original) stehen die Proben, deren Prefix am Options-Event eines Wahlpunkts endet,
k-mal in den Trainingsdaten (wie Gewicht k im Verlust, ohne Eingriff in Loader, Netz und Trainer; _weight_points); die
Validierung bleibt ungewichtet. Beim Ziehen
(draw_cases) ohne die Sperren von sample_suffix (Zeit seit Fallbeginn nie fallend, Abstand neu gezogen bis positiv): sie
verlängerten die gezogene Restzeit auf den RCT-Testfällen um 14 Tage (13,4 tatsächlich, 27,6 gezogen; ohne: 15,9).
Die Aktivität ist Aktivität|Lifecycle (TRAIN lifecycle_tokens, im Original nur die Aktivität): das Netz sagt auch den
Lifecycle voraus, ohne Eingriff in Encoder und Decoder. Das brauchen Wahlpunkte und Optionen, die nicht complete sind.
Ältere Modelle ohne diese Einstellung bleiben nutzbar (lifecycle_tokens False).

Mit TRAIN remaining_head (Variante (b), 2026-11-09; Standard aus wie im Original) lernt ein Restzeit-Kopf (REMAINING_HEAD)
auf dem Zustand des Encoders (h und c der letzten Schicht) log(1 + Restzeit in Tagen) bis zum Fallende als Normalverteilung
mit gelernter Varianz. Sein Verlust geht mit festem Gewicht in den Gesamtverlust ein, außerhalb von GradNorm, das weiter
die drei Aufgaben des Decoders gewichtet (_remaining_trainer); das Ziel kommt aus dem ganzen Fall, auch wenn das Fallende
jenseits des Suffixfensters liegt. Bewertet wird weiter über die gezogenen Suffixe, mit value_from "head" (Attribut oder
TRAIN) über den Kopf: je Ziehung eigene Dropout-Masken des Encoders und ein Wert aus der vorhergesagten Verteilung
(option_remaining_many), mit denselben Zufallszahlen für alle Optionen.

Bei der Abfrage ist die Option das letzte Event des Prefix (das Options-Event):
- control_flow (BPIC17 channel): nach dem Entscheidungs-Event (O_Created) das gewählte Event (O_Sent ...); Optionen sind
  Aktivitäten (ihr complete-Event) oder Aktivität|Lifecycle wie im Simulator (interventions.option_token)
- employee (BPIC17 W_Call incomplete files): das start-Event des Arbeitsschritts mit dem gewählten Mitarbeiter
Outcome je gezogenem Suffix (OUTCOMES), geschätzt als Mittel über samples_per_option Suffixe: die Durchlaufzeit
(negative_cycle_days: negative Tage vom Fallbeginn bis zum letzten Event, gezogen oder im Prefix) oder die Kosten der
Werkstatt (negative_repair_cost, Option kosten: Lohnkosten nach Stundensatz aus den gezogenen start- und complete-Events
plus Satz je Tag Durchlaufzeit, prescriptive/costs/repair/outcome.py SuffixCosts; je Option, das Netz bleibt dasselbe).

    model = PSPModel("bpic17").fit(log)          # log: Event-Log der Trainingsfälle
    model.save("prescriptive/models/psp_model.pkl")
    PSPModel.load(path).values(point, events, options, now)   # geschätzter Outcome je Option
    PSPModel.load(path).outcome_samples(point, events, options, now)   # Outcome je Option und gezogenem Suffix (policies/psp_erm_policy.py)
    PSPModel.load_checkpoint()                   # während des Trainings: das Netz der letzten fertigen Epoche
"""
import io
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import torch
import tqdm.notebook
import tqdm.std


# Loader und Trainer des PSP nutzen tqdm.notebook, das außerhalb von Jupyter abbricht: Ersatz ohne Ausgabe
class _QuietTqdm(tqdm.std.tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("disable", True)
        super().__init__(*args, **kwargs)


tqdm.notebook.tqdm = _QuietTqdm
PSP_SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "probabilistic_suffix_prediction_U-ED-LSTM", "src"))
if PSP_SRC not in sys.path:
    sys.path.insert(0, PSP_SRC)
MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

CASE, ACTIVITY, LIFECYCLE, RESOURCE, TIMESTAMP = "case:concept:name", "concept:name", "lifecycle:transition", "org:resource", "time:timestamp"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
TIME_COLUMNS = ["case_elapsed_time", "event_elapsed_time", "day_in_week", "seconds_in_day"]
DYNAMIC_CATEGORICAL = [ACTIVITY, LIFECYCLE, RESOURCE]
STATIC_CATEGORICAL = ["case:LoanGoal", "case:ApplicationType", "Accepted", "Selected"]
STATIC_NUMERIC = ["case:RequestedAmount", "FirstWithdrawalAmount", "NumberOfTerms", "MonthlyCost", "CreditScore"]
# Andere Datensätze: (kategorial, numerisch); ohne Eintrag die Spalten von BPIC17 oben
STATIC_COLUMNS = {"bpic19": (["case:Spend area text", "case:Sub spend area text", "case:Item Type", "case:Item Category", "case:Document Type",
                              "case:Company", "case:Vendor"], ["Cumulative net worth (EUR)"]),
                  "repair": (["case:device"], [])}
DECODER = [[ACTIVITY], TIME_COLUMNS[:2]]
# gap_classes (Variante (d), 2026-11-10; Standard aus): der Decoder sagt je Schritt zusätzlich die Klasse des Abstands zum
# vorigen Event voraus (GAP_CLASS, kategorial wie die Aktivität, auch Eingang des Encoders; trainiert mit derselben
# attenuierten Kreuzentropie). Gezogen werden je Schritt im Decoder Aktivität, Klasse und Abstand: den Abstand zieht
# draw_cases aus der vom Decoder vorhergesagten Normalverteilung des Abstands, beschnitten auf die Grenzen der gezogenen
# Klasse (gap_draw "decoder", Standard; Event und Zeit kommen so beide aus dem Decoder, Vorgabe 2026-11-10), und gibt ihn mit
# der Zeit seit Fallbeginn zurück ins Netz. Nur zum Vergleich gap_draw "real": der Abstand aus den echten Abständen der
# Trainingsfälle mit derselben Aktivität und Klasse (gap_samples; zu wenige: nur die Klasse). Grund für die Klassen: die
# echten Abstände sind zweigipflig (BPIC17: 45 % genau 0 s, 79 % unter 10 Minuten, zusammen 0,1 % der Durchlaufzeit; die
# Durchlaufzeit kommt aus den 13 % ab einem Tag, bis 7 Tage 32 %, 14 bis 30 Tage 37 %). Eine Normal- oder Log-Normalverteilung
# je Schritt bildet das nicht ab (log_gaps verdoppelte die gezogene Durchlaufzeit).
GAP_CLASS = "gap_class"
GAP_EDGES = (1, 600, 4 * 3600, 86400, 3 * 86400, 7 * 86400, 14 * 86400, 30 * 86400)  # obere Grenzen der Klassen in Sekunden
GAP_NAMES = ("0s", "<10min", "<4h", "<1d", "<3d", "<7d", "<14d", "<30d", ">=30d")
GAP_SAMPLES, GAP_MIN_SAMPLES = 2000, 30  # je (Aktivität, Klasse) höchstens so viele echte Abstände; mit weniger als GAP_MIN_SAMPLES nur die Klasse
GAP_DRAW = ("decoder", "real")  # TRAIN/params gap_draw: Zeit innerhalb der Klasse aus dem Decoder (Standard) oder aus echten Abständen (Vergleich)


def gap_class_names(seconds):
    """Klasse (GAP_NAMES) je Abstand zum vorigen Event in Sekunden; fehlend (erstes Event): 0s."""
    seconds = np.nan_to_num(np.asarray(seconds, dtype=float), nan=0.0)
    return np.asarray(GAP_NAMES, dtype=object)[np.searchsorted(GAP_EDGES, seconds, side="right")]
MIN_SUFFIX_SIZE = 5
VALIDATION_SHARE = 0.15 / 0.8  # Anteil der Validierung an Training und Validierung wie im Original (15 : 65 der Fälle)
MODEL_PARAMS = {"hidden_size": 128, "num_layers": 4, "dropout": 0.1}
# train_cases: so viele Trainingsfälle (None: alle); checkpoints: Ordner, in dem der Trainer jede Epoche speichert;
# lifecycle_tokens: das Netz lernt und zieht Aktivität|Lifecycle statt der Aktivität; suffix_size: Schritte des Decoders im
# Training (Original 5, gezogen wird oft viel weiter: siehe data/psp_varianten); point_weight: Proben am Options-Event eines
# Wahlpunkts so oft im Training (1: wie im Original); device: Gerät fürs Training ("auto": GPU, wenn torch eine sieht; das
# fertige Netz liegt danach wieder auf der CPU, Ziehen und Speichern wie bisher); statics: (kategorial, numerisch) statt der
# Fallattribute oben (None: wie im Original). BPIC17 im Original mit Accepted, Selected, CreditScore und den Angebotswerten: der
# Loader nimmt sie aus dem ganzen Fall, im Training also aus späteren Angeboten (Selected, CreditScore verraten den Ausgang); bei
# der Abfrage vor dem ersten Angebot und im Simulator fehlen sie. STATICS_WITHOUT_FUTURE: nur was zu Fallbeginn feststeht
TRAIN = {"epochs": 100, "learning_rate": 1e-5, "batch_size": 256, "regularization_term": 1e-4, "train_cases": 1000, "threads": 16,
         "checkpoints": os.path.join(MODELS, "psp_training"), "lifecycle_tokens": True, "suffix_size": MIN_SUFFIX_SIZE, "point_weight": 1,
         "device": "auto"}  # checkpoints: BPIC17; andere psp_training_<datensatz>
STATICS_WITHOUT_FUTURE = (["case:LoanGoal", "case:ApplicationType"], ["case:RequestedAmount"])  # BPIC17, für TRAIN statics
# Weitere TRAIN-Einstellungen der Varianten (2026-11-08/09, Standard wie im Original): log_gaps (Abstände log-normal, siehe
# properties), time_from "gaps" (Durchlaufzeit aus der Summe der gezogenen Abstände, siehe outcome)
MAX_GAP_DAYS = 180  # log_gaps: längster gezogener Abstand (wie max_gap_days des Next-Event-Modells, next_event_prediction/datasets.py)
# remaining_head (siehe oben): Breite der verdeckten Schicht des Kopfs, Gewicht seines Verlusts (TRAIN remaining_weight),
# längste gezogene Restzeit (Schutz vor Überlauf im Schwanz der Log-Normalverteilung) und weiche untere Grenze der
# log-Varianz (Streuung auf log(1 + Tage) mindestens etwa 0,22). Ohne Grenze (bis -10) zog der Kopf die Varianz bei Prefixen
# am Fallende (Restzeit genau 0) auf das Minimum; einzelne ähnliche Proben mit etwas Restzeit gaben dann Verluste von
# Tausenden, die das Clipping des Gradienten beherrschten (Lauf vom 2026-11-09 nach 2 Epochen abgebrochen).
REMAINING_HEAD = {"hidden": 64, "weight": 1.0, "max_days": 730, "min_logvar": -3.0}
SAMPLES = 50  # gezogene Suffixe je Option
EVAL_PREFIXES = 1000  # prefix_kpis: so viele zufällige Prefixe jeder Länge je Log
COVERAGES = (50, 75, 95)  # Vorhersageintervalle wie im Paper (Tab. III)
OUTCOMES = ("negative_cycle_days", "negative_repair_cost")
DAY = 86400.0


def _new_remaining_head(hidden_size):
    """Restzeit-Kopf: aus (h, c) der letzten Encoder-Schicht Mittel und log-Varianz von log(1 + Restzeit in Tagen)."""
    return torch.nn.Sequential(torch.nn.Linear(2 * hidden_size, REMAINING_HEAD["hidden"]), torch.nn.ReLU(),
                               torch.nn.Linear(REMAINING_HEAD["hidden"], 2))


def _head_output(net, encoding):
    """Der Restzeit-Kopf des Netzes auf einem Zustand (h, c) des Encoders: (Mittel, log-Varianz) je Probe."""
    h, c = (state[-1] if state.dim() == 3 else state for state in encoding)
    out = net.remaining_head(torch.cat([h, c], dim=-1))
    floor = REMAINING_HEAD["min_logvar"]
    return out[:, 0], (floor + torch.nn.functional.softplus(out[:, 1] - floor)).clamp(max=10.0)


def _remaining_nll(mean, logvar, target):
    """Negative Log-Likelihood der Normalverteilung (ohne Konstante) je Probe."""
    return 0.5 * (torch.exp(-logvar) * (target - mean) ** 2 + logvar)


def _keep_encoding(module, inputs, output):
    """Forward-Hook am Encoder (nur im Training mit remaining_head): sein Zustand (h, c) für den Restzeit-Kopf."""
    module.kept_encoding = output


def _remaining_trainer(trainer_class, weight):
    """
    Der Trainer des Originals mit Restzeit-Kopf (TRAIN remaining_head): train_epoch_gradnorm rechnet wie dort (Verluste von
    Aktivität und Zeiten, GradNorm über diese Aufgaben, L2, Clipping, ein Schritt des Optimierers) und addiert weight mal den
    Restzeit-Verlust zum Gesamtverlust. Das Ziel steht als zusätzlicher letzter Zahlen-Tensor in den Trainingsdaten (fit) und
    wird vor dem Aufruf des Netzes abgetrennt; den Zustand des Encoders hält _keep_encoding fest.
    """
    class RemainingTrainer(trainer_class):
        remaining_sum, remaining_count = 0.0, 0

        def train_epoch_gradnorm(self, prefixes, suffixes, eos_paddings=None, prefix_mask=None, static_inputs=None):
            target = prefixes[1][-1][:, -1]
            prefixes, suffixes = [prefixes[0], prefixes[1][:-1]], [suffixes[0], suffixes[1][:-1]]
            predictions, (h, _), _, (cat_index, num_index) = self.model(
                prefixes=prefixes, static_inputs=static_inputs, suffixes=suffixes, teacher_forcing_ratio=self.teacher_forcing_ratio,
                prefix_mask=prefix_mask)
            predictions_cat, predictions_num = predictions
            cat_losses = {key[:-5]: self.loss_obj.loss_attenuation_cross_entropy(
                pred_logits=predictions_cat[key], pred_logvars=predictions_cat.get(f"{key[:-5]}_var"), T=30,
                targets=suffixes[0][cat_index[key[:-5]]].long(), eos_paddings=eos_paddings) for key in predictions_cat if key.endswith("_mean")}
            num_losses = {key[:-5]: self.loss_obj.loss_attenuation_mse(
                pred_means=predictions_num[key], pred_logvars=predictions_num.get(f"{key[:-5]}_var"),
                targets=suffixes[1][num_index[key[:-5]]], eos_paddings=eos_paddings) for key in predictions_num if key.endswith("_mean")}
            gn_seq_loss = torch.stack(list(cat_losses.values()) + list(num_losses.values()))
            if self.l0 is None:
                self.l0 = gn_seq_loss.detach()
            self.optimizer.zero_grad()
            weight_reg_enc, bias_reg_enc = self.model.encoder.regularizer()
            weight_reg_dec, bias_reg_dec = self.model.decoder.regularizer()
            mean, logvar = _head_output(self.model, self.model.encoder.kept_encoding)
            remaining = _remaining_nll(mean, logvar, target).mean()
            loss = (self.gn_weights * gn_seq_loss).sum() + self.regularization_term * (
                (weight_reg_enc + weight_reg_dec).to(self.device) + (bias_reg_enc + bias_reg_dec).to(self.device)) + weight * remaining
            loss.backward(retain_graph=True)
            # GradNorm wie im Original: Norm des Gradienten jeder Aufgabe am letzten Zustand des Decoders
            gw = torch.stack([torch.norm(torch.autograd.grad(self.gn_weights[i] * loss_i, h, retain_graph=True, create_graph=True)[0])
                              for i, loss_i in enumerate(gn_seq_loss)])
            ratio = (gn_seq_loss / self.l0).detach()
            constant = (gw.mean().detach() * (ratio / ratio.mean()) ** self.gn_alpha).detach()
            self.gn_optimizer.zero_grad()
            torch.abs(gw - constant).sum().backward()
            self.gn_optimizer.step()
            self.gn_weights.data = (self.gn_weights.data / self.gn_weights.data.sum() * self.number_tasks).detach()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.remaining_sum, self.remaining_count = self.remaining_sum + float(remaining.detach()), self.remaining_count + 1
            return [cat_losses, num_losses], loss

        def validation_epoch(self, *args, **kwargs):
            if self.remaining_count:
                print(f"Restzeit-Kopf: Verlust Training {self.remaining_sum / self.remaining_count:.4f} (Gewicht {weight})", flush=True)
            self.remaining_sum, self.remaining_count = 0.0, 0
            return super().validation_epoch(*args, **kwargs)

    return RemainingTrainer


class _SilentWriter:
    """Ersatz für den Tensorboard-Writer des Trainers."""

    def add_scalars(self, *args, **kwargs):
        pass


class PSPModel:
    max_eval_samples = 300  # helpers/training.point_kpis: höchstens so viele Proben je Wahlpunkt und Daten (Ziehen ist teuer)
    lifecycle_tokens = False  # gespeicherte Modelle von vor den Token

    def __init__(self, dataset, points=None, seed=42, params=None):
        """dataset: Schlüssel im Katalog; points: die Wahlpunkte (Standard: interventions.choice_points des Datensatzes)."""
        import interventions
        screening = interventions.catalog()[dataset]["screening"]
        self.dataset, self.seed, self.params = dataset, seed, {**TRAIN, **(params or {})}
        if "checkpoints" not in (params or {}) and dataset != "bpic17":  # eigener Ordner, damit Trainings mehrerer Datensätze parallel laufen können
            self.params["checkpoints"] = os.path.join(MODELS, f"psp_training_{dataset}")
        self.points = {point.name: point for point in (points if points is not None else interventions.choice_points(dataset))}
        self.system_users = screening.get("system_users")
        self.samples_per_option = SAMPLES
        self.lifecycle_tokens = self.params["lifecycle_tokens"]
        self.net, self.encoder_decoder, self.reference, self.summary, self.options = None, None, None, {}, {}

    # ------------------------------------------------------------------ Sicht des PSP

    def statics(self):
        """Die Fallattribute des Datensatzes: (kategorial, numerisch); mit TRAIN statics die dort gesetzten (z. B. ohne Zukunftswissen)."""
        chosen = (getattr(self, "params", None) or {}).get("statics")
        if chosen is not None:
            return list(chosen[0]), list(chosen[1])
        return STATIC_COLUMNS.get(self.dataset, (STATIC_CATEGORICAL, STATIC_NUMERIC))

    def properties(self):
        """Einstellungen des Loaders wie im BPIC17-Loader-Notebook des PSP (ohne Test-Teil, siehe oben)."""
        static_categorical, static_numeric = self.statics()
        # log_gaps: der Abstand zum vorigen Event als positive Spalte (Loader: log1p und standardisiert, also log-normal statt
        # normal auf linearer Skala; Abstände reichen von Sekunden bis Wochen)
        log_gaps = (getattr(self, "params", None) or {}).get("log_gaps", False)
        continuous = [c for c in TIME_COLUMNS if not (log_gaps and c == TIME_COLUMNS[1])]
        return {"case_name": CASE, "concept_name": ACTIVITY, "timestamp_name": TIMESTAMP, "date_format": DATE_FORMAT,
                "time_since_case_start_column": TIME_COLUMNS[0], "time_since_last_event_column": TIME_COLUMNS[1],
                "day_in_week_column": TIME_COLUMNS[2], "seconds_in_day_column": TIME_COLUMNS[3], "min_suffix_size": self.params.get("suffix_size", MIN_SUFFIX_SIZE),
                "train_validation_size": VALIDATION_SHARE, "test_validation_size": 0.0, "window_size": "auto",
                "categorical_columns": self.dynamic_categorical(), "static_categorical_columns": static_categorical,
                "continuous_columns": continuous, "continuous_positive_columns": [TIME_COLUMNS[1]] if log_gaps else [],
                "static_continuous_columns": static_numeric}

    def gap_classes(self):
        """Variante (d): Abstandsklasse als Ausgabe des Decoders (TRAIN gap_classes, siehe GAP_CLASS)."""
        return bool((getattr(self, "params", None) or {}).get("gap_classes", False))

    def gap_draw(self):
        """gap_classes: woher der Abstand innerhalb der gezogenen Klasse kommt (params gap_draw, GAP_DRAW; Standard aus dem Decoder)."""
        mode = (getattr(self, "params", None) or {}).get("gap_draw", GAP_DRAW[0])
        if mode not in GAP_DRAW:
            raise ValueError(f"gap_draw {mode!r}, erlaubt: {GAP_DRAW}")
        return mode

    def dynamic_categorical(self):
        """Kategoriale Eingänge je Event: DYNAMIC_CATEGORICAL, mit gap_classes dazu GAP_CLASS (am Ende: die Stellen der anderen bleiben)."""
        return DYNAMIC_CATEGORICAL + ([GAP_CLASS] if self.gap_classes() else [])

    def decoder(self):
        """Ein- und Ausgänge des Decoders: DECODER, mit gap_classes dazu GAP_CLASS als zweite kategoriale Ausgabe."""
        return [[ACTIVITY, GAP_CLASS], TIME_COLUMNS[:2]] if self.gap_classes() else DECODER

    def labels(self, frame):
        """Die Aktivitäten der Events, wie das Netz sie kennt: mit lifecycle_tokens Aktivität|Lifecycle."""
        labels = frame[ACTIVITY].astype(str)
        if not self.lifecycle_tokens:
            return labels
        return labels + "|" + (frame[LIFECYCLE].astype(str) if LIFECYCLE in frame else "complete")

    def view(self, frame):
        """Die Events (nach Fall und Zeit sortiert) als CSV-Text für den Loader: die Spalten des Originals, Zeit als Text."""
        frame = frame.reset_index(drop=True)
        times = pd.to_datetime(frame[TIMESTAMP])
        if times.dt.tz is not None:
            times = times.dt.tz_convert("UTC").dt.tz_localize(None)
        view = pd.DataFrame({CASE: frame[CASE].astype(str), ACTIVITY: self.labels(frame),
                             LIFECYCLE: frame[LIFECYCLE].astype(str) if LIFECYCLE in frame else "complete",
                             RESOURCE: frame[RESOURCE] if RESOURCE in frame else np.nan, TIMESTAMP: times.dt.strftime(DATE_FORMAT)})
        if self.gap_classes():
            view[GAP_CLASS] = gap_class_names(times.groupby(frame[CASE].astype(str).to_numpy()).diff().dt.total_seconds().to_numpy())
        static_categorical, static_numeric = self.statics()
        for name in static_categorical:
            view[name] = frame[name] if name in frame else np.nan
        for name in static_numeric:
            view[name] = pd.to_numeric(frame[name], errors="coerce") if name in frame else np.nan
        return io.StringIO(view.to_csv(index=False))

    # ------------------------------------------------------------------ Proben

    def samples(self, log):
        """
        Je Wahlpunkt die Proben des Logs, an denselben Stellen wie XGBModel.samples: (Fall, Probe, Option); eine Probe ist
        (Events des Falls, Ende, Zeitpunkt): der Prefix vor dem Options-Event sind die Zeilen bis Ende (ausschließlich).
        """
        from interventions import option_at
        log = log.sort_values([CASE, TIMESTAMP], kind="stable").reset_index(drop=True)
        times = pd.to_datetime(log[TIMESTAMP])
        if times.dt.tz is not None:
            times = times.dt.tz_convert("UTC").dt.tz_localize(None)
        seconds = (times - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
        lifecycle = log[LIFECYCLE].to_numpy() if LIFECYCLE in log else np.full(len(log), "complete")
        resource = log[RESOURCE].to_numpy() if RESOURCE in log else np.full(len(log), None)
        activity, case = log[ACTIVITY].to_numpy(), log[CASE].to_numpy()
        system = re.compile(self.system_users) if self.system_users else None
        control = {(point.activity, point.lifecycle): point for point in self.points.values() if point.kind == "control_flow"}
        employee = {(point.activity, point.lifecycle): point for point in self.points.values() if point.kind == "employee"}
        samples = {name: [] for name in self.points}
        bounds = np.flatnonzero(np.r_[True, case[1:] != case[:-1], True])
        for begin, end in zip(bounds[:-1], bounds[1:]):
            part = log.iloc[begin:end]
            for i in range(begin, end):
                point = employee.get((activity[i], lifecycle[i]))
                if point is not None and isinstance(resource[i], str) and not (system and system.search(resource[i])):
                    # An einem complete-Event (Engine ohne Lifecycles) fragt die Engine am vorigen Event
                    decided = seconds[i - 1] if point.lifecycle == "complete" and i > begin else seconds[i]
                    samples[point.name].append((case[i], (part, i - begin, decided), resource[i]))
                point = control.get((activity[i], lifecycle[i]))
                option = option_at(activity[i + 1], lifecycle[i + 1], point.options) if point is not None and i + 1 < end else None
                if option is not None:
                    samples[point.name].append((case[i], (part, i - begin + 1, seconds[i]), option))
        return samples

    # ------------------------------------------------------------------ Training

    def fit(self, log):
        """Trainiert das PSP wie im Original auf allen Prefixen; log: Event-Log der Trainingsfälle. Liefert self."""
        from event_log_loader.new_event_log_loader_v2 import EventLogLoader
        from loss.losses import Loss
        from model.dropout_uncertainty_enc_dec_LSTM.dropout_uncertainty_model import DropoutUncertaintyEncoderDecoderLSTM
        from trainer.trainer import Trainer
        torch.set_num_threads(self.params["threads"])
        np.random.seed(self.seed)  # auch die Aufteilung des Loaders in Training und Validierung
        torch.manual_seed(self.seed)
        cases = np.sort(log[CASE].astype(str).unique())
        if self.params["train_cases"]:
            wanted = int(round(self.params["train_cases"] / (1 - VALIDATION_SHARE)))
            cases = np.random.default_rng(self.seed).choice(cases, min(wanted, len(cases)), replace=False)
        log = log[log[CASE].astype(str).isin(set(cases))].sort_values([CASE, TIMESTAMP], kind="stable")
        loader = EventLogLoader(event_log_location=self.view(log), event_log_properties=self.properties())
        train, validation = loader.get_dataset("train"), loader.get_dataset("val")
        train, weighted = self._weight_points(train, self.params.get("point_weight", 1))
        categories, numerics = train.all_categories
        static_categories, static_numerics = train.all_static_categories
        net = DropoutUncertaintyEncoderDecoderLSTM(
            data_set_categories=train.all_categories, enc_feat=[[f[0] for f in categories], [f[0] for f in numerics]], dec_feat=self.decoder(),
            seq_len_pred=train.min_suffix_size, static_data_set_categories=train.all_static_categories,
            static_enc_feat=[[f[0] for f in static_categories], [f[0] for f in static_numerics]], **MODEL_PARAMS)
        if self.gap_classes():
            self.gap_samples = self._gap_samples(log)
        remaining_head = self.params.get("remaining_head", False)
        if remaining_head:
            net.remaining_head = _new_remaining_head(net.hidden_size)  # vor dem Optimierer: seine Gewichte lernen mit
        optimizer = torch.optim.AdamW(params=net.parameters(), lr=self.params["learning_rate"], weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=15, min_lr=1e-8)
        self.encoder_decoder = train.encoder_decoder
        # Für ProbabilisticEvaluation (Kategorien, Encoder, window_size) reicht ein kleiner Teil der Trainingsproben
        self.reference = train.subset(range(min(len(train), 64)))
        system = re.compile(self.system_users) if self.system_users else None
        for name, point in self.points.items():
            at = log[(log[ACTIVITY] == point.activity) & (log[LIFECYCLE] == point.lifecycle)]
            self.options[name] = list(point.options) if point.kind == "control_flow" else sorted(
                r for r in at[RESOURCE].dropna().astype(str).unique() if not (system and system.search(r)))
        folder = self.params["checkpoints"]
        os.makedirs(folder, exist_ok=True)
        self.save(os.path.join(folder, "model.pkl"))  # ohne Netz; mit last.pt schon während des Trainings auswertbar (load_checkpoint)
        device = self.params.get("device", "auto")
        device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        trainer_class, hook = Trainer, None
        if remaining_head:  # nach self.reference: die Referenz bleibt ohne den Ziel-Tensor
            train.continuous_tensors.append(self._remaining_targets(train, log))
            hook = net.encoder.register_forward_hook(_keep_encoding)
            trainer_class = _remaining_trainer(Trainer, self.params.get("remaining_weight", REMAINING_HEAD["weight"]))
        trainer = trainer_class(
            device=device, model=net, data_train=train, data_val=validation, loss_obj=Loss(), log_normal_loss_num_feature=[],
            optimize_values={"regularization_term": self.params["regularization_term"], "optimizer": optimizer, "scheduler": scheduler,
                             "epochs": self.params["epochs"], "mini_batches": self.params["batch_size"], "shuffle": True,
                             "min_teacher_forcing_value": 0.0, "max_teacher_forcing_value": 1.0},
            suffix_data_split_value=train.min_suffix_size, writer=_SilentWriter(),
            gradnorm_values={"use_gradnorm": True, "number_tasks": len(self.decoder()[0]) + len(self.decoder()[1]), "gn_alpha": 1.5, "gn_learning_rate": 1e-4},
            save_model_n_th_epoch=1, saving_path=os.path.join(folder, "last.pt"))
        _, validation_losses, _ = trainer.train_model(use_statics=True, use_zero_padd_masking=True, use_eos_padd_masking=True)
        if hook is not None:
            hook.remove()
            net.encoder.__dict__.pop("kept_encoding", None)
            train.continuous_tensors.pop()
        self.net = net.cpu()
        self.net.eval()  # das MC-Dropout der LSTM-Zellen wirkt auch so
        losses = [float(np.mean(list(loss.values()))) if isinstance(loss, dict) else float(loss) for loss in (validation_losses or [])]
        self.summary = {"alle Prefixe": {"Trainingsfälle": loader.train_df[CASE].nunique(), "Validierungsfälle": loader.val_df[CASE].nunique(),
                                         "Proben": len(train), "Proben Validierung": len(validation), "Epochen": len(losses),
                                         "Proben am Wahlpunkt": weighted, "Gewicht am Wahlpunkt": self.params.get("point_weight", 1),
                                         "Validierungsverlust letzte Epoche": losses[-1] if losses else np.nan,
                                         "bester Validierungsverlust": min(losses) if losses else np.nan,
                                         "window_size": self.encoder_decoder.window_size}}
        if remaining_head:
            self.summary["Restzeit-Kopf (Validierung)"] = self._remaining_check(validation, log)
            print(f"Restzeit-Kopf auf der Validierung: {self.summary['Restzeit-Kopf (Validierung)']}", flush=True)
        return self

    def _remaining_targets(self, dataset, log):
        """
        Je Probe log(1 + Restzeit in Tagen) vom letzten Prefix-Event bis zum Fallende, aus dem ganzen Fall (log), als Tensor
        in Fensterbreite (ein View ohne eigenen Speicher; der Trainer schneidet ihn wie die anderen Zahlen-Tensoren).
        """
        times = log.groupby(log[CASE].astype(str))[TIMESTAMP]
        duration = (pd.to_datetime(times.max()) - pd.to_datetime(times.min())).dt.total_seconds()
        names = [entry[0] for entry in dataset.all_categories[1]]
        last = self.encoder_decoder.window_size - self.encoder_decoder.min_suffix_size - 1
        elapsed = self._decode(self.encoder_decoder.continuous_encoders[TIME_COLUMNS[0]], dataset.continuous_tensors[names.index(TIME_COLUMNS[0])][:, last])
        days = np.clip(duration.reindex([str(case) for case in dataset.case_ids]).to_numpy(dtype=float) - elapsed, 0.0, None) / DAY
        if np.isnan(days).any():
            raise ValueError(f"Restzeit-Kopf: {int(np.isnan(days).sum())} Proben ohne ihren Fall im Log")
        return torch.tensor(np.log1p(days), dtype=torch.float32).unsqueeze(1).expand(-1, self.encoder_decoder.window_size)

    def _gap_samples(self, log):
        """gap_classes: je (Aktivität wie in labels, Klasse) und je Klasse allein bis zu GAP_SAMPLES echte Abstände zum vorigen Event (Sekunden)."""
        rng = np.random.default_rng(self.seed)
        frame = log.sort_values([CASE, TIMESTAMP], kind="stable")
        seconds = pd.to_datetime(frame[TIMESTAMP]).groupby(frame[CASE].astype(str).to_numpy()).diff().dt.total_seconds().to_numpy()
        table = pd.DataFrame({"token": self.labels(frame).to_numpy(), "seconds": seconds}).dropna()
        table["class"] = gap_class_names(table["seconds"].to_numpy())

        def pick(values):
            values = values.to_numpy(dtype=np.float32)
            return values if len(values) <= GAP_SAMPLES else rng.choice(values, GAP_SAMPLES, replace=False)

        samples = {(token, name): pick(group["seconds"]) for (token, name), group in table.groupby(["token", "class"]) if len(group) >= GAP_MIN_SAMPLES}
        samples.update({name: pick(group["seconds"]) for name, group in table.groupby("class")})
        return samples

    def _class_gaps(self, labels, class_ids, class_names):
        """
        gap_classes mit gap_draw "real" (nur Vergleich): je Zeile ein echter Abstand in Sekunden aus gap_samples zur
        gezogenen Aktivität und Klasse (zu wenige Werte: nur die Klasse; EOS oder unbekannt: 0). Zufall über torch wie das
        übrige Ziehen, also dieselben Zufallszahlen für alle Optionen (common random numbers).
        """
        draws = torch.rand(len(labels)).numpy()
        gaps = np.zeros(len(labels))
        for k, (label, class_id) in enumerate(zip(labels, class_ids)):
            name = class_names.get(int(class_id))
            values = self.gap_samples.get((label, name))
            values = self.gap_samples.get(name) if values is None else values
            if values is not None and len(values):
                gaps[k] = float(values[min(int(draws[k] * len(values)), len(values) - 1)])
        return gaps

    def _truncated_gaps(self, means, logvars, class_ids, class_names, scaler):
        """
        gap_classes mit gap_draw "decoder" (Standard): je Zeile ein Abstand in Sekunden aus der vom Decoder vorhergesagten
        Normalverteilung des Abstands (kodiert wie im Training; Mittel und Log-Varianz der attenuierten MSE), beschnitten
        auf die Grenzen der gezogenen Klasse (GAP_EDGES, die oberste bis MAX_GAP_DAYS). Gezogen über die Umkehrfunktion der
        beschnittenen Normalverteilung mit einer Zufallszahl je Zeile aus torch wie _class_gaps (dieselben Zufallszahlen
        für alle Optionen). EOS oder unbekannte Klasse: 0.
        """
        from scipy.stats import truncnorm
        key = TIME_COLUMNS[1]
        mean = means[key + "_mean"].reshape(-1).numpy().astype(float)
        std = torch.exp(0.5 * logvars[key + "_var"]).reshape(-1).numpy().astype(float)
        draws = torch.rand(len(mean)).numpy().astype(float)
        if getattr(self, "_class_bounds", None) is None:  # Grenzen der Klassen in Sekunden und kodiert (Klasse i: i bis i + 1)
            seconds = np.array([0.0, *GAP_EDGES, MAX_GAP_DAYS * DAY])
            if hasattr(scaler, "scale_") and hasattr(scaler, "mean_"):
                encoded = (seconds - scaler.mean_[0]) / scaler.scale_[0]
            else:
                encoded = scaler.transform(seconds.reshape(-1, 1)).reshape(-1).astype(float)
            self._class_bounds = (seconds, encoded)
        seconds, encoded = self._class_bounds
        index = np.array([GAP_NAMES.index(class_names.get(int(c))) if class_names.get(int(c)) in GAP_NAMES else -1 for c in class_ids], dtype=int)
        gaps = np.zeros(len(mean))
        known = index >= 0
        if known.any():
            i, m, s = index[known], mean[known], std[known]
            low, high = encoded[i], encoded[i + 1]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                z = truncnorm.ppf(draws[known], (low - m) / s, (high - m) / s)
                value = np.where(np.isfinite(z), m + s * z, np.clip(m, low, high))
            gaps[known] = np.clip(self._decode(scaler, torch.from_numpy(np.clip(value, low, high))), seconds[i], seconds[i + 1])
        return gaps

    @staticmethod
    def _encode_values(scaler, values):
        """Umkehrung von _decode: Sekunden als kodierte Werte des Loaders (Spalte, Tensor rows x 1)."""
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        if hasattr(scaler, "scale_") and hasattr(scaler, "mean_"):
            return torch.tensor((values - scaler.mean_[0]) / scaler.scale_[0], dtype=torch.float32)
        return torch.tensor(scaler.transform(values), dtype=torch.float32)

    def _remaining_check(self, dataset, log, n=5000):
        """Restzeit-Kopf auf bis zu n Proben (z. B. der Validierung): Fehler in Tagen und Rangkorrelation gegen die echte Restzeit."""
        from scipy.stats import spearmanr
        index = torch.tensor(np.sort(np.random.default_rng(self.seed).choice(len(dataset), min(n, len(dataset)), replace=False)))
        cut = self.encoder_decoder.window_size - self.encoder_decoder.min_suffix_size
        cats = [tensor.index_select(0, index)[:, :cut] for tensor in dataset.categorical_tensors]
        nums = [tensor.index_select(0, index)[:, :cut] for tensor in dataset.continuous_tensors[:len(dataset.all_categories[1])]]
        static = tuple(tensor.index_select(0, index) if tensor.numel() > 0 else None
                       for tensor in (dataset.static_categorical_tensor, dataset.static_continuous_tensor))
        mask = dataset.zero_padding.index_select(0, index)[:, :cut]
        truth = np.expm1(self._remaining_targets(dataset, log)[:, 0].index_select(0, index).numpy().astype(float))
        with torch.no_grad():
            mean, logvar = _head_output(self.net, self.net.encoder(input=[cats, nums], static_inputs=None if static == (None, None) else static, mask=mask))
        median = np.expm1(mean.numpy().astype(float))
        expected = np.expm1(mean.numpy().astype(float) + 0.5 * np.exp(logvar.numpy().astype(float)))
        return {"Proben": int(len(index)), "echte Restzeit Mittel (Tage)": float(truth.mean()),
                "Fehler Median (MAE, Tage)": float(np.mean(np.abs(median - truth))), "Mittel vorhergesagt (Tage)": float(expected.mean()),
                "Rangkorrelation": float(spearmanr(median, truth)[0])}

    def _weight_points(self, train, weight):
        """
        Die Trainingsproben, deren Prefix am Options-Event eines Wahlpunkts endet, weight-mal (die übrigen einmal); liefert
        (Datensatz, Zahl dieser Proben). Eine Probe ist eine Zeile der Breite window_size: links aufgefüllt der Prefix, die
        letzten min_suffix_size Stellen das Suffix; das letzte Prefix-Event steht also an Stelle window_size - min_suffix_size - 1.
        Mitarbeiter: dort das Event des Arbeitsschritts (das start-Event mit dem gewählten Mitarbeiter); Control Flow: dort eine
        der Optionen und davor das Entscheidungs-Event.
        """
        from interventions import option_token
        names = [entry[0] for entry in train.all_categories[0]]
        activity = train.categorical_tensors[names.index(ACTIVITY)]
        ids = train.all_categories[0][names.index(ACTIVITY)][2]
        last = train.encoder_decoder.window_size - train.encoder_decoder.min_suffix_size - 1

        def token(name, lifecycle):
            return ids.get(f"{name}|{lifecycle}" if self.lifecycle_tokens else name, -1)

        hit = torch.zeros(len(train), dtype=torch.bool)
        for point in self.points.values():
            if point.kind == "control_flow":
                options = [token(*option_token(option).split("|")) for option in point.options]
                hit |= torch.isin(activity[:, last], torch.tensor(options)) & (activity[:, last - 1] == token(point.activity, point.lifecycle))
            else:
                hit |= activity[:, last] == token(point.activity, point.lifecycle)
        chosen = torch.nonzero(hit).reshape(-1).tolist()
        if weight <= 1 or not chosen:
            return train, len(chosen)
        print(f"PSP: {len(chosen):,} von {len(train):,} Trainingsproben enden am Options-Event eines Wahlpunkts, je {weight}-mal im Training", flush=True)
        return train.subset(list(range(len(train))) + chosen * (weight - 1)), len(chosen)

    @staticmethod
    def load_checkpoint(folder=TRAIN["checkpoints"]):
        """Das Modell eines laufenden Trainings: model.pkl (ohne Netz) mit dem Netz der letzten fertigen Epoche (last.pt)."""
        from model.dropout_uncertainty_enc_dec_LSTM.dropout_uncertainty_model import DropoutUncertaintyEncoderDecoderLSTM
        model = PSPModel.load(os.path.join(folder, "model.pkl"))
        model.net = PSPModel._load_net(os.path.join(folder, "last.pt"))
        model.net.eval()
        return model

    @staticmethod
    def _load_net(path):
        """Das Netz aus einem Checkpoint des Trainers (last.pt), mit Restzeit-Kopf, wenn er darin steht."""
        from model.dropout_uncertainty_enc_dec_LSTM.dropout_uncertainty_model import DropoutUncertaintyEncoderDecoderLSTM
        checkpoint = torch.load(path, weights_only=False, map_location=torch.device("cpu"))
        if not any(key.startswith("remaining_head.") for key in checkpoint["model_state_dict"]):
            return DropoutUncertaintyEncoderDecoderLSTM.load(path)
        kwargs = {"static_data_set_categories": None, "static_enc_feat": None, **checkpoint["kwargs"]}
        kwargs.pop("static_input_size_enc", None)
        net = DropoutUncertaintyEncoderDecoderLSTM(**kwargs)
        net.remaining_head = _new_remaining_head(net.hidden_size)
        net.load_state_dict(checkpoint["model_state_dict"])
        return net

    # ------------------------------------------------------------------ Ziehen

    def _evaluator(self):
        if getattr(self, "_cached_evaluator", None) is None:
            from evaluation_v2.probabilistic_evaluation import ProbabilisticEvaluation
            self._cached_evaluator = ProbabilisticEvaluation(
                model=self.net, dataset=self.reference, concept_name=ACTIVITY, growing_num_values=[TIME_COLUMNS[0]],
                positive_num_values=[TIME_COLUMNS[1]], decoder_cat=self.decoder()[0], decoder_num=self.decoder()[1], num_processes=1)
        return self._cached_evaluator

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_cached_evaluator", None)
        state.pop("_gap_bounds", None)
        return state

    def _encode(self, events, option_event=None):
        """Der Prefix (Events des Falls bis jetzt, dazu das Options-Event) als Tensoren des Encoders."""
        from event_log_loader.new_event_log_loader_v2 import CSV2EventLog
        rows = list(events) + ([option_event] if option_event is not None else [])
        properties = {key: value for key, value in self.properties().items() if key not in ("window_size", "train_validation_size", "test_validation_size")}
        prefix = CSV2EventLog(self.view(pd.DataFrame(rows)), **properties).df
        prefix = prefix[prefix[ACTIVITY] != "EOS"]
        cats, nums, static_cat, static_num, mask = self.encoder_decoder.encode_case(prefix)
        return [list(cats), list(nums)], (static_cat, static_num), mask, len(prefix), float(prefix[TIME_COLUMNS[0]].iloc[-1])

    def _option_event(self, point, events, option, now):
        """Das Options-Event am Ende des Prefix: das gewählte Event bzw. das start-Event mit dem gewählten Mitarbeiter."""
        from interventions import option_token
        first, last = events[0], events[-1]
        activity, lifecycle = option_token(option).split("|") if point.kind == "control_flow" else (point.activity, point.lifecycle)
        event = {CASE: first[CASE], ACTIVITY: activity, LIFECYCLE: lifecycle,
                 RESOURCE: last.get(RESOURCE) if point.kind == "control_flow" else option,
                 # eine Millisekunde nach dem letzten Event: der Loader sortiert nach Zeit
                 TIMESTAMP: pd.Timestamp(now, unit="s") + pd.Timedelta(milliseconds=1)}
        event.update({name: first.get(name) for name in sum(self.statics(), []) if name.startswith("case:")})
        return event

    def draw(self, prefixes, static, mask, prefix_len, n):
        """n Suffixe je Prefix desselben Falls (Liste; die Prefixe unterscheiden sich nur im Options-Event), siehe draw_cases."""
        return self.draw_cases([(prefix, static, mask, prefix_len) for prefix in prefixes], n)

    def draw_cases(self, encoded, n, until=None, stop=None, max_steps=None):
        """
        n Suffixe je kodiertem Prefix (encoded: Liste von (Prefix, Fallattribute, Maske, Prefixlänge) aus _encode, auch
        verschiedener Fälle), alle in einem Batch. Wie ProbabilisticEvaluation.sample_suffix: Aktivität mit Rauschen auf
        den Logits, Zeiten normalverteilt; Ende bei EOS oder nach window_size - min_suffix_size - Prefixlänge Schritten.
        Anders als dort gehen die Zeiten ungesperrt zurück ins Netz (bei Abständen um 0 mit großer Streuung machte das
        Neuziehen bis positiv jeden Abstand im Mittel etwa einen Tag lang); ausgegeben wird als Zeit seit Fallbeginn das
        Maximum bis zu diesem Event (ab dem Prefix), als Abstand der gezogene, mindestens 0. Früher enden (billiger, für
        den Blick auf andere Fälle):
        until je Prefix eine Zeit seit Fallbeginn in Sekunden, das erste Event danach ist das letzte; stop eine Aktivität,
        mit der das Suffix endet; max_steps höchstens so viele Events. Je Prefix eine Liste von n Suffixen, je Suffix eine
        Liste von (Aktivität wie in labels, Sekunden seit Fallbeginn, Sekunden seit dem vorigen Event).
        """
        ev, model = self._evaluator(), self.net
        names = ev.inverted_suffix_categories[0]
        scalers = {name: self.encoder_decoder.continuous_encoders[name] for name in ev.all_num_attributes}
        window, suffix_size = self.encoder_decoder.window_size, self.encoder_decoder.min_suffix_size
        steps = np.repeat([max(window - suffix_size - length + 1, 1) for _, _, _, length in encoded], n)
        steps = np.minimum(steps, max_steps) if max_steps is not None else steps
        limit = np.repeat(np.full(len(encoded), np.inf) if until is None else np.asarray(until, dtype=float), n)
        rows = n * len(encoded)
        classes = self.gap_classes()
        if classes:  # Nummer der Klasse im Encoder -> Name (GAP_NAMES)
            class_names = {v: k for k, v in next(c for c in self.reference.all_categories[0] if c[0] == GAP_CLASS)[2].items()}
        cats, nums, batch_static, mask = self._batch(encoded, n)
        with torch.no_grad():
            prediction, hidden, _ = model.inference(prefix=[cats, nums], static_inputs=batch_static, mask=mask)
            reached = self._decode(scalers[TIME_COLUMNS[0]], nums[ev.prefix_num_attributes.index(TIME_COLUMNS[0])][:, -1])
            suffixes, done = [[] for _ in range(rows)], np.zeros(rows, dtype=bool)
            for step in range(int(steps.max())):
                cat_values = type(ev).sample_cat_predictions_optim(ev.all_cat_attributes, ev.use_variance_cat, ev.sample_argmax, prediction[0][0], prediction[1][0])
                num_values = self._clip_gaps(self._draw_times(ev, prediction[0][1], prediction[1][1]))
                activity = cat_values[ACTIVITY + "_mean"].reshape(-1).numpy()
                done |= (activity == ev.eos_id) | (step >= steps)
                if done.all():
                    break
                labels = [str(names.get(int(a), "<unbekannt>")) for a in activity]
                if classes:  # Abstand in der gezogenen Klasse (gap_draw); beide Zeiten passend dazu zurück ins Netz
                    class_ids = cat_values[GAP_CLASS + "_mean"].reshape(-1).numpy()
                    if self.gap_draw() == "real":
                        gap = self._class_gaps(labels, class_ids, class_names)
                    else:
                        gap = self._truncated_gaps(prediction[0][1], prediction[1][1], class_ids, class_names, scalers[TIME_COLUMNS[1]])
                    reached = reached + gap
                    num_values = {TIME_COLUMNS[0] + "_mean": self._encode_values(scalers[TIME_COLUMNS[0]], reached),
                                  TIME_COLUMNS[1] + "_mean": self._encode_values(scalers[TIME_COLUMNS[1]], gap)}
                else:
                    reached = np.maximum(reached, self._decode(scalers[TIME_COLUMNS[0]], num_values[TIME_COLUMNS[0] + "_mean"]))
                    gap = self._decode(scalers[TIME_COLUMNS[1]], num_values[TIME_COLUMNS[1] + "_mean"])
                for k in np.flatnonzero(~done):
                    suffixes[k].append((labels[k], float(reached[k]), max(float(gap[k]), 0.0)))
                done |= reached > limit
                if stop is not None:
                    done |= np.array([label.split("|")[0] for label in labels], dtype=object) == stop
                prediction, hidden = model.inference(last_event=([cat_values[key].reshape(rows, 1) for key in cat_values],
                                                                 [num_values[key].reshape(rows, 1) for key in num_values]), hx=hidden)
        return [suffixes[k * n:(k + 1) * n] for k in range(len(encoded))]

    @staticmethod
    def _batch(encoded, n):
        """Die kodierten Prefixe (aus _encode, wie in draw_cases) als ein Batch, jeder n-mal hintereinander: (kategorial, Zahlen, Fallattribute, Maske)."""
        first = encoded[0]
        cats = [torch.cat([e[0][0][i] for e in encoded]).repeat_interleave(n, dim=0) for i in range(len(first[0][0]))]
        nums = [torch.cat([e[0][1][i] for e in encoded]).repeat_interleave(n, dim=0) for i in range(len(first[0][1]))]
        static = tuple(torch.cat([e[1][j] for e in encoded]).repeat_interleave(n, dim=0) if torch.is_tensor(first[1][j]) else first[1][j]
                       for j in range(len(first[1])))
        mask = torch.cat([e[2] for e in encoded]).repeat_interleave(n, dim=0)
        return cats, nums, static, mask

    def head_remaining(self, encoded, n, expected=False):
        """
        Restzeit-Kopf (remaining_head): je kodiertem Prefix (wie draw_cases) n Restzeiten in Tagen (Array Prefixe x n). Je
        Ziehung eigene Dropout-Masken des Encoders (MC-Dropout wie beim Ziehen der Suffixe) und ein Wert aus der
        vorhergesagten Normalverteilung von log(1 + Tage); mit expected statt des gezogenen Werts der Erwartungswert dieser
        Verteilung (exp(Mittel + Varianz / 2) - 1, viel weniger Rauschen: der Schwanz der Log-Normalverteilung streut
        stark). Höchstens REMAINING_HEAD max_days.
        """
        cats, nums, static, mask = self._batch(encoded, n)
        with torch.no_grad():
            mean, logvar = _head_output(self.net, self.net.encoder(input=[cats, nums], static_inputs=static, mask=mask))
            log_days = mean + 0.5 * torch.exp(logvar) if expected else torch.normal(mean, torch.exp(0.5 * logvar))
            days = torch.expm1(log_days.clamp(max=float(np.log1p(REMAINING_HEAD["max_days"])))).clamp(min=0.0)
        return days.numpy().astype(float).reshape(len(encoded), n)

    @staticmethod
    def _draw_times(ev, means, variances):
        # Normalverteilt (kodiert), ohne die Sperren von sample_suffix (siehe draw_cases)
        return {name + "_mean": torch.normal(means[name + "_mean"].reshape(-1, 1), torch.exp(0.5 * variances[name + "_var"].reshape(-1, 1)))
                for name in ev.all_num_attributes}

    def _clip_gaps(self, num_values):
        """
        Log-normale Abstände (log_gaps): die gezogenen Werte (kodiert) auf 0 bis MAX_GAP_DAYS Tage begrenzen, bevor sie
        ausgegeben und zurück ins Netz gegeben werden. Der lange Schwanz der Log-Normalverteilung gäbe sonst einzelne
        Abstände von Jahren (die Summe über ein Suffix wäre dann unbrauchbar). Ohne log_gaps unverändert.
        """
        encoder = self.encoder_decoder.continuous_encoders.get(TIME_COLUMNS[1])
        key = TIME_COLUMNS[1] + "_mean"
        if key not in num_values or not hasattr(encoder, "std_"):
            return num_values
        if getattr(self, "_gap_bounds", None) is None:
            self._gap_bounds = tuple(float(encoder.transform(np.array([[x]], dtype=float)).reshape(-1)[0]) for x in (0.0, MAX_GAP_DAYS * DAY))
        low, high = self._gap_bounds
        return {**num_values, key: torch.clamp(num_values[key], min=low, max=high)}

    @staticmethod
    def _decode(scaler, values):
        values = values.reshape(-1).numpy().astype(float)
        if hasattr(scaler, "scale_") and hasattr(scaler, "mean_"):
            return values * scaler.scale_[0] + scaler.mean_[0]
        return scaler.inverse_transform(values.reshape(-1, 1)).reshape(-1)

    @staticmethod
    def outcome(prefix_elapsed, suffix, kind, costs=None, gaps=False):
        """
        Outcome eines Falls aus dem Prefix (Sekunden seit Fallbeginn am letzten Prefix-Event) und einem gezogenen Suffix.
        costs: bei negative_repair_cost die Kosten des Falls mit dieser Option (aus _option_costs). gaps: die Durchlaufzeit
        aus der Summe der gezogenen Abstände statt aus der gezogenen Zeit seit Fallbeginn (TRAIN time_from "gaps"; die
        staucht die Dauern: Diagnose 2026-11-08, scripts/psp_diagnose.py)
        """
        if kind == "negative_cycle_days":
            if gaps:
                return -(prefix_elapsed + sum(gap for _, _, gap in suffix)) / DAY
            return -max([prefix_elapsed] + [elapsed for _, elapsed, _ in suffix]) / DAY
        if kind == "negative_repair_cost":
            return costs(prefix_elapsed, suffix)
        raise NotImplementedError(f"PSP: Outcome {kind} aus gezogenen Suffixen ist noch nicht umgesetzt (möglich: {', '.join(OUTCOMES)})")

    def _outcome_kind(self):
        from preprocessing import chosen_outcome_spec
        return chosen_outcome_spec(self.dataset)["type"]

    def _option_costs(self, point, events, options, now):
        """Je Option die Bewertung der gezogenen Suffixe, wenn das Outcome mehr braucht als die Zeiten (Kosten der Werkstatt), sonst None."""
        from preprocessing import chosen_outcome_spec
        spec = chosen_outcome_spec(self.dataset)
        if spec["type"] != "negative_repair_cost":
            return [None] * len(options)
        from prescriptive.costs.repair.outcome import SuffixCosts
        employee = point.kind != "control_flow"
        return [SuffixCosts(spec, events, self._option_event(point, events, option, now), option if employee else None, point.activity)
                for option in options]

    # ------------------------------------------------------------------ Abfrage

    def knows(self, point):
        """
        Hat das Modell den Wahlpunkt: gleicher Name, oder ein Control-Flow-Punkt am selben Entscheidungs-Event, dessen
        Optionen das Modell dort alle kennt (z. B. channel allein bei einem Modell mit channel+angebot_nachlegen; das
        Options-Event hängt nur von der Option ab).
        """
        if self.net is None:
            return False
        if point.name in self.points:
            return True
        return point.kind == "control_flow" and any(
            known.kind == "control_flow" and (known.activity, known.lifecycle) == (point.activity, point.lifecycle) and set(point.options) <= set(known.options)
            for known in self.points.values())

    def predict_details(self, point, samples, options):
        """
        Je Probe (aus samples: Events des Falls, Ende, Zeitpunkt) mit ihrer Option: geschätzter Outcome (Mittel über die
        gezogenen Suffixe) und die gezogenen Suffixe mit der Zeit seit Fallbeginn am Options-Event.
        """
        kind = self._outcome_kind()
        values, details = [], []
        for (part, cut, now), option in zip(samples, options):
            events = part.iloc[:cut].to_dict("records")
            prefix, static, mask, length, elapsed = self._encode(events, self._option_event(point, events, option, now))
            suffixes = self.draw([prefix], static, mask, length, self.samples_per_option)[0]
            costs = self._option_costs(point, events, [option], now)[0]
            gaps = (getattr(self, "params", None) or {}).get("time_from") == "gaps"
            values.append(float(np.mean([self.outcome(elapsed, suffix, kind, costs, gaps) for suffix in suffixes])))
            details.append({"suffixes": suffixes, "elapsed": elapsed})
        return np.array(values), details

    def predict(self, point, samples, options):
        return self.predict_details(point, samples, options)[0]

    def suffix_kpis(self, samples, details):
        """
        Wie gut die gezogenen Suffixe die tatsächlichen treffen, mit den Kennzahlen des PSP-Papers (Tab. II und III): das
        tatsächliche Suffix sind die Events nach Zeile Ende der Probe (aus samples: Events des Falls, Ende, Zeitpunkt),
        details aus predict_details. Suffixlänge (Mittel der gezogenen Längen), DLS auf den labels (gemittelt über die
        gezogenen Suffixe), Restzeit (Mittel der gezogenen, einmal aus der letzten Zeit seit Fallbeginn, einmal aus der
        Summe der Abstände), PIT und Abdeckung der 50/75/95-%-Intervalle für Restzeit und Suffixlänge.
        """
        from prescriptive.estimators.kpis import dls
        rows = []
        for (part, cut, _), detail in zip(samples, details):
            true = self.labels(part).iloc[cut + 1:].tolist()
            times = pd.to_datetime(part[TIMESTAMP])
            true_rest = (times.iloc[-1] - times.iloc[cut]).total_seconds() / DAY
            elapsed, suffixes = detail["elapsed"], detail["suffixes"]
            rest = np.array([(max([elapsed] + [e for _, e, _ in suffix]) - elapsed) / DAY for suffix in suffixes])
            rest_sum = np.array([sum(g for _, _, g in suffix) / DAY for suffix in suffixes])
            lengths = np.array([len(suffix) for suffix in suffixes])
            row = {"true_len": len(true), "len": lengths.mean(), "dls": np.mean([dls([a for a, _, _ in suffix], true) for suffix in suffixes]),
                   "true_rest": true_rest, "rest": rest.mean(), "rest_sum": rest_sum.mean(), "pit": float(np.mean(rest <= true_rest))}
            for level in COVERAGES:
                bounds = [50 - level / 2, 50 + level / 2]
                low, high = np.percentile(rest, bounds)
                row[f"rest_{level}"] = float(low <= true_rest <= high)
                low, high = np.percentile(lengths, bounds)
                row[f"len_{level}"] = float(low <= len(true) <= high)
            rows.append(row)
        rows = pd.DataFrame(rows)
        if rows.empty:
            return {"Proben": 0}
        return {"Proben": len(rows), "Suffixlänge tatsächlich (Mittel, Events)": rows["true_len"].mean(),
                "Suffixlänge gezogen (Mittel, Events)": rows["len"].mean(), "Suffixlänge MAE (Events)": (rows["len"] - rows["true_len"]).abs().mean(),
                "DLS (Aktivitäten, Mittel über gezogene Suffixe)": rows["dls"].mean(),
                "Restzeit tatsächlich (Mittel, Tage)": rows["true_rest"].mean(), "Restzeit gezogen (Mittel, Tage)": rows["rest"].mean(),
                "Restzeit MAE (Tage, letzte Zeit)": (rows["rest"] - rows["true_rest"]).abs().mean(),
                "Restzeit MAE (Tage, Summe der Abstände)": (rows["rest_sum"] - rows["true_rest"]).abs().mean(),
                "PIT Restzeit (Mittel, ideal 0,5)": rows["pit"].mean(),
                **{f"Abdeckung Restzeit {level} %": rows[f"rest_{level}"].mean() for level in COVERAGES},
                **{f"Abdeckung Suffixlänge {level} %": rows[f"len_{level}"].mean() for level in COVERAGES}}

    def prefix_kpis(self, log, n=EVAL_PREFIXES):
        """
        suffix_kpis auf n zufälligen Prefixen jeder Länge (1 bis Fallende - 1) der Fälle des Logs, ohne Options-Event: wie
        die Auswertung des PSP-Papers (dort alle Prefixe der Testfälle).
        """
        log = log.sort_values([CASE, TIMESTAMP], kind="stable")
        parts = [part.reset_index(drop=True) for _, part in log.groupby(CASE, sort=True) if len(part) > 1]
        pairs = [(c, k) for c, part in enumerate(parts) for k in range(1, len(part))]
        chosen = sorted(np.random.default_rng(0).choice(len(pairs), min(n, len(pairs)), replace=False))
        samples, details = [], []
        for c, k in (pairs[i] for i in chosen):
            prefix, static, mask, length, elapsed = self._encode(parts[c].iloc[:k].to_dict("records"))
            details.append({"suffixes": self.draw([prefix], static, mask, length, self.samples_per_option)[0], "elapsed": elapsed})
            samples.append((parts[c], k - 1, None))
        return self.suffix_kpis(samples, details)

    def _option_ids(self, point, options):
        """Je Eingang (Spalte, Nummern der Optionen im Encoder): beim Control Flow Aktivität und Lifecycle, beim Mitarbeiter der Mitarbeiter."""
        from interventions import option_token

        def ids(column, values):
            encoder = self.encoder_decoder.categorical_encoders[column]
            return DYNAMIC_CATEGORICAL.index(column), (encoder.transform(np.array([[value] for value in values], dtype=object)).reshape(-1) + 1).astype(int)

        if point.kind != "control_flow":
            return [ids(RESOURCE, options)]
        tokens = [option_token(option) for option in options]
        return [ids(ACTIVITY, tokens if self.lifecycle_tokens else [token.split("|")[0] for token in tokens]),
                ids(LIFECYCLE, [token.split("|")[1] for token in tokens])]

    def option_suffixes(self, point, events, options, now, n):
        """
        Je Option n gezogene Suffixe des Falls mit der Option als nächstem Event (Liste in der Reihenfolge von options)
        und die Zeit seit Fallbeginn am Options-Event in Sekunden. events: die Events des Falls bis jetzt (Liste von dicts
        wie engine.events, in Reihenfolge); now: Zeitpunkt der Wahl in Sekunden seit 1970.
        Der Prefix wird einmal kodiert; die Optionen unterscheiden sich nur im Options-Event (Aktivität und Lifecycle bzw. Mitarbeiter),
        dort wird je Option nur dieser Eintrag getauscht. Alle Optionen ziehen mit denselben Zufallszahlen (common random
        numbers): Suffix k jeder Option bekommt dieselben Dropout-Masken und dasselbe Rauschen, der Unterschied zwischen
        den Optionen hängt dann an der Option, nicht am Zufall der Ziehung.
        """
        drawn, elapsed = self.option_suffixes_many(point, [events], options, now, n)
        return [suffixes[0] for suffixes in drawn], elapsed[0]

    def option_suffixes_many(self, point, cases, options, now, n):
        """
        option_suffixes für mehrere Fälle (cases: je Fall seine Events bis jetzt) in einem Batch je Option: je Option je
        Fall n Suffixe, und je Fall die Zeit seit Fallbeginn am Options-Event.
        """
        return self._per_option(point, cases, options, now, n, self.draw_cases)

    def option_remaining_many(self, point, cases, options, now, n, expected=False):
        """
        Wie option_suffixes_many, aber über den Restzeit-Kopf (remaining_head): je Option ein Array Fälle x n Restzeiten in
        Tagen ab dem Options-Event (head_remaining, mit expected die Erwartungswerte), und je Fall die Zeit seit Fallbeginn
        am Options-Event in Sekunden.
        """
        return self._per_option(point, cases, options, now, n, lambda batch, k: self.head_remaining(batch, k, expected))

    def _per_option(self, point, cases, options, now, n, function):
        """
        Je Option function(Batch, n) auf den Prefixen der Fälle mit dieser Option als Options-Event (siehe option_suffixes:
        einmal kodiert, je Option nur der Eintrag am Options-Event getauscht, dieselben Zufallszahlen für alle Optionen);
        dazu je Fall die Zeit seit Fallbeginn am Options-Event.
        """
        encoded = [self._encode(events, self._option_event(point, events, options[0], now)) for events in cases]
        columns = self._option_ids(point, options)
        seed = int(torch.randint(2 ** 31 - 1, (1,)))
        results = []
        for k in range(len(options)):
            batch = []
            for prefix, static, mask, length, _ in encoded:
                cats = [tensor.clone() for tensor in prefix[0]]
                for column, ids in columns:
                    cats[column][:, -1] = int(ids[k])
                batch.append(([cats, prefix[1]], static, mask, length))
            torch.manual_seed(seed)
            results.append(function(batch, n))
        torch.manual_seed(seed + 1)  # danach unabhängig weiter
        return results, [entry[4] for entry in encoded]

    def outcome_samples(self, point, events, options, now, n=None):
        """
        Outcome je Option und gezogenem Suffix (Array Optionen x n, n Standard samples_per_option), aus option_suffixes; mit
        value_from "head" (value_source) je Option n Ziehungen des Restzeit-Kopfs (option_remaining_many).
        """
        kind = self._outcome_kind()
        if self.value_source() == "head":
            if kind != "negative_cycle_days":
                raise ValueError(f"Restzeit-Kopf: nur für die Durchlaufzeit (negative_cycle_days), nicht {kind}")
            remaining, elapsed = self.option_remaining_many(point, [events], options, now, n or self.samples_per_option)
            return np.array([-(elapsed[0] / DAY + days[0]) for days in remaining])
        drawn, elapsed = self.option_suffixes(point, events, options, now, n or self.samples_per_option)
        costs = self._option_costs(point, events, options, now)
        gaps = (getattr(self, "params", None) or {}).get("time_from") == "gaps"
        return np.array([[self.outcome(elapsed, suffix, kind, cost, gaps) for suffix in suffixes] for suffixes, cost in zip(drawn, costs)])

    def value_source(self):
        """Woraus values und outcome_samples bewerten: "suffix" (gezogene Suffixe, Original) oder "head" (Restzeit-Kopf); Attribut value_from vor TRAIN value_from."""
        source = getattr(self, "value_from", None) or (getattr(self, "params", None) or {}).get("value_from") or "suffix"
        if source not in ("suffix", "head"):
            raise ValueError(f"value_from: suffix oder head, nicht {source}")
        if source == "head" and getattr(self.net, "remaining_head", None) is None:
            raise ValueError("value_from head: das Netz hat keinen Restzeit-Kopf (TRAIN remaining_head)")
        return source

    def values(self, point, events, options, now):
        """
        Geschätzter Outcome je Option (Array in der Reihenfolge von options): Mittel über die Suffixe aus option_suffixes;
        mit value_from "head" Mittel der Erwartungswerte des Restzeit-Kopfs über samples_per_option Dropout-Masken.
        """
        if self.value_source() == "head":
            if self._outcome_kind() != "negative_cycle_days":
                raise ValueError(f"Restzeit-Kopf: nur für die Durchlaufzeit (negative_cycle_days), nicht {self._outcome_kind()}")
            remaining, elapsed = self.option_remaining_many(point, [events], options, now, self.samples_per_option, expected=True)
            return np.array([-(elapsed[0] / DAY + days[0].mean()) for days in remaining])
        return self.outcome_samples(point, events, options, now).mean(axis=1)

    def set_threads(self, n):
        torch.set_num_threads(n)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
