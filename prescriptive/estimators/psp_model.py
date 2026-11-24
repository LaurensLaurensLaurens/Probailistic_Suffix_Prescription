"""
PSP-Modell (probabilistic_suffix_prediction_U-ED-LSTM, mit optionaler Erweiterung für Variante (e): EventLogLoader,
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
# gap_condition_on_event (Variante (e)): wie (d), aber der Kopf für GAP_CLASS bekommt zusätzlich das aktuelle
# Aktivität|Lifecycle-Token. Training und Validierungsverlust: das Ziel-Event desselben Schritts; autoregressives
# Feedback: das vorhergesagte Event; Ziehen: das tatsächlich gezogene bzw. erzwungene Options-Event. Die stetigen
# Zeitköpfe bleiben wie bei (d); der Abstand wird aus ihrer Normalverteilung innerhalb der gezogenen Klasse gezogen.


def gap_class_names(seconds):
    """Klasse (GAP_NAMES) je Abstand zum vorigen Event in Sekunden; fehlend (erstes Event): 0s."""
    seconds = np.nan_to_num(np.asarray(seconds, dtype=float), nan=0.0)
    return np.asarray(GAP_NAMES, dtype=object)[np.searchsorted(GAP_EDGES, seconds, side="right")]
# Stufe 3 (2026-11-11), TRAIN-Einstellungen, Standard jeweils aus bzw. wie bisher:
# - beta_nll (2026-11-18): β-NLL für die Zeitköpfe (Loss(beta_nll), Seitzer et al. 2022): jeder Term des attenuierten Verlusts
#   mal stop_grad(σ^(2β)); lange, verrauschte Abstände fallen dann nicht mehr aus dem Mittelwert-Fit. 0.0 = wie bisher.
# - time_event_head (2026-11-18, mit time_on_event): "offset" = eigener Versatz je Event für jede Zeitausgabe (Mittel und
#   log-Varianz getrennt), "slope" = zusätzlich eine eigene Steigung je Event auf dem Decoder-Zustand (Wechselwirkung);
#   dropout_uncertainty_decoder.add_event_time_parameters, mit null begonnen. Anlass: das gemeinsame Event-Embedding der
#   Zeitköpfe verschob fast nur die log-Varianz (Versatz des Mittels vor A_Cancelled ~0, Kopf 3,6 statt ~20 Tage).
#   head_only: nur die Zeitköpfe lernen (Warmstart init_from nötig, Rest eingefroren, _head_only_trainer ohne GradNorm);
#   offset_learning_rate: Lernrate der Versätze je Event (Standard wie learning_rate).
# - teacher_forcing: unterste Teacher-Forcing-Quote (Trainer min_teacher_forcing_value). Standard 0.0 wie im Original: die
#   Quote fällt bis zur Hälfte der Epochen von 1 auf 0, danach bekommt der Decoder nur noch seine eigenen Vorhersagen
#   (argmax) zurück, die Ziele bleiben der echte Pfad. Im Original über 5 Schritte; über 60 Schritte ist der Decoder dann
#   meist neben dem echten Pfad und lernt Events und EOS nach der Position statt nach seinem Zustand (gemessen: 40 bis 48 %
#   der gezogenen Suffixe enden mit EOS ohne A_Pending/A_Denied/A_Cancelled, echt 0 bis 1 %). Mit 1.0 sieht der Decoder
#   immer die echten vorigen Events (Maximum-Likelihood je Schritt); gezogen wird wie bisher Schritt für Schritt.
# - gap_tokens: das Token des Decoders ist Aktivität|Lifecycle|Abstandsklasse (GAP_NAMES) statt zweier getrennt gezogener
#   Ausgaben wie bei gap_classes: Event und Klasse seines Abstands werden zusammen gezogen (ein langer Abstand hängt daran,
#   welches Event folgt, z. B. A_Cancelled nach 30 Tagen). Der Abstand kommt wie bei gap_classes aus der Normalverteilung des
#   Decoders, beschnitten auf die Klasse. Die Klasse bleibt zusätzlich ein Eingang des Encoders (GAP_CLASS). Ausgegeben
#   werden die Token ohne Klasse (wie labels).
# - option_as "decoder": an Control-Flow-Punkten ist die Option nicht das letzte Event des Prefix (mit künstlichem Abstand
#   1 ms, bei Klassen also immer "0s", das es bei den meisten Optionen im Log nie gibt), sondern das erste Event des
#   Suffix: der erste Schritt des Decoders zieht nur unter den Token der Option (bei gap_tokens also ihre Abstandsklasse),
#   ihr Abstand wird gezogen und zählt zur Durchlaufzeit. Im Training zählen dann auch die Proben point_weight-fach, deren
#   Prefix am Entscheidungs-Event endet.
# - option_as "encoder" (2026-11-11, nach den Pilotläufen): die Option wirkt nur, wenn der Encoder sie sieht (dasselbe Netz
#   mit der Option erst im Decoder: Vorzeichen wie AIPW 54/65 statt 86/90 %). Deshalb in zwei Schritten: der Decoder zieht
#   aus dem Prefix bis zum Entscheidungs-Event nur den ersten Schritt unter den Token der Option (ihr Abstand, bei
#   gap_classes/gap_tokens mit Klasse); dann steht die Option mit diesem gezogenen Abstand als letztes Event im Prefix
#   (Token bzw. Klasse und Zeiten passend), und der Rest des Suffix wird von dort gezogen. Kein künstlicher Abstand von
#   1 ms, kein Wert aus dem Log; der Abstand der Option zählt zur Durchlaufzeit. Geht auch nachträglich (params setzen).
# - decoder_context (2026-11-11): der Decoder bekommt in jedem Schritt zusätzlich den Zustand des Encoders nach dem Prefix
#   (h der obersten Schicht, linear abgebildet und auf die Projektion des Eingangs-Events addiert, _ContextProjection).
#   Grund: mit Teacher Forcing 1.0 stimmen die Fallenden, aber die Option wirkt nicht mehr (Pilot: Vorzeichen wie AIPW um
#   50 %); der Decoder müsste sich die Option über 20 bis 30 Schritte merken. Ohne Teacher Forcing sagt er nach der Position
#   aus dem Encoder-Zustand vorher (Signal da, Fallenden falsch). Mit dem Kontext in jedem Schritt muss er sich nichts merken.
# - static_in_decoder (2026-11-24): die statischen Merkmale des Falls (wie im Encoder kodiert: Einbettungen der kategorialen
#   und die Zahlen, static_input_size Werte) gehen in jeden Decoder-Schritt, linear abgebildet und auf die Projektion des
#   Eingangs-Events addiert (_StaticProjection; Forward-Hook _keep_static am Encoder hält sie je Batch fest). Gewichte mit 0
#   begonnen: ein vorhandenes Netz rechnet anfangs genau wie zuvor. Grund (Sonde 23.11., diagnose/sonde_betrag.md): der
#   Betrag ist im Encoder-Zustand ablesbar, verblasst im Decoder aber nach 6–15 Schritten, bevor die Events gezogen werden,
#   an denen der Effekt hängt. Verlust, Dropout-Masken und das Ziehen bleiben unverändert.
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
# keyed_noise (2026-11-19, params, nur beim Ziehen, Standard aus): Zufallszahlen je Ereignis statt je Schrittnummer, siehe
# PSPModel._keyed_draw. Hash splitmix64 auf (Saat, Ziehung, Schlüssel ...), damit beide Optionen dieselbe Zahl erhalten.
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _mix64(x):
    """splitmix64-Finalizer auf uint64-Werten (Überlauf gewollt)."""
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def _keyed_uniform(*keys):
    """Gleichverteilte Zahlen in (0, 1) als feste Funktion ganzzahliger Schlüssel (Arrays, broadcastbar): gleiche Schlüssel, gleiche Zahl."""
    h = np.uint64(0)
    with np.errstate(over="ignore"):
        for key in keys:
            h = _mix64(h ^ (np.asarray(key).astype(np.uint64) + _GOLDEN))
    return ((np.asarray(h) >> np.uint64(11)).astype(np.float64) + 0.5) / float(2 ** 53)


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


class _ContextProjection(torch.nn.Module):
    """
    decoder_context: input_proj des Decoders plus eine lineare Abbildung des Encoder-Zustands nach dem Prefix (kept_context
    am Encoder aus dem Forward-Hook _keep_context; je Batch neu). Die Abbildung beginnt bei 0: am Anfang rechnet das Netz
    wie das Original. weight und bias wie nn.Linear (der Regularisierer des Decoders liest sie).
    """

    def __init__(self, projection, encoder, hidden_size):
        super().__init__()
        self.projection = projection
        self.context_projection = torch.nn.Linear(hidden_size, projection.out_features)
        torch.nn.init.zeros_(self.context_projection.weight)
        torch.nn.init.zeros_(self.context_projection.bias)
        self.__dict__["encoder"] = encoder  # ohne Eintrag als Teilmodul (sonst stünde der Encoder doppelt im Netz)

    @property
    def weight(self):
        return self.projection.weight

    @property
    def bias(self):
        return self.projection.bias

    def forward(self, event):  # event: Schritte (1) x Batch x Merkmale
        out = self.projection(event)
        context = self.__dict__["encoder"].__dict__.get("kept_context")
        if context is not None and context.shape[0] == out.shape[1]:
            out = out + self.context_projection(context).unsqueeze(0)
        return out


def _keep_context(module, inputs, output):
    """Forward-Hook am Encoder (decoder_context): h der obersten Schicht nach dem Prefix für _ContextProjection."""
    module.__dict__["kept_context"] = output[0]


def _attach_context(net):
    """decoder_context am Netz einrichten (vor dem Optimierer: die Abbildung lernt mit)."""
    net.decoder.input_proj = _ContextProjection(net.decoder.input_proj, net.encoder, net.hidden_size)
    net.encoder.register_forward_hook(_keep_context)
    return net


class _StaticProjection(torch.nn.Module):
    """
    static_in_decoder: input_proj des Decoders (ggf. schon mit _ContextProjection) plus eine lineare Abbildung der statischen
    Merkmale des Falls (kept_static am Encoder aus dem Forward-Hook _keep_static; je Batch neu). Die Abbildung beginnt bei 0: am
    Anfang rechnet das Netz wie das Original. weight und bias wie nn.Linear (der Regularisierer des Decoders liest sie).
    Fehlen die festgehaltenen Merkmale oder passt ihre Zeilenzahl nicht zum Batch, bricht es ab (kein stilles Weglassen).
    """

    def __init__(self, projection, encoder, hidden_size):
        super().__init__()
        self.projection = projection
        self.static_projection = torch.nn.Linear(int(encoder.static_input_size), hidden_size, bias=False)
        torch.nn.init.zeros_(self.static_projection.weight)
        self.__dict__["encoder"] = encoder  # ohne Eintrag als Teilmodul (sonst stünde der Encoder doppelt im Netz)

    @property
    def weight(self):
        return self.projection.weight

    @property
    def bias(self):
        return self.projection.bias

    def forward(self, event):  # event: Schritte (1) x Batch x Merkmale
        out = self.projection(event)
        static = self.__dict__["encoder"].__dict__.get("kept_static")
        if static is None or static.shape[0] != out.shape[1]:
            raise RuntimeError(f"static_in_decoder: statische Merkmale fehlen oder passen nicht zum Batch "
                               f"({None if static is None else tuple(static.shape)} gegen {tuple(out.shape)})")
        return out + self.static_projection(static.to(out.dtype)).unsqueeze(0)


def _keep_static(module, args, kwargs, output):
    """Forward-Hook am Encoder (static_in_decoder): die kodierten statischen Merkmale des Batches (Zeilen x static_input_size)."""
    static_inputs = kwargs.get("static_inputs", args[1] if len(args) > 1 else None)
    module.__dict__["kept_static"] = module._DropoutUncertaintyLSTMEncoder__static_data_enc_model(
        static_inputs, device=output[0].device, dtype=output[0].dtype)


def _attach_static(net):
    """static_in_decoder am Netz einrichten (nach dem Warmstart, vor dem Optimierer: die Abbildung lernt mit)."""
    if not int(getattr(net.encoder, "static_input_size", 0) or 0):
        raise ValueError("static_in_decoder: das Netz hat keine statischen Merkmale")
    net.decoder.input_proj = _StaticProjection(net.decoder.input_proj, net.encoder, net.hidden_size)
    net.encoder.register_forward_hook(_keep_static, with_kwargs=True)
    return net


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


def _time_head_groups(net, learning_rate, offset_learning_rate=None):
    """
    head_only (2026-11-18): alle Gewichte einfrieren außer den Zeitköpfen des Decoders (die linearen Köpfe für Mittel und
    log-Varianz jeder Zeit, das Event-Embedding der Zeitköpfe mit seinen Gewichten, die eigenen Parameter je Event aus
    time_event_head). Liefert die Parametergruppen für AdamW: die Versätze je Event mit offset_learning_rate (sie müssen weit
    wandern, z. B. der Abstand vor A_Cancelled um rund 10 Standardabweichungen des Abstands), der Rest mit learning_rate.
    """
    decoder = net.decoder
    names = [f"{key}_{suffix}" for key in decoder.output_sizes[1] for suffix in ("mean", "var")]
    head = [p for name in names for p in decoder.output_layers[name].parameters()]
    head += list(decoder.num_condition_embedding.parameters()) + list(decoder.num_condition_weights.parameters())
    if getattr(decoder, "num_event_slopes", None) is not None:
        head += list(decoder.num_event_slopes.parameters())
    offsets = list(decoder.num_event_offsets.parameters()) if getattr(decoder, "num_event_offsets", None) is not None else []
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for parameter in head + offsets:
        parameter.requires_grad_(True)
    groups = [{"params": head, "lr": learning_rate}]
    if offsets:
        groups.append({"params": offsets, "lr": offset_learning_rate or learning_rate})
    print(f"PSP head_only: {sum(p.numel() for p in head + offsets):,} von {sum(p.numel() for p in net.parameters()):,} Gewichten lernen "
          f"(Lernrate {learning_rate}, Versätze je Event {offset_learning_rate or learning_rate})", flush=True)
    return groups


def _static_only_groups(net, params):
    """
    static_only (2026-11-24, Lernbarkeitstest nach dem nicht bestandenen Tor von static_in_decoder, Nutzer: „ja“): alle Gewichte
    einfrieren außer der Abbildung der Statik im Decoder (_StaticProjection.static_projection); mit static_only_output dazu der
    Event-Kopf des Decoders (output_layers der kategorialen Ausgaben, Mittel und log-Varianz). Lernrate der Abbildung
    static_learning_rate, des Event-Kopfs learning_rate.
    """
    if not params.get("static_in_decoder", False) or not params.get("init_from") or params.get("head_only", False):
        raise ValueError("static_only: nur mit static_in_decoder und init_from, ohne head_only")
    decoder = net.decoder
    statik = list(decoder.input_proj.static_projection.parameters())
    kopf = []
    if params.get("static_only_output", False):
        names = [f"{key}_{suffix}" for key in decoder.output_sizes[0] for suffix in ("mean", "var")]
        kopf = [p for name in names for p in decoder.output_layers[name].parameters()]
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for parameter in statik + kopf:
        parameter.requires_grad_(True)
    lr = float(params.get("static_learning_rate") or params["learning_rate"])
    print(f"PSP static_only: {sum(p.numel() for p in statik + kopf):,} von {sum(p.numel() for p in net.parameters()):,} Gewichten lernen "
          f"(Statik-Abbildung Lernrate {lr}; Event-Kopf {'Lernrate ' + str(params['learning_rate']) if kopf else 'eingefroren'})", flush=True)
    return [{"params": statik, "lr": lr}] + ([{"params": kopf, "lr": float(params["learning_rate"])}] if kopf else [])


def _fix_task_weights(trainer, weights, n_cat, n_num):
    """task_weights (2026-11-24, nach der Gradienten-Zerlegung statik_decoder_20261124/gradienten.md: der Zeitverlust bestimmt die
    Gradienten am Betrag und arbeitet gegen den Effekt-Übergang): feste Gewichte der Aufgaben statt der GradNorm-Anpassung.
    weights {"cat": a, "num": b}: alle Event-Verluste (kategorial) a, alle Zeit-/Zahlenverluste b. Der GradNorm-Rechenweg des Trainers
    bleibt (gleiche Reihenfolge der Verluste: kategorial, dann numerisch); seine Gewichte werden gesetzt und nicht mehr verändert (Lernrate 0;
    die Normierung auf Summe = Zahl der Aufgaben im Trainer lässt das Verhältnis gleich). Abschwächung und Dropout-Masken unverändert."""
    if not getattr(trainer, "use_gradnorm", False):
        raise ValueError("task_weights: nur mit dem GradNorm-Trainer")
    cat, num = float(weights.get("cat", 1.0)), float(weights.get("num", 1.0))
    if min(cat, num) < 0 or cat + num <= 0 or trainer.number_tasks != n_cat + n_num:
        raise ValueError(f"task_weights: ungültig ({weights}, {trainer.number_tasks} Aufgaben, {n_cat} kategorial + {n_num} numerisch)")
    w = torch.tensor([cat] * n_cat + [num] * n_num, dtype=torch.float, device=trainer.gn_weights.device)
    trainer.gn_weights.data = (w / w.sum() * trainer.number_tasks).detach()
    trainer.gn_optimizer = torch.optim.AdamW(params=[trainer.gn_weights], lr=0.0, weight_decay=0.0)
    print(f"PSP task_weights: Event-Verluste {cat:g}, Zahlenverluste {num:g} → feste Gewichte {[round(x, 4) for x in trainer.gn_weights.tolist()]}",
          flush=True)


def _head_only_trainer(trainer_class):
    """
    head_only: der Trainer des Originals, aber nur die Zeitköpfe lernen (_time_head_groups). Verluste wie dort (attenuierter
    Verlust der Zeiten, Teacher Forcing nach Plan); der Verlust der Aktivität wird nur fürs Protokoll gerechnet (eingefroren).
    Ohne GradNorm: es misst den Gradienten am Zustand des Decoders, der hier nicht lernt; jede Zeitausgabe hängt nur an
    ihrem eigenen Verlust, Adam gleicht die Skalierung je Gewicht aus. Clipping wie im Original (Norm 1) über die lernenden Gewichte.
    """
    class HeadOnlyTrainer(trainer_class):
        def train_epoch_gradnorm(self, prefixes, suffixes, eos_paddings=None, prefix_mask=None, static_inputs=None):
            predictions, _, _, (cat_index, num_index) = self.model(
                prefixes=prefixes, static_inputs=static_inputs, suffixes=suffixes, teacher_forcing_ratio=self.teacher_forcing_ratio,
                prefix_mask=prefix_mask)
            predictions_cat, predictions_num = predictions
            with torch.no_grad():
                cat_losses = {key[:-5]: self.loss_obj.loss_attenuation_cross_entropy(
                    pred_logits=predictions_cat[key], pred_logvars=predictions_cat.get(f"{key[:-5]}_var"), T=30,
                    targets=suffixes[0][cat_index[key[:-5]]].long(), eos_paddings=eos_paddings) for key in predictions_cat if key.endswith("_mean")}
            num_losses = {key[:-5]: self.loss_obj.loss_attenuation_mse(
                pred_means=predictions_num[key], pred_logvars=predictions_num.get(f"{key[:-5]}_var"),
                targets=suffixes[1][num_index[key[:-5]]], eos_paddings=eos_paddings) for key in predictions_num if key.endswith("_mean")}
            loss = torch.stack(list(num_losses.values())).sum()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for group in self.optimizer.param_groups for p in group["params"]], max_norm=1.0)
            self.optimizer.step()
            return [cat_losses, {key: value.detach() for key, value in num_losses.items()}], loss.detach()

    return HeadOnlyTrainer


# sequence_loss (2026-11-19, Nutzer: „ja“): Verlust auf ganzen, frei gezogenen Suffixen. Anlass: mit den echten Events trifft der
# Zeitkopf die längeren Wartezeiten nach einem Zusatzangebot (Sonde angebote_sonde.py), in den selbst gezogenen Suffixen geht die
# späte Wirkung verloren (nachlegen_verlaeufe.py); mit Teacher Forcing 1 sieht das Netz im Training nie seine eigenen Verläufe.
# Je Batch zieht das Netz für bis zu batch Prefixe je draws Suffixe genau wie draw_cases (Zweig time_on_event): Event aus den
# verrauschten Logits (ohne Gradient), Zeiten Mittel + σ·z aus den Köpfen, die das gezogene Event sehen (mit Gradient), neue
# Dropout-Masken in jedem Schritt, die gezogenen Werte gehen ohne Gradient zurück ins Netz; Summe der Abstände (mindestens 0, wie
# outcome mit time_from gaps) bis EOS bzw. bis zur Schrittgrenze von draw_cases (höchstens max_steps). Verlust: CRPS dieser
# gezogenen Restzeiten gegen die echte Restzeit des Falls ab dem letzten Prefix-Event (Tage, geteilt durch scale_days),
# weight-fach zum Verlust des Originals; Loss Attenuation je Schritt, GradNorm und Teacher Forcing bleiben unverändert.
# Bewertet wird weiter nur über gezogene Suffixe. TRAIN/params sequence_loss: True oder ein dict mit Einträgen aus SEQUENCE.
# calibration > 0 (2026-11-19, Nutzer: „Dann mach das mal“; Befund: der CRPS allein verschiebt jeden Arm als Ganzes, nicht je Punkt):
# Gruppe einer Probe = (Wahlpunkt, Option), an deren Options-Event ihr Prefix endet (_sample_groups). Die Prefixe des Suffix-Verlusts
# werden dann je Gruppe gleich häufig gezogen, und je Gruppe im Batch kommt (Mittel der gezogenen Restzeit − echtes Mittel der Gruppe
# über alle Trainingsproben)² / scale_days hinzu, calibration-fach (Kalibrierung im Großen je Gruppe; datensatzunabhängig, die Gruppen
# kommen aus den konfigurierten Wahlpunkten).
# calibration_mode (2026-11-20, Nutzer: „ja“ zur Korrektur): "global" = wie oben (Stand svk). Das Mittel von ≈ 8 Prefixen × 16 Ziehungen
# gegen das globale Gruppenmittel bestraft im Mittel auch die Streuung der Fallmittel innerhalb der Gruppe und die Streuung der Ziehungen
# (Schrumpfdruck gegen fallabhängige Effekte). "paired" = Zielwert ist das echte Mittel DERSELBEN Prefixe im Batch, und die quadrierte
# Abweichung wird als Produkt zweier unabhängiger Hälften der Ziehungen geschätzt ((Mittel Hälfte A − echt) · (Mittel Hälfte B − echt)),
# unverzerrt für das Quadrat der Abweichung des Modellmittels: für das wahre bedingte Modell im Erwartungswert ohne Gradient, kein Druck
# auf Streuung zwischen Fällen oder Ziehungen. Ergebnis svp: gescheitert (online −0,48), das Rauschen der einzelnen echten Ausgänge
# (≈ 4,6 d je Gruppe) dominiert den Gradienten.
# "split" (2026-11-20, Nutzer: „Starte den zuerst“): die Prefixe einer Gruppe im Batch abwechselnd in zwei disjunkte Hälften,
# Strafe (Mittel Hälfte 1 − globales Gruppenmittel) · (Mittel Hälfte 2 − globales Gruppenmittel): unverzerrt für das Quadrat der
# Abweichung des Gruppenmittels (Hälften unabhängig), daher kein Druck auf Streuung zwischen Fällen oder Ziehungen, und kein Rauschen
# der einzelnen echten Ausgänge (Zielwert wie global). Gruppen mit nur einem Prefix im Batch fallen weg.
SEQUENCE ={"weight": 1.0, "draws": 16, "batch": 32, "max_steps": 100, "scale_days": 10.0, "calibration": 0.0, "calibration_mode": "global"}


def _vocabulary_from(source_path):
    """
    vocab_from (2026-11-20): Kontext, in dem der EventLogLoader nach dem Anpassen seiner Encoder für jede kategoriale Spalte (dynamisch
    und statisch) genau die Kategorien des Modells source_path übernimmt (OrdinalEncoder sortiert, gleiche Menge → gleiche Nummern und
    gleiche Netzform). Für den strengen Warmstart (init_from) auf einem Log mit weniger Kategorien, z. B. einem Simulator-Log, der nur
    einen Teil der Mitarbeiter enthält. Enthält der neue Log eine Kategorie, die die Quelle nicht kennt: Abbruch.
    """
    import contextlib
    from event_log_loader.new_event_log_loader_v2 import TensorEncoderDecoder
    quelle = PSPModel.load(source_path).encoder_decoder.categorical_encoders
    original = TensorEncoderDecoder.train_imputers_encoders

    def train_mit_vokabular(self_enc):
        original(self_enc)
        fehlend = {}
        for spalte, encoder in self_enc.categorical_encoders.items():
            if spalte not in quelle:
                raise ValueError(f"vocab_from: Spalte {spalte} fehlt im Quellmodell")
            eigen, fremd = encoder.categories_[0], quelle[spalte].categories_[0]
            leer = lambda x: x is None or (isinstance(x, float) and x != x) or str(x) == "nan"
            extra = {x for x in eigen.tolist() if not leer(x)} - set(fremd.tolist())  # leere Werte: unbekannt (−1, wie Auffüllung)
            if extra:
                raise ValueError(f"vocab_from: {spalte} hat Kategorien, die das Quellmodell nicht kennt: {sorted(map(str, extra))[:10]}")
            fehlend[spalte] = len(fremd) - len(eigen)
            encoder.categories_ = [fremd.copy()]
            if hasattr(encoder, "_missing_indices"):
                encoder._missing_indices = {}  # leere Werte gelten jetzt als unbekannt
        print(f"PSP vocab_from: Kategorien aus {source_path} übernommen; im Log fehlend je Spalte {fehlend}", flush=True)

    @contextlib.contextmanager
    def kontext():
        TensorEncoderDecoder.train_imputers_encoders = train_mit_vokabular
        try:
            yield
        finally:
            TensorEncoderDecoder.train_imputers_encoders = original
    return kontext()


def _sequence_draws(net, prefixes, static_inputs, prefix_mask, settings):
    """
    sequence_loss: je Zeile von prefixes settings["draws"] frei gezogene Restzeiten in Tagen (Zeilen x draws), mit Gradient über
    die gezogenen Zeiten (siehe SEQUENCE). settings: eos_id, window, suffix, gap_mean, gap_scale (Kodierung des Abstands).
    """
    draws = int(settings["draws"])

    def repeat(tensor):
        return tensor.repeat_interleave(draws, dim=0) if torch.is_tensor(tensor) else tensor

    cats, nums = [repeat(t) for t in prefixes[0]], [repeat(t) for t in prefixes[1]]
    static = None if static_inputs is None else tuple(repeat(t) for t in static_inputs)
    mask = repeat(prefix_mask)
    rows, device = cats[0].shape[0], cats[0].device
    names = list(TIME_COLUMNS[:2])  # Reihenfolge der Zahlen-Eingänge wie in draw_cases (ev.all_num_attributes)
    if list(net.output_sizes[0]) != [ACTIVITY] or set(net.output_sizes[1]) != set(names):
        raise ValueError("sequence_loss: nur für Decoder mit der Aktivität als einziger kategorialer und den beiden Zeiten als Zahlen-Ausgabe")
    lengths = mask.reshape(rows, -1).sum(dim=1) if mask is not None else torch.full((rows,), cats[0].shape[1], device=device)
    steps = (settings["window"] - settings["suffix"] - lengths + 1).clamp(min=1, max=int(settings["max_steps"]))
    h, c = net.encoder(input=[cats, nums], static_inputs=static, mask=mask)
    counts = net._state_of_prefix(cats, nums)
    preds, (h, c), _ = net.decoder(input=[[t[:, -1:] for t in cats], [t[:, -1:] for t in nums]], hx=(h, c), z=None, pred=False,
                                   **net._count_kwargs(counts))
    total = torch.zeros(rows, device=device)
    done = torch.zeros(rows, dtype=torch.bool, device=device)
    gap = names.index(TIME_COLUMNS[1])
    for step in range(int(steps.max())):
        with torch.no_grad():  # Event wie sample_cat_predictions_optim (use_variance_cat, aus der Softmax)
            logits = preds[0][0][ACTIVITY + "_mean"].reshape(rows, -1)
            std = torch.exp(0.5 * torch.clamp(preds[1][0][ACTIVITY + "_var"].reshape(rows, -1), min=-6.0, max=6.0))
            event = torch.multinomial(torch.softmax(torch.normal(logits, std), dim=-1), 1).reshape(-1)
        done = done | (event == settings["eos_id"]) | (step >= steps)
        if bool(done.all()):
            break
        means, logvars = net.decoder.numeric_prediction(h, event)
        values = [means[name + "_mean"].reshape(-1) + torch.exp(0.5 * logvars[name + "_var"].reshape(-1)) * torch.randn(rows, device=device)
                  for name in names]
        days = (values[gap] * settings["gap_scale"] + settings["gap_mean"]) / DAY
        total = total + torch.relu(days) * (~done).to(days.dtype)
        last = [[event.reshape(rows, 1)], [value.detach().reshape(rows, 1) for value in values]]
        if counts is not None:
            counts = net._advance(counts, last[0][net.count_dec_position], last[1][net.clock_dec_position] if net._clocking() else None)
        preds, (h, c), _ = net.decoder(input=last, hx=(h, c), z=None, pred=True, **net._count_kwargs(counts))
    return total.reshape(-1, draws)


def _crps(draws, target):
    """CRPS je Zeile aus Ziehungen (Zeilen x K, unverzerrte Form mit K - 1) gegen target (Zeilen), in den Einheiten der Werte."""
    k = draws.shape[1]
    first = (draws - target.unsqueeze(1)).abs().mean(dim=1)
    spread = (draws.unsqueeze(2) - draws.unsqueeze(1)).abs().sum(dim=(1, 2)) / (2 * k * (k - 1))
    return first - spread


def _sequence_trainer(trainer_class, settings):
    """
    Der Trainer des Originals mit sequence_loss: train_epoch_gradnorm rechnet wie dort (Verluste von Aktivität und Zeiten mit Loss
    Attenuation, GradNorm über diese Aufgaben, L2, Clipping, ein Schritt des Optimierers) und addiert weight mal den CRPS der frei
    gezogenen Restzeiten (_sequence_draws) zum Gesamtverlust. Die echte Restzeit (Tage, -1 = keine) steht als zusätzlicher letzter
    Zahlen-Tensor in den Trainingsdaten (fit) und wird vor dem Aufruf des Netzes abgetrennt.
    """
    class SequenceTrainer(trainer_class):
        sequence_stats = None

        def train_epoch_gradnorm(self, prefixes, suffixes, eos_paddings=None, prefix_mask=None, static_inputs=None):
            grouped = settings.get("group_means") is not None  # calibration: Gruppe als zweiter Zusatz-Tensor
            extra = 2 if grouped else 1
            target = prefixes[1][-extra][:, -1]
            group = prefixes[1][-1][:, -1].round().long() if grouped else None
            prefixes, suffixes = [prefixes[0], prefixes[1][:-extra]], [suffixes[0], suffixes[1][:-extra]]
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
            sequence = torch.zeros((), device=gn_seq_loss.device)
            usable = torch.nonzero((target >= 0) & (group >= 0) if grouped else target >= 0).reshape(-1)
            if len(usable):
                size = min(int(settings["batch"]), len(usable))
                if grouped:  # je Gruppe gleich häufig
                    members = group.index_select(0, usable)
                    share = 1.0 / torch.bincount(members).float()[members]
                    pick = usable[torch.multinomial(share, size, replacement=False)]
                else:
                    pick = usable[torch.randperm(len(usable), device=usable.device)[:size]]

                def sub(tensor):
                    return tensor.index_select(0, pick) if torch.is_tensor(tensor) else tensor
                drawn = _sequence_draws(self.model, [[sub(t) for t in prefixes[0]], [sub(t) for t in prefixes[1]]],
                                        None if static_inputs is None else tuple(sub(t) for t in static_inputs), sub(prefix_mask), settings)
                real = target.index_select(0, pick).to(drawn.dtype)
                crps = _crps(drawn, real).mean() / float(settings["scale_days"])
                stats = self.sequence_stats or {"crps": 0.0, "drawn": 0.0, "real": 0.0, "batches": 0, "bias": {}}
                sequence = crps
                if grouped and float(settings["calibration"]) > 0:
                    picked = group.index_select(0, pick)
                    targets = torch.as_tensor(settings["group_means"], dtype=drawn.dtype, device=drawn.device)
                    mode = settings.get("calibration_mode", "global")
                    if mode not in ("global", "paired", "split"):
                        raise ValueError(f"sequence_loss calibration_mode {mode!r}: global, paired oder split")
                    paired = mode == "paired"
                    if paired and drawn.shape[1] < 2:
                        raise ValueError("sequence_loss calibration_mode paired: mindestens 2 Ziehungen je Prefix")
                    half = drawn.shape[1] // 2
                    terms = []
                    for g in picked.unique().tolist():
                        rows = picked == g
                        if mode == "split":  # disjunkte Hälften der Prefixe der Gruppe gegen das globale Gruppenmittel
                            members = torch.nonzero(rows).reshape(-1)
                            bias = drawn[members].mean() - targets[g]
                            if len(members) >= 2:
                                terms.append((drawn[members[0::2]].mean() - targets[g]) * (drawn[members[1::2]].mean() - targets[g]))
                        elif paired:  # Zielwert: echtes Mittel derselben Prefixe; Quadrat unverzerrt aus zwei Hälften der Ziehungen
                            aim = real[rows].mean()
                            bias = drawn[rows].mean() - aim
                            terms.append((drawn[rows][:, :half].mean() - aim) * (drawn[rows][:, half:2 * half].mean() - aim))
                        else:
                            bias = drawn[rows].mean() - targets[g]
                            terms.append(bias ** 2)
                        total, count = stats["bias"].get(g, (0.0, 0))
                        stats["bias"][g] = (total + float(bias.detach()), count + 1)
                    if terms:  # split: Gruppen mit nur einem Prefix im Batch tragen nichts bei
                        sequence = crps + float(settings["calibration"]) * torch.stack(terms).mean() / float(settings["scale_days"])
                stats["crps"] += float(crps.detach()) * float(settings["scale_days"])
                stats["drawn"] += float(drawn.detach().mean())
                stats["real"] += float(real.mean())
                stats["batches"] += 1
                self.sequence_stats = stats
            weight_reg_enc, bias_reg_enc = self.model.encoder.regularizer()
            weight_reg_dec, bias_reg_dec = self.model.decoder.regularizer()
            loss = (self.gn_weights * gn_seq_loss).sum() + self.regularization_term * (
                (weight_reg_enc + weight_reg_dec).to(self.device) + (bias_reg_enc + bias_reg_dec).to(self.device)) + float(settings["weight"]) * sequence
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
            return [cat_losses, num_losses], loss

        def validation_epoch(self, *args, **kwargs):
            stats = self.sequence_stats
            if stats and stats["batches"]:
                n = stats["batches"]
                print(f"Suffix-Verlust (CRPS, Training): {stats['crps'] / n:.3f} Tage; gezogene Restzeit im Mittel {stats['drawn'] / n:.2f}, "
                      f"echte {stats['real'] / n:.2f} Tage ({n} Batches, Gewicht {settings['weight']})", flush=True)
                if stats.get("bias"):
                    names = settings.get("group_names") or []
                    against = ("echtes Mittel derselben Prefixe" if settings.get("calibration_mode", "global") == "paired"
                               else "echtes Gruppenmittel")
                    print(f"Kalibrierung je Gruppe (gezogen − {against}, Tage, Mittel über Batches): " + ", ".join(
                        f"{names[g] if g < len(names) else g} {total / count:+.2f}" for g, (total, count) in sorted(stats["bias"].items())), flush=True)
            self.sequence_stats = None
            return super().validation_epoch(*args, **kwargs)

    return SequenceTrainer


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
        if self.gap_condition_on_event() and (self.gap_tokens() or self.params.get("log_gaps", False)):
            raise ValueError("Variante (e) braucht getrennte Event-/Zeitklassen-Ausgaben und lineare Abstände")
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
        """Varianten (d)/(e): Abstandsklasse als eigene Ausgabe des Decoders."""
        return bool((getattr(self, "params", None) or {}).get("gap_classes", False) or self.gap_condition_on_event())

    def gap_condition_on_event(self):
        """Variante (e): die Zeitklasse hängt zusätzlich vom aktuellen Aktivität|Lifecycle-Token ab."""
        return bool((getattr(self, "params", None) or {}).get("gap_condition_on_event", False))

    def gap_draw(self):
        """gap_classes: woher der Abstand innerhalb der gezogenen Klasse kommt (params gap_draw, GAP_DRAW; Standard aus dem Decoder)."""
        mode = (getattr(self, "params", None) or {}).get("gap_draw", GAP_DRAW[0])
        if mode not in GAP_DRAW:
            raise ValueError(f"gap_draw {mode!r}, erlaubt: {GAP_DRAW}")
        return mode

    def gap_tokens(self):
        """Stufe 3: Token des Decoders mit Abstandsklasse (TRAIN gap_tokens, siehe oben)."""
        return bool((getattr(self, "params", None) or {}).get("gap_tokens", False))

    def counters(self):
        """Zähler (2026-11-16, params counters): je Aktivität|Lifecycle-Token die Zahl bisheriger Events als Eingabe in Encoder
        (kumuliert je Position) und Decoder (Prefix plus zurückgegebene bzw. gezogene Events); reine Eingabe, kein Verlust."""
        return bool((getattr(self, "params", None) or {}).get("counters", False))

    def time_on_event(self):
        """Zeit je Event (2026-11-16, params time_on_event): die Zeitköpfe sehen das Event desselben Schritts (im Verlust das
        echte, beim Ziehen das gezogene bzw. die erzwungene Option); ohne Zeitklassen und Abstands-Token."""
        return bool((getattr(self, "params", None) or {}).get("time_on_event", False))

    def clocks(self):
        """Uhren (2026-11-17, params clocks, nur mit counters): je Aktivität|Lifecycle-Token die Tage seit seinem letzten
        Auftreten als Eingabe in Encoder und Decoder (aus den Abständen, beim Ziehen aus den gezogenen), z. B. Zeit seit dem
        letzten O_Sent für die 30-Tage-Frist bis A_Cancelled; reine Eingabe, kein Verlust; nur mit stetigen Abständen."""
        return bool((getattr(self, "params", None) or {}).get("clocks", False))

    def option_in_decoder(self):
        """Stufe 3: die Option ist an Control-Flow-Punkten das erste Event des Suffix (TRAIN/params option_as "decoder")."""
        return (getattr(self, "params", None) or {}).get("option_as") == "decoder"

    def option_in_encoder(self):
        """Die Option als letztes Event des Prefix mit vom Decoder gezogenem Abstand (TRAIN/params option_as "encoder", siehe oben)."""
        return (getattr(self, "params", None) or {}).get("option_as") == "encoder"

    @staticmethod
    def _split_gap_token(label):
        """Token mit Abstandsklasse -> (Token ohne Klasse, Nummer der Klasse in GAP_NAMES oder -1)."""
        base, _, name = str(label).rpartition("|")
        return (base, GAP_NAMES.index(name)) if base and name in GAP_NAMES else (str(label), -1)

    def dynamic_categorical(self):
        """Kategoriale Eingänge je Event: DYNAMIC_CATEGORICAL, mit gap_classes oder gap_tokens dazu GAP_CLASS (am Ende: die Stellen der anderen bleiben)."""
        return DYNAMIC_CATEGORICAL + ([GAP_CLASS] if self.gap_classes() or self.gap_tokens() else [])

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
        if self.gap_classes() or self.gap_tokens():
            view[GAP_CLASS] = gap_class_names(times.groupby(frame[CASE].astype(str).to_numpy()).diff().dt.total_seconds().to_numpy())
        if self.gap_tokens():
            view[ACTIVITY] = view[ACTIVITY].astype(str) + "|" + view[GAP_CLASS].astype(str)
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
        keep = self.params.get("only_options")  # T-Learner (2026-11-17): nur Proben am Options-Event dieser Optionen (Token Aktivität|Lifecycle)
        chunks = int(self.params.get("load_chunks") or 1)
        if chunks > 1 and not (self.params.get("only_points", False) and self.params.get("only_points_validation", False)):
            raise ValueError("load_chunks: nur mit only_points und only_points_validation (sonst wären alle Prefixe nötig)")

        def datasets(loader):
            if chunks == 1:
                return loader.get_dataset("train"), loader.get_dataset("val")
            # 2026-11-21: in Stücken von Fällen kodieren und sofort auf die Wahlpunkt-Proben filtern (Speicher: der Loader legt jeden
            # Prefix als Python-Liste an); gleiche Encoder, Fensterbreite und Aufteilung wie ohne Stücke
            return (self._chunked_points(loader, "train", chunks, keep), self._chunked_points(loader, "val", chunks, keep))

        if self.params.get("vocab_from"):  # 2026-11-20: Vokabular eines anderen Modells (Warmstart auf einem Simulator-Log)
            with _vocabulary_from(self.params["vocab_from"]):
                loader = EventLogLoader(event_log_location=self.view(log), event_log_properties=self.properties())
                train, validation = datasets(loader)
        else:
            loader = EventLogLoader(event_log_location=self.view(log), event_log_properties=self.properties())
            train, validation = datasets(loader)
        train, weighted = self._weight_points(train, self.params.get("point_weight", 1), only=self.params.get("only_points", False), keep=keep)
        if self.params.get("only_points_validation", False):  # 2026-11-17: Validierung ebenfalls nur am Options-Event (schnellere Epochen)
            validation, _ = self._weight_points(validation, 1, only=True, keep=keep)
        categories, numerics = train.all_categories
        static_categories, static_numerics = train.all_static_categories
        clock_kwargs = {}
        if self.clocks():
            if not self.counters() or self.gap_classes() or self.gap_tokens() or self.params.get("log_gaps", False):
                raise ValueError("clocks: nur mit counters und stetigen Abständen (ohne gap_classes, gap_tokens, log_gaps)")
            scaler = train.encoder_decoder.continuous_encoders[TIME_COLUMNS[1]]
            clock_kwargs = {"clock_feature": TIME_COLUMNS[1], "clock_stats": (float(scaler.mean_[0]), float(scaler.scale_[0]))}
        net = DropoutUncertaintyEncoderDecoderLSTM(
            data_set_categories=train.all_categories, enc_feat=[[f[0] for f in categories], [f[0] for f in numerics]], dec_feat=self.decoder(),
            seq_len_pred=train.min_suffix_size, static_data_set_categories=train.all_static_categories,
            static_enc_feat=[[f[0] for f in static_categories], [f[0] for f in static_numerics]],
            categorical_conditioning=(GAP_CLASS, ACTIVITY) if self.gap_condition_on_event() else None,
            count_feature=ACTIVITY if self.counters() else None,
            numeric_conditioning=ACTIVITY if self.time_on_event() else None, **clock_kwargs, **MODEL_PARAMS)
        if self.time_on_event() and (self.gap_classes() or self.gap_tokens() or self.params.get("log_gaps", False)):
            raise ValueError("time_on_event: nur mit stetigen Abständen (ohne gap_classes, gap_tokens, log_gaps)")
        if self.params.get("init_from"):  # Warmstart (2026-11-17, T-Learner): Gewichte eines fertigen PSP mit gleicher Netzstruktur
            source = PSPModel.load(self.params["init_from"]).net
            net.load_state_dict(source.state_dict(), strict=True)
            print(f"PSP: Gewichte übernommen aus {self.params['init_from']}", flush=True)
        event_head = self.params.get("time_event_head")  # 2026-11-18: eigene Parameter je Event für die Zeitköpfe (siehe oben)
        if event_head:
            if not self.time_on_event() or event_head not in ("offset", "slope"):
                raise ValueError("time_event_head: 'offset' oder 'slope', nur mit time_on_event")
            net.decoder.add_event_time_parameters(slopes=event_head == "slope")  # nach dem Warmstart, mit null begonnen
        head_only = self.params.get("head_only", False)
        if head_only and (not self.params.get("init_from") or self.params.get("remaining_head") or not self.time_on_event()):
            raise ValueError("head_only: nur mit init_from und time_on_event, ohne remaining_head")
        if self.gap_classes():
            self.gap_samples = self._gap_samples(log)
        if self.params.get("decoder_context", False):
            _attach_context(net)
        if self.params.get("static_in_decoder", False):  # 2026-11-24, nach dem Warmstart (init_from lädt die Gewichte ohne die neue Abbildung)
            _attach_static(net)
        remaining_head = self.params.get("remaining_head", False)
        if remaining_head:
            net.remaining_head = _new_remaining_head(net.hidden_size)  # vor dem Optimierer: seine Gewichte lernen mit
        if head_only:  # nur die Zeitköpfe lernen, der Rest bleibt eingefroren (Ereignisfolgen unverändert)
            optimizer = torch.optim.AdamW(params=_time_head_groups(net, self.params["learning_rate"], self.params.get("offset_learning_rate")),
                                          lr=self.params["learning_rate"], weight_decay=0)
        elif self.params.get("static_only", False):  # 2026-11-24 Lernbarkeitstest: nur die Statik-Abbildung (± Event-Kopf) lernt
            optimizer = torch.optim.AdamW(params=_static_only_groups(net, self.params), lr=self.params["learning_rate"], weight_decay=0)
        elif self.params.get("static_in_decoder", False) and self.params.get("static_learning_rate"):
            # 2026-11-24: eigene Lernrate für die neue, mit 0 begonnene Abbildung der Statik (sonst wächst sie beim Warmstart mit
            # kleiner Lernrate kaum: Probelauf 1 Epoche, LR 1e-4 → |Gewicht| ≤ 4e-4)
            neu = list(net.decoder.input_proj.static_projection.parameters())
            ids = {id(p) for p in neu}
            optimizer = torch.optim.AdamW(params=[{"params": [p for p in net.parameters() if id(p) not in ids]},
                                                  {"params": neu, "lr": float(self.params["static_learning_rate"])}],
                                          lr=self.params["learning_rate"], weight_decay=0)
        else:
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
        if head_only:
            trainer_class = _head_only_trainer(Trainer)
        if remaining_head:  # nach self.reference: die Referenz bleibt ohne den Ziel-Tensor
            train.continuous_tensors.append(self._remaining_targets(train, log))
            hook = net.encoder.register_forward_hook(_keep_encoding)
            trainer_class = _remaining_trainer(Trainer, self.params.get("remaining_weight", REMAINING_HEAD["weight"]))
        sequence = self.params.get("sequence_loss")  # 2026-11-19: Verlust auf frei gezogenen Suffixen (SEQUENCE)
        if sequence:
            if not self.time_on_event() or remaining_head or head_only or self.params.get("decoder_context", False):
                raise ValueError("sequence_loss: nur mit time_on_event, ohne remaining_head, head_only und decoder_context")
            scaler = train.encoder_decoder.continuous_encoders[TIME_COLUMNS[1]]
            if not (hasattr(scaler, "mean_") and hasattr(scaler, "scale_")):
                raise ValueError("sequence_loss: der Abstand muss linear standardisiert sein (ohne log_gaps)")
            names = [entry[0] for entry in train.all_categories[0]]
            sequence_settings = {**SEQUENCE, **(sequence if isinstance(sequence, dict) else {}),
                                 "eos_id": int(train.all_categories[0][names.index(ACTIVITY)][2]["EOS"]),
                                 "window": int(train.encoder_decoder.window_size), "suffix": int(train.encoder_decoder.min_suffix_size),
                                 "gap_mean": float(scaler.mean_[0]), "gap_scale": float(scaler.scale_[0])}
            train.continuous_tensors.append(self._sequence_targets(train, log))
            if float(sequence_settings["calibration"]) > 0:  # Gruppe je Probe als zweiter Zusatz-Tensor, echtes Mittel je Gruppe
                groups, group_names = self._sample_groups(train)
                days = train.continuous_tensors[-1][:, 0]
                means = [float(days[(groups == g) & (days >= 0)].mean()) if bool(((groups == g) & (days >= 0)).any()) else float("nan")
                         for g in range(len(group_names))]
                sequence_settings.update(group_means=means, group_names=group_names)
                train.continuous_tensors.append(groups.float().unsqueeze(1).expand(-1, train.encoder_decoder.window_size))
                print("PSP sequence_loss Gruppen: " + ", ".join(f"{name} {int((groups == g).sum()):,} Proben, echt {mean:.2f} d"
                                                                 for g, (name, mean) in enumerate(zip(group_names, means))), flush=True)
            print(f"PSP sequence_loss: {sequence_settings}", flush=True)
            trainer_class = _sequence_trainer(Trainer, sequence_settings)
        trainer = trainer_class(
            device=device, model=net, data_train=train, data_val=validation, loss_obj=Loss(beta_nll=float(self.params.get("beta_nll", 0.0))), log_normal_loss_num_feature=[],
            optimize_values={"regularization_term": self.params["regularization_term"], "optimizer": optimizer, "scheduler": scheduler,
                             "epochs": self.params["epochs"], "mini_batches": self.params["batch_size"], "shuffle": True,
                             "min_teacher_forcing_value": float(self.params.get("teacher_forcing", 0.0)), "max_teacher_forcing_value": 1.0},
            suffix_data_split_value=train.min_suffix_size, writer=_SilentWriter(),
            gradnorm_values={"use_gradnorm": True, "number_tasks": len(self.decoder()[0]) + len(self.decoder()[1]), "gn_alpha": 1.5, "gn_learning_rate": 1e-4},
            save_model_n_th_epoch=1, saving_path=os.path.join(folder, "last.pt"))
        if self.params.get("task_weights"):  # 2026-11-24: feste Aufgabengewichte statt GradNorm-Anpassung (siehe _fix_task_weights)
            _fix_task_weights(trainer, self.params["task_weights"], len(self.decoder()[0]), len(self.decoder()[1]))
        _, validation_losses, _ = trainer.train_model(use_statics=True, use_zero_padd_masking=True, use_eos_padd_masking=True)
        if head_only or self.params.get("static_only", False):
            for parameter in net.parameters():
                parameter.requires_grad_(True)
        if hook is not None:
            hook.remove()
            net.encoder.__dict__.pop("kept_encoding", None)
            train.continuous_tensors.pop()
        if sequence:
            train.continuous_tensors.pop()
            if sequence_settings.get("group_means") is not None:
                train.continuous_tensors.pop()
        net.encoder.__dict__.pop("kept_context", None)
        net.encoder.__dict__.pop("kept_static", None)  # static_in_decoder: Merkmale des letzten Batches (ggf. auf der GPU)
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

    def _sequence_targets(self, dataset, log):
        """
        sequence_loss: je Probe die echte Restzeit in Tagen vom letzten Prefix-Event bis zum Fallende (aus dem ganzen Fall, wie
        _remaining_targets, aber ohne log), als Tensor in Fensterbreite (View); der Trainer schneidet ihn wie die anderen Zahlen.
        """
        times = log.groupby(log[CASE].astype(str))[TIMESTAMP]
        duration = (pd.to_datetime(times.max()) - pd.to_datetime(times.min())).dt.total_seconds()
        names = [entry[0] for entry in dataset.all_categories[1]]
        encoder_decoder = dataset.encoder_decoder
        last = encoder_decoder.window_size - encoder_decoder.min_suffix_size - 1
        elapsed = self._decode(encoder_decoder.continuous_encoders[TIME_COLUMNS[0]], dataset.continuous_tensors[names.index(TIME_COLUMNS[0])][:, last])
        days = np.clip(duration.reindex([str(case) for case in dataset.case_ids]).to_numpy(dtype=float) - elapsed, 0.0, None) / DAY
        if np.isnan(days).any():
            raise ValueError(f"sequence_loss: {int(np.isnan(days).sum())} Proben ohne ihren Fall im Log")
        return torch.tensor(days, dtype=torch.float32).unsqueeze(1).expand(-1, encoder_decoder.window_size)

    def _sample_groups(self, dataset):
        """
        sequence_loss calibration: je Probe die Gruppe (Wahlpunkt, Option), an deren Options-Event ihr Prefix endet (-1 = keine),
        erkannt wie in _weight_points (Option an der letzten Prefix-Stelle nach dem Entscheidungs-Event, oder mit Option im Encoder /
        Decoder als erstes Suffix-Event); dazu die Namen der Gruppen. Nur Gruppen mit mindestens einer Probe.
        """
        from interventions import option_token
        names = [entry[0] for entry in dataset.all_categories[0]]
        activity = dataset.categorical_tensors[names.index(ACTIVITY)]
        ids = dataset.all_categories[0][names.index(ACTIVITY)][2]
        last = dataset.encoder_decoder.window_size - dataset.encoder_decoder.min_suffix_size - 1

        def token(name, lifecycle):
            base = f"{name}|{lifecycle}" if self.lifecycle_tokens else name
            if self.gap_tokens():
                return torch.tensor([number for label, number in ids.items() if self._split_gap_token(label)[0] == base], dtype=activity.dtype)
            return torch.tensor([ids.get(base, -1)], dtype=activity.dtype)

        groups = torch.full((len(dataset),), -1, dtype=torch.long)
        labels = []
        for name, point in self.points.items():
            if point.kind != "control_flow":
                continue
            decision = token(point.activity, point.lifecycle)
            for option in point.options:
                chosen = token(*option_token(option).split("|"))
                hit = torch.isin(activity[:, last], chosen) & torch.isin(activity[:, last - 1], decision)
                if self.option_in_decoder() or self.option_in_encoder():
                    hit |= torch.isin(activity[:, last + 1], chosen) & torch.isin(activity[:, last], decision)
                hit &= groups < 0
                if bool(hit.any()):
                    groups[hit] = len(labels)
                    labels.append(f"{name}:{option_token(option)}")
        return groups, labels

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

    def _truncated_gaps(self, means, logvars, index, scaler):
        """
        gap_classes mit gap_draw "decoder" (Standard): je Zeile ein Abstand in Sekunden aus der vom Decoder vorhergesagten
        Normalverteilung des Abstands (kodiert wie im Training; Mittel und Log-Varianz der attenuierten MSE), beschnitten
        auf die Grenzen der gezogenen Klasse (GAP_EDGES, die oberste bis MAX_GAP_DAYS). Gezogen über die Umkehrfunktion der
        beschnittenen Normalverteilung mit einer Zufallszahl je Zeile aus torch wie _class_gaps (dieselben Zufallszahlen
        für alle Optionen). index: je Zeile die Nummer der Klasse in GAP_NAMES; EOS oder unbekannte Klasse (-1): 0.
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
        index = np.asarray(index, dtype=int)
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

    def _chunked_points(self, loader, kind, chunks, keep):
        """
        load_chunks (2026-11-21): den Datensatz kind ("train" / "val") des Loaders in chunks Stücken von Fällen kodieren, jedes Stück
        sofort auf die Proben am Options-Event filtern (_weight_points only) und die Stücke zusammenfügen. Gleiches Ergebnis wie
        get_dataset + Filter (bis auf die Reihenfolge der Proben), aber mit einem Bruchteil des Speichers.
        """
        from event_log_loader.new_event_log_loader_v2 import EventLogDataset
        df = loader.get_encoded_dataframe(kind)
        spalte = loader.encoder_decoder.case_name
        faelle = df[spalte].drop_duplicates().to_numpy()
        teile = []
        for i, stueck in enumerate(np.array_split(faelle, chunks)):
            bundle, cats, statics = loader.encoder_decoder.encode_df(df[df[spalte].isin(set(stueck.tolist()))])
            gefiltert, _ = self._weight_points(EventLogDataset(bundle, cats, statics, loader.encoder_decoder), 1, only=True, keep=keep)
            teile.append(gefiltert)
            del bundle
            print(f"PSP load_chunks {kind}: Stück {i + 1}/{chunks}, {len(stueck):,} Fälle, {len(gefiltert):,} Proben am Options-Event", flush=True)
        erster = teile[0]
        for teil in teile[1:]:
            if teil.all_categories != erster.all_categories or teil.all_static_categories != erster.all_static_categories:
                raise ValueError("load_chunks: Kategorien der Stücke verschieden")
        bundle = {"categorical": [torch.cat([t.categorical_tensors[k] for t in teile]) for k in range(len(erster.categorical_tensors))],
                  "continuous": [torch.cat([t.continuous_tensors[k] for t in teile]) for k in range(len(erster.continuous_tensors))],
                  "eos_padding": torch.cat([t.eos_padding for t in teile]), "zero_padding": torch.cat([t.zero_padding for t in teile]),
                  "case_ids": tuple(c for t in teile for c in t.case_ids),
                  "static_categorical": torch.cat([t.static_categorical_tensor for t in teile]),
                  "static_continuous": torch.cat([t.static_continuous_tensor for t in teile])}
        ganz = EventLogDataset(bundle, erster.all_categories, erster.all_static_categories, loader.encoder_decoder)
        ganz.prefixes_petri_net_marking = [m for t in teile for m in t.prefixes_petri_net_marking]
        return ganz

    def _weight_points(self, train, weight, only=False, keep=None):
        """
        Die Trainingsproben, deren Prefix am Options-Event eines Wahlpunkts endet, weight-mal (die übrigen einmal); liefert
        (Datensatz, Zahl dieser Proben). only (TRAIN only_points, 2026-11-17): nur diese Proben, je einmal (weight egal); die
        Validierungsproben bleiben unverändert, außer mit TRAIN only_points_validation (dann ebenso gefiltert). keep (TRAIN
        only_options, T-Learner): nur Optionen mit diesen Token (Aktivität|Lifecycle); Mitarbeiterpunkte zählen dann nicht. Eine Probe ist eine Zeile der Breite window_size: links aufgefüllt der Prefix, die
        letzten min_suffix_size Stellen das Suffix; das letzte Prefix-Event steht also an Stelle window_size - min_suffix_size - 1.
        Mitarbeiter: dort das Event des Arbeitsschritts (das start-Event mit dem gewählten Mitarbeiter); Control Flow: dort eine
        der Optionen und davor das Entscheidungs-Event.
        """
        from interventions import option_token
        names = [entry[0] for entry in train.all_categories[0]]
        activity = train.categorical_tensors[names.index(ACTIVITY)]
        ids = train.all_categories[0][names.index(ACTIVITY)][2]
        last = train.encoder_decoder.window_size - train.encoder_decoder.min_suffix_size - 1

        def token(name, lifecycle):  # Nummern des Tokens (gap_tokens: mit jeder Abstandsklasse)
            base = f"{name}|{lifecycle}" if self.lifecycle_tokens else name
            if self.gap_tokens():
                return torch.tensor([number for label, number in ids.items() if self._split_gap_token(label)[0] == base], dtype=activity.dtype)
            return torch.tensor([ids.get(base, -1)], dtype=activity.dtype)

        hit = torch.zeros(len(train), dtype=torch.bool)
        nur_punkte = self.params.get("only_point_names")  # 2026-11-22: ein Netz je Wahlpunkt und Option (nur Proben dieser Punkte)
        for point in self.points.values():
            if nur_punkte and point.name not in set(nur_punkte):
                continue
            if point.kind == "control_flow":
                kept = [option for option in point.options if keep is None or option_token(option) in set(keep)]
                if not kept:
                    continue
                options = torch.cat([token(*option_token(option).split("|")) for option in kept])
                decision = token(point.activity, point.lifecycle)
                hit |= torch.isin(activity[:, last], options) & torch.isin(activity[:, last - 1], decision)
                if self.option_in_decoder() or self.option_in_encoder():  # die Option als erstes Event des Suffix
                    hit |= torch.isin(activity[:, last + 1], options) & torch.isin(activity[:, last], decision)
            elif keep is None:
                hit |= torch.isin(activity[:, last], token(point.activity, point.lifecycle))
        chosen = torch.nonzero(hit).reshape(-1).tolist()
        if only:
            if not chosen:
                raise ValueError("only_points: keine Trainingsprobe endet am Options-Event eines Wahlpunkts")
            print(f"PSP: nur die {len(chosen):,} von {len(train):,} Trainingsproben, die am Options-Event eines Wahlpunkts enden", flush=True)
            return train.subset(chosen), len(chosen)
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
        head = any(key.startswith("remaining_head.") for key in checkpoint["model_state_dict"])
        static = any(key.startswith("decoder.input_proj.static_projection.") for key in checkpoint["model_state_dict"])  # 2026-11-24
        context = any(key.startswith(("decoder.input_proj.context_projection.", "decoder.input_proj.projection.context_projection."))
                      for key in checkpoint["model_state_dict"])
        offsets = any(key.startswith("decoder.num_event_offsets.") for key in checkpoint["model_state_dict"])
        slopes = any(key.startswith("decoder.num_event_slopes.") for key in checkpoint["model_state_dict"])
        if not head and not context and not offsets and not static:
            return DropoutUncertaintyEncoderDecoderLSTM.load(path)
        kwargs = {"static_data_set_categories": None, "static_enc_feat": None, **checkpoint["kwargs"]}
        kwargs.pop("static_input_size_enc", None)
        net = DropoutUncertaintyEncoderDecoderLSTM(**kwargs)
        if context:
            _attach_context(net)
        if static:
            _attach_static(net)
        if offsets:  # time_event_head (2026-11-18)
            net.decoder.add_event_time_parameters(slopes=slopes)
        if head:
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
        state.pop("_transition_cache", None)
        state.pop("last_option_gap_draws", None)
        state.pop("_work_state_mask_cache", None)
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

    def _transition_masks(self, ev):
        """Übergangsregeln, optional: nur beobachtete Aktivität|Lifecycle-Paare und Fallenden.

        Zeitklassen schränken das Paar nicht ein. Prefix- und Decoder-IDs werden
        unabhängig über ihre Labels zugeordnet; Padding bleibt gesperrt.
        """
        rules = getattr(self, "transition_rules", None)
        if rules is None:
            return None
        cached = getattr(self, "_transition_cache", None)
        if cached is not None and cached[0] is rules:
            return cached[1]
        names = ev.inverted_suffix_categories[0]
        width = max(names) + 1
        base = lambda label: self._split_gap_token(label)[0] if self.gap_tokens() else str(label)
        following = {}
        for previous, successor in rules["pairs"]:
            following.setdefault(previous, set()).add(successor)
        ends = set(rules["ends"])

        def row(label):
            label = base(label)
            mask = torch.zeros(width, dtype=torch.bool)
            for number, successor in names.items():
                if int(number) == ev.eos_id:
                    mask[int(number)] = label in ends
                else:
                    mask[int(number)] = base(successor) in following.get(label, ())
            return mask

        decoder = torch.stack([row(names.get(number, "<padding>")) for number in range(width)])
        # Bereits beendete Zeilen bleiben bei EOS; sie werden nicht mehr ausgegeben.
        decoder[ev.eos_id, ev.eos_id] = True
        categories = self.encoder_decoder.categorical_encoders[ACTIVITY].categories_[0]
        prefix = torch.stack([torch.zeros(width, dtype=torch.bool), *[row(label) for label in categories]])
        masks = {"prefix": prefix, "decoder": decoder,
                 "by_token": {label: row(label) for label in set(following) | ends}}
        self._transition_cache = (rules, masks)
        return masks

    def draw_cases(self, encoded, n, until=None, stop=None, max_steps=None, first=None, last=None,
                   completion_steps=None, previous_tokens=None, work_states=None, force=None, _state_retry=0):
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
        first (option_as "decoder"): je Prefix das Token (wie labels), das das erste Event des Suffix sein muss; der erste
        Schritt zieht dann nur unter diesem Token (gap_tokens: unter seinen Abstandsklassen).
        last (option_as "encoder"): Werte des letzten Prefix-Events je Zeile (Prefixe x n) statt der kodierten:
        {"cats": {Stelle: Nummern}, "nums": {Stelle: kodierte Werte}}.
        completion_steps: für vollständige Fallansichten bis EOS, mit dieser Obergrenze statt des Trainingsfensters.
        Ohne diese Angabe bleibt das Längenlimit für Policy und Evaluation unverändert.
        transition_rules am Modell: unbeobachtete Übergänge erhalten Wahrscheinlichkeit 0,
        einschließlich EOS nach Events, nach denen kein Trainingsfall endet.
        previous_tokens: tatsächliches letztes Aktivität|Lifecycle-Token je Prefix.
        force (2026-11-23, Rollout, policies/ev_policy/rollout.py): Funktion (labels, aktive Zeilen, suffixes) -> {Zeile: Token}, nach jedem
        Schritt aufgerufen; diese Token werden im nächsten Schritt gezogen (Maske auf die Logits; Zeit wie sonst aus dem Decoder).
        Ohne force unverändert.
        Damit bleiben die Regeln auch dann unabhängig von Zeitklassen, wenn eine
        neue Aktivität-Zeitklasse-Kombination im Encoder als unbekannt kodiert ist.
        """
        ev, model = self._evaluator(), self.net
        names = ev.inverted_suffix_categories[0]
        scalers = {name: self.encoder_decoder.continuous_encoders[name] for name in ev.all_num_attributes}
        window, suffix_size = self.encoder_decoder.window_size, self.encoder_decoder.min_suffix_size
        steps = np.repeat([max(window - suffix_size - length + 1, 1) for _, _, _, length in encoded], n)
        if completion_steps is not None:
            if type(completion_steps) is not int or completion_steps < 1:
                raise ValueError("completion_steps muss eine positive ganze Zahl sein")
            steps = np.full(len(steps), completion_steps)
        steps = np.minimum(steps, max_steps) if max_steps is not None else steps
        limit = np.repeat(np.full(len(encoded), np.inf) if until is None else np.asarray(until, dtype=float), n)
        rows = n * len(encoded)
        classes, tokens = self.gap_classes(), self.gap_tokens()
        stateful = bool(getattr(self, "work_state_constraints", False))
        conditional_gaps = getattr(self, "conditional_gaps", None)
        states = None
        if stateful:
            from interventions.work_state import WorkState
            if work_states is None:
                column = ev.prefix_cat_attributes.index(ACTIVITY)
                vocabulary = self.encoder_decoder.categorical_encoders[ACTIVITY].categories_[0]
                work_states = []
                for entry in encoded:
                    state = WorkState()
                    for number in entry[0][0][column].reshape(-1).tolist():
                        if int(number) > 0:
                            label = str(vocabulary[int(number)-1])
                            state.push(self._split_gap_token(label)[0] if tokens else label)
                    work_states.append(state)
            if len(work_states) != len(encoded):
                raise ValueError("One complete work state is required per prefix")
            initial_states = [state.copy() for state in work_states for _ in range(n)]
            states = [state.copy() for state in initial_states]
        elif conditional_gaps is not None:
            raise ValueError("Conditional gap calibration requires work-state constraints")
        conditional = self.gap_condition_on_event()
        if conditional and getattr(model.decoder, "categorical_conditioning", None) != (GAP_CLASS, ACTIVITY):
            raise ValueError("Variante (e) braucht ein mit gap_condition_on_event trainiertes Netz")
        if classes:  # Nummer der Klasse im Encoder -> Name (GAP_NAMES)
            class_names = {v: k for k, v in next(c for c in self.reference.all_categories[0] if c[0] == GAP_CLASS)[2].items()}
        allowed = None
        if first is not None:  # je Zeile die Token, die das erste Event sein dürfen
            width = max(names) + 1
            masks = {}
            for token in set(first):
                masks[token] = torch.zeros(width, dtype=torch.bool)
                for number, name in names.items():
                    if (self._split_gap_token(name)[0] if tokens else str(name)) == token:
                        masks[token][int(number)] = True
                if not masks[token].any():
                    raise ValueError(f"PSP: das Modell kennt das Token {token!r} nicht (option_as decoder)")
            allowed = torch.stack([masks[token] for token in first]).repeat_interleave(n, dim=0)
        cats, nums, batch_static, mask = self._batch(encoded, n)
        if last is not None:
            for column, ids in last.get("cats", {}).items():
                cats[column][:, -1] = torch.as_tensor(np.asarray(ids), dtype=cats[column].dtype)
            for column, values in last.get("nums", {}).items():
                nums[column][:, -1] = torch.as_tensor(values, dtype=nums[column].dtype).reshape(-1)
        transitions = self._transition_masks(ev)
        if transitions is not None:
            if previous_tokens is not None:
                if len(previous_tokens) != len(encoded):
                    raise ValueError("previous_tokens muss genau ein Token je Prefix enthalten")
                try:
                    transition_allowed = torch.stack([transitions["by_token"][token] for token in previous_tokens]).repeat_interleave(n, dim=0)
                except KeyError as error:
                    raise ValueError(f"PSP-Übergangsregeln: unbekanntes letztes Token {error.args[0]!r}") from error
            else:
                previous = cats[ev.prefix_cat_attributes.index(ACTIVITY)][:, -1].long()
                transition_allowed = transitions["prefix"][previous]
        keyed = None
        if (self.params or {}).get("keyed_noise", False):
            if not self.time_on_event() or classes or tokens or stateful or conditional or list(ev.all_cat_attributes) != [ACTIVITY]:
                raise ValueError("keyed_noise: nur für time_on_event-Modelle ohne Abstandsklassen, Klassen-Token oder Arbeitszustände")
            keyed = {"seed": int(torch.initial_seed()) & 0xFFFFFFFFFFFFFFFF, "rows": np.arange(rows, dtype=np.int64),
                     # erster Schlüssel: letztes Prefix-Event (Encoder-Nummer, eigener Bereich), danach das zuletzt gezogene Event
                     "previous": cats[ev.prefix_cat_attributes.index(ACTIVITY)][:, -1].long().numpy().astype(np.int64) + 100000,
                     "previous_count": np.zeros(rows, dtype=np.int64), "counts": np.zeros((rows, max(names) + 1), dtype=np.int64)}
        with torch.no_grad():
            prediction, hidden, _ = model.inference(prefix=[cats, nums], static_inputs=batch_static, mask=mask)
            reached = self._decode(scalers[TIME_COLUMNS[0]], nums[ev.prefix_num_attributes.index(TIME_COLUMNS[0])][:, -1])
            suffixes, done = [[] for _ in range(rows)], np.zeros(rows, dtype=bool)
            ended, dead = np.zeros(rows, dtype=bool), np.zeros(rows, dtype=bool)
            erzwungen = {}  # force: Zeile -> Token, das in diesem Schritt gezogen werden muss
            if force is not None:
                label_ids = {str(label): int(number) for number, label in names.items()}
            for step in range(int(steps.max())):
                cat_means = prediction[0][0]
                if stateful:
                    permitted = torch.zeros((rows, max(names)+1), dtype=torch.bool)
                    cache = getattr(self, "_work_state_mask_cache", None)
                    if cache is None:
                        self._work_state_mask_cache = cache = {}
                    grouped_names = {}
                    for number, label in names.items():
                        if int(number) != ev.eos_id:
                            base = self._split_gap_token(label)[0] if tokens else str(label)
                            grouped_names.setdefault(base, []).append(int(number))
                    for k, state in enumerate(states):
                        cache_key = (tuple(sorted(state.states.items())), state.business_end, state.previous)
                        row = cache.get(cache_key)
                        if row is None:
                            row = torch.zeros(max(names)+1, dtype=torch.bool)
                            row[ev.eos_id] = state.complete
                            for base, numbers in grouped_names.items():
                                if state.permits(base) and (conditional_gaps is None or conditional_gaps.distribution(state, base) is not None):
                                    row[numbers] = True
                            cache[cache_key] = row
                        permitted[k] = row
                    if transitions is not None:
                        permitted &= transition_allowed
                    if step == 0 and allowed is not None:
                        permitted &= allowed
                    permitted[torch.from_numpy(done), ev.eos_id] = True
                    missing = ~permitted.any(dim=1)
                    dead |= missing.numpy() & ~done
                    done |= missing.numpy()
                    permitted[missing, ev.eos_id] = True
                    key = ACTIVITY+"_mean"
                    cat_means = {**cat_means, key: cat_means[key].masked_fill(
                        ~permitted.reshape(cat_means[key].shape), float("-inf"))}
                if transitions is not None:
                    permitted = transition_allowed.clone()
                    if step == 0 and allowed is not None:
                        permitted &= allowed
                    inactive = torch.from_numpy(done.copy())
                    permitted[inactive] = False
                    permitted[inactive, ev.eos_id] = True
                    missing = ~permitted.any(dim=1)
                    if missing.any() and not stateful:
                        raise ValueError(f"PSP-Übergangsregeln: kein erlaubter Nachfolger für Zeilen "
                                         f"{torch.nonzero(missing, as_tuple=True)[0].tolist()} (Schritt {step}, first={first})")
                    key = ACTIVITY + "_mean"
                    cat_means = {**cat_means, key: cat_means[key].masked_fill(~permitted.reshape(cat_means[key].shape), float("-inf"))}
                if erzwungen:  # force (Rollout): in diesen Zeilen nur das erzwungene Token
                    if conditional or classes or tokens or stateful:
                        raise NotImplementedError("force: nur für Modelle ohne Abstandsklassen, Klassen-Token oder Arbeitszustände")
                    key = ACTIVITY + "_mean"
                    zeilen = torch.as_tensor(list(erzwungen), dtype=torch.long)
                    erlaubt_jetzt = torch.ones((rows, max(names) + 1), dtype=torch.bool)
                    erlaubt_jetzt[zeilen] = False
                    erlaubt_jetzt[zeilen, torch.as_tensor([label_ids[t] for t in erzwungen.values()], dtype=torch.long)] = True
                    cat_means = {**cat_means, key: cat_means[key].masked_fill(~erlaubt_jetzt.reshape(cat_means[key].shape), float("-inf"))}
                if conditional:
                    # (e): choose the event first, including a forced candidate;
                    # only then predict and draw its conditional gap class.
                    cat_values = type(ev).sample_cat_predictions_optim(
                        [ACTIVITY], ev.use_variance_cat, ev.sample_argmax, cat_means, prediction[1][0])
                    if step == 0 and allowed is not None and transitions is None and not stateful:
                        cat_values[ACTIVITY + "_mean"] = self._draw_allowed(
                            ev, cat_means[ACTIVITY + "_mean"], prediction[1][0][ACTIVITY + "_var"], allowed)
                    mean, logvar = model.decoder.conditioned_prediction(hidden[0], cat_values[ACTIVITY + "_mean"])
                    cat_values.update(type(ev).sample_cat_predictions_optim(
                        [GAP_CLASS], ev.use_variance_cat, ev.sample_argmax,
                        {GAP_CLASS + "_mean": mean}, {GAP_CLASS + "_var": logvar}))
                    num_values = self._clip_gaps(self._draw_times(ev, prediction[0][1], prediction[1][1]))
                elif keyed is not None:
                    cat_values, num_values = self._keyed_draw(ev, model, hidden, cat_means, prediction[1][0],
                                                              allowed if step == 0 and transitions is None else None, keyed)
                    num_values = self._clip_gaps(num_values)
                elif self.time_on_event():
                    # Zeit je Event: erst das Event (bzw. die erzwungene Option), dann die Zeiten aus den Köpfen, die dieses
                    # Event sehen; gleiche Zahl Zufallszahlen je Schritt für alle Optionen (gemeinsame Zufallszahlen)
                    cat_values = type(ev).sample_cat_predictions_optim(ev.all_cat_attributes, ev.use_variance_cat, ev.sample_argmax, cat_means, prediction[1][0])
                    if step == 0 and allowed is not None and transitions is None and not stateful:
                        cat_values[ACTIVITY + "_mean"] = self._draw_allowed(ev, prediction[0][0][ACTIVITY + "_mean"], prediction[1][0][ACTIVITY + "_var"], allowed)
                    means, logvars = model.decoder.numeric_prediction(hidden[0], cat_values[ACTIVITY + "_mean"].reshape(-1))
                    num_values = self._clip_gaps(self._draw_times(ev, means, logvars))
                else:
                    cat_values = type(ev).sample_cat_predictions_optim(ev.all_cat_attributes, ev.use_variance_cat, ev.sample_argmax, cat_means, prediction[1][0])
                    num_values = self._clip_gaps(self._draw_times(ev, prediction[0][1], prediction[1][1]))
                    if step == 0 and allowed is not None and transitions is None and not stateful:
                        cat_values[ACTIVITY + "_mean"] = self._draw_allowed(ev, prediction[0][0][ACTIVITY + "_mean"], prediction[1][0][ACTIVITY + "_var"], allowed)
                activity = cat_values[ACTIVITY + "_mean"].reshape(-1).numpy()
                if transitions is not None:
                    transition_allowed = transitions["decoder"][torch.as_tensor(activity, dtype=torch.long)]
                ended |= (activity == ev.eos_id) & ~dead & ~done
                done |= (activity == ev.eos_id) | (step >= steps)
                if done.all():
                    break
                labels = [str(names.get(int(a), "<unbekannt>")) for a in activity]
                if classes:  # Abstand in der gezogenen Klasse (gap_draw); beide Zeiten passend dazu zurück ins Netz
                    class_ids = cat_values[GAP_CLASS + "_mean"].reshape(-1).numpy()
                    if self.gap_draw() == "real":
                        gap = self._class_gaps(labels, class_ids, class_names)
                    else:
                        index = [GAP_NAMES.index(class_names.get(int(c))) if class_names.get(int(c)) in GAP_NAMES else -1 for c in class_ids]
                        gap = self._truncated_gaps(prediction[0][1], prediction[1][1], index, scalers[TIME_COLUMNS[1]])
                    reached = reached + gap
                    num_values = {TIME_COLUMNS[0] + "_mean": self._encode_values(scalers[TIME_COLUMNS[0]], reached),
                                  TIME_COLUMNS[1] + "_mean": self._encode_values(scalers[TIME_COLUMNS[1]], gap)}
                elif tokens:  # die Klasse steht im gezogenen Token; ausgegeben wird das Token ohne Klasse
                    split = [self._split_gap_token(label) for label in labels]
                    labels = [base for base, _ in split]
                    gap = self._truncated_gaps(prediction[0][1], prediction[1][1], [number for _, number in split], scalers[TIME_COLUMNS[1]])
                    reached = reached + gap
                    num_values = {TIME_COLUMNS[0] + "_mean": self._encode_values(scalers[TIME_COLUMNS[0]], reached),
                                  TIME_COLUMNS[1] + "_mean": self._encode_values(scalers[TIME_COLUMNS[1]], gap)}
                else:
                    reached = np.maximum(reached, self._decode(scalers[TIME_COLUMNS[0]], num_values[TIME_COLUMNS[0] + "_mean"]))
                    gap = self._decode(scalers[TIME_COLUMNS[1]], num_values[TIME_COLUMNS[1] + "_mean"])
                if conditional_gaps is not None:
                    active = np.flatnonzero(~done)
                    calibrated = np.zeros(rows, dtype=float)
                    calibrated[active] = conditional_gaps.draw([states[k] for k in active], [labels[k] for k in active])
                    # reached currently includes the original decoder gap.
                    reached = reached-gap+calibrated
                    gap = calibrated
                    num_values = {TIME_COLUMNS[0]+"_mean": self._encode_values(scalers[TIME_COLUMNS[0]], reached),
                                  TIME_COLUMNS[1]+"_mean": self._encode_values(scalers[TIME_COLUMNS[1]], gap)}
                    actual_classes = gap_class_names(gap)
                    if classes:
                        ids = {name: number for number, name in class_names.items()}
                        cat_values[GAP_CLASS+"_mean"] = torch.tensor([ids[name] for name in actual_classes]).reshape(rows, 1)
                    if tokens:
                        ids = {str(label): int(number) for number, label in names.items()}
                        for k in active:
                            activity[k] = ids[f"{labels[k]}|{actual_classes[k]}"]
                        cat_values[ACTIVITY+"_mean"] = torch.as_tensor(activity).reshape(rows, 1)
                        if transitions is not None:
                            transition_allowed = transitions["decoder"][torch.as_tensor(activity, dtype=torch.long)]
                for k in np.flatnonzero(~done):
                    suffixes[k].append((labels[k], float(reached[k]), max(float(gap[k]), 0.0)))
                    if stateful:
                        states[k].push(labels[k])
                done |= reached > limit
                if stop is not None:
                    done |= np.array([label.split("|")[0] for label in labels], dtype=object) == stop
                if force is not None:
                    for k, t in erzwungen.items():  # Prüfung: die erzwungenen Zeilen haben das Token gezogen
                        if str(labels[k]) != t:
                            raise RuntimeError(f"force: Zeile {k} zog {labels[k]!r} statt {t!r}")
                    erzwungen = force(labels, np.flatnonzero(~done), suffixes)
                prediction, hidden = model.inference(last_event=([cat_values[key].reshape(rows, 1) for key in cat_values],
                                                                 [num_values[key].reshape(rows, 1) for key in num_values]), hx=hidden)
        if stateful and max_steps is None and until is None and stop is None:
            invalid = np.flatnonzero(~ended | dead)
            audit = getattr(self, "sampling_audit", None)
            if audit is not None:
                audit["draw_attempts"] = audit.get("draw_attempts", 0)+rows
                audit["rejected_incomplete_paths"] = audit.get("rejected_incomplete_paths", 0)+len(invalid)
            if len(invalid):
                from prescriptive.estimators.conditional_gaps import SuffixSamplingError
                if _state_retry >= 20:
                    raise SuffixSamplingError("No complete state-consistent suffix after 21 attempts")
                prefix_ids = [int(k)//n for k in invalid]
                subset_last = None if last is None else {kind: {col: np.asarray(values)[invalid]
                    for col, values in columns.items()} for kind, columns in last.items()}
                replacements = self.draw_cases([encoded[k] for k in prefix_ids], 1,
                    completion_steps=completion_steps, last=subset_last,
                    previous_tokens=None if previous_tokens is None else [previous_tokens[k] for k in prefix_ids],
                    work_states=[initial_states[k] for k in invalid], _state_retry=_state_retry+1)
                for k, replacement in zip(invalid, replacements):
                    suffixes[k] = replacement[0]
        return [suffixes[k * n:(k + 1) * n] for k in range(len(encoded))]

    @staticmethod
    def _keyed_draw(ev, model, hidden, cat_means, cat_vars, allowed, state):
        """
        keyed_noise: ein Schritt von draw_cases (Zweig time_on_event) mit Zufallszahlen als feste Funktion von (Saat, Ziehung,
        Ereignis-Schlüssel) statt aus dem fortlaufenden Strom. Event: Logit-Rauschen (use_variance_cat) und Gumbel-Max über die
        Logits, Schlüssel = voriges Event und sein Vorkommen im gezogenen Suffix (erster Schritt: letztes Prefix-Event); Zeiten:
        Normal aus den Köpfen, Schlüssel = gezogenes Event und sein Vorkommen. Die Verteilung jeder Ziehung bleibt gleich (Gumbel-
        Max = Ziehen aus der Softmax); zwei Optionen teilen aber die Zufallszahl desselben Ereignisses (z. B. der ersten
        Kundenantwort), auch wenn es bei ihnen an verschiedenen Schrittnummern liegt. state wird fortgeschrieben.
        """
        key = ACTIVITY + "_mean"
        rows = len(state["rows"])
        logits = cat_means[key].reshape(rows, -1).double()
        component = np.arange(logits.shape[1], dtype=np.int64)[None, :]
        base = (state["seed"], state["rows"][:, None], state["previous"][:, None], state["previous_count"][:, None])
        if ev.use_variance_cat:
            std = torch.exp(0.5 * torch.clamp(cat_vars[ACTIVITY + "_var"], min=-6.0, max=6.0)).reshape(rows, -1).double()
            logits = logits + std * torch.special.ndtri(torch.from_numpy(_keyed_uniform(*base, 1, component)))
        if allowed is not None:
            logits = logits.masked_fill(~allowed.reshape(rows, -1), float("-inf"))
        if ev.sample_argmax:
            choice = logits.argmax(dim=1)
        else:
            choice = (logits - torch.log(-torch.log(torch.from_numpy(_keyed_uniform(*base, 2, component))))).argmax(dim=1)
        activity = choice.numpy().astype(np.int64)
        index = np.arange(rows)
        state["counts"][index, activity] += 1
        occurrence = state["counts"][index, activity].copy()
        means, logvars = model.decoder.numeric_prediction(hidden[0], torch.as_tensor(activity))
        num_values = {}
        for c, name in enumerate(ev.all_num_attributes):
            mean = means[name + "_mean"].reshape(-1, 1)
            z = torch.special.ndtri(torch.from_numpy(_keyed_uniform(state["seed"], state["rows"], activity + 10000, occurrence, 3, c)))
            num_values[name + "_mean"] = mean + torch.exp(0.5 * logvars[name + "_var"].reshape(-1, 1)) * z.reshape(-1, 1).to(mean.dtype)
        state["previous"][:] = activity
        state["previous_count"][:] = occurrence
        return {key: choice.reshape(rows, 1)}, num_values

    @staticmethod
    def _draw_allowed(ev, logits, logvars, allowed):
        """Die Aktivität wie sample_cat_predictions_optim (Rauschen auf den Logits, dann aus der Softmax), aber nur unter den erlaubten Token (Zeilen x Token)."""
        if ev.use_variance_cat:
            logits = torch.normal(logits, torch.exp(0.5 * torch.clamp(logvars, min=-6.0, max=6.0)))
        shape = logits.shape
        probabilities = torch.softmax(logits.reshape(allowed.shape).masked_fill(~allowed, float("-inf")), dim=-1)
        return torch.multinomial(probabilities, num_samples=1, replacement=True).reshape(*shape[:-1], 1) if len(shape) > 2 else \
            torch.multinomial(probabilities, num_samples=1, replacement=True)

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
        if point.kind == "control_flow" and getattr(self, "option_gap_distributions", None) is not None:
            return self._options_with_shared_gap(point, cases, options, now, n)
        if point.kind == "control_flow" and self.option_in_decoder():
            return self._options_in_decoder(cases, options, n)
        if point.kind == "control_flow" and self.option_in_encoder():
            return self._options_in_encoder(point, cases, options, now, n)
        return self._per_option(point, cases, options, now, n, self.draw_cases)

    def _options_with_shared_gap(self, point, cases, options, now, n):
        """Ein eigener Normal-Abstand je Fall/Option; alle n Suffixe teilen diesen Prefix.

        Lage und Skala stammen aus beobachteten Trainingsabständen je Wahlpunkt
        und Option. Die Normalverteilung wird bei 0 trunkiert. Zeitziehungen
        verwenden einen separaten Generator; die Suffixe behalten dieselbe
        Start-Saat für beide Optionen wie in _per_option.
        """
        from scipy.stats import truncnorm
        from interventions import option_token
        if self.gap_classes() or self.gap_tokens() or self.params.get("log_gaps", False):
            raise ValueError("Geteilter Optionsabstand ist für das normale Modell (c) vorgesehen")
        ev = self._evaluator()
        numeric_at = {name: i for i, name in enumerate(ev.prefix_num_attributes)}
        scalers = self.encoder_decoder.continuous_encoders
        elapsed = [self._encode(events)[4] for events in cases]
        encoded = [self._encode(events, self._option_event(point, events, options[0], now)) for events in cases]
        columns = self._option_ids(point, options)
        seed = int(torch.randint(2 ** 31 - 1, (1,)))
        timing_rng = np.random.default_rng(seed)
        distributions = self.option_gap_distributions[point.name]
        results, observed_gaps = [], {}
        for k, option in enumerate(options):
            distribution = distributions[option]
            mean, std = float(distribution["mean_seconds"]), float(distribution["std_seconds"])
            if not np.isfinite(mean) or not np.isfinite(std) or std < 0:
                raise ValueError(f"Ungültige Normalparameter für {point.name}: {option}")
            gaps = (np.full(len(cases), max(mean, 0.0)) if std == 0 else
                    truncnorm.ppf(timing_rng.random(len(cases)), -mean / std, np.inf, loc=mean, scale=std))
            observed_gaps[str(option)] = gaps.tolist()
            stamps = pd.Timestamp(now, unit="s") + pd.to_timedelta(gaps, unit="s")
            batch = []
            for j, (prefix, static, mask, length, _) in enumerate(encoded):
                cats, nums = [tensor.clone() for tensor in prefix[0]], [tensor.clone() for tensor in prefix[1]]
                for column, ids in columns:
                    cats[column][:, -1] = int(ids[k])
                values = (elapsed[j] + gaps[j], gaps[j], stamps[j].weekday(),
                          stamps[j].hour * 3600 + stamps[j].minute * 60 + stamps[j].second)
                for name, value in zip(TIME_COLUMNS, values):
                    nums[numeric_at[name]][:, -1] = self._encode_values(scalers[name], [value]).reshape(-1)
                batch.append(([cats, nums], static, mask, length))
            torch.manual_seed(seed)
            token = option_token(option) if self.lifecycle_tokens else option_token(option).split("|")[0]
            rest = self.draw_cases(batch, n, previous_tokens=[token] * len(cases))
            results.append([[[(token, float(elapsed[j] + gaps[j]), float(gaps[j])), *suffix]
                             for suffix in suffixes] for j, suffixes in enumerate(rest)])
        torch.manual_seed(seed + 1)
        self.last_option_gap_draws = {"point": point.name, "gaps_seconds": observed_gaps,
                                      "draws_per_case_and_option": 1}
        return results, elapsed

    def _options_in_decoder(self, cases, options, n):
        """
        option_as "decoder": je Option je Fall n Suffixe, deren erstes Event die Option ist (draw_cases first), aus dem
        Prefix bis zum Entscheidungs-Event; dieselben Zufallszahlen für alle Optionen. Die Zeit seit Fallbeginn ist die am
        Entscheidungs-Event: der gezogene Abstand der Option zählt zur Durchlaufzeit.
        """
        from interventions import option_token
        encoded = [self._encode(events) for events in cases]
        batch = [entry[:4] for entry in encoded]
        seed = int(torch.randint(2 ** 31 - 1, (1,)))
        results = []
        for option in options:
            token = option_token(option) if self.lifecycle_tokens else option_token(option).split("|")[0]
            torch.manual_seed(seed)
            results.append(self.draw_cases(batch, n, first=[token] * len(batch), previous_tokens=self._last_activity_tokens(cases)))
        torch.manual_seed(seed + 1)  # danach unabhängig weiter
        return results, [entry[4] for entry in encoded]

    def _options_in_encoder(self, point, cases, options, now, n):
        """
        option_as "encoder": je Option je Fall n Suffixe in zwei Schritten mit denselben Zufallszahlen für alle Optionen.
        1. Aus dem Prefix bis zum Entscheidungs-Event zieht der Decoder n-mal nur das erste Event unter den Token der
           Option: ihr Abstand (bei gap_classes/gap_tokens mit seiner Klasse).
        2. Der Prefix mit dem Options-Event wird einmal kodiert; je Ziehung stehen an seiner letzten Stelle die Zeiten zum
           gezogenen Abstand (seit Fallbeginn, seit dem vorigen Event, Wochentag, Sekunden des Tages) und die passende
           Klasse bzw. das Token mit Klasse. Von dort wird der Rest gezogen.
        Mit params shared_option_gap wird nur ein erster Abstand je Fall/Option gezogen und für alle n Fortsetzungen
        wiederholt. Die übrige Suffixziehung bleibt unverändert; das vollständige Options-Prefix ist dann geteilt.
        Ausgegeben wie _options_in_decoder: die Option ist das erste Event des Suffix (ihr Abstand zählt zur Durchlaufzeit),
        die Zeit seit Fallbeginn ist die am Entscheidungs-Event.
        """
        from interventions import option_token
        if (self.params or {}).get("log_gaps", False):
            raise NotImplementedError("option_as encoder: nicht für log_gaps")
        ev = self._evaluator()
        encoders, scalers = self.encoder_decoder.categorical_encoders, self.encoder_decoder.continuous_encoders
        cat_at = {name: k for k, name in enumerate(ev.prefix_cat_attributes)}
        num_at = {name: k for k, name in enumerate(ev.prefix_num_attributes)}
        classes, tokens = self.gap_classes(), self.gap_tokens()

        def number(column, value):
            return int(encoders[column].transform(np.array([[value]], dtype=object)).reshape(-1)[0]) + 1

        plain = [self._encode(events) for events in cases]
        elapsed = np.array([entry[4] for entry in plain], dtype=float)
        base = pd.Timestamp(now, unit="s")
        seed = int(torch.randint(2 ** 31 - 1, (1,)))
        shared_gap = bool((self.params or {}).get("shared_option_gap", False))
        first_n = 1 if shared_gap else n
        results, observed_gaps = [], {}
        for option in options:
            token = option_token(option) if self.lifecycle_tokens else option_token(option).split("|")[0]
            torch.manual_seed(seed)
            firsts = self.draw_cases([entry[:4] for entry in plain], first_n, first=[token] * len(plain), max_steps=1,
                                    previous_tokens=self._last_activity_tokens(cases),
                                    work_states=self._work_states(cases))
            if getattr(self, "work_state_constraints", False) and any(not suffix for group in firsts for suffix in group):
                from prescriptive.estimators.conditional_gaps import SuffixSamplingError
                raise SuffixSamplingError(f"Unsupported or infeasible first option event: {token}")
            gaps = np.array([[suffix[0][2] if suffix else 0.0 for suffix in suffixes] for suffixes in firsts], dtype=float).reshape(-1)  # Fälle x n
            if shared_gap:
                observed_gaps[str(option)] = gaps.tolist()
                gaps = np.repeat(gaps, n)
            stamps = base + pd.to_timedelta(gaps, unit="s")
            reached = np.repeat(elapsed, n) + gaps
            last = {"cats": {}, "nums": {
                num_at[TIME_COLUMNS[0]]: self._encode_values(scalers[TIME_COLUMNS[0]], reached),
                num_at[TIME_COLUMNS[1]]: self._encode_values(scalers[TIME_COLUMNS[1]], gaps),
                num_at[TIME_COLUMNS[2]]: self._encode_values(scalers[TIME_COLUMNS[2]], np.asarray(stamps.weekday, dtype=float)),
                num_at[TIME_COLUMNS[3]]: self._encode_values(scalers[TIME_COLUMNS[3]], np.asarray(stamps.hour * 3600 + stamps.minute * 60 + stamps.second, dtype=float))}}
            if classes or tokens:
                names = gap_class_names(gaps)
                ids = {name: number(GAP_CLASS, name) for name in set(names)}
                last["cats"][cat_at[GAP_CLASS]] = np.array([ids[name] for name in names])
                if tokens:
                    ids = {name: number(ACTIVITY, f"{token}|{name}") for name in set(names)}
                    last["cats"][cat_at[ACTIVITY]] = np.array([ids[name] for name in names])
            batch = [self._encode(events, self._option_event(point, events, option, now))[:4] for events in cases]
            torch.manual_seed(seed + 2)
            states = self._work_states(cases)
            if states is not None:
                for state in states:
                    if not state.permits(token):
                        raise ValueError(f"Infeasible option {token} for the current work state")
                    state.push(token)
            ro = getattr(self, "rollout", None)  # 2026-11-23: Rollout mit Causal Forest an späteren Wahlpunkten (policies/ev_policy/rollout.py)
            force = ro.kontext(cases, token, reached, n) if ro is not None else None
            rest = self.draw_cases(batch, n, last=last, previous_tokens=[token] * len(cases), work_states=states, force=force)
            results.append([[[(token, float(reached[c * n + i]), float(gaps[c * n + i])), *rest[c][i]] for i in range(n)] for c in range(len(cases))])
        torch.manual_seed(seed + 1)  # danach unabhängig weiter
        if shared_gap:
            self.last_option_gap_draws = {"point": point.name, "gaps_seconds": observed_gaps,
                                          "draws_per_case_and_option": 1, "source": "decoder"}
        return results, [float(e) for e in elapsed]

    def _work_states(self, cases):
        if not getattr(self, "work_state_constraints", False):
            return None
        from interventions.work_state import WorkState
        return [WorkState.from_events(events) for events in cases]

    def _last_activity_tokens(self, cases):
        """Letztes Event aus dem Verlauf, unabhängig von seiner kodierten Zeitklasse."""
        return [str(events[-1][ACTIVITY]) + ("|" + str(events[-1].get(LIFECYCLE, "complete")) if self.lifecycle_tokens else "")
                for events in cases]

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
        if getattr(self, "net", None) is not None:
            self.net.encoder.__dict__.pop("kept_context", None)  # decoder_context: Zustand des letzten Batches
            self.net.encoder.__dict__.pop("kept_static", None)  # static_in_decoder: Merkmale des letzten Batches
            self.net.__dict__.pop("_inference_counts", None)  # Zähler: Zustand der letzten Ziehung
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
