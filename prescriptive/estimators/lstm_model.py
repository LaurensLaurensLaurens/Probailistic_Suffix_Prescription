"""
LSTM-Modell der Policy lstm (policies/lstm_policy.py), als S-Learner wie estimators/xgb_model.py; gespeichert in
models/lstm_model.pkl (mit gewähltem Outcome z. B. lstm_model_gewinn.pkl): je Wahlpunkt (interventions.choice_points)
ein Netz, das den Outcome des Falls aus der Event-Sequenz bis zum Wahlpunkt, den Fallattributen, dem Zeitpunkt der Wahl
und der Option schätzt. Die Policy nimmt die Option mit dem höchsten geschätzten Outcome.

Proben (fit) wie beim XGBoost-Modell, aus einem Event-Log mit der Spalte outcome, jedes Erreichen eines Wahlpunkts eine:
- control_flow (BPIC17 channel): jedes complete-Event der Aktivität des Punkts, nach dem direkt eine der Optionen kommt;
  die Sequenz bis einschließlich dieses Events, Option = die folgende Aktivität
- employee (BPIC17 W_Call incomplete files, start): jeder Beginn des Arbeitsschritts durch einen Mitarbeiter (kein
  Systemnutzer); die Sequenz bis vor dieses Event, Option = der Mitarbeiter, jeder mit eigener Option (ein Mitarbeiter,
  der im Training an diesem Wahlpunkt nie vorkam, bekommt die Option "unbekannt" mit festem Null-Vektor)

Eingaben (event_arrays und case_vector, für Training und Abfrage dieselben):
- je Event (die letzten MAX_LENGTH): Token (Aktivität|Lifecycle, Embedding), Stunden seit Fallbeginn und seit dem
  vorigen Event (log1p), Wochentag, Stunde, die aktuellen Werte der Event-Attribute (screening im Katalog:
  event_numeric; BPIC17 das letzte Angebot; log1p, 0 vor dem ersten) und wie oft sie gesetzt wurden
- je Probe: Fallattribute (case_attributes als Embeddings, numeric_attributes log1p), zum Zeitpunkt der Wahl die Stunden
  seit Fallbeginn und seit dem letzten Event (log1p), Wochentag und Stunde, die Option (Embedding)
Ein LSTM fasst die Sequenz zusammen, ein kleines MLP schätzt aus Zusammenfassung, Fall und Option den standardisierten
Outcome (MSE, frühes Stoppen auf VALIDATION_SHARE der Fälle).

    model = LSTMModel("bpic17").fit(log)          # log: Event-Log mit Spalte outcome
    model.save("prescriptive/models/lstm_model.pkl")
    LSTMModel.load(path).values(point, events, options, now)   # geschätzter Outcome je Option
"""
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from torch import nn

CASE, ACTIVITY, LIFECYCLE, RESOURCE, TIMESTAMP = "case:concept:name", "concept:name", "lifecycle:transition", "org:resource", "time:timestamp"
MAX_LENGTH = 100  # höchstens so viele letzte Events je Sequenz
VALIDATION_SHARE = 0.1  # Anteil der Fälle für das frühe Stoppen
TRAIN = {"hidden": 64, "epochs": 40, "batch_size": 512, "learning_rate": 1e-3, "weight_decay": 1e-5, "patience": 4, "threads": 8}


def _seconds(values):
    """Zeitstempel als Sekunden seit 1970 (UTC; ohne Zeitzone gelten sie als UTC, wie in der Engine)."""
    times = pd.to_datetime(pd.Series(values))
    if times.dt.tz is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    return (times - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()


def _calendar(seconds):
    """Wochentag / 6 und Stunde / 24 zu Sekunden seit 1970."""
    moments = pd.to_datetime(np.asarray(seconds, dtype=float), unit="s")
    return np.asarray(moments.weekday, dtype=float) / 6, (np.asarray(moments.hour, dtype=float) + np.asarray(moments.minute, dtype=float) / 60) / 24


def _hours(seconds):
    return np.log1p(np.clip(np.asarray(seconds, dtype=float), 0, None) / 3600)


class OutcomeNet(nn.Module):
    """LSTM über die Events, dann ein MLP auf (letzter Zustand, Fallattribute, Zahlen der Probe, Option) -> Outcome."""

    def __init__(self, n_tokens, category_sizes, n_event_nums, n_case_nums, n_options, hidden):
        super().__init__()
        self.token = nn.Embedding(n_tokens + 2, 16, padding_idx=0)  # 0: Auffüllen, n_tokens + 1: unbekannt
        self.lstm = nn.LSTM(16 + n_event_nums, hidden, batch_first=True)
        self.attributes = nn.ModuleList([nn.Embedding(size + 1, min(8, size + 1)) for size in category_sizes])  # 0: unbekannt
        self.option = nn.Embedding(n_options + 1, 8, padding_idx=0)  # 0: unbekannte Option (fester Null-Vektor)
        width = hidden + sum(min(8, size + 1) for size in category_sizes) + n_case_nums + 8
        self.head = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, ids, nums, lengths, categories, case_nums, option):
        x = torch.cat([self.token(ids), nums], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        parts = [h[-1]] + [embedding(categories[:, i]) for i, embedding in enumerate(self.attributes)] + [case_nums, self.option(option)]
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


class LSTMModel:
    def __init__(self, dataset, points=None, seed=42, params=None):
        """dataset: Schlüssel im Katalog; points: die Wahlpunkte (Standard: interventions.choice_points des Datensatzes)."""
        import interventions
        screening = interventions.catalog()[dataset]["screening"]
        self.dataset, self.seed, self.params = dataset, seed, {**TRAIN, **(params or {})}
        self.points = {point.name: point for point in (points if points is not None else interventions.choice_points(dataset))}
        self.categorical, self.numeric = list(screening["case_attributes"]), list(screening.get("numeric_attributes") or [])
        self.event_numeric = list(screening.get("event_numeric") or [])
        self.system_users = screening.get("system_users")
        self.vocab, self.categories = None, None
        self.nets, self.scales, self.options, self.summary = {}, {}, {}, {}

    # ------------------------------------------------------------------ Eingaben

    def event_arrays(self, frame):
        """
        Je Event (frame: Events eines oder mehrerer Fälle, nach Fall und Zeit sortiert) die Token-Nummer und die Zahlen;
        dazu die Sekunden seit 1970 je Event.
        """
        seconds = _seconds(frame[TIMESTAMP])
        case = frame[CASE].to_numpy()
        lifecycle = frame[LIFECYCLE].astype(str) if LIFECYCLE in frame else pd.Series("complete", index=frame.index)
        tokens = (frame[ACTIVITY].astype(str) + "|" + lifecycle).to_numpy()
        ids = np.array([self.vocab.get(token, len(self.vocab) + 1) for token in tokens], dtype=np.int64)
        by_case = pd.Series(seconds, index=frame.index).groupby(case)
        start = by_case.transform("min").to_numpy()
        previous = by_case.shift(1).fillna(pd.Series(seconds, index=frame.index)).to_numpy()
        values = frame.reindex(columns=self.event_numeric).apply(pd.to_numeric, errors="coerce")
        current = values.groupby(case).ffill().fillna(0.0).to_numpy(dtype=float)
        n_set = values.notna().any(axis=1).astype(float).groupby(case).cumsum().to_numpy()
        weekday, hour = _calendar(seconds)
        nums = np.column_stack([_hours(seconds - start), _hours(seconds - previous), weekday, hour,
                                np.log1p(np.clip(current, 0, None)), n_set]).astype(np.float32)
        return ids, nums, seconds

    def case_vector(self, first, start, last, now):
        """Fallattribute (Nummern der Kategorien, 0 = unbekannt) und die Zahlen der Probe zum Zeitpunkt now."""
        categories = np.array([self.categories[name].get(str(first.get(name)), 0) for name in self.categorical], dtype=np.int64)
        numeric = [np.log1p(max(float(first.get(name)), 0.0)) if first.get(name) is not None and not pd.isna(first.get(name)) else 0.0
                   for name in self.numeric]
        weekday, hour = _calendar([now])
        case_nums = np.array(numeric + [_hours(now - start), _hours(now - last), weekday[0], hour[0]], dtype=np.float32)
        return categories, case_nums

    def _sample(self, ids, nums, seconds, first, begin, cut, now):
        """Eine Probe: die letzten MAX_LENGTH Events vor cut (ab begin, dem ersten Event des Falls) und der Fall zum Zeitpunkt now."""
        low = max(begin, cut - MAX_LENGTH)
        last = seconds[cut - 1] if cut > begin else now
        return (ids[low:cut], nums[low:cut]) + self.case_vector(first, seconds[begin], last, now)

    # ------------------------------------------------------------------ Training

    def samples(self, log):
        """Je Wahlpunkt die Proben des Logs: Liste von (Fall, Probe, Option); dieselben Stellen wie XGBModel.samples."""
        log = log.sort_values([CASE, TIMESTAMP], kind="stable").reset_index(drop=True)
        ids, nums, seconds = self.event_arrays(log)
        lifecycle = log[LIFECYCLE].to_numpy() if LIFECYCLE in log else np.full(len(log), "complete")
        resource = log[RESOURCE].to_numpy() if RESOURCE in log else np.full(len(log), None)
        activity, case = log[ACTIVITY].to_numpy(), log[CASE].to_numpy()
        first_rows = log.groupby(CASE, sort=False).head(1).set_index(CASE)
        system = re.compile(self.system_users) if self.system_users else None
        control = {point.activity: point for point in self.points.values() if point.kind == "control_flow"}
        employee = {(point.activity, point.lifecycle): point for point in self.points.values() if point.kind == "employee"}
        samples = {name: [] for name in self.points}
        bounds = np.flatnonzero(np.r_[True, case[1:] != case[:-1], True])
        for begin, end in zip(bounds[:-1], bounds[1:]):
            first = first_rows.loc[case[begin]]
            for i in range(begin, end):
                point = employee.get((activity[i], lifecycle[i]))
                if point is not None and isinstance(resource[i], str) and not (system and system.search(resource[i])):
                    # Mitarbeiterwahl: der Fall vor dem Beginn, so wie die Engine fragt. An einem complete-Event (Engine ohne
                    # Lifecycles) fragt sie am vorigen Event: die Zeit bis hierher hängt schon von der Wahl ab
                    decided = seconds[i - 1] if point.lifecycle == "complete" and i > begin else seconds[i]
                    samples[point.name].append((case[i], self._sample(ids, nums, seconds, first, begin, i, decided), resource[i]))
                point = control.get(activity[i]) if lifecycle[i] == "complete" else None
                if point is not None and i + 1 < end and activity[i + 1] in point.options:
                    # Control Flow: der Fall bis einschließlich des Entscheidungs-Events, Option = das folgende Event
                    samples[point.name].append((case[i], self._sample(ids, nums, seconds, first, begin, i + 1, seconds[i]), activity[i + 1]))
        return samples

    def _batch(self, point_name, samples, options):
        """Tensoren eines Stapels: Sequenzen aufgefüllt, Zahlen standardisiert, Optionen als Nummern (0 = unbekannt)."""
        scale, index = self.scales[point_name], {option: k + 1 for k, option in enumerate(self.options[point_name])}
        lengths = np.array([max(len(sample[0]), 1) for sample in samples])
        width = int(lengths.max())
        ids = np.zeros((len(samples), width), dtype=np.int64)
        nums = np.zeros((len(samples), width, len(scale["nums_mean"])), dtype=np.float32)
        for row, sample in enumerate(samples):
            n = len(sample[0])
            ids[row, :n] = sample[0]
            nums[row, :n] = (sample[1] - scale["nums_mean"]) / scale["nums_std"]
        categories = np.stack([sample[2] for sample in samples])
        case_nums = (np.stack([sample[3] for sample in samples]) - scale["case_mean"]) / scale["case_std"]
        option = np.array([index.get(option, 0) for option in options], dtype=np.int64)
        return (torch.from_numpy(ids), torch.from_numpy(nums), torch.from_numpy(lengths), torch.from_numpy(categories),
                torch.from_numpy(case_nums.astype(np.float32)), torch.from_numpy(option))

    def fit(self, log):
        """Ein Netz je Wahlpunkt mit Proben; log: Event-Log mit Spalte outcome. Liefert self (summary: je Punkt Proben, Epochen, RMSE)."""
        torch.set_num_threads(self.params["threads"])
        lifecycle = log[LIFECYCLE].astype(str) if LIFECYCLE in log else pd.Series("complete", index=log.index)
        self.vocab = {token: k + 1 for k, token in enumerate(sorted((log[ACTIVITY].astype(str) + "|" + lifecycle).unique()))}
        self.categories = {name: {value: k + 1 for k, value in enumerate(sorted(log[name].dropna().astype(str).unique()))} for name in self.categorical}
        outcome = log.groupby(CASE)["outcome"].first()
        for name, rows in self.samples(log).items():
            point = self.points[name]
            if not rows:
                continue
            cases, samples, options = zip(*rows)
            # Control Flow: die Optionen des Punkts; Mitarbeiter: jeder, der hier im Training vorkam
            self.options[name] = list(point.options) if point.kind == "control_flow" else sorted(set(options))
            y = outcome.reindex(list(cases)).to_numpy(dtype=float)
            steps = np.concatenate([sample[1] for sample in samples if len(sample[1])])
            case_nums = np.stack([sample[3] for sample in samples])
            self.scales[name] = {"nums_mean": steps.mean(0), "nums_std": steps.std(0) + 1e-6, "case_mean": case_nums.mean(0),
                                 "case_std": case_nums.std(0) + 1e-6, "y_mean": float(y.mean()), "y_std": float(y.std() + 1e-6)}
            # Frühes Stoppen auf einem Teil der Fälle (alle Proben eines Falls auf derselben Seite)
            unique = sorted(set(cases))
            validation = set(np.array(unique, dtype=object)[np.random.default_rng(self.seed).random(len(unique)) < VALIDATION_SHARE])
            held = np.array([c in validation for c in cases])
            torch.manual_seed(self.seed)
            net = OutcomeNet(len(self.vocab), [len(self.categories[c]) for c in self.categorical], steps.shape[1], case_nums.shape[1],
                             len(self.options[name]), self.params["hidden"])
            optimizer = torch.optim.Adam(net.parameters(), lr=self.params["learning_rate"], weight_decay=self.params["weight_decay"])
            target = torch.from_numpy(((y - self.scales[name]["y_mean"]) / self.scales[name]["y_std"]).astype(np.float32))
            train_index, val_index = np.flatnonzero(~held), np.flatnonzero(held)
            val_batch = self._batch(name, [samples[k] for k in val_index], [options[k] for k in val_index])
            rng = np.random.default_rng(self.seed)
            best, best_state, waited, epochs = float("inf"), None, 0, 0
            for epoch in range(self.params["epochs"]):
                net.train()
                for part in np.array_split(rng.permutation(train_index), max(1, len(train_index) // self.params["batch_size"])):
                    batch = self._batch(name, [samples[k] for k in part], [options[k] for k in part])
                    optimizer.zero_grad()
                    loss = nn.functional.mse_loss(net(*batch), target[part])
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()
                net.eval()
                with torch.no_grad():
                    val_loss = float(nn.functional.mse_loss(net(*val_batch), target[val_index]))
                epochs = epoch + 1
                if val_loss < best - 1e-4:
                    best, best_state, waited = val_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
                else:
                    waited += 1
                    if waited >= self.params["patience"]:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.nets[name] = net
            predicted = self.predict(point, [samples[k] for k in val_index], [options[k] for k in val_index])
            self.summary[name] = {"Proben": len(y), "Fälle": len(set(cases)), "Optionen": len(self.options[name]), "Epochen": epochs,
                                  "RMSE Validierung": float(np.sqrt(np.mean((predicted - y[held]) ** 2))), "Streuung Outcome": float(np.std(y[held]))}
        return self

    # ------------------------------------------------------------------ Abfrage

    def knows(self, point):
        return point.name in self.nets

    def predict(self, point, samples, options):
        """Geschätzter Outcome je Probe (samples aus samples bzw. values) mit der Option je Probe."""
        result = []
        for begin in range(0, len(samples), 4096):
            batch = self._batch(point.name, samples[begin:begin + 4096], options[begin:begin + 4096])
            with torch.no_grad():
                result.append(self.nets[point.name](*batch).numpy())
        scale = self.scales[point.name]
        return np.concatenate(result) * scale["y_std"] + scale["y_mean"] if result else np.zeros(0)

    def values(self, point, events, options, now):
        """
        Geschätzter Outcome je Option (Array in der Reihenfolge von options). events: die Events des Falls bis jetzt
        (Liste von dicts wie engine.events, in Reihenfolge); now: Zeitpunkt der Wahl in Sekunden seit 1970.
        """
        frame = pd.DataFrame(events)
        ids, nums, seconds = self.event_arrays(frame)
        sample = self._sample(ids, nums, seconds, events[0], 0, len(events), now)
        return self.predict(point, [sample] * len(options), list(options))

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
