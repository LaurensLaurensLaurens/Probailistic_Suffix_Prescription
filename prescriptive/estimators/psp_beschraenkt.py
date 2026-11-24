"""
Beschränktes Ziehen für PSP (Experiment 2026-11-17, Anregung des Nutzers): beim Ziehen der Suffixe nur Abläufe, wie sie im echten
Log vorkommen. Zwei aus den Trainingsfällen gelernte Übergangsmodelle:
- zwischen Events: Token (Aktivität|Lifecycle) → nächstes Token; Fallende (EOS) nur nach Tokens, nach denen ein Trainingsfall
  endet (vorhandener Mechanismus transition_rules von PSPModel)
- innerhalb einer Aktivität: letzter Lifecycle derselben Aktivität im Fall (oder „keins“) → erlaubte nächste Lifecycles
  (z. B. suspend → resume, kein suspend nach ate_abort, A_Pending nur einmal), Zustand aus Prefix und gezogenen Events
Die Wahrscheinlichkeiten des Netzes werden auf die erlaubten Tokens renormiert (Logits der übrigen auf −∞); die Zeit zieht weiter
der Decoder je Schritt (bei time_on_event aus den Köpfen, die das gezogene Event sehen) → Loss Attenuation, Dropout-Masken und
im Decoder gezogene Suffixe mit Event und Zeit bleiben. Kein Neuziehen unvollständiger Verläufe: gleiche Zahl Zufallszahlen je
Schritt für alle Optionen, die gemeinsamen Zufallszahlen bleiben gekoppelt. Gibt es für eine Zeile kein erlaubtes Token, endet
sie (gezählt in self.ablauf_statistik["ohne_erlaubtes_token"]).

Nur für Modelle mit normalen Abständen (wie (c)) und time_on_event (ZZ); nicht für Zeitklassen/Zeit-Token/Variante (e) und nicht
zusammen mit work_state_constraints. PSPModel bleibt unverändert; ein beschränktes Modell ist eine gespeicherte Kopie mit dieser
Klasse:
    python -m prescriptive.estimators.psp_beschraenkt <modell.pkl> <ziel.pkl>      (im Projektordner, Pfade wie add_paths)
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

from prescriptive.estimators.psp_model import ACTIVITY, CASE, LIFECYCLE, TIMESTAMP, TIME_COLUMNS, PSPModel

KEINS = "keins"
GESCHAEFTSENDE = ("A_Pending", "A_Denied", "A_Cancelled")  # jeder Trainingsfall enthält eines (regeln_lernen prüft den Anteil)


def regeln_lernen(log, cases):
    """Übergangsregeln aus den Fällen cases des Logs: Paare zwischen Events, Endtokens, Lifecycle-Folgen je Aktivität."""
    part = log[log[CASE].astype(str).isin({str(c) for c in cases})].sort_values([CASE, TIMESTAMP], kind="stable")
    tokens = part[ACTIVITY].astype(str).str.strip()+"|"+part[LIFECYCLE].astype(str)
    pairs, ends, within = set(), set(), {}
    for _, seq in tokens.groupby(part[CASE], sort=False):
        seq = list(seq)
        pairs.update(zip(seq[:-1], seq[1:]))
        ends.add(seq[-1])
        last = {}
        for tok in seq:
            activity, _, lifecycle = tok.rpartition("|")
            within.setdefault((activity, last.get(activity, KEINS)), set()).add(lifecycle)
            last[activity] = lifecycle
    share = part.groupby(CASE)[ACTIVITY].apply(lambda a: a.astype(str).str.strip().isin(GESCHAEFTSENDE).any()).mean()
    print(f"regeln_lernen: Anteil Trainingsfälle mit Geschäftsende {share:.4f}")
    return {"pairs": sorted(pairs), "ends": sorted(ends)}, {key: sorted(value) for key, value in within.items()}


class BeschraenktesPSP(PSPModel):
    """PSPModel, dessen draw_cases nur gültige Übergänge zwischen und innerhalb von Aktivitäten zieht (siehe Modulbeschreibung)."""

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_lifecycle_cache", None)
        state.pop("ablauf_statistik", None)
        return state

    def _lifecycle_mask(self, state, width, eos_id, names):
        """Erlaubte Decoder-Tokens für einen Lifecycle-Zustand (dict Aktivität → letzter Lifecycle), zwischengespeichert."""
        cache = self.__dict__.setdefault("_lifecycle_cache", {})
        key = tuple(sorted(state.items()))
        row = cache.get(key)
        if row is None:
            row = torch.zeros(width, dtype=torch.bool)
            for number, label in names.items():
                number = int(number)
                if number == eos_id:
                    row[number] = True  # das Fallende regelt die Maske zwischen Events (nur nach Endtokens)
                    continue
                activity, _, lifecycle = str(label).rpartition("|")
                if getattr(self, "ohne_lifecycle_regeln", False) or \
                        lifecycle in self.lifecycle_rules.get((activity, state.get(activity, KEINS)), ()):
                    row[number] = True
            cache[key] = row
        return row

    def _sample_cats(self, ev, cat_means, cat_variances):
        """Eventwahl wie sample_cat_predictions_optim; mit gumbel_kopplung über argmax(Logits + Gumbel-Rauschen je Token) statt
        multinomial: dieselbe Verteilung, aber beide Optionen teilen das Rauschen je Token (gemeinsame Zufallszahlen) und ziehen
        dasselbe Token, solange sich die Rangfolge nicht ändert → engere Kopplung."""
        if not getattr(self, "gumbel_kopplung", False) or ev.sample_argmax:
            return type(ev).sample_cat_predictions_optim(ev.all_cat_attributes, ev.use_variance_cat, ev.sample_argmax, cat_means, cat_variances)
        result = {}
        for c in ev.all_cat_attributes:
            logits = cat_means[c + "_mean"]
            if ev.use_variance_cat:
                logvar = torch.clamp(cat_variances[c + "_var"], min=-6.0, max=6.0)
                logits = torch.normal(logits, torch.exp(0.5 * logvar))
            uniform = torch.rand_like(logits).clamp_(1e-12, 1 - 1e-12)
            result[c + "_mean"] = torch.argmax(logits - torch.log(-torch.log(uniform)), dim=-1, keepdim=True)
        return result

    def draw_cases(self, encoded, n, until=None, stop=None, max_steps=None, first=None, last=None,
                   completion_steps=None, previous_tokens=None, work_states=None, _state_retry=0):
        """Wie PSPModel.draw_cases (normale Abstände, optional time_on_event), dazu die Lifecycle-Maske je Zeile."""
        if self.gap_classes() or self.gap_tokens() or self.gap_condition_on_event() or getattr(self, "work_state_constraints", False) \
                or getattr(self, "conditional_gaps", None) is not None or self.params.get("log_gaps", False):
            raise ValueError("BeschraenktesPSP: nur normale Abstände (wie (c)) bzw. time_on_event, ohne Zustandsregeln")
        ev, model = self._evaluator(), self.net
        names = ev.inverted_suffix_categories[0]
        width = max(names) + 1
        scalers = {name: self.encoder_decoder.continuous_encoders[name] for name in ev.all_num_attributes}
        window, suffix_size = self.encoder_decoder.window_size, self.encoder_decoder.min_suffix_size
        steps = np.repeat([max(window - suffix_size - length + 1, 1) for _, _, _, length in encoded], n)
        if completion_steps is not None:
            steps = np.full(len(steps), completion_steps)
        steps = np.minimum(steps, max_steps) if max_steps is not None else steps
        limit = np.repeat(np.full(len(encoded), np.inf) if until is None else np.asarray(until, dtype=float), n)
        rows = n * len(encoded)
        stats = self.__dict__.setdefault("ablauf_statistik", {"zeilen": 0, "ohne_erlaubtes_token": 0, "schritte": 0,
                                                               "masse_gesperrt": 0.0})
        allowed = None
        if first is not None:  # option_as decoder: das erste Event ist vorgegeben
            masks = {}
            for token in set(first):
                masks[token] = torch.zeros(width, dtype=torch.bool)
                for number, name in names.items():
                    if str(name) == token:
                        masks[token][int(number)] = True
            allowed = torch.stack([masks[token] for token in first]).repeat_interleave(n, dim=0)
        cats, nums, batch_static, mask = self._batch(encoded, n)
        if last is not None:
            for column, ids in last.get("cats", {}).items():
                cats[column][:, -1] = torch.as_tensor(np.asarray(ids), dtype=cats[column].dtype)
            for column, values in last.get("nums", {}).items():
                nums[column][:, -1] = torch.as_tensor(values, dtype=nums[column].dtype).reshape(-1)
        # Lifecycle-Zustand je Zeile aus dem Prefix (nach dem Einsetzen der Option)
        column = ev.prefix_cat_attributes.index(ACTIVITY)
        vocabulary = self.encoder_decoder.categorical_encoders[ACTIVITY].categories_[0]
        states = []
        for ids in cats[column].tolist():
            state = {}
            for number in ids:
                if int(number) > 0:
                    activity, _, lifecycle = str(vocabulary[int(number)-1]).rpartition("|")
                    state[activity] = lifecycle
            states.append(state)
        business_end = np.array([any(a in state for a in GESCHAEFTSENDE) for state in states])
        need_end = getattr(self, "ende_erst_nach_geschaeftsende", True)
        transitions = self._transition_masks(ev)  # zwischen Events (self.transition_rules)
        if previous_tokens is not None:
            transition_allowed = torch.stack([transitions["by_token"][token] for token in previous_tokens]).repeat_interleave(n, dim=0)
        else:
            transition_allowed = transitions["prefix"][cats[column][:, -1].long()]
        with torch.no_grad():
            prediction, hidden, _ = model.inference(prefix=[cats, nums], static_inputs=batch_static, mask=mask)
            reached = self._decode(scalers[TIME_COLUMNS[0]], nums[ev.prefix_num_attributes.index(TIME_COLUMNS[0])][:, -1])
            suffixes, done = [[] for _ in range(rows)], np.zeros(rows, dtype=bool)
            for step in range(int(steps.max())):
                cat_means = prediction[0][0]
                key = ACTIVITY + "_mean"
                permitted = transition_allowed.clone()
                for k in np.flatnonzero(~done):
                    permitted[k] &= self._lifecycle_mask(states[k], width, ev.eos_id, names)
                if need_end:  # Fallende erst nach einem Geschäftsende
                    permitted[torch.from_numpy(~business_end), ev.eos_id] = False
                if step == 0 and allowed is not None:
                    permitted &= allowed
                inactive = torch.from_numpy(done.copy())
                permitted[inactive] = False
                permitted[inactive, ev.eos_id] = True
                missing = ~permitted.any(dim=1)
                if missing.any():
                    stats["ohne_erlaubtes_token"] += int(missing.sum())
                    permitted[missing, ev.eos_id] = True
                logits = cat_means[key].reshape(rows, -1)
                active = torch.from_numpy(~done)
                if active.any():  # Anteil der Wahrscheinlichkeit, die das Netz auf gesperrte Tokens legt (Diagnose)
                    probs = torch.softmax(logits[active], dim=-1)
                    stats["masse_gesperrt"] += float((probs * (~permitted[active]).float()).sum())
                    stats["schritte"] += int(active.sum())
                cat_means = {**cat_means, key: cat_means[key].masked_fill(~permitted.reshape(cat_means[key].shape), float("-inf"))}
                cat_values = self._sample_cats(ev, cat_means, prediction[1][0])
                if self.time_on_event():
                    means, logvars = model.decoder.numeric_prediction(hidden[0], cat_values[key].reshape(-1))
                    num_values = self._clip_gaps(self._draw_times(ev, means, logvars))
                else:
                    num_values = self._clip_gaps(self._draw_times(ev, prediction[0][1], prediction[1][1]))
                activity = cat_values[key].reshape(-1).numpy()
                transition_allowed = transitions["decoder"][torch.as_tensor(activity, dtype=torch.long)]
                done |= (activity == ev.eos_id) | (step >= steps)
                if done.all():
                    break
                labels = [str(names.get(int(a), "<unbekannt>")) for a in activity]
                reached = np.maximum(reached, self._decode(scalers[TIME_COLUMNS[0]], num_values[TIME_COLUMNS[0] + "_mean"]))
                gap = self._decode(scalers[TIME_COLUMNS[1]], num_values[TIME_COLUMNS[1] + "_mean"])
                for k in np.flatnonzero(~done):
                    suffixes[k].append((labels[k], float(reached[k]), max(float(gap[k]), 0.0)))
                    act, _, lifecycle = labels[k].rpartition("|")
                    states[k][act] = lifecycle
                    business_end[k] |= act in GESCHAEFTSENDE
                done |= reached > limit
                if stop is not None:
                    done |= np.array([label.split("|")[0] for label in labels], dtype=object) == stop
                prediction, hidden = model.inference(last_event=([cat_values[k].reshape(rows, 1) for k in cat_values],
                                                                 [num_values[k].reshape(rows, 1) for k in num_values]), hx=hidden)
        stats["zeilen"] += rows
        return [suffixes[k * n:(k + 1) * n] for k in range(len(encoded))]


def beschraenken(source, target, gumbel=False, masken=True):
    """Kopie des Modells source mit beschränktem Ziehen; Regeln aus den Trainingsfällen (alle Fälle des Logs außer test_cases).
    gumbel: Gumbel-Kopplung der Eventwahl; masken=False: nur Gumbel-Kopplung, ohne Übergangsregeln (zum Vergleich)."""
    model = PSPModel.load(source)
    raw = pd.read_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cache", "bpic17_raw.pkl"))
    raw[TIMESTAMP] = pd.to_datetime(raw[TIMESTAMP], utc=True)
    test = {str(c) for c in model.test_cases}
    training = [c for c in raw[CASE].astype(str).unique() if c not in test]
    model.transition_rules, model.lifecycle_rules = regeln_lernen(raw, training)
    if not masken:  # alles erlaubt: Paare aller Tokens, jedes Token darf enden, jeder Lifecycle-Wechsel
        tokens = sorted({t for pair in model.transition_rules["pairs"] for t in pair})
        model.transition_rules = {"pairs": [(a, b) for a in tokens for b in tokens], "ends": tokens}
        model.lifecycle_rules = {}
        model.ende_erst_nach_geschaeftsende = False
        model.ohne_lifecycle_regeln = True
    model.gumbel_kopplung = bool(gumbel)
    model.__class__ = BeschraenktesPSP
    model.save(target)
    return model


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from prescriptive.helpers.datasets import add_paths
    add_paths()
    built = beschraenken(sys.argv[1], sys.argv[2])
    print(f"{sys.argv[2]}: {len(built.transition_rules['pairs'])} Übergänge, {len(built.transition_rules['ends'])} Endtokens, "
          f"{len(built.lifecycle_rules)} Lifecycle-Zustände")
