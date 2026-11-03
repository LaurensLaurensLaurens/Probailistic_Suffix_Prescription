# Auswertung der Suffix-Vorhersage des PSP, nach dem Vorbild von probabilistic_suffix_prediction_U-ED-LSTM
# (src/evaluation_v2/probabilistic_evaluation.py und src/evaluation_metrics/metrics.py).
#
# Zu jedem Prefix gibt es wie dort zwei Vorhersagen:
#   mean -> ein Suffix ohne Dropout, je Schritt die wahrscheinlichste Aktivität und der vorhergesagte Mittelwert
#   prob -> samples_per_case gezogene Suffixe (PSP.rollout)
# Darauf werden die Metriken des PSP berechnet (Suffixlänge, DLS, letzter Wert wachsender Attribute, PIT, Intervalle),
# dazu dieselben Kennzahlen für das Outcome am EOS-Event. Alle Werte bleiben in der skalierten Einheit der Tensoren.
from collections import Counter
from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def damerau_levenshtein_distance(a, b):
    # Optimal-String-Alignment-Distanz zweier Tupel (Einfügen, Löschen, Ersetzen, Vertauschen von Nachbarn),
    # wie pyxdameraulevenshtein im PSP
    previous_previous, previous = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                current[j] = min(current[j], previous_previous[j - 2] + 1)
        previous_previous, previous = previous, current
    return previous[len(b)]


def damerau_levenshtein_similarity(a, b):
    # DLS wie NormalizedDamerauLevenshteinDistance im PSP: 1 = identische Folge. Zwei leere Folgen sind identisch.
    longest = max(len(a), len(b))
    return 1.0 if longest == 0 else 1 - damerau_levenshtein_distance(a, b) / longest


class PSPSuffixEvaluation:
    def __init__(self, model, model_params, samples_per_case=100, batch_size=256, growing_num_values=None, percentiles=(0.5, 0.75, 0.95)):
        """
        model: trained PSP network, model_params: its model parameters.
        samples_per_case: number of sampled suffixes per prefix.
        growing_num_values: numerical event attributes that never decrease (e.g. elapsed_time);
            their last value is evaluated like the remaining time in the PSP.
        percentiles: widths of the central intervals whose coverage is reported.
        """
        self.model = model
        self.samples_per_case = samples_per_case
        self.batch_size = batch_size
        self.percentiles = percentiles
        self.n_classes = model_params["dim_t"] if model_params["dim_t"] > 1 else 2
        suffix_cols = model_params["suffix_cols_encoded"]
        self.num_names = [suffix_cols[i] for i in model.suffix_num_idx]
        self.growing_num_values = [name for name in (growing_num_values or []) if name in self.num_names]

    def interval_coverage(self, samples, true_value):
        # Liegt der echte Wert im zentralen Intervall der Stichproben? (wie LastValueInterval / SuffixCountInterval)
        coverage = {}
        for percentile in self.percentiles:
            bottom = torch.quantile(samples, 0.5 - percentile / 2, dim=1)
            top = torch.quantile(samples, 0.5 + percentile / 2, dim=1)
            coverage[int(round(percentile * 100))] = ((true_value >= bottom) & (true_value <= top)).float()
        return coverage

    @torch.no_grad()
    def evaluate(self, data):
        """
        data: dict with X_case, X_event, prefix_len, T, Y, X_suffix and suffix_len (as in ModelTrainer).
        Returns a dict with
            "per_prefix": metric name -> tensor [n_prefixes],
            "summary": metric name -> mean over all prefixes,
            "by_prefix_len" / "by_suffix_len": length -> {"count": ..., metric name -> mean}.
        Metric names ending in "/mean" belong to the most likely suffix, "/prob" to the sampled suffixes.
        """
        model = self.model
        was_training = model.training
        model.eval()

        n = data["X_event"].shape[0]
        max_steps = data["X_suffix"].shape[2]
        t = data["T"][:, 1:] if self.n_classes > 2 else data["T"]
        y = data["Y"].view(n, -1).float()
        true_len = data["suffix_len"].long()

        # Echte Suffixe wie in forward_suffix: [n, max_steps, C] ohne die Outcome-Spalte
        suffix = data["X_suffix"][:, :-1].permute(0, 2, 1)
        true_activity_one_hot = suffix[:, :, model.suffix_activity_idx]
        true_activity = (true_activity_one_hot.argmax(dim=2) + 1) * (true_activity_one_hot.sum(dim=2) > 0).long()
        true_nums = suffix[:, :, model.suffix_num_idx]
        position_correct = torch.zeros(max_steps, dtype=torch.long)
        position_count = torch.zeros(max_steps, dtype=torch.long)
        activity_counts = {}
        activity_correct = {}

        def last_value(event_attributes, suffix_len, prefix_last, j):
            # Wert des Attributs j am letzten echten Event; ohne Events der Wert des letzten Prefix-Events
            position = (suffix_len - 1).clamp(min=0)
            value = event_attributes[..., j].gather(-1, position.unsqueeze(-1)).squeeze(-1)
            return torch.where(suffix_len > 0, value, prefix_last)

        batches = []
        for start in range(0, n, self.batch_size):
            batch = slice(start, start + self.batch_size)
            inputs = dict(x_case=data["X_case"][batch], x_process=data["X_event"][batch], prefix_len=data["prefix_len"][batch], t=t[batch],
                          max_steps=max_steps, growing_num_values=self.growing_num_values)
            # Wahrscheinlichstes Suffix ohne Dropout und die gezogenen Suffixe
            with model.dropout_disabled():
                mean_roll = model.rollout(n_samples=1, sample=False, **inputs)
            roll = model.rollout(n_samples=self.samples_per_case, **inputs)
            metrics = {}
            batch_len = true_len[batch]

            # Suffixlänge (wie SuffixCountMAE)
            mean_len, sampled_len = mean_roll["suffix_len"][:, 0], roll["suffix_len"]
            metrics["suffix_length_mae/mean"] = (batch_len - mean_len).abs().float()
            metrics["suffix_length_mae/prob"] = (batch_len - sampled_len.float().mean(dim=1)).abs()
            for percentile, covered in self.interval_coverage(sampled_len.float(), batch_len.float()).items():
                metrics[f"suffix_length_interval_{percentile}"] = covered

            # Aktivitätsfolge (wie NormalizedDamerauLevenshteinDistanceMeanVar). Gleiche gezogene Folgen werden nur einmal verglichen.
            dls_mean, dls_prob = [], []
            position_mask = torch.arange(max_steps, device=batch_len.device).unsqueeze(0) < batch_len.unsqueeze(1)
            predicted_activities = mean_roll["activities"][:, 0]
            position_correct += ((predicted_activities == true_activity[batch]) & position_mask).sum(dim=0).cpu()
            position_count += position_mask.sum(dim=0).cpu()
            true_labels = true_activity[batch][position_mask].cpu()
            predicted_labels = predicted_activities[position_mask].cpu()
            for label in torch.unique(true_labels).tolist():
                selected = true_labels == label
                activity_counts[label] = activity_counts.get(label, 0) + int(selected.sum())
                activity_correct[label] = activity_correct.get(label, 0) + int((predicted_labels[selected] == label).sum())
            for i in range(batch_len.shape[0]):
                true_sequence = tuple(true_activity[start + i, :batch_len[i]].tolist())
                dls_mean.append(damerau_levenshtein_similarity(tuple(mean_roll["activities"][i, 0, :mean_len[i]].tolist()), true_sequence))
                sampled_sequences = Counter(tuple(roll["activities"][i, s, :sampled_len[i, s]].tolist()) for s in range(self.samples_per_case))
                dls_prob.append(sum(count * damerau_levenshtein_similarity(sequence, true_sequence) for sequence, count in sampled_sequences.items()) / self.samples_per_case)
            metrics["dls/mean"] = torch.tensor(dls_mean)
            metrics["dls/prob"] = torch.tensor(dls_prob)

            # Letzter Wert der wachsenden Attribute, z.B. die Zeit am Fallende (wie LastValueMean2VarMAE, LastValuePIT, LastValueInterval)
            for name in self.growing_num_values:
                j = self.num_names.index(name)
                prefix_last = roll["last_prefix_attributes"][:, :, j]
                true_last = last_value(true_nums[batch], batch_len, prefix_last[:, 0], j)
                mean_last = last_value(mean_roll["event_attributes"][:, 0], mean_len, prefix_last[:, 0], j)
                sampled_last = last_value(roll["event_attributes"], sampled_len, prefix_last, j)
                metrics[f"last_{name}_mae/mean"] = (mean_last - true_last).abs()
                metrics[f"last_{name}_mae/prob"] = (sampled_last - true_last.unsqueeze(1)).abs().median(dim=1).values
                metrics[f"last_{name}_pit"] = (sampled_last <= true_last.unsqueeze(1)).float().mean(dim=1)
                for percentile, covered in self.interval_coverage(sampled_last, true_last).items():
                    metrics[f"last_{name}_interval_{percentile}"] = covered

            # Outcome am EOS-Event (gibt es im PSP nicht): Mittelwert über die Suffixe, Kalibrierung über gezogene Outcomes
            if y.shape[1] == 1:
                batch_y = y[batch, 0]
                outcome_mean, outcome_var = roll["outcome_mean"][:, :, 0], roll["outcome_var"][:, :, 0]
                sampled_outcome = outcome_mean + outcome_var.sqrt() * torch.randn_like(outcome_mean)
                metrics["outcome_mae/mean"] = (mean_roll["outcome_mean"][:, 0, 0] - batch_y).abs()
                metrics["outcome_mae/prob"] = (outcome_mean.mean(dim=1) - batch_y).abs()
                metrics["outcome_pit"] = (sampled_outcome <= batch_y.unsqueeze(1)).float().mean(dim=1)
                for percentile, covered in self.interval_coverage(sampled_outcome, batch_y).items():
                    metrics[f"outcome_interval_{percentile}"] = covered

            # Anteil der gezogenen Suffixe ohne EOS bis max_steps
            metrics["truncated"] = roll["truncated"].float().mean(dim=1)
            batches.append(metrics)

        if was_training:
            model.train()

        per_prefix = {name: torch.cat([metrics[name] for metrics in batches]) for name in batches[0]}
        results = {"per_prefix": per_prefix, "summary": {name: values.mean().item() for name, values in per_prefix.items()}, "n_prefixes": n}
        results["sequence_diagnostics"] = {
            "by_position": {
                position + 1: {"count": int(position_count[position]), "accuracy": position_correct[position].item() / position_count[position].item()}
                for position in range(max_steps) if position_count[position] > 0
            },
            "by_activity": {
                label: {"count": count, "accuracy": activity_correct[label] / count}
                for label, count in activity_counts.items()
            },
        }
        # Aufgeschlüsselt nach Prefix- und Suffixlänge, wie in den Plots des PSP
        for key, lengths in [("by_prefix_len", data["prefix_len"].long()), ("by_suffix_len", true_len)]:
            results[key] = {}
            for length in sorted(lengths.unique().tolist()):
                selected = lengths == length
                results[key][length] = {"count": int(selected.sum()), **{name: values[selected].mean().item() for name, values in per_prefix.items()}}
        return results

    def print_summary(self, results):
        summary = results["summary"]
        print(f"    PSP-Suffix-Auswertung: {results['n_prefixes']} Prefixe, {self.samples_per_case} gezogene Suffixe je Prefix (mean = wahrscheinlichstes Suffix, prob = gezogene Suffixe)")
        for name in sorted({key.split("/")[0] for key in summary if "/" in key}):
            print(f"      {name:28s} mean {summary[name + '/mean']:8.4f} | prob {summary[name + '/prob']:8.4f}")
        for name in sorted({key[:-len("_pit")] for key in summary if key.endswith("_pit")} | {"suffix_length"}):
            line = f"      {name:28s} Abdeckung der Intervalle: " + ", ".join(
                f"{int(round(p * 100))} % -> {summary[f'{name}_interval_{int(round(p * 100))}']:.3f}" for p in self.percentiles)
            if name + "_pit" in summary:
                line += f" | PIT-Mittel {summary[name + '_pit']:.3f} (ideal 0.5)"
            print(line)
        print("      DLS nach echter Suffixlänge (Länge: n, Greedy, Sampling):")
        for length, metrics in sorted(results["by_suffix_len"].items()):
            print(f"        {length:3d}: n={metrics['count']:4d}, Greedy {metrics['dls/mean']:.3f}, Sampling {metrics['dls/prob']:.3f}")
        diagnostics = results["sequence_diagnostics"]
        positions = " | ".join(
            f"{position}: {metrics['accuracy']:.3f} (n={metrics['count']})"
            for position, metrics in diagnostics["by_position"].items()
        )
        print(f"      Greedy-Aktivitätsgenauigkeit nach Position: {positions}")
        activities = " | ".join(
            f"{label}: {metrics['accuracy']:.3f} (n={metrics['count']})"
            for label, metrics in sorted(diagnostics["by_activity"].items())
        )
        print(f"      Greedy-Aktivitätsgenauigkeit nach Klasse: {activities}")
        print(f"      Abgeschnittene Suffixe (kein EOS): {summary['truncated']:.4f}")
