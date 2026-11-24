"""Gradienten-Zerlegung ohne Training (2026-11-24, Nutzer: „ja mach beides“): Warum lernt das Netz den betragsabhängigen Übergang
A_Complete → A_Complete an konzept (vervollständigen, < 11k 30 % doppelt, ≥ 11k nie) nicht, obwohl der Betrag ankommt (Lernbarkeitstest
sn/sno, befunde.md 11:20)?

Genau der Datenweg des Trainings: PSPModel.fit mit den Parametern von k3x4ol_konzept_bearbeiten (init_from dieses Netz, static_in_decoder
mit null begonnener Abbildung, nur Proben am Options-Event), aber der Trainer rechnet statt Optimierungsschritten nur Gradienten auf
N_BATCHES Batches (Teacher Forcing 1, train-Modus mit Dropout-Masken wie im Training, GradNorm-Gewichte 1 = Startwerte). Gemessen:
  1. Anteil der Fokus-Schritte (Ziel direkt nach dem ersten A_Complete im Suffix) an allen gültigen Token und am Event-Verlust.
  2. Gradient an der Betragsspalte der Statik-Abbildung (und am Event-Kopf) je Verlustteil: Fokus-Schritte, übrige Event-Token, jede
     weitere Aufgabe (Zeiten usw.); Normen, Kosinus zum Gesamtgradienten, Anteil des Gesamtgradienten in Fokus-Richtung.
  3. Gelernte Logit-Varianz an den Fokus-Schritten je Gruppe und Dämpfung durch die Abschwächung: Gradient des abgeschwächten Fokus-
     Verlusts nach den mittleren Logits gegen den ohne Rauschen (Norm-Verhältnis, Kosinus).
  4. Möglicher Gewinn: Verlust an den Fokus-Schritten mit der jetzigen Vorhersage gegen die mit der wahren Gruppenrate (nats je Fokus-Token
     und umgerechnet auf den Event-Verlust je Token).
Ausgabe gradienten.md / gradienten.json hier. Aufruf im Ordner process-simulation-engine:
  .venv/bin/python -u <dieses Skript> [Fälle=20000] [Batches=20]
"""
from pathlib import Path
import json
import os
import shutil
import sys
import time
import warnings

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
P = ROOT/"prescriptive"
LOG = P/"scripts/results_100k/kuenstliche_effekte_20261120/log_effekte_v3_x4/log_raw.pkl"
QUELLE = P/"models/psp_model_durchlaufzeit_k3x4ol_konzept_bearbeiten.pkl"
TMP = HERE/"gradienten_tmp"
N_FAELLE = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
N_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 20
AC = "A_Complete|complete"
GRENZE_Z = (11000 - 15217.0) / 9898.0  # StandardScaler des Betrags (befunde 11-24 01:40)
os.environ["PSP_CATALOG"] = str(P/"data/catalog_neue_punkte_zuteilung_random.yaml")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(P))
from prescriptive.helpers.datasets import add_paths  # noqa: E402
add_paths()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402


class Fertig(Exception):
    pass


def diag_trainer(basis):
    from torch.utils.data import DataLoader
    from prescriptive.estimators import psp_model as pm

    class Diag(basis):
        def train_model(self, use_statics=False, use_zero_padd_masking=False, use_eos_padd_masking=False):
            torch.manual_seed(5)
            net = self.model
            net.train()
            names = [e[0] for e in self.data_train.all_categories[0]]
            ac_id = int(self.data_train.all_categories[0][names.index(pm.ACTIVITY)][2][AC])
            W = net.decoder.input_proj.static_projection.weight          # 128 × static_input_size, mit 0 begonnen
            kopf = list(net.decoder.output_layers[f"{pm.ACTIVITY}_mean"].parameters())
            acc = {}                                                     # Teil -> [Gradient Betragsspalte, Gradient Abbildung, Gradient Kopf]
            stat = {"token": 0.0, "fokus": 0, "fokus_ac": {"< 11k": [0, 0], "≥ 11k": [0, 0]}, "p_ac": {"< 11k": [], "≥ 11k": []},
                    "logvar_ac": {"< 11k": [], "≥ 11k": []}, "logvar_ziel": [], "daempfung": [], "kosinus_logits": [],
                    "ce_jetzt": {"< 11k": [], "≥ 11k": []}, "logits_ac": {"< 11k": [], "≥ 11k": []}, "verluste": {}, "proben": 0}
            loader = DataLoader(dataset=self.data_train, batch_size=self.mini_batches, shuffle=True, num_workers=0)
            start = time.time()
            for b, data in enumerate(loader):
                if b >= N_BATCHES:
                    break
                _, cats, nums, eos, zero, cats_static, nums_static, _ = data
                static_inputs = self._prepare_static_inputs(cats_static, nums_static)
                s = self.suffix_data_split_value
                prefixes = [[c[:, :-s] for c in cats], [n[:, :-s] for n in nums]]
                suffixes = [[c[:, -s:] for c in cats], [n[:, -s:] for n in nums]]
                mask, eos_s = zero[:, :-s], eos[:, -s:]
                preds, _, _, (cat_idx, num_idx) = net(prefixes=prefixes, static_inputs=static_inputs, suffixes=suffixes,
                                                      teacher_forcing_ratio=1.0, prefix_mask=mask)
                pc, pn = preds
                betrag = nums_static[:, 0].float()
                klein = betrag < GRENZE_Z
                # Aufgaben-Verluste wie train_epoch_gradnorm (GradNorm-Gewichte 1)
                teile = {}
                for key in pc:
                    if key.endswith("_mean"):
                        f = key[:-5]
                        teile[f"kat:{f}"] = self.loss_obj.loss_attenuation_cross_entropy(
                            pred_logits=pc[key], pred_logvars=pc[f"{f}_var"], T=30, targets=suffixes[0][cat_idx[f]].long(), eos_paddings=eos_s)
                for key in pn:
                    if key.endswith("_mean"):
                        f = key[:-5]
                        teile[f"num:{f}"] = self.loss_obj.loss_attenuation_mse(pred_means=pn[key], pred_logvars=pn[f"{f}_var"],
                                                                               targets=suffixes[1][num_idx[f]], eos_paddings=eos_s)
                # Fokus: Ziel direkt nach dem ersten A_Complete im Suffix
                ziel = suffixes[0][cat_idx[pm.ACTIVITY]].long()                     # Batch × Schritte
                gueltig = eos_s.bool()
                fokus = torch.zeros_like(gueltig)
                for r in range(ziel.shape[0]):
                    pos = torch.nonzero((ziel[r] == ac_id) & gueltig[r]).flatten()
                    if len(pos) and int(pos[0]) + 1 < ziel.shape[1] and bool(gueltig[r, int(pos[0]) + 1]):
                        fokus[r, int(pos[0]) + 1] = True
                logits = pc[f"{pm.ACTIVITY}_mean"]                                   # Schritte × Batch × Klassen
                logvars = torch.clamp(pc[f"{pm.ACTIVITY}_var"], -6.0, 6.0)
                n_token = float(gueltig.sum())
                # abgeschwächte CE je Token (wie Loss.loss_attenuation_cross_entropy, ohne Reduktion)
                ce = 0
                for _ in range(30):
                    noisy = (logits + torch.exp(0.5 * logvars) * torch.randn_like(logits)).permute(1, 2, 0)
                    ce = ce + torch.nn.functional.cross_entropy(noisy, ziel, reduction="none")
                ce = ce / 30                                                          # Batch × Schritte
                l_fokus = (ce * fokus).sum() / (n_token + 1e-8)
                teile["kat:fokus"] = l_fokus
                teile[f"kat:{pm.ACTIVITY}"] = teile[f"kat:{pm.ACTIVITY}"] - l_fokus  # übrige Event-Token (andere Rauschziehung, Mittel gleich)
                for name, loss in teile.items():
                    g = torch.autograd.grad(loss, [W] + kopf, retain_graph=True, allow_unused=True)
                    g = [torch.zeros_like(x) if y is None else y for x, y in zip([W] + kopf, g)]
                    a = acc.setdefault(name, [0, 0, 0])
                    acc[name] = [a[0] + g[0][:, -1].detach(), a[1] + g[0].detach(), a[2] + torch.cat([x.flatten() for x in g[1:]]).detach()]
                    stat["verluste"].setdefault(name, []).append(float(loss))
                # Logits an den Fokus-Schritten: Gruppe, P(AC), Logit-Varianz, Dämpfung durch das Rauschen
                fr, fs = torch.nonzero(fokus, as_tuple=True)
                if len(fr):
                    z = logits[fs, fr].detach().clone().requires_grad_(True)          # Fokus × Klassen
                    lv = logvars[fs, fr].detach()
                    y = ziel[fr, fs]
                    plain = torch.nn.functional.cross_entropy(z, y, reduction="none")
                    att = sum(torch.nn.functional.cross_entropy(z + torch.exp(0.5 * lv) * torch.randn_like(z), y, reduction="none")
                              for _ in range(200)) / 200
                    g_plain = torch.autograd.grad(plain.sum(), z, retain_graph=True)[0]
                    g_att = torch.autograd.grad(att.sum(), z)[0]
                    stat["daempfung"].append(float(g_att.norm() / (g_plain.norm() + 1e-12)))
                    stat["kosinus_logits"].append(float(torch.nn.functional.cosine_similarity(g_att.flatten(), g_plain.flatten(), dim=0)))
                    p = torch.softmax(z.detach(), dim=1)
                    for i, r in enumerate(fr.tolist()):
                        gname = "< 11k" if bool(klein[r]) else "≥ 11k"
                        stat["fokus_ac"][gname][0] += 1
                        stat["fokus_ac"][gname][1] += int(y[i] == ac_id)
                        stat["p_ac"][gname].append(float(p[i, ac_id]))
                        stat["logvar_ac"][gname].append(float(lv[i, ac_id]))
                        stat["logits_ac"][gname].append(float(z[i, ac_id]))
                        stat["ce_jetzt"][gname].append((float(plain[i]), int(y[i] == ac_id), float(p[i, ac_id]), float(p[i, y[i]])))
                    stat["logvar_ziel"] += lv[torch.arange(len(y)), y].tolist()
                stat["token"] += n_token
                stat["fokus"] += int(fokus.sum())
                stat["proben"] += ziel.shape[0]
                print(f"Batch {b + 1}/{N_BATCHES}: {int(fokus.sum())} Fokus-Schritte, {time.time() - start:.0f} s", flush=True)
            raise Fertig((acc, stat))
    return Diag


def auswerten(acc, stat):
    gesamt = sum(v[0] for v in acc.values())
    gesamt_map = sum(v[1] for v in acc.values())
    gesamt_kopf = sum(v[2] for v in acc.values())
    fok = acc["kat:fokus"]
    richtung = fok[0] / (fok[0].norm() + 1e-12)
    zeilen = []
    for name, (g_b, g_m, g_k) in acc.items():
        zeilen.append({"Teil": name, "Verlust (Mittel je Batch)": np.mean(stat["verluste"][name]),
                       "|grad Betragsspalte|": float(g_b.norm()), "Kosinus zu gesamt (Betrag)": float(torch.nn.functional.cosine_similarity(g_b, gesamt, dim=0)),
                       "Anteil in Fokus-Richtung": float(g_b @ richtung / (gesamt @ richtung + 1e-12)),
                       "|grad Abbildung|": float(g_m.norm()), "|grad Event-Kopf|": float(g_k.norm())})
    tab = pd.DataFrame(zeilen)
    gruppen = {}
    for g in ("< 11k", "≥ 11k"):
        n, n_ac = stat["fokus_ac"][g]
        rate = n_ac / max(n, 1)
        rows = stat["ce_jetzt"][g]
        # Gewinn, wenn P(AC) auf die Gruppenrate gesetzt wird (übrige Klassen anteilig): je Fokus-Token
        r = min(max(rate, 1e-4), 1 - 1e-4)
        gewinn = [ce - (-np.log(r) if is_ac else -np.log(p_y * (1 - r) / max(1 - p_ac, 1e-9))) for ce, is_ac, p_ac, p_y in rows]
        gruppen[g] = {"Fokus-Schritte": n, "wahr A_Complete": round(rate, 3), "P(AC) Netz": round(float(np.mean(stat["p_ac"][g])), 3),
                      "Logit AC": round(float(np.mean(stat["logits_ac"][g])), 3),
                      "logvar AC (Mittel)": round(float(np.mean(stat["logvar_ac"][g])), 3),
                      "σ AC (Median)": round(float(np.median(np.exp(0.5 * np.array(stat["logvar_ac"][g])))), 3),
                      "CE jetzt": round(float(np.mean([x[0] for x in rows])), 4), "Gewinn mit Gruppenrate (nats je Fokus-Token)": round(float(np.mean(gewinn)), 4)}
    anteil = stat["fokus"] / stat["token"]
    gewinn_token = sum(gruppen[g]["Gewinn mit Gruppenrate (nats je Fokus-Token)"] * gruppen[g]["Fokus-Schritte"] for g in gruppen) / stat["token"]
    kopf = {"Proben": stat["proben"], "gültige Token": int(stat["token"]), "Fokus-Schritte": stat["fokus"], "Anteil Fokus an Token": round(anteil, 4),
            "Gewinn umgerechnet auf Event-Verlust je Token (nats)": round(gewinn_token, 5),
            "Event-Verlust je Token (Mittel)": round(float(np.mean(stat["verluste"]["kat:" + "concept:name"]) + np.mean(stat["verluste"]["kat:fokus"])), 4),
            "Dämpfung durch Abschwächung (|grad att| / |grad ohne Rauschen|, Fokus-Logits)": round(float(np.mean(stat["daempfung"])), 3),
            "Kosinus grad att / ohne Rauschen": round(float(np.mean(stat["kosinus_logits"])), 3),
            "σ Ziel-Klasse an Fokus (Median)": round(float(np.median(np.exp(0.5 * np.array(stat["logvar_ziel"])))), 3),
            "|grad gesamt Betragsspalte|": float(gesamt.norm()), "|grad gesamt Abbildung|": float(gesamt_map.norm()),
            "|grad gesamt Event-Kopf|": float(gesamt_kopf.norm())}
    return tab, gruppen, kopf


def main():
    from helpers.training import test_cases
    from preprocessing import preprocess
    from prescriptive.estimators.psp_model import PSPModel  # setzt den Pfad zum U-ED-LSTM (trainer, loss, model)
    import trainer.trainer as tt
    start = time.time()
    quelle = PSPModel.load(str(QUELLE))
    raw = pd.read_pickle(LOG)
    held = test_cases(raw)
    faelle = np.sort(raw["case:concept:name"].astype(str).unique())
    faelle = [c for c in faelle if c not in held]
    wahl = set(np.random.default_rng(11).choice(faelle, min(N_FAELLE, len(faelle)), replace=False))
    raw = raw[raw["case:concept:name"].astype(str).isin(wahl)]
    log = preprocess("bpic17", raw, verbose=False)
    print(f"{log['case:concept:name'].nunique():,} Fälle ({time.time() - start:.0f} s)", flush=True)
    params = {k: v for k, v in quelle.params.items() if k != "option_as"}  # option_as encoder setzt train_psp_real erst nach fit
    params.update(epochs=1, train_cases=0, load_chunks=4, checkpoints=str(TMP), init_from=str(QUELLE), vocab_from=str(QUELLE),
                  static_in_decoder=True, threads=16, device="cpu")
    orig = tt.Trainer
    tt.Trainer = diag_trainer(orig)
    try:
        PSPModel("bpic17", list(quelle.points.values()), seed=42, params=params).fit(log)
        raise RuntimeError("Diagnose-Trainer nicht aufgerufen")
    except Fertig as f:
        acc, stat = f.args[0]
    finally:
        tt.Trainer = orig
        shutil.rmtree(TMP, ignore_errors=True)
    tab, gruppen, kopf = auswerten(acc, stat)
    with pd.option_context("display.width", 250):
        text = (f"# Gradienten-Zerlegung an konzept / vervollständigen (k3x4ol, ohne Training), {time.strftime('%Y-%m-%d %H:%M')}\n\n"
                f"{N_FAELLE} Fälle des x4-Logs (ohne Testfälle), {N_BATCHES} Batches à 256, Teacher Forcing 1, train-Modus, GradNorm-Gewichte 1; "
                f"Statik-Abbildung mit 0 begonnen (Gradient = Richtung des ersten Schritts). Fokus = Ziel direkt nach dem ersten A_Complete.\n\n"
                f"## Überblick\n```\n{json.dumps(kopf, indent=1, ensure_ascii=False)}\n```\n\n## Fokus-Schritte je Gruppe\n```\n"
                f"{pd.DataFrame(gruppen).T.to_string()}\n```\n\n## Gradient je Verlustteil\n"
                f"(Anteil in Fokus-Richtung: Projektion des Teils auf die Richtung des Fokus-Gradienten, relativ zum Gesamtgradienten)\n```\n"
                f"{tab.round(5).to_string(index=False)}\n```\n\nLaufzeit {time.time() - start:.0f} s\n")
    (HERE/"gradienten.md").write_text(text)
    (HERE/"gradienten.json").write_text(json.dumps({"kopf": kopf, "gruppen": gruppen, "teile": tab.to_dict("records")}, indent=1, ensure_ascii=False, default=float))
    print(text, flush=True)


if __name__ == "__main__":
    main()
