"""
Das PSP (estimators/psp_model.py) auf dem echten Event-Log statt auf den RCT-Daten des Simulators trainieren, sonst wie
data/bpic17_psp_w3_10k.sh: dieselben 12 Wahlpunkte (8 Mitarbeiter, 4 Control Flow; hier fest gesetzt, nicht aus dem
Katalog gelesen, der während anderer Läufe umgestellt sein kann), train_cases Fälle, Epochen, Lernrate, Gewicht der
Wahlpunkte, GPU (TRAIN device auto).

Geteilt wird wie in helpers/training.train_estimators: TEST_SHARE der Fälle des echten Logs bleiben als Testfälle draußen
(test_cases, gleiche Saat), aus den übrigen zieht PSPModel.fit train_cases Fälle und Validierungsfälle wie im Original.
Eigener Checkpoint-Ordner und eigener Modellname, damit dieser Lauf parallel zu den Läufen der Pipeline laufen kann
(models/psp_model.pkl, models/psp_training und der Katalog bleiben unberührt).

Mit --catalog-points die Wahlpunkte des Katalogs (z. B. eine Kopie über PSP_CATALOG, wie data/bpic17_psp_real_neue_punkte.sh).
Mit --remaining-head lernt ein Restzeit-Kopf auf dem Encoder mit (psp_model REMAINING_HEAD, Gewicht --remaining-weight).
Mit --gap-classes sagt der Decoder je Schritt auch die Klasse des Abstands zum vorigen Event voraus (psp_model GAP_CLASS).
Mit --gap-condition-on-event (Variante e) wird die Klasse zusätzlich auf das aktuelle Aktivität|Lifecycle-Token bedingt.
--option-in-encoder-after-training setzt die Optionsziehung erst nach dem Training um, bei unveränderter Probengewichtung.
Stufe 3 (psp_model, Kommentar "Stufe 3"): --teacher-forcing 1.0, --gap-tokens, --option-in-decoder.
2026-11-16: --counters (Zähler je Token in Encoder und Decoder), --time-on-event (Zeitköpfe sehen das Event desselben Schritts).
2026-11-17: --clocks (Tage seit dem letzten Auftreten je Token, mit --counters); --only-points (nur Trainingsproben am Options-Event
eines Wahlpunkts, psp_model only_points); --only-points-validation, --arm aktion|referenz und --init-from für den T-Learner
(policies/ev_policy/t_learner_policy.py). 2026-11-18: --log-pickle (Simulator-/RCT-Log statt echtem Log), --test-cases-from.
2026-11-20: --vocab-from (Vokabular eines anderen Modells, für den Warmstart auf einem Simulator-Log; psp_model vocab_from).
2026-11-22: --point / --option (ein Netz je Wahlpunkt und Option; psp_model only_point_names / only_options).
2026-11-21: --load-chunks N (Kodieren in Stücken, Filter je Stück auf Wahlpunkt-Proben; psp_model load_chunks; für große Simulator-Logs).
2026-11-18: --beta-nll β (β-NLL der Zeitköpfe, psp_model beta_nll; Standard 0 = wie bisher).
2026-11-18: --time-event-head offset|slope (eigene Parameter je Event für die Zeitköpfe), --head-only (nur die Zeitköpfe nachtrainieren,
mit --init-from), --offset-learning-rate (Lernrate der Versätze je Event).
2026-11-19: --sequence-loss (CRPS frei gezogener Restzeiten gegen die echte Restzeit, psp_model SEQUENCE), --sequence-weight,
--sequence-draws, --sequence-batch, --sequence-steps; --sequence-calibration c (Kalibrierung je Wahlpunkt und Option, SEQUENCE calibration).
2026-11-20: --sequence-calibration-mode global|paired (SEQUENCE calibration_mode; paired = gegen das echte Mittel derselben Prefixe,
Quadrat aus zwei Ziehungshälften, ohne Schrumpfdruck gegen fallabhängige Effekte; split = zwei disjunkte Prefix-Hälften gegen das
globale Gruppenmittel).

Aufruf (im Ordner process-simulation-engine):
    .venv/bin/python -u ../prescriptive/scripts/train_psp_real.py --tag real_w3_10k --cases 10000 --epochs 100 \
        --learning-rate 1e-5 --point-weight 3
Ergebnis: models/psp_model_durchlaufzeit_<tag>.pkl, models/psp_training_<tag>/ (last.pt jede Epoche).
"""
import argparse
import copy
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
PRESCRIPTIVE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PRESCRIPTIVE)

from helpers.datasets import CASE, add_paths

# Die Wahlpunkte von bpic17_psp_w5.sh / bpic17_psp_w3_10k.sh (lauf.mitarbeiterwahl und lauf.control_flow)
EMPLOYEES = ["Mitarbeiter: W_Call incomplete files", "Mitarbeiter: W_Complete application", "Mitarbeiter: W_Handle leads",
             "Mitarbeiter: W_Validate application", "Mitarbeiter: W_Call after offers (resume)", "Mitarbeiter: W_Call incomplete files (resume)",
             "Mitarbeiter: W_Complete application (resume)", "Mitarbeiter: W_Validate application (resume)"]
CONTROL = ["handle_leads", "annehmen_nach_vervollstaendigung", "angebot_nachlegen", "neues_angebot_beim_nachfassen"]


def fixed_points(dataset):
    """Die Wahlpunkte mit EMPLOYEES und CONTROL, unabhängig vom Stand von catalog.yaml (Katalog im Speicher ersetzt)."""
    import interventions
    fixed = copy.deepcopy(interventions.catalog())
    lauf = fixed[dataset].setdefault("lauf", {})
    lauf["mitarbeiterwahl"], lauf["control_flow"] = list(EMPLOYEES), list(CONTROL)
    interventions.catalog = lambda: fixed
    return interventions.choice_points(dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="bpic17")
    parser.add_argument("--tag", required=True, help="Name: models/psp_model_durchlaufzeit_<tag>.pkl, models/psp_training_<tag>")
    parser.add_argument("--cases", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--point-weight", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix-size", type=int, default=None, help="Schritte des Decoders im Training (Standard wie im Original: 5)")
    parser.add_argument("--log-gaps", action="store_true", help="Abstände zum vorigen Event log-normal statt normal (psp_model properties)")
    parser.add_argument("--time-from", choices=("elapsed", "gaps"), default=None,
                        help="Durchlaufzeit aus der gezogenen Zeit seit Fallbeginn (Original) oder der Summe der gezogenen Abstände")
    parser.add_argument("--statics-without-future", action="store_true",
                        help="nur die Fallattribute, die zu Fallbeginn feststehen (psp_model.STATICS_WITHOUT_FUTURE), statt der des Originals")
    parser.add_argument("--remaining-head", action="store_true",
                        help="Restzeit-Kopf auf dem Zustand des Encoders mittrainieren (psp_model REMAINING_HEAD, TRAIN remaining_head)")
    parser.add_argument("--gap-classes", action="store_true",
                        help="Abstandsklasse als zusätzliche kategoriale Ausgabe; Abstand aus der auf die Klasse trunkierten Decoder-Normalverteilung")
    parser.add_argument("--gap-condition-on-event", action="store_true",
                        help="Variante (e): erst Aktivität|Lifecycle, dann die darauf bedingte Zeitklasse; schließt --gap-classes ein")
    parser.add_argument("--remaining-weight", type=float, default=None, help="Gewicht des Restzeit-Verlusts (Standard REMAINING_HEAD weight)")
    parser.add_argument("--teacher-forcing", type=float, default=None,
                        help="unterste Teacher-Forcing-Quote (Standard 0.0 wie im Original; 1.0: der Decoder sieht im Training immer die echten vorigen Events)")
    parser.add_argument("--gap-tokens", action="store_true",
                        help="Token des Decoders mit Abstandsklasse: Event und Klasse seines Abstands werden zusammen gezogen (psp_model gap_tokens)")
    parser.add_argument("--option-in-decoder", action="store_true",
                        help="Control Flow: die Option als erstes Event des Suffix statt als letztes des Prefix (psp_model option_as decoder)")
    parser.add_argument("--option-in-encoder", action="store_true",
                        help="Control Flow: die Option als letztes Event des Prefix mit vom Decoder gezogenem Abstand (psp_model option_as encoder)")
    parser.add_argument("--option-in-encoder-after-training", action="store_true",
                        help="option_as encoder erst nach fit setzen, damit Trainingsproben wie beim ursprünglichen (d) gewichtet bleiben")
    parser.add_argument("--decoder-context", action="store_true",
                        help="der Decoder bekommt in jedem Schritt den Zustand des Encoders nach dem Prefix (psp_model decoder_context)")
    parser.add_argument("--static-in-decoder", action="store_true",
                        help="die statischen Merkmale des Falls gehen in jeden Decoder-Schritt (psp_model static_in_decoder, 2026-11-24)")
    parser.add_argument("--static-learning-rate", type=float, default=None,
                        help="eigene Lernrate der neuen Statik-Abbildung (psp_model static_learning_rate; mit --static-in-decoder)")
    parser.add_argument("--static-only", action="store_true",
                        help="Lernbarkeitstest 2026-11-24: nur die Statik-Abbildung lernt, Rest eingefroren (psp_model static_only); "
                             "mit --static-in-decoder und --init-from")
    parser.add_argument("--static-only-output", action="store_true",
                        help="mit --static-only: dazu der Event-Kopf des Decoders (psp_model static_only_output)")
    parser.add_argument("--task-weights", default=None,
                        help="2026-11-24: feste Aufgabengewichte EVENT,ZAHLEN (z. B. 10,1) statt der GradNorm-Anpassung (psp_model task_weights)")
    parser.add_argument("--catalog-points", action="store_true",
                        help="die Wahlpunkte des Katalogs (interventions.choice_points, auch über PSP_CATALOG) statt EMPLOYEES und CONTROL")
    parser.add_argument("--counters", action="store_true",
                        help="Zähler je Aktivität|Lifecycle-Token als Eingabe in Encoder und Decoder (psp_model counters)")
    parser.add_argument("--clocks", action="store_true",
                        help="Uhren: Tage seit dem letzten Auftreten je Token als Eingabe (psp_model clocks); nur mit --counters und stetigen Abständen")
    parser.add_argument("--time-on-event", action="store_true",
                        help="Zeitköpfe sehen das Event desselben Schritts (psp_model time_on_event); nur mit stetigen Abständen")
    parser.add_argument("--only-points", action="store_true",
                        help="nur die Trainingsproben, deren Prefix am Options-Event eines Wahlpunkts endet (psp_model only_points; --point-weight egal)")
    parser.add_argument("--point", default=None,
                        help="nur Proben an diesem Wahlpunkt (Name im Katalog; psp_model only_point_names; mit --only-points)")
    parser.add_argument("--option", default=None,
                        help="nur Proben am Options-Event dieser Option (Token Aktivität|Lifecycle, z. B. 'A_Accepted|complete'; psp_model only_options; mit --only-points, statt --arm)")
    parser.add_argument("--arm", choices=("aktion", "referenz"), default=None,
                        help="T-Learner: nur Proben am Options-Event der Aktion (annehmen, annehmen, weiteres Angebot, Lead bearbeiten) "
                             "bzw. der Referenz (psp_model only_options; mit --only-points)")
    parser.add_argument("--log-pickle", default=None,
                        help="statt des echten Logs dieses Simulator-Log (Events wie Simulator.run, z. B. RCT-Daten); preprocess wie beim echten Log")
    parser.add_argument("--test-cases-from", default=None,
                        help="nach dem Training test_cases von diesem Modell übernehmen (z. B. echte Testfälle von (c) bei Training auf RCT-Daten)")
    parser.add_argument("--init-from", default=None, help="Warmstart: Gewichte dieses fertigen PSP (gleiche Netzstruktur, psp_model init_from)")
    parser.add_argument("--load-chunks", type=int, default=None,
                        help="Log in so vielen Stücken kodieren und je Stück auf die Wahlpunkt-Proben filtern (Speicher; nur mit --only-points und --only-points-validation; psp_model load_chunks)")
    parser.add_argument("--vocab-from", default=None,
                        help="Kategorien (Vokabular) dieses Modells übernehmen, z. B. für --init-from auf einem Simulator-Log mit weniger Kategorien (psp_model vocab_from)")
    parser.add_argument("--beta-nll", type=float, default=None,
                        help="β-NLL für die Zeitköpfe (psp_model beta_nll, Seitzer et al. 2022), z. B. 0.5; Standard 0 = attenuierter Verlust wie bisher")
    parser.add_argument("--only-points-validation", action="store_true",
                        help="auch die Validierungsproben nur am Options-Event eines Wahlpunkts (psp_model only_points_validation)")
    parser.add_argument("--time-event-head", choices=("offset", "slope"), default=None,
                        help="eigene Parameter je Event für die Zeitköpfe (psp_model time_event_head): Versatz bzw. Versatz und Steigung; mit --time-on-event")
    parser.add_argument("--head-only", action="store_true",
                        help="nur die Zeitköpfe nachtrainieren, Rest eingefroren (psp_model head_only); mit --init-from")
    parser.add_argument("--offset-learning-rate", type=float, default=None,
                        help="Lernrate der Versätze je Event (psp_model offset_learning_rate; Standard --learning-rate)")
    parser.add_argument("--sequence-loss", action="store_true",
                        help="Verlust auf frei gezogenen Suffixen: CRPS der gezogenen gegen die echte Restzeit (psp_model SEQUENCE, mit --time-on-event)")
    parser.add_argument("--sequence-weight", type=float, default=None, help="Gewicht des Suffix-Verlusts (Standard SEQUENCE weight)")
    parser.add_argument("--sequence-draws", type=int, default=None, help="Ziehungen je Prefix im Suffix-Verlust (Standard SEQUENCE draws)")
    parser.add_argument("--sequence-batch", type=int, default=None, help="Prefixe je Batch im Suffix-Verlust (Standard SEQUENCE batch)")
    parser.add_argument("--sequence-steps", type=int, default=None, help="höchstens so viele gezogene Schritte (Standard SEQUENCE max_steps)")
    parser.add_argument("--sequence-calibration-mode", choices=["global", "paired", "split"], default=None,
                        help="Kalibrierung: global (Gruppenmittel, Stand svk), paired (dieselben Prefixe, svp) oder split (disjunkte Prefix-Hälften gegen das Gruppenmittel)")
    parser.add_argument("--sequence-calibration", type=float, default=None,
                        help="Gewicht der Kalibrierung je (Wahlpunkt, Option) im Suffix-Verlust, Prefixe je Gruppe gleich häufig (Standard 0 = aus)")
    args = parser.parse_args()
    if (args.time_event_head or args.head_only) and not args.time_on_event:
        parser.error("--time-event-head und --head-only nur mit --time-on-event")
    if args.head_only and not args.init_from:
        parser.error("--head-only nur mit --init-from")
    if args.time_on_event and (args.gap_classes or args.gap_condition_on_event or args.gap_tokens or args.log_gaps):
        parser.error("--time-on-event nur mit stetigen Abständen (ohne --gap-classes, --gap-condition-on-event, --gap-tokens, --log-gaps)")
    if args.gap_condition_on_event and (args.gap_tokens or args.log_gaps):
        parser.error("--gap-condition-on-event ist nicht mit --gap-tokens oder --log-gaps kombinierbar")
    if args.option_in_encoder_after_training and (args.option_in_encoder or args.option_in_decoder):
        parser.error("--option-in-encoder-after-training ist nicht mit den anderen Optionsmodi kombinierbar")
    add_paths()

    from helpers.datasets import load_dataset
    from helpers.training import MODELS, test_cases
    from preprocessing import preprocess
    from prescriptive.estimators.psp_model import PSPModel

    path = os.path.join(MODELS, f"psp_model_durchlaufzeit_{args.tag}.pkl")
    checkpoints = os.path.join(MODELS, f"psp_training_{args.tag}")
    for taken in (path, checkpoints):
        if os.path.exists(taken):
            sys.exit(f"{taken} gibt es schon, Abbruch")

    print(f"######## Echter Log {args.dataset}: Laden und Preprocessing ({time.strftime('%H:%M:%S')})", flush=True)
    if args.log_pickle:  # 2026-11-18: Training auf Simulatordaten (RCT), sonst unverändert
        import pandas as pd
        raw = pd.read_pickle(args.log_pickle)
        print(f"Simulator-Log statt echtem Log: {args.log_pickle}", flush=True)
    else:
        raw, _, _ = load_dataset(args.dataset)
    log = preprocess(args.dataset, raw)
    del raw
    held_out = test_cases(log)
    train = log[~log[CASE].astype(str).isin(held_out)]
    print(f"Echter Log: {log[CASE].nunique():,} Fälle, davon {len(held_out):,} Testfälle; Training zieht {args.cases:,} aus "
          f"{train[CASE].nunique():,} Fällen", flush=True)

    if args.catalog_points:
        import interventions
        points = interventions.choice_points(args.dataset)
        print(f"Katalog: {interventions.CANDIDATES_FILE}", flush=True)
    else:
        points = fixed_points(args.dataset)
    print(f"Wahlpunkte ({len(points)}): {', '.join(point.name for point in points)}", flush=True)
    params = {"train_cases": args.cases, "epochs": args.epochs, "learning_rate": args.learning_rate,
              "point_weight": args.point_weight, "checkpoints": checkpoints}
    if args.suffix_size:
        params["suffix_size"] = args.suffix_size
    if args.log_gaps:
        params["log_gaps"] = True
    if args.time_from:
        params["time_from"] = args.time_from
    if args.remaining_head:
        params["remaining_head"] = True
    if args.gap_classes or args.gap_condition_on_event:
        params["gap_classes"] = True
    if args.gap_condition_on_event:
        params["gap_condition_on_event"] = True
    if args.remaining_weight is not None:
        params["remaining_weight"] = args.remaining_weight
    if args.teacher_forcing is not None:
        params["teacher_forcing"] = args.teacher_forcing
    if args.counters:
        params["counters"] = True
    if args.time_on_event:
        params["time_on_event"] = True
    if args.clocks:
        if not args.counters or args.gap_classes or args.gap_tokens or args.log_gaps:
            parser.error("--clocks nur mit --counters und stetigen Abständen")
        params["clocks"] = True
    if args.gap_tokens:
        params["gap_tokens"] = True
    if args.option_in_decoder:
        params["option_as"] = "decoder"
    if args.option_in_encoder:
        params["option_as"] = "encoder"
    if args.decoder_context:
        params["decoder_context"] = True
    if args.static_in_decoder:
        params["static_in_decoder"] = True
    if args.static_learning_rate is not None:
        if not args.static_in_decoder:
            parser.error("--static-learning-rate nur mit --static-in-decoder")
        params["static_learning_rate"] = args.static_learning_rate
    if args.static_only or args.static_only_output:
        if not (args.static_in_decoder and args.init_from and args.static_only) or args.head_only:
            parser.error("--static-only nur mit --static-in-decoder und --init-from, ohne --head-only; --static-only-output nur mit --static-only")
        params["static_only"] = True
        params["static_only_output"] = bool(args.static_only_output)
    if args.task_weights:
        try:
            cat, num = (float(x) for x in args.task_weights.split(","))
        except ValueError:
            parser.error("--task-weights: zwei Zahlen EVENT,ZAHLEN, z. B. 10,1")
        if min(cat, num) < 0 or cat + num <= 0:
            parser.error("--task-weights: nicht negativ, Summe > 0")
        params["task_weights"] = {"cat": cat, "num": num}
    if args.only_points:
        params["only_points"] = True
    if args.only_points_validation:
        params["only_points_validation"] = True
    if args.arm:
        if not args.only_points:
            parser.error("--arm nur mit --only-points")
        from prescriptive.policies.ev_policy.t_learner_policy import ARME
        params["only_options"] = sorted(ARME[args.arm])
    if args.option or args.point:  # 2026-11-22: ein Netz je Wahlpunkt und Option
        if not args.only_points or args.arm:
            parser.error("--option / --point nur mit --only-points und ohne --arm")
        if args.option:
            params["only_options"] = [args.option]
        if args.point:
            params["only_point_names"] = [args.point]
    if args.beta_nll:
        params["beta_nll"] = args.beta_nll
    if args.init_from:
        params["init_from"] = os.path.abspath(args.init_from)
    if args.vocab_from:
        params["vocab_from"] = os.path.abspath(args.vocab_from)
    if args.load_chunks:
        params["load_chunks"] = args.load_chunks

    if args.time_event_head:
        params["time_event_head"] = args.time_event_head
    if args.head_only:
        params["head_only"] = True
    if args.offset_learning_rate:
        params["offset_learning_rate"] = args.offset_learning_rate
    if args.sequence_loss:
        if not args.time_on_event or args.head_only or args.remaining_head:
            parser.error("--sequence-loss nur mit --time-on-event, ohne --head-only und --remaining-head")
        params["sequence_loss"] = {key: value for key, value in (("weight", args.sequence_weight), ("draws", args.sequence_draws),
                                                                  ("batch", args.sequence_batch), ("max_steps", args.sequence_steps),
                                                                  ("calibration", args.sequence_calibration),
                                                                  ("calibration_mode", args.sequence_calibration_mode)) if value is not None}
        params["sequence_loss"] = params["sequence_loss"] or True
    if args.statics_without_future:
        from prescriptive.estimators.psp_model import STATICS_WITHOUT_FUTURE
        params["statics"] = STATICS_WITHOUT_FUTURE
    print(f"######## Training: {params} ({time.strftime('%H:%M:%S')})", flush=True)
    model = PSPModel(args.dataset, points, seed=args.seed, params=params).fit(train)
    if args.option_in_encoder_after_training:
        model.params["option_as"] = "encoder"
    model.test_cases, model.trained_on = held_out, {"echter Log": args.dataset, "outcome": "durchlaufzeit"}
    if args.log_pickle:
        model.trained_on = {"simulator_log": os.path.abspath(args.log_pickle), "outcome": "durchlaufzeit"}
    if args.test_cases_from:
        model.test_cases = PSPModel.load(args.test_cases_from).test_cases
    model.save(path)
    print(f"Zusammenfassung: {model.summary}", flush=True)
    print(f"######## Gespeichert in {path} ({time.strftime('%H:%M:%S')})", flush=True)


if __name__ == "__main__":
    main()
