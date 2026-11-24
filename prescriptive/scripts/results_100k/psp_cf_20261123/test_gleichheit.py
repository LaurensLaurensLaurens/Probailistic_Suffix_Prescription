"""Prüfung psp_cf_policy (2026-11-23): Aufbau F, Seed 3, 12 bewertete Fälle. (1) z = 0 (Forest nie gefragt) → dieselben Wahlen wie
PSPPolicy mit olsv (gleiche Ziehungen); (2) z = 1,96 → Einträge mit unsicher / Forest-Wahl. Aufruf im Ordner process-simulation-engine."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"modellvergleich_20261116"))
import f_zwoelf_seeds as fz  # noqa: E402  (Simulator, Modelle, FOREST)
import pandas as pd  # noqa: E402

N = 12


def lauf(art):
    import torch
    torch.set_num_threads(1)
    if art == "psp":
        from prescriptive.policies.ev_policy.psp_policy import PSPPolicy
        policy = PSPPolicy(model_path=str(fz.MODELS["olsv"]), seed=3, samples=300, first_only=fz.FIRST_ONLY)
    else:
        from prescriptive.policies.ev_policy.psp_cf_policy import PSPCausalForestPolicy
        policy = PSPCausalForestPolicy(str(fz.MODELS["olsvcf"]), str(fz.FOREST), seed=3, samples=300, first_only=fz.FIRST_ONLY,
                                       z=0.0 if art == "z0" else 1.96)
    _, choices = fz.simulator().run(700 + N + 50, policy=policy, window=(700, 700 + N), seed=3)
    return choices, pd.DataFrame(getattr(policy, "entscheidungen", []))


if __name__ == "__main__":
    with ProcessPoolExecutor(3, mp_context=multiprocessing.get_context("spawn")) as pool:
        (cp, _), (c0, e0), (cz, ez) = pool.map(lauf, ["psp", "z0", "z196"])
    gleich = cp[["case", "point", "choice"]].astype(str).equals(c0[["case", "point", "choice"]].astype(str))
    print(f"(1) z=0 gleich PSPPolicy: {gleich} ({len(cp)} / {len(c0)} Wahlen, Forest gefragt {int(e0.forest.notna().sum()) if len(e0) else 0})")
    print(f"(2) z=1,96: {len(ez)} Entscheidungen, unsicher {int(ez.unsicher.sum())}, Forest entscheidet {int(ez.forest.notna().sum())}, "
          f"Forest anders als PSP {int((ez.forest.notna() & (ez.forest != ez.psp)).sum())}")
    with pd.option_context("display.width", 250, "display.max_columns", 12):
        print(ez[["point", "case", "delta", "se", "unsicher", "psp", "forest", "wahl"]].round(2).to_string())
    print("OK" if gleich else "FEHLER")
