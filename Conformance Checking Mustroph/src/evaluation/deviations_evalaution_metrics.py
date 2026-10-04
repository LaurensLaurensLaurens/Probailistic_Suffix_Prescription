from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import textwrap
from collections import Counter, defaultdict
from typing import List, Tuple, Union, Iterable, Optional, Set

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from matplotlib.ticker import MaxNLocator

def load_results(path: Union[str, Path]) -> dict:
    with Path(path).open("rb") as f:
        return pickle.load(f)

class DeviationEvaluation:
    """
    1) Macro average precision and recall per true deviaiton label
    2) Get suffixes with target deviations and occurrence/position in predicted suffix samples -> updated to use SEQUENCE-INDEX positions (more robust)
    3) Likelihood evaluation:
       - keeps mean likelihood at target positions.
       - adds hitProb (fraction of samples that hit any true position).
    4) Visualization
    """
    def __init__(self, deviation_results: Iterable[dict]):
        self.deviation_results = list(deviation_results)
    
    # updated
    def precision_recall_macro_by_label_dev(self,
                                            weighted_macro: bool = False,
                                            return_counts: bool = False):
        """
        Compute per-label precision and recall (BINARY presence per case) only for labels appearing in the target set,
        then return macro-averages across those target labels.

        Aggregation:
        - weighted_macro=False (default): unweighted macro across labels (each label counts equally).
        - weighted_macro=True: support-weighted macro across labels, where support is the number of cases
          where the label is present in the TARGET (binary presence per case).

        Per label (per case):
        TP: pred 1, tgt 1
        FP: pred 1, tgt 0
        FN: pred 0, tgt 1
        TN: pred 0, tgt 0   (not needed for prec/rec)

        Notes:
        - Uses set() per case (binary) -> Only chek if label is in dev at least once!
        - Only evaluates labels that appear in the TARGET set (total_tgt > 0).
        """
        tgt_deviations = [dr.get('tgt_deviations', []) for dr in self.deviation_results]
        # pred_deviations: is dev in samples has higher prob of occurence than threshold
        pred_deviations = [dr.get('pred_deviations', []) for dr in self.deviation_results]

        # label-level accumulators across cases (binary presence)
        tp_label = defaultdict(int)
        fp_label = defaultdict(int)
        fn_label = defaultdict(int)
        total_tgt = defaultdict(int)  # number of cases where label is present in target (for filtering)

        # first pass: determine target labels (binary presence)
        for tgt_list in tgt_deviations:
            t_set = set(tgt_list)
            for lbl in t_set:
                total_tgt[lbl] += 1

        target_label_list = sorted(k for k, v in total_tgt.items() if v > 0)

        # second pass: compute TP/FP/FN per label (binary per case)
        for tgt_list, pred_list in zip(tgt_deviations, pred_deviations):
            t_set = set(tgt_list)
            p_set = set(pred_list)
            for lbl in target_label_list:
                t = (lbl in t_set)
                p = (lbl in p_set)
                if p and t:
                    tp_label[lbl] += 1
                elif p and (not t):
                    fp_label[lbl] += 1
                elif (not p) and t:
                    fn_label[lbl] += 1

        precision_per_label = {}
        recall_per_label = {}
        for lbl in target_label_list:
            denom_p = tp_label[lbl] + fp_label[lbl]
            denom_r = tp_label[lbl] + fn_label[lbl]
            precision_per_label[lbl] = tp_label[lbl] / denom_p if denom_p > 0 else 0.0
            recall_per_label[lbl] = tp_label[lbl] / denom_r if denom_r > 0 else 0.0

        if not weighted_macro:
            precision_macro = float(np.mean(list(precision_per_label.values()))) if precision_per_label else 0.0
            recall_macro = float(np.mean(list(recall_per_label.values()))) if recall_per_label else 0.0
        else:
            total_support = sum(total_tgt[lbl] for lbl in target_label_list)
            precision_macro = (
                float(sum(precision_per_label[lbl] * total_tgt[lbl] for lbl in target_label_list) / total_support)
                if total_support > 0 else 0.0
            )
            recall_macro = (
                float(sum(recall_per_label[lbl] * total_tgt[lbl] for lbl in target_label_list) / total_support)
                if total_support > 0 else 0.0
            )
            
        if not return_counts:
            return precision_macro, recall_macro, precision_per_label, recall_per_label

        pred_support = defaultdict(int)
        for pred_list in pred_deviations:
            for lbl in set(pred_list):
                pred_support[lbl] += 1

        counts = {"tgt_support": dict(total_tgt), "pred_support": dict(pred_support)}
        
        return precision_macro, recall_macro, precision_per_label, recall_per_label, counts
        
    # updated
    def precision_recall_macro_by_label_no_dev(self,
                                               zero_division: float = 1.0,
                                               weighted_macro: bool = False,
                                               return_counts: bool = False):
        """
        Compute per-label precision and recall for the no-deviation (opposite) using BINARY presence per case.
        Positive event = label is NOT present in prediction AND NOT present in target (per case).

        Per label (per case):
        TP: pred_no=1, tgt_no=1   (both absent)
        FP: pred_no=1, tgt_no=0   (absent in pred, present in tgt)
        FN: pred_no=0, tgt_no=1   (present in pred, absent in tgt)
        TN: pred_no=0, tgt_no=0

        same filter as dev).
        """
        tgt_deviations = [dr.get('tgt_deviations', []) for dr in self.deviation_results]
        pred_deviations = [dr.get('pred_deviations', []) for dr in self.deviation_results]

        # determine target labels (binary presence)
        total_tgt = defaultdict(int)
        for tgt_list in tgt_deviations:
            for lbl in set(tgt_list):
                total_tgt[lbl] += 1
        target_label_list = sorted(k for k, v in total_tgt.items() if v > 0)

        # accumulators for "no deviation" positives (binary per case)
        tp_no = defaultdict(int)
        pred_no_count = defaultdict(int)  # cases where pred says "no deviation" (lbl absent)
        true_no_count = defaultdict(int)  # cases where true is "no deviation" (lbl absent)

        for tgt_list, pred_list in zip(tgt_deviations, pred_deviations):
            t_set = set(tgt_list)
            p_set = set(pred_list)
            for lbl in target_label_list:
                pred_has = (lbl in p_set)
                tgt_has = (lbl in t_set)

                pred_no = (not pred_has)
                true_no = (not tgt_has)

                if pred_no:
                    pred_no_count[lbl] += 1
                if true_no:
                    true_no_count[lbl] += 1
                if pred_no and true_no:
                    tp_no[lbl] += 1

        precision_per_label = {}
        recall_per_label = {}
        for lbl in target_label_list:
            denom_p = pred_no_count[lbl]
            denom_r = true_no_count[lbl]

            precision_per_label[lbl] = (tp_no[lbl] / denom_p) if denom_p > 0 else float(zero_division)
            recall_per_label[lbl] = (tp_no[lbl] / denom_r) if denom_r > 0 else float(zero_division)

        if not weighted_macro:
            precision_macro = float(np.mean(list(precision_per_label.values()))) if precision_per_label else 0.0
            recall_macro = float(np.mean(list(recall_per_label.values()))) if recall_per_label else 0.0
        else:
            total_support = sum(true_no_count[lbl] for lbl in target_label_list)
            precision_macro = (
                float(sum(precision_per_label[lbl] * true_no_count[lbl] for lbl in target_label_list) / total_support)
                if total_support > 0 else 0.0
            )
            recall_macro = (
                float(sum(recall_per_label[lbl] * true_no_count[lbl] for lbl in target_label_list) / total_support)
                if total_support > 0 else 0.0
            )

        if not return_counts:
            return precision_macro, recall_macro, precision_per_label, recall_per_label

        # For no-dev, the positive class is "label absent".
        counts = {"pred_no": dict(pred_no_count), "true_no": dict(true_no_count)}
        
        return precision_macro, recall_macro, precision_per_label, recall_per_label, counts
    
    # data selection (used by plotting)
    def _cases_with_target_deviations(self) -> List[dict]:
        return [
            {
                'case_id': dr.get('case_id', None),
                'label': dr.get('label', None),
                'prefix': dr.get('prefix', []),
                
                'tgt_suffix': dr.get('tgt_suffix', []),
                'pred_suffix': dr.get('pred_suffix', []),
                
                'tgt_cleaned_aligns': dr.get('tgt_cleaned_aligns', []),
                'pred_cleaned_aligns': dr.get('pred_cleaned_aligns', []),
                
                'tgt_deviations': dr.get('tgt_deviations', []),
                'pred_deviations': dr.get('pred_deviations', []),
                
                'tgt_model_moves': dr.get('tgt_model_moves', None),
                'tgt_log_moves': dr.get('tgt_log_moves', None),
                
                'pred_model_moves': dr.get('pred_model_moves', None),
                'pred_log_moves': dr.get('pred_log_moves', None),
                'pred_model_moves_pos': dr.get('pred_model_moves_pos', None),
                'pred_log_moves_pos': dr.get('pred_log_moves_pos', None),
                'num_samples': dr.get('num_samples', None),
            }
            for dr in self.deviation_results
            if len(dr.get('tgt_deviations', []) or []) > 0
        ]

    # risk-only selection
    def _cases_risk(self) -> List[dict]:
        return [
            {
                'case_id': dr.get('case_id', None),
                'label': int(dr.get('label', 1)) if dr.get('label', None) is not None else None,
                'prefix': dr.get('prefix', []),
                
                'tgt_suffix': dr.get('tgt_suffix', []),
                'pred_suffix': dr.get('pred_suffix', []),
                
                'tgt_cleaned_aligns': dr.get('tgt_cleaned_aligns', []),
                'pred_cleaned_aligns': dr.get('pred_cleaned_aligns', []),
                
                'tgt_deviations': dr.get('tgt_deviations', []),
                'pred_deviations': dr.get('pred_deviations', []),
                
                'tgt_model_moves': dr.get('tgt_model_moves', None),
                'tgt_log_moves': dr.get('tgt_log_moves', None),
                
                'pred_model_moves': dr.get('pred_model_moves', None),
                'pred_log_moves': dr.get('pred_log_moves', None),
                'pred_model_moves_pos': dr.get('pred_model_moves_pos', None),
                'pred_log_moves_pos': dr.get('pred_log_moves_pos', None),
                'num_samples': dr.get('num_samples', None),
            }
            for dr in self.deviation_results
            if int(dr.get('label', 1)) == 0
        ]

    # Likelihood at target deviation positions
    def likelihood_at_target_positions(self):
        """
        Compute likelihood at target deviation positions for risk cases only.

        For each risk case:
        - If both tgt_model_moves and tgt_log_moves are empty: skip.
        - For each target move + target position, check if the corresponding predicted move exists.
        - If yes and the position exists in the predicted move's position-prob list, store the prob.

        Returns
        - (model_hits, log_hits, model_macro_avg, log_macro_avg), where each hits list is: [(((a,b), pos), prob), ...]

        Macro averages are computed per list (model vs log) as:
        - for each risk case, take the mean probability over target positions (misses contribute 0.0), then average those per-case means.
        """
        risk_cases = self._cases_risk()

        model_hits: List[Tuple[Tuple[Tuple[str, str], int], float]] = []
        log_hits: List[Tuple[Tuple[Tuple[str, str], int], float]] = []

        model_case_means: List[float] = []
        log_case_means: List[float] = []

        for rc in risk_cases:
            tgt_model = rc.get('tgt_model_moves') or {} # get target moves
            tgt_log = rc.get('tgt_log_moves') or {}
            if (not tgt_model) and (not tgt_log):
                continue

            pred_model_pos = rc.get('pred_model_moves_pos') or {} # get prediction moves
            pred_log_pos = rc.get('pred_log_moves_pos') or {}
            if not isinstance(pred_model_pos, dict):
                pred_model_pos = {}
            if not isinstance(pred_log_pos, dict):
                pred_log_pos = {}

            model_scores_at_targets: List[float] = [] # store the likelihoods per case for macro average
            log_scores_at_targets: List[float] = [] # 

            # iterate model moves: ('>>', label)
            for label, positions in (tgt_model or {}).items():# label: move, positions: list of pos in target suffix
                move = label
                pos_list = pred_model_pos.get(move) or [] # list of positions: [(pos, prob across samples), ...]
                pos_map = {int(p): float(prob) for (p, prob) in pos_list} # dict of list
                for pos in positions or []:
                    pos_i = int(pos) # get position
                    prob = float(pos_map.get(pos_i, 0.0)) # get probability
                    model_scores_at_targets.append(prob) # add prob for macro averaging
                    if pos_i in pos_map:
                        model_hits.append(((move, pos_i), prob)) # store move with right position and probability

            # same logic as for model moves
            # iterate through log moves: (label, '>>')
            for label, positions in (tgt_log or {}).items(): 
                # move as string
                move = label
                pos_list = pred_log_pos.get(move) or []
                pos_map = {int(p): float(prob) for (p, prob) in pos_list}
                for pos in positions or []:
                    pos_i = int(pos)
                    prob = float(pos_map.get(pos_i, 0.0))
                    log_scores_at_targets.append(prob)
                    if pos_i in pos_map:
                        log_hits.append(((move, pos_i), prob))

            if model_scores_at_targets:
                model_case_means.append(sum(model_scores_at_targets) / len(model_scores_at_targets))
            if log_scores_at_targets:
                log_case_means.append(sum(log_scores_at_targets) / len(log_scores_at_targets))

        model_macro_avg = (sum(model_case_means) / len(model_case_means)) if model_case_means else None
        log_macro_avg = (sum(log_case_means) / len(log_case_means)) if log_case_means else None

        return model_hits, log_hits, model_macro_avg, log_macro_avg


####### PLOTTING Prototype ########

    def plot_suffix_deviation_distribution(
        self,
        suffix_index: int,
        label: str,
        move: str,
        tgt_suff_move,
        pred_suff_move,
        pred_suffix_samples,
        pred_suff_sets=None,
        tgt_suffixes=None,
        figsize=(16.0, 8.0),
        dpi: int = 220,
        scale: float = 1.0,
        match_header_font: bool = False,
        style: str = "paper",   # "paper" | "dense" | "auto"
        show: bool = True,
        return_objects: bool = False):
        
        if not (0 <= suffix_index < len(tgt_suff_move)):
            raise IndexError("suffix_index out of range.")

        cases = self._cases_with_target_deviations()
        case = cases[suffix_index] if suffix_index < len(cases) else {}

        prefix = case.get("prefix") or []
        if tgt_suffixes is None:
            tgt_suffixes = [(c.get("tgt_suffix") or []) for c in cases]
        tgt_suffix = tgt_suffixes[suffix_index] if suffix_index < len(tgt_suffixes) else []

        samples_case = pred_suffix_samples[suffix_index] if suffix_index < len(pred_suffix_samples) else []
        sets_case = (pred_suff_sets[suffix_index] if (pred_suff_sets and suffix_index < len(pred_suff_sets)) else None)

        is_log = (move == "log")
        # Use the *actual* number of available samples for normalization.
        # Using the num_samples argument here can artificially deflate rates when fewer samples exist.
        denom_req = max(1, len(sets_case) if (sets_case is not None) else len(samples_case))
        highlight_tok = str(label) if is_log else None

        # Bigger typography for paper-ready figures (requested)
        # NOTE: These are intentionally large so the figure remains readable
        # when scaled down in a paper.
        scale = float(scale) if scale is not None else 1.0
        scale = 1.0 if not np.isfinite(scale) else max(0.5, scale)

        # Scale figure size and typography together to preserve proportions.
        figsize = (float(figsize[0]) * scale, float(figsize[1]) * scale)
        PAPER = dict(
            fs_title=int(round(28 * scale)),
            fs_h=int(round(22 * scale)),
            fs_t=int(round(19 * scale)),
            fs_s=int(round(16 * scale)),
            # Slightly smaller line widths -> earlier line breaks (prevents crowding into the right panel)
            max_chars=78,
            max_chars_s=72,
            # Force heavier line breaks in the left text block.
            # This keeps the right plot from being squeezed when exported.
            max_toks_per_line=3,
            max_ev_prefix=26,
            max_ev_sample=38,
            gap_h=0.038 * scale,
            gap_b=0.026 * scale,
        )
        DENSE = dict(
            fs_title=int(round(24 * scale)),
            fs_h=int(round(19 * scale)),
            fs_t=int(round(16 * scale)),
            fs_s=int(round(14 * scale)),
            max_chars=104,
            max_chars_s=96,
            max_toks_per_line=4,
            max_ev_prefix=32,
            max_ev_sample=46,
            gap_h=0.032 * scale,
            gap_b=0.022 * scale,
        )

        if style not in {"paper", "dense", "auto"}:
            style = "paper"
        if style == "auto":
            longish = (len(prefix) + len(tgt_suffix) >= 55) or any(len(s or []) >= 40 for s in (samples_case[:10] or []))
            cfg = DENSE if longish else PAPER
        else:
            cfg = PAPER if style == "paper" else DENSE

        # For paper figures, use a stacked layout (text block on top, plot below) so all
        # elements share the same width and we can avoid aggressive line breaking.
        paper_like = (style == "paper") or (style == "auto" and cfg == PAPER)
        stacked_layout = bool(paper_like)

        if stacked_layout:
            # Relax the paper layout's forced wrapping so long sequences can stay on one line.
            cfg = dict(cfg)
            cfg["max_chars"] = max(int(cfg.get("max_chars", 80)), 160)
            cfg["max_chars_s"] = max(int(cfg.get("max_chars_s", 72)), 160)
            cfg["max_toks_per_line"] = None

        fs_title, fs_h, fs_t, fs_s = cfg["fs_title"], cfg["fs_h"], cfg["fs_t"], cfg["fs_s"]
        if bool(match_header_font):
            # Make all text (prefix/sample tokens + plot labels/ticks) match the section-header size.
            fs_t = fs_h
            fs_s = fs_h
        GAP_H, GAP_B = cfg["gap_h"], cfg["gap_b"]
        USE_TEX = bool(plt.rcParams.get("text.usetex", False))

        def _maybe_bold_lines(s: str) -> str:
            """Make text bold; for usetex use \textbf{...} per line."""
            s = str(s)
            if not USE_TEX:
                return s
            return "\n".join([r"\textbf{" + ln + "}" for ln in s.split("\n")])

        # ---------- helpers ----------
        def _wrap(s: str, w: int) -> str:
            s = str(s)
            return s if len(s) <= w else "\n".join(textwrap.wrap(s, width=w, break_long_words=False, break_on_hyphens=False))

        def _fmt_key_for_title(k: Tuple[str, str]) -> str:
            """Format the move tuple for display; keep '>>' readable under LaTeX."""
            a, b = str(k[0]), str(k[1])
            use_tex = bool(plt.rcParams.get("text.usetex", False))
            if use_tex and b == ">>":
                # Use \textgreater to reliably render '>' in LaTeX text mode.
                b = r"\textgreater\textgreater"
            return f"('{a}', '{b}')"

        def _clip_indexed(tokens, n, tail=False):
            t = [(i, str(x)) for i, x in enumerate(tokens or [])]
            if len(t) <= n:
                return t
            ell = [(-1, "…")]
            return (ell + t[-(n - 1):]) if tail else (t[: n - 1] + ell)

        def _token_lines(
            tokens,
            *,
            max_events,
            max_chars,
            tail: bool = False,
            max_toks_per_line: Optional[int] = None,
        ):
            toks = _clip_indexed(tokens, max_events, tail=tail)
            out: List[List[Tuple[int, str]]] = []
            cur: List[Tuple[int, str]] = []
            cur_len = 0
            for idx, tok in toks:
                add = len(tok) + (3 if cur else 0)  # " → "
                if cur and (
                    (cur_len + add > max_chars)
                    or (max_toks_per_line is not None and len(cur) >= int(max_toks_per_line))
                ):
                    out.append(cur)
                    cur, cur_len = [(idx, tok)], len(tok)
                else:
                    cur.append((idx, tok))
                    cur_len += add
            if cur:
                out.append(cur)
            return out or [[(-1, "(empty)")]]

        def _draw_tokens(
            ax,
            x,
            y_top,
            tokens,
            *,
            max_events,
            max_chars,
            max_toks_per_line: Optional[int] = None,
            tail=False,
            fontsize=10,
            hi_token: Optional[str] = None,
            hi_positions: Optional[Set[int]] = None,
            accent="tab:orange",
        ):
            base = "black"
            lines = _token_lines(
                tokens,
                max_events=max_events,
                max_chars=max_chars,
                tail=tail,
                max_toks_per_line=max_toks_per_line,
            )

            v = []
            for ln in lines:
                h = []
                for j, (idx, tok) in enumerate(ln):
                    if j:
                        h.append(TextArea(" → ", textprops={"fontsize": fontsize, "color": base}))
                    col = (
                        accent
                        if (
                            hi_token is not None
                            and hi_positions is not None
                            and idx in hi_positions
                            and tok == str(hi_token)
                        )
                        else base
                    )
                    h.append(TextArea(tok, textprops={"fontsize": fontsize, "color": col}))
                v.append(HPacker(children=h, align="baseline", pad=0, sep=0))

            box = VPacker(children=v, align="left", pad=0, sep=2)
            abox = AnchoredOffsetbox(
                loc="upper left",
                child=box,
                frameon=False,
                bbox_to_anchor=(x, y_top),
                bbox_transform=ax.transAxes,
                borderpad=0.0,
            )
            abox.set_clip_on(True)
            abox.set_clip_path(ax.patch)
            ax.add_artist(abox)
            return abox

        def _after_artist(fig, ax, artist, gap_axes: float) -> float:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = artist.get_window_extent(renderer=renderer)
            y0 = ax.transAxes.inverted().transform((bbox.x0, bbox.y0))[1]
            return y0 - gap_axes

        def _samples_with_token_at_pos(samples, pos: int, token: str):
            if not samples:
                return []
            out = []
            tok = str(token)
            for s in samples:
                if not s or len(s) <= pos:
                    continue
                if str(s[pos]) == tok:
                    out.append(list(s))
            return out

        # Target move positions are keyed by full move tuples, e.g. (label, '>>') or ('>>', label)
        tgt_case = tgt_suff_move[suffix_index] if (tgt_suff_move and suffix_index < len(tgt_suff_move)) else {}
        key = (str(label), ">>") if is_log else (">>", str(label))
        true_pos = sorted({int(p) for p in ((tgt_case or {}).get(key) or [])})

        # Predicted move position probabilities (preferred):
        # pred_*_moves_pos[move] == [(pos, prob), ...]
        pred_pos_list: List[Tuple[int, float]] = []
        if pred_suff_move and (suffix_index < len(pred_suff_move)):
            pred_case = pred_suff_move[suffix_index] or {}
            if isinstance(pred_case, dict):
                pred_pos_list = [(int(p), float(pr)) for (p, pr) in (pred_case.get(key) or [])]

        def _rate_from_sets(label_sets):
            pos_counts = defaultdict(int)
            for smap in (label_sets or []):
                for p in (smap.get(label) or set()):
                    pos_counts[int(p)] += 1
            pos = sorted(pos_counts)
            denom = max(1, len(label_sets or []))
            return pos, [pos_counts[p] / denom for p in pos], dict(pos_counts), denom

        if sets_case is not None:
            positions, probs, counts_shown, denom_shown = _rate_from_sets(sets_case)
        elif pred_pos_list:
            positions = [p for (p, _) in pred_pos_list]
            probs = [pr for (_, pr) in pred_pos_list]
            counts_shown = {int(p): int(round(float(pr) * denom_req)) for (p, pr) in pred_pos_list}
            denom_shown = denom_req
        else:
            positions, probs, counts_shown, denom_shown = [], [], {}, denom_req

        # ---------- seaborn “camera-ready” theme (local) ----------
        palette = sns.color_palette("colorblind")
        # Keep bars blue, but render highlighted tokens (e.g., the deviating label) in red.
        # Use the palette's red entry when available (colorblind-friendly).
        accent_idx = 3 if len(palette) > 3 else 1
        bar_color, accent = palette[0], palette[accent_idx]

        rc = {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "grid.linestyle": ":",
            "grid.alpha": 0.22,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
        ctx = "paper" if style != "dense" else "notebook"
        font_scale = (1.65 if style == "paper" else 1.35) * scale

        with sns.axes_style("ticks", rc=rc), sns.plotting_context(ctx, font_scale=font_scale, rc=rc):
            # deterministic layout (avoid constrained_layout quirks with offsetboxes)
            fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=False)
            if stacked_layout:
                # Use a dedicated header row + equal spacer rows for a clean, paper-like layout.
                fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.08, hspace=0.0)
            else:
                # Side-by-side layout spacing.
                wspace = 0.3 if paper_like else 0.04
                fig.subplots_adjust(left=0.035, right=0.99, top=0.89, bottom=0.12, wspace=wspace)

            if stacked_layout:
                # Layout order:
                # 1) header
                # 2) spacer (header → text)
                # 3) text block (Prefix + sampled event sequence)
                # 4) spacer (text → plot)
                # 5) plot (position distribution)
                # The two spacer rows have equal height, ensuring equal vertical gaps.
                hdr_h = 0.26
                gap_h = 0.14
                txt_h = 0.92
                plot_h = 1.25
                gs = fig.add_gridspec(5, 1, height_ratios=[hdr_h, gap_h, txt_h, gap_h, plot_h])

                ax_hdr = fig.add_subplot(gs[0, 0]); ax_hdr.axis("off")
                # Leave gs[1, 0] empty as a spacer row.
                ax_txt = fig.add_subplot(gs[2, 0])
                # Leave gs[3, 0] empty as a spacer row.
                ax_prob = fig.add_subplot(gs[4, 0])

                ax_hdr.text(
                    0.5,
                    0.52,
                    _maybe_bold_lines(f"Deviation position distribution -- {_fmt_key_for_title(key)}"),
                    ha="center",
                    va="center",
                    fontsize=fs_title,
                    fontweight="bold" if not USE_TEX else "normal",
                    transform=ax_hdr.transAxes,
                )
            else:
                # Slightly larger blocks (less wrapping/clipping)
                # Add a dedicated spacer column to enforce visible separation.
                # Layout: keep a small gap and give the right plot more width.
                # A too-large spacer column will squeeze the right plot.
                # Give the y-axis label/ticks some breathing room.
                spacer = 0.32 if paper_like else 0.18
                gs = fig.add_gridspec(1, 3, width_ratios=[1.05, spacer, 1.45])

                ax_txt = fig.add_subplot(gs[0, 0])
                ax_gap = fig.add_subplot(gs[0, 1])
                ax_gap.axis("off")
                ax_prob = fig.add_subplot(gs[0, 2])

            if not stacked_layout:
                # shrink + lift the bar axis -> visually “shorter” plot and better balance
                shrink = 0.70 if (len(true_pos) <= 1 and len(prefix) <= 2) else 0.78
                lift   = 0.14 if shrink < 0.78 else 0.10
                p = ax_prob.get_position()
                ax_prob.set_position([p.x0, p.y0 + p.height * lift, p.width, p.height * shrink])

            if not stacked_layout:
                fig.suptitle(
                    _maybe_bold_lines(f"Deviation position distribution -- {_fmt_key_for_title(key)}"),
                    fontsize=fs_title,
                    fontweight="bold" if not USE_TEX else "normal",
                    y=0.965,
                )

            if not stacked_layout:
                # ---------- aligned subtitles (left + right) ----------
                # Draw both subtitles at the same figure y-position to align their height.
                # Place them below the suptitle to increase vertical spacing.
                bbox_txt = ax_txt.get_position()
                bbox_prob = ax_prob.get_position()
                y_subtitle = min(0.93, float(fig.subplotpars.top) + 0.012)

                fig.text(
                    bbox_txt.x0 + 0.002,
                    y_subtitle,
                    _maybe_bold_lines(f"Prefix (len={len(prefix)}):"),
                    ha="left",
                    va="bottom",
                    fontsize=fs_h,
                    fontweight="bold" if not USE_TEX else "normal",
                )

                fig.text(
                    (bbox_prob.x0 + bbox_prob.x1) / 2,
                    y_subtitle,
                    _maybe_bold_lines(f"Predicted rate of {_fmt_key_for_title(key)}"),
                    ha="center",
                    va="bottom",
                    fontsize=fs_h,
                    fontweight="bold" if not USE_TEX else "normal",
                )

            # ---------- left text panel ----------
            ax_txt.axis("off")
            ax_txt.set_xlim(0, 1)
            ax_txt.set_ylim(0, 1)

            X, y = 0.01, 0.985  # tiny padding looks better than flush-left

            def add_header(text: str, y: float) -> float:
                hdr = _wrap(text, cfg["max_chars"])
                hdr = _maybe_bold_lines(hdr)
                t = ax_txt.text(
                    X,
                    y,
                    hdr,
                    ha="left",
                    va="top",
                    fontsize=fs_h,
                    fontweight="bold" if not USE_TEX else "normal",
                    clip_on=True,
                )
                return _after_artist(fig, ax_txt, t, GAP_H)

            def add_tokens(
                tokens,
                y: float,
                *,
                max_events,
                max_chars,
                tail: bool = False,
                hi_token: Optional[str] = None,
                hi_positions: Optional[Set[int]] = None,
            ) -> float:
                ab = _draw_tokens(
                    ax_txt,
                    X,
                    y,
                    tokens,
                    max_events=max_events,
                    max_chars=max_chars,
                    max_toks_per_line=cfg.get("max_toks_per_line"),
                    tail=tail,
                    fontsize=fs_t,
                    hi_token=hi_token,
                    hi_positions=hi_positions,
                    accent=accent,
                )
                return _after_artist(fig, ax_txt, ab, GAP_B)

            if stacked_layout:
                y = add_header(f"Prefix (len={len(prefix)}):", y)
            y = add_tokens(prefix, y, max_events=cfg["max_ev_prefix"], max_chars=cfg["max_chars"], tail=True)
            y -= GAP_B * 0.20

            sample_hdr = "Sampled event sequence:"
            # For the common single-position case, inline the Top-1 summary next to the header.
            # This avoids an extra standalone "Top 1..." line and looks cleaner in papers.
            if samples_case and true_pos and len(true_pos) == 1:
                p0 = int(true_pos[0])
                samples_at_pos = _samples_with_token_at_pos(samples_case, pos=p0, token=str(label))
                if samples_at_pos:
                    tgt_tuple = tuple(tgt_suffix or [])
                    ctr = Counter(tuple(s) for s in samples_at_pos if tuple(s) != tgt_tuple)
                    mc = ctr.most_common(1)
                    if mc:
                        c = int(mc[0][1])
                        sample_hdr = f"Sampled event sequence: Top 1. {c}/{denom_req} ({c})"

            y = add_header(sample_hdr, y)
            if not samples_case:
                t = ax_txt.text(X, y, "(no samples available)", ha="left", va="top", fontsize=fs_t, alpha=0.85, clip_on=True)
                y = _after_artist(fig, ax_txt, t, GAP_B)
            elif not true_pos:
                t = ax_txt.text(X, y, "(no deviating positions for this label)", ha="left", va="top", fontsize=fs_t, alpha=0.85, clip_on=True)
                y = _after_artist(fig, ax_txt, t, GAP_B)
            else:
                show_pos_header = len(true_pos) > 1
                for p0 in true_pos:
                    if y < 0.08:
                        ax_txt.text(X, max(0.02, y), "(truncated to fit figure)",
                                    ha="left", va="bottom", fontsize=fs_s, alpha=0.6, clip_on=True)
                        break

                    if show_pos_header:
                        y = add_header(f"Position {p0}", y)

                    # Consider only samples that actually contain the deviating label at this position.
                    samples_at_pos = _samples_with_token_at_pos(samples_case, pos=int(p0), token=str(label))
                    if not samples_at_pos:
                        t = ax_txt.text(X, y, "(none)", ha="left", va="top", fontsize=fs_t, alpha=0.85, clip_on=True)
                        y = _after_artist(fig, ax_txt, t, GAP_B)
                        continue

                    tgt_tuple = tuple(tgt_suffix or [])
                    match_count = sum(1 for s in samples_at_pos if tuple(s) == tgt_tuple)

                    # 1) Target-matching sampled suffix (if present)
                    if match_count > 0 and tgt_suffix:
                        t = ax_txt.text(
                            X,
                            y,
                            _maybe_bold_lines(f"Target match. {match_count}/{denom_req} ({match_count})"),
                            ha="left",
                            va="top",
                            fontsize=fs_t,
                            fontweight="bold" if not USE_TEX else "normal",
                            clip_on=True,
                        )
                        y = _after_artist(fig, ax_txt, t, GAP_B * 0.65)
                        y = add_tokens(
                            tgt_suffix,
                            y,
                            max_events=cfg["max_ev_sample"],
                            max_chars=cfg["max_chars_s"],
                            tail=False,
                            hi_token=highlight_tok,
                            hi_positions={int(p0)},
                        )
                        y -= GAP_B * 0.10

                    # 2) Top-1 most frequent sample (excluding target suffix if already shown)
                    ctr = Counter(tuple(s) for s in samples_at_pos if tuple(s) != tgt_tuple)
                    topk = [(c, list(seq)) for seq, c in ctr.most_common(1)]
                    if not topk:
                        if match_count == 0:
                            t = ax_txt.text(X, y, "(no samples to rank)", ha="left", va="top", fontsize=fs_t, alpha=0.85, clip_on=True)
                            y = _after_artist(fig, ax_txt, t, GAP_B)
                        y -= GAP_B * 0.15
                        continue

                    for i, (c, seq) in enumerate(topk, 1):
                        # If we already inlined Top-1 into the header (single-position),
                        # skip the redundant "Top 1..." label line.
                        if not (len(true_pos) == 1 and i == 1):
                            t = ax_txt.text(
                                X,
                                y,
                                _maybe_bold_lines(f"Top {i}. {c}/{denom_req} ({c})"),
                                ha="left",
                                va="top",
                                fontsize=fs_t,
                                fontweight="bold" if not USE_TEX else "normal",
                                clip_on=True,
                            )
                            y = _after_artist(fig, ax_txt, t, GAP_B * 0.65)
                        y = add_tokens(
                            seq,
                            y,
                            max_events=cfg["max_ev_sample"],
                            max_chars=cfg["max_chars_s"],
                            tail=False,
                            hi_token=highlight_tok,
                            hi_positions={int(p0)},
                        )
                    y -= GAP_B * 0.15

            # --- right bar panel (equal spacing) ---
            # In stacked layout, keep the plot free of an extra title to avoid collisions with the main header.
            # In side-by-side layout, the title is drawn as a figure-level subtitle above (aligned with left panel).
            ax_prob.set_xlabel("Position (start with index 0)", fontsize=fs_t)
            # Reduce padding so the y-axis label/ticks don't spill into the left panel.
            ax_prob.set_ylabel("Rate", fontsize=fs_t, labelpad=2)
            ax_prob.tick_params(axis="both", labelsize=fs_t)
            ax_prob.tick_params(axis="y", pad=2)
            ax_prob.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax_prob.yaxis.set_major_locator(MaxNLocator(6))
            ax_prob.grid(axis="y", linestyle=":", alpha=0.22)
            ax_prob.grid(axis="x", visible=False)

            if positions:
                pos_vals = list(map(int, positions))
                xs = list(range(len(pos_vals)))             # equally spaced bar locations

                bars = ax_prob.bar(xs, probs, color=bar_color, edgecolor="0.25", linewidth=0.8, width=0.78, zorder=3)
                ax_prob.set_xticks(xs)
                ax_prob.set_xticklabels([str(p) for p in pos_vals])

                ymax = max(probs) if probs else 0.0
                ax_prob.set_ylim(0, min(1.0, max(0.10, ymax * 1.18)))
                ax_prob.set_xlim(-0.6, len(xs) - 0.4)

                # Count labels with subtle white background (prevents collisions)
                for b, p in zip(bars, pos_vals):
                    c = int(counts_shown.get(int(p), 0))
                    ax_prob.text(
                        b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.004,
                        f"{c}/{denom_shown}",
                        ha="center", va="bottom",
                        fontsize=fs_s, alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
                        zorder=4,
                    )
            else:
                ax_prob.text(0.5, 0.5, "(no predicted occurrences)", ha="center", va="center", fontsize=fs_h)
                ax_prob.set_xticks([]); ax_prob.set_yticks([])

            sns.despine(ax=ax_prob)

            if show:
                plt.show()

            if return_objects:
                return fig, (ax_txt, ax_prob)
            return None