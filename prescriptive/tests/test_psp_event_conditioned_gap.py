"""Variant (e): the current event conditions the gap class in training and sampling."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from prescriptive.helpers.datasets import add_paths
add_paths()
from prescriptive.estimators.psp_model import ACTIVITY, CASE, GAP_CLASS, GAP_NAMES, LIFECYCLE, RESOURCE, TIME_COLUMNS, TIMESTAMP, PSPModel
from evaluation_v2.probabilistic_evaluation import ProbabilisticEvaluation
from model.dropout_uncertainty_enc_dec_LSTM.dropout_uncertainty_model import DropoutUncertaintyEncoderDecoderLSTM


EVENTS = {"PAD": 0, "A|start": 1, "A|complete": 2, "Decision|complete": 3, "EOS": 4}
CLASSES = {name: i + 1 for i, name in enumerate(GAP_NAMES)}


def network(conditional=True):
    with contextlib.redirect_stdout(io.StringIO()):
        return DropoutUncertaintyEncoderDecoderLSTM(
            data_set_categories=[[(ACTIVITY, len(EVENTS), EVENTS), (GAP_CLASS, len(CLASSES) + 1, CLASSES)],
                                 [(name, 1) for name in TIME_COLUMNS]],
            enc_feat=[[ACTIVITY, GAP_CLASS], TIME_COLUMNS], dec_feat=[[ACTIVITY, GAP_CLASS], TIME_COLUMNS[:2]],
            seq_len_pred=3, hidden_size=8, num_layers=1, dropout=0.0,
            categorical_conditioning=(GAP_CLASS, ACTIVITY) if conditional else None)


def inputs():
    prefixes = [[torch.full((2, 3), 3), torch.ones(2, 3, dtype=torch.long)],
                [torch.zeros(2, 3) for _ in TIME_COLUMNS]]
    suffixes = [[torch.tensor([[2, 2, 1], [1, 2, 1]]), torch.tensor([[5, 5, 1], [1, 5, 1]])],
                [torch.zeros(2, 3) for _ in TIME_COLUMNS]]
    return prefixes, suffixes


def distinguish_lifecycles(net):
    """Same activity, different lifecycle: force class 0s vs <3d through the event input."""
    with torch.no_grad():
        net.decoder.condition_embedding.weight.zero_()
        net.decoder.condition_embedding.weight[1, 0] = 1
        net.decoder.condition_embedding.weight[2, 1] = 1
        head = net.decoder.output_layers[GAP_CLASS + "_mean"]
        head.weight.zero_()
        head.bias.fill_(-100)
        head.weight[1, net.hidden_size] = 200
        head.weight[5, net.hidden_size + 1] = 200
        for key in (GAP_CLASS, ACTIVITY):
            net.decoder.output_layers[key + "_var"].weight.zero_()
            net.decoder.output_layers[key + "_var"].bias.fill_(-20)
        event = net.decoder.output_layers[ACTIVITY + "_mean"]
        event.weight.zero_()
        event.bias.fill_(-100)
        event.bias[1] = 100


def sampling_model(conditional=True):
    model = PSPModel.__new__(PSPModel)
    model.params = {"gap_classes": True, "gap_condition_on_event": conditional, "time_from": "gaps"}
    model.net = network(conditional)
    if conditional:
        distinguish_lifecycles(model.net)
    model.net.eval()
    model.reference = SimpleNamespace(all_categories=[[(GAP_CLASS, len(CLASSES) + 1, CLASSES)], []])
    scalers = {name: StandardScaler().fit(np.array([[-1.], [1.]])) for name in TIME_COLUMNS}
    model.encoder_decoder = SimpleNamespace(continuous_encoders=scalers, window_size=8, min_suffix_size=2)
    ev = ProbabilisticEvaluation.__new__(ProbabilisticEvaluation)
    ev.inverted_suffix_categories = [{value: name for name, value in EVENTS.items()}]
    ev.all_cat_attributes, ev.all_num_attributes = [ACTIVITY, GAP_CLASS], TIME_COLUMNS[:2]
    ev.prefix_num_attributes = TIME_COLUMNS
    ev.use_variance_cat, ev.sample_argmax, ev.eos_id = True, False, EVENTS["EOS"]
    model._cached_evaluator = ev
    prefixes, _ = inputs()
    encoded = [([prefixes[0][0][:1], prefixes[0][1][:1]], [v[:1] for v in prefixes[1]]),
               (None, None), torch.zeros(1, 3), 3]
    return model, [encoded]


class EventConditionedGapTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_current_event_changes_class_without_changing_event_or_continuous_heads(self):
        net = network()
        distinguish_lifecycles(net)
        prefixes, suffixes = inputs()
        other = [[v.clone() for v in suffixes[0]], suffixes[1]]
        other[0][0][:, 0] = 1
        net.train()
        torch.manual_seed(11)
        first = net(prefixes, suffixes=suffixes, teacher_forcing_ratio=1)[0]
        torch.manual_seed(11)
        second = net(prefixes, suffixes=other, teacher_forcing_ratio=1)[0]
        self.assertTrue(torch.equal(first[0][ACTIVITY + "_mean"][0], second[0][ACTIVITY + "_mean"][0]))
        for key in first[1]:
            self.assertTrue(torch.equal(first[1][key][0], second[1][key][0]))
        self.assertEqual(first[0][GAP_CLASS + "_mean"][0, 0].argmax().item(), 5)
        self.assertEqual(second[0][GAP_CLASS + "_mean"][0, 0].argmax().item(), 1)

    def test_training_and_validation_use_current_target_but_feedback_uses_predicted_event(self):
        net = network()
        distinguish_lifecycles(net)
        prefixes, suffixes = inputs()
        for training, ratio in ((True, 0.0), (True, 1.0), (False, 0.0)):
            with self.subTest(training=training, teacher_forcing=ratio):
                calls = []
                def observe(module, args, kwargs):
                    calls.append(kwargs)
                handle = net.decoder.register_forward_pre_hook(observe, with_kwargs=True)
                net.train(training)
                try:
                    net(prefixes, suffixes=suffixes, teacher_forcing_ratio=ratio)
                finally:
                    handle.remove()
                self.assertEqual(len(calls), 3)
                for step, call in enumerate(calls):
                    self.assertTrue(torch.equal(call["condition_event"], suffixes[0][0][:, step:step+1]))
                if not training or ratio == 0:
                    # The event head predicts A|start even when the target is A|complete.
                    self.assertTrue(torch.all(calls[1]["input"][0][0] == 1))
                    self.assertTrue(torch.all(calls[1]["input"][0][1] == 1))
                else:
                    self.assertTrue(torch.equal(calls[1]["input"][0][1], suffixes[0][1][:, :1]))

    def test_conditional_head_and_event_embedding_receive_gradients(self):
        net = network()
        prefixes, suffixes = inputs()
        result = net(prefixes, suffixes=suffixes, teacher_forcing_ratio=0)[0]
        result[0][GAP_CLASS + "_mean"].square().mean().backward()
        for param in (net.decoder.condition_embedding.weight, net.decoder.output_layers[GAP_CLASS + "_mean"].weight):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.abs().sum().item(), 0)

    def test_forced_option_conditions_class_before_gap_is_drawn(self):
        model, encoded = sampling_model()
        for token, low, high in (("A|start", 0., 1.), ("A|complete", 86400., 3*86400.)):
            with self.subTest(token=token):
                drawn = model.draw_cases(encoded, 24, first=[token], max_steps=1)[0]
                self.assertEqual(len(drawn), 24)
                gaps = [suffix[0][2] for suffix in drawn]
                self.assertTrue(all(suffix[0][0] == token for suffix in drawn))
                self.assertTrue(all(low <= gap <= high for gap in gaps))
                self.assertGreater(len(set(gaps)), 1)

    def test_sampled_event_conditions_every_step(self):
        model, encoded = sampling_model()
        drawn = model.draw_cases(encoded, 12, max_steps=3)[0]
        self.assertTrue(all(len(suffix) == 3 for suffix in drawn))
        self.assertTrue(all(token == "A|start" and 0 <= gap <= 1 for suffix in drawn for token, _, gap in suffix))

    def test_e_setting_cannot_relabel_a_d_network(self):
        model, encoded = sampling_model(conditional=False)
        model.params["gap_condition_on_event"] = True
        with self.assertRaisesRegex(ValueError, "trainiertes Netz"):
            model.draw_cases(encoded, 1)

    def test_checkpoint_roundtrip_and_legacy_defaults(self):
        prefixes, suffixes = inputs()
        for conditional in (False, True):
            with self.subTest(conditional=conditional), tempfile.TemporaryDirectory() as directory:
                net = network(conditional)
                path = str(Path(directory) / "net.pt")
                net.save(path)
                if not conditional:
                    # Legacy checkpoints do not have the new optional keyword.
                    checkpoint = torch.load(path, weights_only=False)
                    checkpoint["kwargs"].pop("categorical_conditioning")
                    torch.save(checkpoint, path)
                    del net.categorical_conditioning
                    del net.decoder.categorical_conditioning
                with contextlib.redirect_stdout(io.StringIO()):
                    loaded = DropoutUncertaintyEncoderDecoderLSTM.load(path)
                self.assertEqual(set(net.state_dict()), set(loaded.state_dict()))
                for mode in (True, False):
                    net.train(mode)
                    loaded.train(mode)
                    torch.manual_seed(21)
                    before = net(prefixes, suffixes=suffixes, teacher_forcing_ratio=0)[0]
                    torch.manual_seed(21)
                    after = loaded(prefixes, suffixes=suffixes, teacher_forcing_ratio=0)[0]
                    for part in (0, 1):
                        for key in before[part]:
                            self.assertTrue(torch.equal(before[part][key], after[part][key]))

    def test_e_implies_separate_classes_and_rejects_joint_tokens(self):
        model = PSPModel("bpic17", points=[], params={"gap_condition_on_event": True})
        self.assertTrue(model.gap_classes())
        self.assertEqual(model.decoder()[0], [ACTIVITY, GAP_CLASS])
        with self.assertRaises(ValueError):
            PSPModel("bpic17", points=[], params={"gap_condition_on_event": True, "gap_tokens": True})

    def test_fit_checkpoint_and_encoder_option_sampling(self):
        from interventions import ChoicePoint
        import prescriptive.estimators.psp_model as psp_module
        import trainer.trainer as trainer_module

        records = []
        for case in range(16):
            now = pd.Timestamp("2025-01-01") + pd.Timedelta(days=case * 4)
            path = [("Begin", "complete", 0), ("Decision", "complete", 10),
                    ("A", "start" if case % 2 else "complete", 20 if case % 2 else 90000),
                    ("Middle", "complete", 600 + case), ("Middle", "complete", 30),
                    ("End", "complete", 86400 + case)]
            for activity, lifecycle, gap in path:
                now += pd.Timedelta(seconds=gap)
                records.append({CASE: str(case), ACTIVITY: activity, LIFECYCLE: lifecycle,
                                RESOURCE: "worker", TIMESTAMP: now})
        log = pd.DataFrame(records)
        point = ChoicePoint("test", "control_flow", "Decision", "complete", ("A|start", "A|complete"))
        original_loader = trainer_module.DataLoader

        def serial_loader(*args, **kwargs):
            kwargs.update(num_workers=0, pin_memory=False)
            return original_loader(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings(), \
                contextlib.redirect_stdout(io.StringIO()), \
                patch.dict(psp_module.MODEL_PARAMS, hidden_size=8, num_layers=1), \
                patch.object(trainer_module, "DataLoader", serial_loader):
            warnings.simplefilter("ignore", FutureWarning)
            model = PSPModel("bpic17", points=[point], params={
                "gap_condition_on_event": True, "epochs": 1, "train_cases": 0, "suffix_size": 3,
                "threads": 1, "batch_size": 16, "statics": ([], []), "device": "cpu",
                "point_weight": 3, "time_from": "gaps", "checkpoints": directory}).fit(log)
            self.assertEqual(model.summary["alle Prefixe"]["Epochen"], 1)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.net.parameters()))
            checkpoint = PSPModel.load_checkpoint(directory)
            self.assertEqual(checkpoint.net.categorical_conditioning, (GAP_CLASS, ACTIVITY))
            for key, value in model.net.state_dict().items():
                self.assertTrue(torch.equal(value, checkpoint.net.state_dict()[key]))
            model.params["option_as"] = "encoder"
            events = log[log[CASE] == "0"].iloc[:2].to_dict("records")
            now = events[-1][TIMESTAMP].timestamp()
            torch.manual_seed(10)
            before = model.option_suffixes_many(point, [events], list(point.options), now, 6)
            for option, cases in zip(point.options, before[0]):
                self.assertEqual(len(cases[0]), 6)
                self.assertTrue(all(suffix[0][0] == option for suffix in cases[0]))
                self.assertTrue(all(np.isfinite(suffix[0][2]) and suffix[0][2] >= 0 for suffix in cases[0]))
                self.assertGreater(len(set(suffix[0][2] for suffix in cases[0])), 1)
            path = str(Path(directory) / "fitted.pkl")
            model.save(path)
            loaded = PSPModel.load(path)
            torch.manual_seed(10)
            after = loaded.option_suffixes_many(point, [events], list(point.options), now, 6)
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
