"""Lock published report statistics to the artifacts that produced them.

Three defects reached published PHL-DAM reports, all from numbers typed into
markdown by hand:

* a results table that mixed canonical W=12 rows with canonical W=16 rows;
* a run count that did not match the artifact set;
* an "N of twelve" tally that was off by two.

Each is pinned below against the raw JSON. If an artifact set changes, these
fail rather than letting a report quietly drift out of agreement with it.
"""

import math
import unittest
from pathlib import Path

import phl_dam_report_stats as stats

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def _available(pattern: str) -> list[dict]:
    return stats.load(pattern, OUTPUTS)


class SettingSeparationTests(unittest.TestCase):
    """A recall figure must never be quoted without its evaluation setting."""

    def test_dilation_and_full_scale_share_exactly_one_setting(self) -> None:
        runs = _available("phl_dam_004c_*.json")
        if not runs:
            self.skipTest("004C artifacts not present")
        dilation = [r for r in runs if "dilate" in str(r["configuration"]["scale"])]
        full = [r for r in runs if r["configuration"]["scale"] == "full"]
        self.assertTrue(dilation and full)
        shared = set(stats.settings_of(dilation)) & set(stats.settings_of(full))
        self.assertEqual(
            shared,
            {"canonical_w16"},
            "canonical_w16 is the only setting both families expose; any table "
            "comparing them must use it",
        )

    def test_per_setting_table_never_merges_settings(self) -> None:
        runs = _available("phl_dam_004bs_*.json")
        if not runs:
            self.skipTest("004B-S artifacts not present")
        table = stats.per_setting_table(runs)
        for setting, block in table.items():
            for arm, row in block.items():
                for run in runs:
                    if run["model"] != arm:
                        continue
                    seed = run["configuration"]["seed"]
                    self.assertAlmostEqual(
                        row["per_seed_recall"][seed],
                        run["metrics"][setting][arm]["recall"],
                        places=12,
                        msg=f"{arm} seed{seed} {setting}",
                    )


class RunAccountingTests(unittest.TestCase):
    def test_004c_run_counts_match_the_artifact_set(self) -> None:
        runs = _available("phl_dam_004c_*.json")
        if not runs:
            self.skipTest("004C artifacts not present")
        report = stats.health(runs)
        dilation = [r for r in runs if "dilate" in str(r["configuration"]["scale"])]
        full = [r for r in runs if r["configuration"]["scale"] == "full"]
        self.assertEqual(report["total_runs"], 10)
        self.assertEqual(len(dilation), 8)
        self.assertEqual(len(full), 2)
        self.assertEqual(report["non_finite_runs"], 1)

    def test_finite_and_non_finite_partition_the_runs(self) -> None:
        for pattern in ("phl_dam_004c_*.json", "phl_dam_004bs_*.json", "phl_dam_004d_*.json"):
            runs = _available(pattern)
            if not runs:
                continue
            report = stats.health(runs)
            self.assertEqual(
                report["finite_runs"] + report["non_finite_runs"], report["total_runs"], pattern
            )


class RunIdentityTests(unittest.TestCase):
    def test_run_ids_are_unique_across_conditions(self) -> None:
        """`model_seedN` collides when one arm runs at several write levels."""
        runs = _available("phl_dam_004d_*.json")
        if not runs:
            self.skipTest("004D artifacts not present")
        ids = [stats.run_id(r) for r in runs]
        self.assertEqual(len(ids), len(set(ids)), "run ids must identify a run uniquely")
        report = stats.health(runs)
        self.assertEqual(
            len(report["non_finite_ids"]),
            len(set(report["non_finite_ids"])),
            "non-finite listing must not merge distinct runs",
        )


class TallyTests(unittest.TestCase):
    def test_timing_auroc_tally_is_computed_not_asserted(self) -> None:
        runs = _available("phl_dam_004bs_*.json")
        if not runs:
            self.skipTest("004B-S artifacts not present")
        values = [
            r["metrics"]["canonical_w16"][r["model"]]["timing_head"]["live_vs_never_auroc"]
            for r in runs
        ]
        tally = stats.count_within(values, target=0.8490, tolerance=0.005)
        self.assertEqual(tally["total"], 12)
        self.assertEqual(
            tally["within"], 8, "v1 of the report claimed ten; the artifacts say eight"
        )
        self.assertEqual(tally["phrase"], "8 of 12")


class PairedStatisticsTests(unittest.TestCase):
    def test_paired_reports_signs_and_a_bootstrap_interval(self) -> None:
        result = stats.paired([-0.3682, 0.0003, 0.0119])
        self.assertEqual(result["n"], 3)
        self.assertEqual((result["positive"], result["negative"], result["ties"]), (2, 1, 0))
        self.assertLess(result["mean"], 0.0)
        self.assertGreater(result["median"], 0.0)
        low, high = result["bootstrap_95"]
        self.assertLess(low, result["mean"])
        self.assertGreaterEqual(high, result["median"] - 1e-9)

    def test_a_mean_and_a_median_can_disagree_in_sign(self) -> None:
        """The reason raw means alone were not good enough for these contrasts."""
        result = stats.paired([-0.3682, 0.0003, 0.0119])
        self.assertNotEqual(result["mean"] > 0, result["median"] > 0)

    def test_learned_uses_the_final_value_not_a_transient_dip(self) -> None:
        dipped = {"training_history": [{"step": 1, "recall_ce": 0.5}, {"step": 2, "recall_ce": 2.4}]}
        held = {"training_history": [{"step": 1, "recall_ce": 2.4}, {"step": 2, "recall_ce": 1.2}]}
        self.assertFalse(stats.learned(dipped))
        self.assertTrue(stats.learned(held))
        self.assertEqual(stats.breakthrough_step(dipped), 1)

    def test_summarise_ignores_non_finite_values(self) -> None:
        row = stats.summarise([1.0, 2.0, float("nan"), None, 3.0])
        self.assertEqual(row["n"], 3)
        self.assertAlmostEqual(row["mean"], 2.0)
        self.assertAlmostEqual(row["median"], 2.0)


if __name__ == "__main__":
    unittest.main()
