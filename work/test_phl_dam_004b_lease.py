"""Invariants for the finite-capacity, lease-carrying PHL-DAM.

These encode the preregistered claims of PHL-DAM-004B: capacity is real,
eviction is real, key/value content never moves through the lease lattice,
the lease is causal, and no arm but the oracle ever sees the future.
"""

import copy
import io
import unittest

import torch
from torch import nn

import phl_dam_pressure_task as task
import phl_dam_004b_lease as lease


def _batch(count: int = 4, writes: int = 24) -> lease.Batch:
    episodes = task.generate_episodes(7, count, writes, "canonical")
    return lease.pack_batch(episodes, torch.device("cpu"))


def _eager_writer(arm: str = "phl_lease") -> lease.PHLDAMLease:
    """A model whose write gate is saturated, so writes actually commit."""
    torch.manual_seed(3)
    model = lease.PHLDAMLease(arm=arm)
    nn.init.constant_(model.write_gate.bias, 12.0)
    nn.init.zeros_(model.write_gate.weight)
    return model


class CapacityTests(unittest.TestCase):
    def test_occupancy_never_exceeds_the_slot_count(self) -> None:
        model = _eager_writer()
        batch = _batch()
        with torch.no_grad():
            _, diagnostics = model(batch.tokens, collect=True)
        occupied = diagnostics["occupancy"].gt(0.5).sum(dim=-1)
        self.assertLessEqual(int(occupied.max()), model.num_slots)
        owner = diagnostics["owner_token"]
        self.assertLessEqual(int(owner.ge(0).sum(dim=-1).max()), model.num_slots)

    def test_a_free_slot_is_taken_before_any_occupied_slot(self) -> None:
        """The occupancy term must dominate the eviction term while a slot is free.

        Allocation is a softmax, so the order in which several empty slots are
        claimed is not fixed. What must hold is that an unoccupied slot always
        outranks every occupied one, whatever the eviction score says - no
        occupant is evicted while capacity remains.
        """
        model = _eager_writer()
        device = torch.device("cpu")
        for free_slot in (0, 3, 7):
            for arm in ("lru", "fifo", "phl_lease"):
                state = model.init_state(1, device)
                state.dam.occupancy[:] = 1.0
                state.dam.occupancy[0, free_slot] = 0.0
                state.dam.owner_token[0] = torch.arange(
                    task.KEY_START, task.KEY_START + model.num_slots
                )
                state.dam.owner_token[0, free_slot] = -1
                # Make the free slot look maximally attractive to keep, so only
                # the occupancy term can be responsible for choosing it.
                state.dam.inserted_at[0] = torch.full((model.num_slots,), 90.0)
                state.dam.last_access[0] = torch.full((model.num_slots,), 90.0)
                state.dam.inserted_at[0, free_slot] = 99.0
                state.dam.last_access[0, free_slot] = 99.0
                state.step_index = 100
                token = torch.randn(1, model.d_model)
                _, new_state, trace = model.step(
                    torch.zeros(1, model.d_model),
                    token,
                    token,
                    state,
                    arm=arm,
                    previous_token_ids=torch.tensor([task.KEY_START + 40]),
                    current_token_ids=torch.tensor([task.VALUE_START]),
                )
                self.assertEqual(int(trace.victim_slot[0]), free_slot, (free_slot, arm))
                self.assertEqual(
                    int(new_state.dam.owner_token[0, free_slot]), task.KEY_START + 40
                )

    def test_eviction_replaces_exactly_the_selected_slot(self) -> None:
        model = _eager_writer()
        device = torch.device("cpu")
        state = model.init_state(1, device)
        state.dam.occupancy[:] = 1.0
        state.dam.owner_token[0] = torch.arange(
            task.KEY_START, task.KEY_START + model.num_slots
        )
        state.dam.inserted_at[0] = torch.tensor([5.0, 1.0, 9.0, 2.0, 8.0, 7.0, 6.0, 4.0])
        state.dam.last_access[0] = state.dam.inserted_at[0].clone()
        before = state.dam.owner_token.clone()

        token = torch.randn(1, model.d_model)
        new_key = task.KEY_START + 40
        _, state, trace = model.step(
            torch.zeros(1, model.d_model),
            token,
            token,
            state,
            arm="fifo",
            previous_token_ids=torch.tensor([new_key]),
            current_token_ids=torch.tensor([task.VALUE_START]),
        )
        victim = int(trace.victim_slot[0])
        self.assertEqual(victim, 1)  # inserted_at is smallest at slot 1
        after = state.dam.owner_token
        self.assertEqual(int(after[0, victim]), new_key)
        for slot in range(model.num_slots):
            if slot != victim:
                self.assertEqual(int(after[0, slot]), int(before[0, slot]))


class PolicyChoiceTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = lease.PHLDAMLease(arm="phl_lease")
        self.state = self.model.init_state(1, torch.device("cpu"))
        self.state.dam.occupancy[:] = 1.0
        self.state.dam.owner_token[0] = torch.arange(
            task.KEY_START, task.KEY_START + self.model.num_slots
        )
        self.state.dam.inserted_at[0] = torch.tensor(
            [5.0, 1.0, 9.0, 2.0, 8.0, 7.0, 6.0, 4.0]
        )
        self.state.dam.last_access[0] = torch.tensor(
            [50.0, 90.0, 95.0, 20.0, 80.0, 70.0, 60.0, 40.0]
        )
        self.state.step_index = 100

    def test_fifo_scores_pick_the_oldest_insertion(self) -> None:
        score = self.model.eviction_score(self.state, "fifo", None, None)
        self.assertEqual(int(score.argmin()), 1)

    def test_lru_scores_pick_the_least_recently_accessed(self) -> None:
        score = self.model.eviction_score(self.state, "lru", None, None)
        self.assertEqual(int(score.argmin()), 3)

    def test_oracle_prefers_a_never_needed_slot(self) -> None:
        next_use = torch.full((1, self.model.num_slots), 150.0)
        next_use[0, 5] = float(lease.INFINITY)
        score = self.model.eviction_score(self.state, "oracle", None, next_use)
        self.assertEqual(int(score.argmin()), 5)

    def test_oracle_picks_the_farthest_next_use_when_all_are_needed(self) -> None:
        next_use = torch.tensor([[110.0, 400.0, 150.0, 130.0, 200.0, 180.0, 170.0, 160.0]])
        score = self.model.eviction_score(self.state, "oracle", None, next_use)
        self.assertEqual(int(score.argmin()), 1)

    def test_random_scores_stay_inside_the_unit_interval(self) -> None:
        generator = torch.Generator().manual_seed(4)
        score = self.model.eviction_score(self.state, "random", generator, None)
        self.assertTrue(bool((score >= 0).all() and (score <= 1).all()))


class LeaseSeparationTests(unittest.TestCase):
    def test_lease_content_never_enters_the_key_or_value_path(self) -> None:
        """K/V must be a function of tokens and allocation, never of the lease."""
        model = _eager_writer(arm="content_only")
        device = torch.device("cpu")
        left = model.init_state(2, device)
        left.dam.occupancy[:] = 1.0
        left.dam.keys.normal_()
        left.dam.values.normal_()
        right = copy.deepcopy(left)
        right.dam.leases.copy_(torch.rand_like(right.dam.leases))
        token = torch.randn(2, model.d_model)
        context = torch.randn(2, model.d_model)
        args = dict(
            arm="content_only",
            previous_token_ids=torch.tensor([task.KEY_START, task.KEY_START + 1]),
            current_token_ids=torch.tensor([task.VALUE_START, task.VALUE_START]),
        )
        _, left_state, _ = model.step(context, token, token, left, **args)
        _, right_state, _ = model.step(context, token, token, right, **args)
        self.assertTrue(torch.equal(left_state.dam.keys, right_state.dam.keys))
        self.assertTrue(torch.equal(left_state.dam.values, right_state.dam.values))
        self.assertFalse(torch.equal(left.dam.leases, right.dam.leases))

    def test_lease_does_influence_the_eviction_score(self) -> None:
        """The separation above must not be because the lease is inert."""
        torch.manual_seed(11)
        model = lease.PHLDAMLease(arm="phl_lease")
        state = model.init_state(1, torch.device("cpu"))
        state.dam.occupancy[:] = 1.0
        first = model.eviction_score(state, "phl_lease", None, None).clone()
        state.dam.leases.copy_(torch.rand_like(state.dam.leases))
        second = model.eviction_score(state, "phl_lease", None, None)
        self.assertFalse(torch.allclose(first, second))

    def test_content_never_migrates_between_slots(self) -> None:
        model = _eager_writer(arm="lru")
        device = torch.device("cpu")
        base = model.init_state(1, device)
        base.dam.occupancy[:] = 1.0
        base.dam.keys.normal_()
        base.dam.values.normal_()
        perturbed = copy.deepcopy(base)
        perturbed.dam.values[0, 3] += 5.0
        token = torch.randn(1, model.d_model)
        context = torch.randn(1, model.d_model)
        args = dict(
            arm="lru",
            previous_token_ids=torch.tensor([task.KEY_START]),
            current_token_ids=torch.tensor([task.VALUE_START]),
        )
        _, first, trace = model.step(context, token, token, base, **args)
        _, second, _ = model.step(context, token, token, perturbed, **args)
        victim = int(trace.victim_slot[0])
        for slot in range(model.num_slots):
            if slot in (3, victim):
                continue
            self.assertTrue(
                torch.equal(first.dam.values[0, slot], second.dam.values[0, slot]),
                slot,
            )

    def test_lease_evolves_while_the_transport_leaves_content_alone(self) -> None:
        model = lease.PHLDAMLease(arm="phl_lease")
        state = model.init_state(1, torch.device("cpu"))
        state.dam.leases[0, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        transport = model.lease_transport
        current = state.dam.leases[0, 0].clone()
        history = [current.clone()]
        for _ in range(400):
            current = transport.T @ current
            history.append(current.clone())
        self.assertAlmostEqual(float(history[-1].sum()), 1.0, places=4)
        # A "far" lease must migrate towards "due" as its horizon approaches.
        self.assertGreater(float(history[-1][0]) + float(history[-1][5]),
                           float(history[0][0]) + float(history[0][5]))
        self.assertLess(float(history[-1][4]), float(history[0][4]))

    def test_identity_transport_makes_phl_lease_equal_static_priority(self) -> None:
        """Lease transport OFF must reproduce the static-priority arm exactly."""
        torch.manual_seed(5)
        transported = lease.PHLDAMLease(arm="phl_lease")
        static = lease.PHLDAMLease(arm="static_priority")
        static.load_state_dict(transported.state_dict())
        with torch.no_grad():
            transported.lease_transport.copy_(
                torch.eye(lease.NUM_LEASE_BINS)
            )
        batch = _batch(2)
        with torch.no_grad():
            left, _ = transported(batch.tokens)
            right, _ = static(batch.tokens)
        self.assertTrue(torch.equal(left, right))


class CausalityTests(unittest.TestCase):
    def test_future_tokens_cannot_change_earlier_outputs(self) -> None:
        model = _eager_writer()
        batch = _batch(2)
        cut = 200
        mutated = batch.tokens.clone()
        mutated[:, cut:] = task.FILL
        with torch.no_grad():
            original, first = model(batch.tokens, collect=True)
            changed, second = model(mutated, collect=True)
        self.assertTrue(torch.equal(original[:, :cut - 2], changed[:, :cut - 2]))
        self.assertTrue(
            torch.equal(first["leases"][:, : cut - 2], second["leases"][:, : cut - 2])
        )

    def test_learned_arms_refuse_future_information(self) -> None:
        model = lease.PHLDAMLease(arm="phl_lease")
        batch = _batch(2)
        for arm in lease.LEARNED_ARMS + ("random", "fifo", "lru"):
            with self.assertRaises(ValueError, msg=arm):
                model(batch.tokens, arm=arm, oracle_future=batch.key_use_positions)

    def test_the_oracle_arm_requires_future_information(self) -> None:
        model = lease.PHLDAMLease(arm="phl_lease")
        batch = _batch(2)
        with self.assertRaises(ValueError):
            model(batch.tokens, arm="oracle")

    def test_future_query_schedule_does_not_reach_a_learned_arm(self) -> None:
        """Changing only the oracle table must leave learned outputs untouched."""
        model = _eager_writer()
        batch = _batch(2)
        with torch.no_grad():
            first, _ = model(batch.tokens, arm="phl_lease")
        batch.key_use_positions.fill_(lease.INFINITY)
        with torch.no_grad():
            second, _ = model(batch.tokens, arm="phl_lease")
        self.assertTrue(torch.equal(first, second))


class RuntimeTests(unittest.TestCase):
    # Parameters that actually feed each arm's eviction score. lease_head is
    # excluded for the non-lease arms: they carry it only so that timing
    # supervision reaches every arm equally, and it is deliberately not wired
    # into their eviction decision, so the recall loss alone cannot touch it.
    SCORER_PARAMETERS = {
        "content_only": ("content_scorer",),
        "learned_utility": ("utility_scorer",),
        "static_priority": ("lease_readout", "eviction_recency", "eviction_bias", "lease_head"),
        "phl_lease": ("lease_readout", "eviction_recency", "eviction_bias", "lease_head"),
    }

    def test_gradients_are_finite_and_reach_the_eviction_scorer(self) -> None:
        for arm in lease.LEARNED_ARMS:
            with self.subTest(arm=arm):
                torch.manual_seed(2)
                model = _eager_writer(arm=arm)
                batch = _batch(2)
                logits, _ = model(batch.tokens)
                loss, _, _ = lease.common_objective(logits, batch)
                loss.backward()
                scorer = [
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if name.startswith(self.SCORER_PARAMETERS[arm])
                ]
                self.assertTrue(scorer)
                touched = 0
                for name, parameter in scorer:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    touched += int(parameter.grad.abs().sum() > 0)
                self.assertGreater(touched, 0, arm)
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_the_recall_loss_alone_leaves_a_non_lease_timing_head_untouched(self) -> None:
        """Confirms the exclusion above is a real property, not a test excuse."""
        for arm in ("content_only", "learned_utility"):
            with self.subTest(arm=arm):
                model = _eager_writer(arm=arm)
                logits, _ = model(_batch(2).tokens)
                lease.common_objective(logits, _batch(2))[0].backward()
                self.assertIsNone(model.lease_head.weight.grad)

    def test_runtime_state_stays_finite_for_every_arm(self) -> None:
        model = _eager_writer()
        batch = _batch(2)
        for arm in lease.ALL_ARMS:
            if arm in lease.LEARNED_ARMS and arm != model.arm:
                continue
            with self.subTest(arm=arm):
                future = batch.key_use_positions if arm == "oracle" else None
                with torch.no_grad():
                    logits, diagnostics = model(
                        batch.tokens,
                        arm=arm,
                        rng=torch.Generator().manual_seed(1),
                        oracle_future=future,
                        collect=True,
                    )
                self.assertTrue(torch.isfinite(logits).all())
                state = diagnostics["final_state"]
                for tensor in (
                    state.phl,
                    state.dam.keys,
                    state.dam.values,
                    state.dam.occupancy,
                    state.dam.leases,
                ):
                    self.assertTrue(torch.isfinite(tensor).all())

    def test_checkpoint_round_trip_reproduces_outputs(self) -> None:
        model = _eager_writer()
        batch = _batch(2)
        with torch.no_grad():
            before, _ = model(batch.tokens)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = lease.PHLDAMLease(arm=model.arm)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            after, _ = restored(batch.tokens)
        self.assertTrue(torch.equal(before, after))

    def test_evaluation_is_deterministic_under_a_fixed_seed(self) -> None:
        model = _eager_writer()
        common = dict(
            seed=1, writes=24, condition="canonical", episodes=32,
            batch_size=16, device=torch.device("cpu"),
        )
        for arm in ("random", "lru", "oracle", "phl_lease"):
            with self.subTest(arm=arm):
                first = lease.evaluate_arm(model, arm, **common)
                second = lease.evaluate_arm(model, arm, **common)
                self.assertEqual(first["recall"], second["recall"])
                self.assertEqual(first["residency"], second["residency"])

    def test_parameter_accounting_matches_the_arm(self) -> None:
        counts = {
            arm: lease.parameter_report(lease.PHLDAMLease(arm=arm))
            for arm in lease.LEARNED_ARMS
        }
        self.assertEqual(
            counts["static_priority"]["eviction_policy_parameters"],
            counts["phl_lease"]["eviction_policy_parameters"],
        )
        self.assertEqual(
            counts["static_priority"]["total_parameters"],
            counts["phl_lease"]["total_parameters"],
        )

    def test_only_the_lease_arms_let_the_lease_reach_the_eviction_score(self) -> None:
        """Every arm owns a timing head; only two consume it when evicting."""
        for arm in lease.LEARNED_ARMS:
            names = {name for name, _ in lease.PHLDAMLease(arm=arm).named_parameters()}
            self.assertIn("lease_head.weight", names, arm)
            if arm in ("static_priority", "phl_lease"):
                self.assertIn("lease_readout", names, arm)
            else:
                self.assertNotIn("lease_readout", names, arm)


class TimingSupervisionTests(unittest.TestCase):
    def tearDown(self) -> None:
        task.set_scale("full")

    def test_lease_bins_follow_the_active_scale(self) -> None:
        task.set_scale("full")
        self.assertEqual(lease.lease_bin_edges()[1], task.FIRST_USE_DELAY[task.NEAR])
        self.assertEqual(lease.lease_bin_edges()[4], task.FIRST_USE_DELAY[task.FAR])
        full = lease.lease_bin_edges()
        task.set_scale("compact")
        compact = lease.lease_bin_edges()
        self.assertNotEqual(full, compact)
        self.assertEqual(compact[4], task.FIRST_USE_DELAY[task.FAR])

    def test_transport_is_calibrated_to_the_active_scale(self) -> None:
        """Far-horizon mass must reach "due" as the far delay actually elapses."""
        for scale in ("full", "compact"):
            task.set_scale(scale)
            transport = lease.build_lease_transport()
            state = torch.zeros(lease.NUM_LEASE_BINS)
            state[4] = 1.0
            far_low, far_high = task.FIRST_USE_DELAY[task.FAR]
            best_step, best_due = 0, -1.0
            for step in range(1, far_high * 2):
                state = transport.T @ state
                if float(state[0]) > best_due:
                    best_due, best_step = float(state[0]), step
            self.assertGreater(best_due, 0.20, scale)
            self.assertLess(abs(best_step - far_high), far_high * 0.6, scale)

    def test_timing_labels_mark_write_positions_with_true_horizons(self) -> None:
        episodes = task.generate_episodes(11, 3, 24, "canonical")
        batch = lease.pack_batch(episodes, torch.device("cpu"))
        labelled = batch.timing_label != lease.TIMING_IGNORE_INDEX
        self.assertEqual(int(labelled.sum()), sum(len(e.items) for e in episodes))
        for row, episode in enumerate(episodes):
            for item in episode.items:
                got = int(batch.timing_label[row, item.write_value_position])
                if item.is_never:
                    self.assertEqual(got, lease.NUM_LEASE_BINS - 1)
                else:
                    delay = item.query_key_positions[0] - item.write_value_position
                    self.assertEqual(got, lease.delay_to_lease_bin(delay))
                    self.assertNotEqual(got, lease.NUM_LEASE_BINS - 1)

    def test_timing_supervision_reaches_every_learned_arm_equally(self) -> None:
        batch = _batch(2)
        for arm in lease.LEARNED_ARMS:
            with self.subTest(arm=arm):
                torch.manual_seed(4)
                model = lease.PHLDAMLease(arm=arm)
                loss = lease.timing_objective(lease.timing_logits(model, batch.tokens), batch)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(model.lease_head.weight.grad.abs().sum().item(), 0.0)
                self.assertGreater(
                    model.context_encoder[0].weight.grad.abs().sum().item(), 0.0
                )

    def test_timing_labels_never_reach_the_model_forward(self) -> None:
        """Supervision is a loss term only; forward still sees tokens alone."""
        model = _eager_writer()
        batch = _batch(2)
        with torch.no_grad():
            before, _ = model(batch.tokens, arm="phl_lease")
        batch.timing_label.fill_(lease.TIMING_IGNORE_INDEX)
        with torch.no_grad():
            after, _ = model(batch.tokens, arm="phl_lease")
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
