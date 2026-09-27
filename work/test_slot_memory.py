import unittest

import torch

from slot_memory import SlotMemory, unit
import lm_hybrid as lm


class SlotMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.m = SlotMemory(32, num_slots=8, d_key=16, d_value=16)

    def test_causal(self):
        h = torch.randn(2, 40, 32)
        g = h.clone(); g[:, 25:] = torch.randn(2, 15, 32)
        with torch.no_grad():
            a, _ = self.m(h); b, _ = self.m(g)
        self.assertTrue(torch.equal(a[:, :25], b[:, :25]))

    def test_unit_is_bounded_at_zero(self):
        x = torch.zeros(3, 4, requires_grad=True)
        unit(x).sum().backward()
        self.assertLessEqual(x.grad.abs().max().item(), 20.0 + 1e-4)

    def test_reset_isolates_documents(self):
        h = torch.randn(1, 30, 32)
        reset = torch.zeros(1, 30, dtype=torch.bool); reset[0, 12] = True
        with torch.no_grad():
            joined, _ = self.m(h, reset)
            alone, _ = self.m(h[:, 12:])
        self.assertTrue(torch.allclose(joined[:, 12:], alone, atol=1e-5))

    def test_a_written_binding_is_retrieved(self):
        """Gate forced open: write at t=1 (key from h_0), query with h_0 later."""
        with torch.no_grad():
            self.m.gates.bias.copy_(torch.tensor([20.0, 20.0, 0.0]))
            self.m.gates.weight.zero_(); self.m.read_gate.bias.fill_(20.0)
        key, value = torch.randn(32), torch.randn(32)
        h = torch.randn(1, 10, 32) * 0.01
        h[0, 0], h[0, 1], h[0, 9] = key, value, key
        with torch.no_grad():
            _, m = self.m(h)
        self.assertGreater(torch.cosine_similarity(m[0, 9], self.m.value(value), 0).item(), 0.9)

    def test_gradients_finite_from_zero_state(self):
        h = torch.randn(2, 64, 32, requires_grad=True)
        out, m = self.m(h)
        (out.sum() + m.sum()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.m.parameters() if p.grad is not None))


class HybridLMTests(unittest.TestCase):
    def test_arms_are_parameter_matched(self):
        counts = {a: lm.parameter_count(lm.LM(a, 64, 16)) for a in ("window", "window_sml", "window_ssm", "full")}
        self.assertEqual(counts["window"], counts["full"])
        self.assertLess(abs(counts["window_sml"] - counts["window"]) / counts["window"], 0.005)
        self.assertLess(abs(counts["window_ssm"] - counts["window"]) / counts["window"], 0.005)

    def test_local_control_cannot_carry_memory_beyond_window(self):
        torch.manual_seed(0)
        model = lm.LM("window_sml_local", 64, 8)
        x = torch.randint(0, 256, (1, 64)); y = x.clone(); y[:, :4] = 3
        with torch.no_grad():
            # attention reach 4*7=28, memory wiped every 8: nothing from 0..3 past 32
            self.assertTrue(torch.equal(model(x)[:, 40:], model(y)[:, 40:]))

    def test_surprise_gated_lm_is_causal(self):
        torch.manual_seed(0)
        model = lm.LM("window_sml", 64, 8, memory_layer=3, surprise_gate=True)
        x = torch.randint(0, 256, (2, 64)); y = x.clone(); y[:, 40:] = 5
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x)[:, :40], model(y)[:, :40]))

    def test_lm_is_causal_for_every_arm(self):
        for arm in ("window", "window_sml", "window_sml_local", "window_ssm", "full"):
            torch.manual_seed(0)
            model = lm.LM(arm, 48, 8)
            x = torch.randint(0, 256, (2, 48)); y = x.clone(); y[:, 30:] = 7
            with torch.no_grad():
                self.assertTrue(torch.equal(model(x)[:, :30], model(y)[:, :30]), arm)

    def test_window_arm_cannot_see_beyond_its_stacked_reach(self):
        """4 layers of window 8 reach 4*7 = 28 back; nothing earlier leaks."""
        torch.manual_seed(0)
        model = lm.LM("window", 48, 8)
        x = torch.randint(0, 256, (1, 48)); y = x.clone(); y[:, :5] = 3
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x)[:, 5 + 28:], model(y)[:, 5 + 28:]))
            self.assertFalse(torch.equal(model(x)[:, 5:5 + 28], model(y)[:, 5:5 + 28]))

    def test_long_range_mask(self):
        text = list(b"abcdefghijXYZ" + b"." * 40 + b"abcdefghijQ")
        flags = lm.long_range_mask(text, window=16)
        start = 13 + 40
        self.assertTrue(any(flags[start + 8:start + 10]))
        self.assertFalse(any(flags[:13]))


if __name__ == "__main__":
    unittest.main()
