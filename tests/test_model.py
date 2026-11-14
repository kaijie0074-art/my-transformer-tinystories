import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from model import GPT, GPTConfig


class GPTBehaviorTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(1234)
        self.config = GPTConfig(
            vocab_size=64, context_length=16, n_layer=2,
            n_head=4, n_embd=32, ffn_hidden=64,
        )
        self.model = GPT(self.config)

    def test_future_tokens_do_not_affect_prefix_logits(self):
        first = mx.array([[1, 7, 9, 11, 13, 15]], dtype=mx.int32)
        second = mx.array([[1, 7, 9, 11, 21, 23]], dtype=mx.int32)
        first_logits = self.model(first)
        second_logits = self.model(second)
        mx.eval(first_logits, second_logits)
        np.testing.assert_allclose(
            np.asarray(first_logits[:, :4]), np.asarray(second_logits[:, :4]),
            rtol=1e-5, atol=1e-5,
        )

    def test_last_token_projection_matches_full_logits(self):
        tokens = mx.array([[1, 8, 3, 14, 2], [1, 6, 5, 4, 2]], dtype=mx.int32)
        full = self.model(tokens)
        last = self.model(tokens, last_token_only=True)
        mx.eval(full, last)
        self.assertEqual(last.shape, (2, 1, self.config.vocab_size))
        np.testing.assert_allclose(
            np.asarray(full[:, -1:, :]), np.asarray(last), rtol=1e-4, atol=1e-4
        )

    def test_one_sequence_context_and_invalid_inputs(self):
        one = self.model(mx.array([[1]], dtype=mx.int32))
        mx.eval(one)
        self.assertEqual(one.shape, (1, 1, self.config.vocab_size))
        with self.assertRaises(ValueError):
            self.model(mx.zeros((1, self.config.context_length + 1), dtype=mx.int32))
        with self.assertRaises((ValueError, TypeError)):
            self.model(mx.array([1, 2], dtype=mx.int32))
        with self.assertRaises(ValueError):
            GPTConfig(n_embd=31, n_head=4)

    def test_gradient_step_reduces_next_token_loss(self):
        x = mx.array([[1, 5, 6, 7], [1, 5, 6, 8]], dtype=mx.int32)
        y = mx.array([[5, 6, 7, 2], [5, 6, 8, 2]], dtype=mx.int32)

        def loss_fn(model, inputs, labels):
            return nn.losses.cross_entropy(model(inputs), labels, reduction="mean")

        initial = loss_fn(self.model, x, y)
        mx.eval(initial)
        optimizer = optim.SGD(learning_rate=0.03)
        loss_and_grad = nn.value_and_grad(self.model, loss_fn)
        for _ in range(8):
            _, gradients = loss_and_grad(self.model, x, y)
            optimizer.update(self.model, gradients)
            mx.eval(self.model.parameters(), optimizer.state)
        final = loss_fn(self.model, x, y)
        mx.eval(final)
        self.assertTrue(np.isfinite(float(final.item())))
        self.assertLess(float(final.item()), float(initial.item()))


if __name__ == "__main__":
    unittest.main()
