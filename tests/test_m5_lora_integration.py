"""可选PEFT CPU集成：随机极小Qwen，绝不读取真实基座或个人数据。"""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from minigpt_cn import m5_lora_run as runner

try:
    import peft
except ImportError:
    peft = None


@unittest.skipIf(peft is None, "PEFT is optional in the Windows project environment")
class TestPeftCPU(unittest.TestCase):
    def setUp(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(11)
        torch.set_num_threads(1)
        self.config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                  head_dim=8, max_position_embeddings=64, pad_token_id=0,
                                  bos_token_id=1, eos_token_id=2, tie_word_embeddings=True)
        self.config._attn_implementation = "sdpa"
        base = Qwen3ForCausalLM(self.config).eval().requires_grad_(False)
        self.base_state = copy.deepcopy(base.state_dict())
        self.reference = copy.deepcopy(base)
        self.model = peft.get_peft_model(base, peft.LoraConfig(
            task_type="CAUSAL_LM", r=2, lora_alpha=4, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            layers_to_transform=[0, 1], init_lora_weights=True))
        self.items = [
            {"sample_id": "a", "input_ids": [1, 3, 4, 5, 2], "labels": [-100, -100, -100, 5, 2]},
            {"sample_id": "b", "input_ids": [1, 6, 7, 2], "labels": [-100, -100, -100, 2]},
        ]

    def test_zero_initialization_is_noop_and_base_is_frozen(self):
        result = runner.check_adapter(self.model, {"expected_target_module_count": 14,
                                                   "expected_trainable_parameters": 1024})
        self.assertTrue(result["base_frozen"])
        batch = runner.collate(self.items, "cpu", pad_token_id=0)
        with torch.no_grad():
            before = self.reference(**batch, use_cache=False).logits
            after = self.model.eval()(**batch, use_cache=False).logits
        torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_real_peft_group_loss_and_gradients_match_batched_loss(self):
        self.model.eval()
        batched = copy.deepcopy(self.model)
        loss, count = runner.accumulate(self.model, self.items, "cpu")
        batch = runner.collate(self.items, "cpu", pad_token_id=0)
        expected = batched(**batch, use_cache=False, logits_to_keep=0).loss
        expected.backward()
        self.assertEqual(count, 3)
        self.assertAlmostEqual(loss, expected.item(), places=6)
        for (name, actual), (_, reference) in zip(self.model.named_parameters(), batched.named_parameters()):
            if actual.requires_grad:
                torch.testing.assert_close(actual.grad, reference.grad, atol=2e-6, rtol=5e-5)
            else:
                self.assertIsNone(actual.grad, name)

    def test_checkpointed_backward_updates_only_adapter(self):
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            "use_reentrant": False, "preserve_rng_state": True})
        self.model.train()
        runner.accumulate(self.model, self.items, "cpu")
        self.assertTrue(runner.finite_gradients(self.model))
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-3)
        optimizer.step()
        changed_adapter = False
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                changed_adapter |= not torch.equal(parameter, before[name])
            else:
                torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
        self.assertTrue(changed_adapter)

    def test_adapter_safetensors_save_and_reload_preserves_logits(self):
        from transformers import Qwen3ForCausalLM

        runner.accumulate(self.model, self.items, "cpu")
        torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=0.01).step()
        batch = runner.collate(self.items, "cpu", pad_token_id=0)
        with tempfile.TemporaryDirectory(prefix="m5-peft-cpu-", dir=runner.PROJECT / "logs") as folder:
            self.model.save_pretrained(folder, safe_serialization=True, save_embedding_layers=False)
            self.assertTrue((Path(folder) / "adapter_model.safetensors").is_file())
            self.assertFalse((Path(folder) / "model.safetensors").exists())
            fresh = Qwen3ForCausalLM(self.config).requires_grad_(False)
            fresh.load_state_dict(self.base_state)
            reloaded = peft.PeftModel.from_pretrained(fresh, folder, is_trainable=False,
                                                      local_files_only=True).eval()
            with torch.no_grad():
                expected = self.model.eval()(**batch, use_cache=False).logits
                observed = reloaded(**batch, use_cache=False).logits
            torch.testing.assert_close(expected, observed, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
