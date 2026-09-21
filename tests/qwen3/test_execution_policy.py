import unittest

from demo.qwen3.execution.workload import WorkloadDescriptor
from demo.qwen3.policy import make_policy


class ExecutionPolicyTest(unittest.TestCase):
    def test_static_phase_decisions(self):
        prefill = WorkloadDescriptor.for_prefill(1, 128)
        decode = WorkloadDescriptor.for_decode(1, 128, 0)
        expected = {
            "always": (True, True),
            "decode-only": (False, True),
            "prefill-only": (True, False),
        }
        for name, decisions in expected.items():
            with self.subTest(policy=name):
                policy = make_policy(name)
                self.assertEqual(policy.should_use_mpk(prefill), decisions[0])
                self.assertEqual(policy.should_use_mpk(decode), decisions[1])

    def test_workload_aware_requires_measured_map(self):
        policy = make_policy("workload-aware")
        with self.assertRaises(NotImplementedError):
            policy.should_use_mpk(WorkloadDescriptor.for_decode(1, 128, 0))


if __name__ == "__main__":
    unittest.main()
