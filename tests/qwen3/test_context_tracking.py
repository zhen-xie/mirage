import unittest

from demo.qwen3.execution.workload import WorkloadDescriptor


class ContextTrackingTest(unittest.TestCase):
    def test_context_advances_with_generated_tokens(self):
        workloads = [WorkloadDescriptor.for_decode(8, 1024, step)
                     for step in range(16)]
        self.assertEqual([w.context_length for w in workloads],
                         list(range(1024, 1040)))
        self.assertEqual([w.decode_step for w in workloads], list(range(16)))
        self.assertTrue(all(w.active_tokens == 8 for w in workloads))

    def test_prefill_active_tokens(self):
        workload = WorkloadDescriptor.for_prefill(8, 1024)
        self.assertEqual(workload.context_length, 1024)
        self.assertEqual(workload.active_tokens, 8192)

    def test_invalid_workload_rejected(self):
        with self.assertRaises(ValueError):
            WorkloadDescriptor.for_decode(1, 1024, -1)


if __name__ == "__main__":
    unittest.main()
