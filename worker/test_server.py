import threading
import unittest

import torch

import scheduler_worker_pb2

from server import SchedulerWorkerServicer


class CPUFakeTorch:
    long = torch.long

    @staticmethod
    def no_grad():
        return torch.no_grad()

    @staticmethod
    def tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return torch.tensor(*args, **kwargs)

    @staticmethod
    def ones(*args, **kwargs):
        kwargs.pop("device", None)
        return torch.ones(*args, **kwargs)

    @staticmethod
    def cat(*args, **kwargs):
        return torch.cat(*args, **kwargs)

    @staticmethod
    def argmax(*args, **kwargs):
        return torch.argmax(*args, **kwargs)

    @staticmethod
    def where(*args, **kwargs):
        return torch.where(*args, **kwargs)

    @staticmethod
    def arange(*args, **kwargs):
        return torch.arange(*args, **kwargs)


class FakeCache:
    def __init__(self, keys=None, ddp_cache_data=None, config=None):
        if ddp_cache_data is not None:
            keys = ddp_cache_data[0][0]
            values = ddp_cache_data[0][1]
        else:
            values = keys
        self.keys = keys
        self.values = values

    def batch_select_indices(self, indices):
        self.keys = self.keys.index_select(0, indices)
        self.values = self.values.index_select(0, indices)

    def __iter__(self):
        yield self.keys, self.values, None


class FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values=None,
        use_cache=True,
    ):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "position_ids": position_ids.clone(),
                "past_key_values": past_key_values,
            }
        )
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros((batch_size, sequence_length, 64))
        next_token_ids = input_ids[:, -1] + 1
        logits[torch.arange(batch_size), -1, next_token_ids] = 1

        current_keys = input_ids.reshape(batch_size, 1, sequence_length, 1).float()
        if past_key_values is not None:
            current_keys = torch.cat(
                [past_key_values.keys, current_keys],
                dim=2,
            )
        return type(
            "ModelOutput",
            (),
            {
                "logits": logits,
                "past_key_values": FakeCache(current_keys),
            },
        )()


class ActiveContext:
    @staticmethod
    def is_active():
        return True

    @staticmethod
    def abort(code, details):
        raise AssertionError(f"unexpected worker abort: {code} {details}")


class SchedulerWorkerCancelTests(unittest.TestCase):
    def setUp(self):
        self.servicer = SchedulerWorkerServicer.__new__(SchedulerWorkerServicer)
        cohort = {
            "request_ids": ["active-request"],
            "kv": FakeCache(torch.zeros((1, 1, 2, 1))),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "counts": [4],
        }
        self.servicer.caches = {"active-request": cohort}
        self.servicer.torch = CPUFakeTorch
        self.servicer.lock = threading.Lock()

    def test_cancel_releases_cache_and_is_idempotent(self):
        request = scheduler_worker_pb2.CancelGenerationRequest(
            request_id="active-request",
        )

        first = self.servicer.CancelGeneration(request, None)
        second = self.servicer.CancelGeneration(request, None)

        self.assertTrue(first.cache_released)
        self.assertFalse(second.cache_released)
        self.assertNotIn("active-request", self.servicer.caches)

    def test_prefill_and_decode_batch_variable_length_prompts(self):
        servicer = SchedulerWorkerServicer.__new__(SchedulerWorkerServicer)
        servicer.torch = CPUFakeTorch
        servicer.model = FakeModel()
        servicer.stop_token_ids = {12}
        servicer.caches = {}
        servicer.lock = threading.Lock()
        context = ActiveContext()

        prefill = scheduler_worker_pb2.ScheduleTaskBatch(
            requests=[
                scheduler_worker_pb2.ScheduleTaskRequest(
                    request_id="long",
                    phase=scheduler_worker_pb2.PREFILL,
                    token_ids=[10, 11],
                ),
                scheduler_worker_pb2.ScheduleTaskRequest(
                    request_id="short",
                    phase=scheduler_worker_pb2.PREFILL,
                    token_ids=[20],
                ),
            ],
        )
        prefill_response = servicer.ScheduleTask(prefill, context)

        self.assertEqual(
            [(token.request_id, token.token_id, token.finished) for token in prefill_response.sampled_tokens],
            [("long", 12, True), ("short", 21, False)],
        )
        self.assertNotIn("long", servicer.caches)
        self.assertIn("short", servicer.caches)
        self.assertEqual(servicer.caches["short"]["request_ids"], ["short"])
        self.assertEqual(servicer.caches["short"]["kv"].keys.shape[0], 1)
        prefill_call = servicer.model.calls[0]
        self.assertEqual(prefill_call["input_ids"].tolist(), [[10, 11], [0, 20]])
        self.assertEqual(prefill_call["attention_mask"].tolist(), [[1, 1], [0, 1]])
        self.assertEqual(prefill_call["position_ids"].tolist(), [[0, 1], [0, 0]])

        decode = scheduler_worker_pb2.ScheduleTaskBatch(
            requests=[
                scheduler_worker_pb2.ScheduleTaskRequest(
                    request_id="short",
                    phase=scheduler_worker_pb2.DECODE,
                    token_ids=[21],
                ),
            ],
        )
        decode_response = servicer.ScheduleTask(decode, context)

        self.assertEqual(
            [(token.request_id, token.token_id) for token in decode_response.sampled_tokens],
            [("short", 22)],
        )
        decode_call = servicer.model.calls[1]
        self.assertEqual(decode_call["input_ids"].tolist(), [[21]])
        self.assertEqual(decode_call["attention_mask"].tolist(), [[0, 1, 1]])
        self.assertEqual(decode_call["position_ids"].tolist(), [[1]])
        self.assertIsInstance(decode_call["past_key_values"], FakeCache)
        self.assertEqual(servicer.caches["short"]["counts"], [2])

    def test_decode_keeps_multiple_requests_in_one_cache_object(self):
        servicer = SchedulerWorkerServicer.__new__(SchedulerWorkerServicer)
        servicer.torch = CPUFakeTorch
        servicer.model = FakeModel()
        servicer.stop_token_ids = {-1}
        servicer.caches = {}
        servicer.lock = threading.Lock()
        context = ActiveContext()

        prefill = scheduler_worker_pb2.ScheduleTaskBatch(
            requests=[
                scheduler_worker_pb2.ScheduleTaskRequest(
                    request_id="first",
                    phase=scheduler_worker_pb2.PREFILL,
                    token_ids=[10, 11],
                ),
                scheduler_worker_pb2.ScheduleTaskRequest(
                    request_id="second",
                    phase=scheduler_worker_pb2.PREFILL,
                    token_ids=[20],
                ),
            ],
        )
        servicer.ScheduleTask(prefill, context)
        cohort = servicer.caches["first"]
        self.assertIs(servicer.caches["second"], cohort)
        self.assertEqual(cohort["kv"].keys.shape[0], 2)

        decode = scheduler_worker_pb2.ScheduleTaskBatch(
            requests=[
                scheduler_worker_pb2.ScheduleTaskRequest(
                    request_id="first",
                    phase=scheduler_worker_pb2.DECODE,
                    token_ids=[12],
                ),
                scheduler_worker_pb2.ScheduleTaskRequest(
                    request_id="second",
                    phase=scheduler_worker_pb2.DECODE,
                    token_ids=[21],
                ),
            ],
        )
        response = servicer.ScheduleTask(decode, context)

        self.assertEqual(
            [(token.request_id, token.token_id) for token in response.sampled_tokens],
            [("first", 13), ("second", 22)],
        )
        decode_call = servicer.model.calls[1]
        self.assertIsInstance(decode_call["past_key_values"], FakeCache)
        self.assertEqual(decode_call["past_key_values"].keys.shape[0], 2)
        self.assertEqual(servicer.caches["first"]["counts"], [2, 2])

    def test_decode_merges_new_prefill_with_active_cache(self):
        servicer = SchedulerWorkerServicer.__new__(SchedulerWorkerServicer)
        servicer.torch = CPUFakeTorch
        servicer.model = FakeModel()
        servicer.stop_token_ids = {-1}
        servicer.caches = {}
        servicer.lock = threading.Lock()
        context = ActiveContext()

        servicer.ScheduleTask(
            scheduler_worker_pb2.ScheduleTaskBatch(
                requests=[
                    scheduler_worker_pb2.ScheduleTaskRequest(
                        request_id="active",
                        phase=scheduler_worker_pb2.PREFILL,
                        token_ids=[10, 11],
                    ),
                ],
            ),
            context,
        )
        servicer.ScheduleTask(
            scheduler_worker_pb2.ScheduleTaskBatch(
                requests=[
                    scheduler_worker_pb2.ScheduleTaskRequest(
                        request_id="late",
                        phase=scheduler_worker_pb2.PREFILL,
                        token_ids=[20],
                    ),
                ],
            ),
            context,
        )

        response = servicer.ScheduleTask(
            scheduler_worker_pb2.ScheduleTaskBatch(
                requests=[
                    scheduler_worker_pb2.ScheduleTaskRequest(
                        request_id="active",
                        phase=scheduler_worker_pb2.DECODE,
                        token_ids=[12],
                    ),
                    scheduler_worker_pb2.ScheduleTaskRequest(
                        request_id="late",
                        phase=scheduler_worker_pb2.DECODE,
                        token_ids=[21],
                    ),
                ],
            ),
            context,
        )

        self.assertEqual(
            [(token.request_id, token.token_id) for token in response.sampled_tokens],
            [("active", 13), ("late", 22)],
        )
        decode_call = servicer.model.calls[-1]
        self.assertEqual(decode_call["attention_mask"].tolist(), [[1, 1, 1], [0, 1, 1]])
        self.assertEqual(decode_call["position_ids"].tolist(), [[2], [1]])
        self.assertEqual(
            decode_call["past_key_values"].keys[:, 0, :, 0].tolist(),
            [[10, 11], [0, 20]],
        )
        self.assertIs(servicer.caches["active"], servicer.caches["late"])
        self.assertEqual(servicer.caches["active"]["request_ids"], ["active", "late"])

    def test_logs_initial_cache_layout_once(self):
        servicer = SchedulerWorkerServicer.__new__(SchedulerWorkerServicer)
        cache = FakeCache(torch.zeros((1, 2, 3, 4)))

        with self.assertLogs(level="INFO") as logs:
            servicer._log_cache_layout(cache)
            servicer._log_cache_layout(cache)

        self.assertEqual(len(logs.output), 1)
        self.assertIn("layer 0", logs.output[0])
        self.assertIn("None", logs.output[0])
        self.assertIn("(1, 2, 3, 4)", logs.output[0])

    def test_cancelling_one_request_prunes_shared_batch_cache(self):
        other_cohort = {
            "request_ids": ["other-request"],
            "kv": FakeCache(torch.ones((1, 1, 2, 1))),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "counts": [3],
        }
        cohort = self.servicer.caches["active-request"]
        cohort["request_ids"].append("other-request")
        cohort["kv"].keys = torch.cat(
            [cohort["kv"].keys, other_cohort["kv"].keys],
            dim=0,
        )
        cohort["kv"].values = torch.cat(
            [cohort["kv"].values, other_cohort["kv"].values],
            dim=0,
        )
        cohort["attention_mask"] = torch.ones((2, 2), dtype=torch.long)
        cohort["counts"].append(3)
        self.servicer.caches["other-request"] = cohort

        response = self.servicer.CancelGeneration(
            scheduler_worker_pb2.CancelGenerationRequest(request_id="active-request"),
            None,
        )

        self.assertTrue(response.cache_released)
        self.assertNotIn("active-request", self.servicer.caches)
        self.assertIs(self.servicer.caches["other-request"], cohort)
        self.assertEqual(cohort["request_ids"], ["other-request"])
        self.assertEqual(cohort["kv"].keys.shape[0], 1)
        self.assertEqual(cohort["counts"], [3])


if __name__ == "__main__":
    unittest.main()
