import threading
import unittest

import scheduler_worker_pb2

from server import SchedulerWorkerServicer


class SchedulerWorkerCancelTests(unittest.TestCase):
    def setUp(self):
        self.servicer = SchedulerWorkerServicer.__new__(SchedulerWorkerServicer)
        self.servicer.caches = {
            "active-request": {"kv": object(), "count": 4},
        }
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


if __name__ == "__main__":
    unittest.main()
