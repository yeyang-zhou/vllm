# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput


def test_connector_output_commits_resident_kv():
    scheduler = Scheduler.__new__(Scheduler)
    request = object()

    scheduler.connector = Mock()
    scheduler.connector.update_connector_output.return_value = {"req": 17}
    scheduler.kv_cache_manager = Mock()
    scheduler.requests = {"req": request}

    scheduler._update_from_kv_xfer_finished(KVConnectorOutput())

    scheduler.kv_cache_manager.commit_resident_kv.assert_called_once_with(request, 17)
