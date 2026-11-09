import random
from unittest.mock import Mock
import pytest
from diagnostic_storage.observer_adapter_v1 import FailIsolatedSink
from diagnostic_storage.observer_invocation_bridge_v1 import invoke_with_adapter
from phase3_mask_edit_refine.passive_observer import observe


def adapter():
    core = Mock()
    core.sink = FailIsolatedSink(core)
    return core


def test_existing_observe_routes_in_order_and_closes_before_receipt_finalize():
    core = adapter()
    before = random.getstate()

    def business():
        observe("qualification.begin", case="a")
        observe("qualification.complete", failures=["underfill"])
        return 17

    def finish(**kwargs):
        assert core.sink.closed
        assert kwargs == {
            "normal_invocation_end": True,
            "business_receipt_path": "receipt.json",
        }

    core.finalize.side_effect = finish
    assert (
        invoke_with_adapter(business, core, business_receipt_path="receipt.json") == 17
    )
    assert [c.args[0] for c in core.record.call_args_list] == [
        "qualification.begin",
        "qualification.complete",
    ]
    assert random.getstate() == before
    observe("after_session")
    assert core.record.call_count == 2


def test_business_exception_survives_double_recorder_failure():
    core = adapter()
    core.record.side_effect = OSError("disk")
    core.invalidate.side_effect = RuntimeError("fault handler")
    core.finalize.side_effect = RuntimeError("seal")
    error = ValueError("business failure")

    def business():
        observe("qualification.begin")
        raise error

    with pytest.raises(ValueError) as caught:
        invoke_with_adapter(business, core, business_receipt_path="receipt.json")
    assert caught.value is error and core.sink.closed and core.sink.failed
    assert core.finalize.call_args.kwargs["normal_invocation_end"] is False


def test_normal_business_return_survives_recorder_and_finalize_failure():
    core = adapter()
    core.record.side_effect = OSError("disk")
    core.finalize.side_effect = OSError("seal")

    def business():
        observe("qualification.begin")
        return "unchanged"

    assert (
        invoke_with_adapter(business, core, business_receipt_path="receipt.json")
        == "unchanged"
    )
    assert core.sink.failed and core.sink.closed
