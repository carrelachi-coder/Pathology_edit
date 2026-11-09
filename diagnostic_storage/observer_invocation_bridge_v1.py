"""Isolated adapter invocation through the existing business observation API."""
from phase3_mask_edit_refine.passive_observer import observation_session


def invoke_with_adapter(invoke, adapter, *, business_receipt_path):
    normal_end = False
    try:
        with observation_session(adapter.sink):
            result = invoke()
        normal_end = True
    finally:
        try:
            adapter.finalize(
                normal_invocation_end=normal_end,
                business_receipt_path=business_receipt_path,
            )
        except Exception as exc:
            adapter.sink.invalidate("bridge_finalize", exc)
    return result
