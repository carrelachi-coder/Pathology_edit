"""Saved diagnostic text must not promote invalid areas or execute payloads."""
from scripts.diagnose_native_topology_failures import floor_trials,literals_after


def test_preserve_topology_status_instead_of_treating_largest_area_as_safe():
    unsafe="minimum=70779, realized=70732, topology={'passed': False, 'source_holes_before': 13, 'source_holes_after': 10}"
    safe="minimum=70779, realized=22287, topology={'passed': True, 'source_holes_before': 13, 'source_holes_after': 13}"
    states,errors=floor_trials([unsafe+'; '+safe+'; '+unsafe])
    assert not errors and len(states)==2
    assert max(s['realized_pixels'] for s in states)==70732
    assert max(s['realized_pixels'] for s in states if s['topology']['passed'])==22287


def test_literal_braces_inside_strings_and_unbounded_spacing():
    text="{'passed': True, 'note': 'a } is text', 'spacing': inf, 'components': [2, 400]}"
    result=literals_after(text,0)
    assert result['passed'] is True and result['note']=='a } is text'
    assert result['spacing'] is None and result['components']==[2,400]


def test_saved_error_payload_cannot_execute_python(tmp_path):
    path=tmp_path/'must_not_exist'
    text="minimum=70779, realized=1, topology={'passed': __import__('pathlib').Path("+repr(str(path))+").write_text('unsafe')}"
    states,errors=floor_trials([text])
    assert states==[] and errors and not path.exists()
