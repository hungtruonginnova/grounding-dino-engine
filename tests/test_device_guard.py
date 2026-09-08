"""The device guard. Imports torch, so it lives in the sidecar suite only.

The failure this protects against is silent: a CPU-only wheel on a GPU host
leaves torch.cuda.is_available() False, pick_device falls back to cpu, and
inference runs at a fraction of the speed with nothing in the logs to say so.
"""

import pytest

from app.detector import DeviceUnavailable, pick_device, require_device, torch_build


def test_unset_requirement_accepts_any_device(monkeypatch):
    monkeypatch.delenv("GROUNDING_REQUIRE_DEVICE", raising=False)
    for device in ("cpu", "mps", "cuda"):
        require_device(device)  # must not raise


def test_matching_requirement_passes(monkeypatch):
    monkeypatch.setenv("GROUNDING_REQUIRE_DEVICE", "cuda")
    require_device("cuda")


@pytest.mark.parametrize("actual", ["cpu", "mps"])
def test_mismatch_refuses_to_start(monkeypatch, actual):
    """A container that will not start is far easier to notice than one that
    quietly runs on CPU."""
    monkeypatch.setenv("GROUNDING_REQUIRE_DEVICE", "cuda")
    with pytest.raises(DeviceUnavailable, match="GROUNDING_REQUIRE_DEVICE=cuda"):
        require_device(actual)


def test_the_error_names_both_causes(monkeypatch):
    """Whoever hits this needs to know which of the two mistakes they made."""
    monkeypatch.setenv("GROUNDING_REQUIRE_DEVICE", "cuda")
    with pytest.raises(DeviceUnavailable) as exc:
        require_device("cpu")
    message = str(exc.value)
    assert "CPU-only wheel" in message
    assert "nvidia-container-toolkit" in message
    assert "cudaBuild=" in message


def test_requirement_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setenv("GROUNDING_REQUIRE_DEVICE", "  CUDA ")
    require_device("cuda")


def test_forced_device_overrides_detection(monkeypatch):
    monkeypatch.setenv("GROUNDING_DEVICE", "cpu")
    assert pick_device() == "cpu"


def test_torch_build_reports_what_readyz_needs():
    build = torch_build()
    assert set(build) == {"torch", "cudaBuild", "cudaAvailable", "cudaDevices"}
    # cudaBuild is None on a CPU-only wheel; that is the tell.
    assert build["cudaBuild"] is None or isinstance(build["cudaBuild"], str)
    assert isinstance(build["cudaAvailable"], bool)
