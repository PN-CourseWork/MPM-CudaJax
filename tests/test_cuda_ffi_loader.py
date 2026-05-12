from pathlib import Path

from mpm_jax.cuda import p2g_cuda


def test_register_missing_so_returns_false(monkeypatch, tmp_path):
    """When the prebuilt .so is absent we should fail gracefully (False, no raise)."""
    monkeypatch.setattr(p2g_cuda, "_LIB_DIR", tmp_path)
    p2g_cuda._REGISTERED.clear()

    assert (
        p2g_cuda._register(
            "unit_test_missing", "libdoes_not_exist.so", "MissingSymbol"
        )
        is False
    )


def test_register_loads_prebuilt_so_and_calls_ffi(monkeypatch, tmp_path):
    """Happy path: .so exists in _LIB_DIR, gets loaded, FFI target registered."""
    so_path = tmp_path / "libp2g_scatter.so"
    so_path.write_bytes(b"")  # only existence matters; LoadLibrary is faked
    monkeypatch.setattr(p2g_cuda, "_LIB_DIR", tmp_path)

    class FakeLibrary:
        P2GScatter = object()

    loaded = {}
    calls = {}

    def fake_load_library(path):
        loaded["path"] = path
        return FakeLibrary()

    monkeypatch.setattr(p2g_cuda.ctypes.cdll, "LoadLibrary", fake_load_library)
    monkeypatch.setattr(p2g_cuda.jax.ffi, "pycapsule", lambda symbol: symbol)

    def fake_register_ffi_target(name, capsule, **kwargs):
        calls["name"] = name
        calls["capsule"] = capsule
        calls.update(kwargs)

    monkeypatch.setattr(
        p2g_cuda.jax.ffi, "register_ffi_target", fake_register_ffi_target
    )
    p2g_cuda._REGISTERED.clear()

    assert p2g_cuda._register(
        "unit_test_p2g_scatter_cuda", "libp2g_scatter.so", "P2GScatter"
    )

    assert Path(loaded["path"]) == so_path
    assert calls["name"] == "unit_test_p2g_scatter_cuda"
    assert calls["capsule"] is FakeLibrary.P2GScatter
    assert calls["platform"] == "CUDA"
    assert calls["api_version"] == 1


def test_register_is_cached(monkeypatch, tmp_path):
    """Second _register() call for the same name should not re-load the library."""
    so_path = tmp_path / "libp2g_scatter.so"
    so_path.write_bytes(b"")
    monkeypatch.setattr(p2g_cuda, "_LIB_DIR", tmp_path)

    calls = []

    class FakeLibrary:
        P2GScatter = object()

    monkeypatch.setattr(
        p2g_cuda.ctypes.cdll,
        "LoadLibrary",
        lambda path: (calls.append(path), FakeLibrary())[1],
    )
    monkeypatch.setattr(p2g_cuda.jax.ffi, "pycapsule", lambda s: s)
    monkeypatch.setattr(p2g_cuda.jax.ffi, "register_ffi_target", lambda *a, **k: None)
    p2g_cuda._REGISTERED.clear()

    name = "unit_test_cache_check"
    assert p2g_cuda._register(name, "libp2g_scatter.so", "P2GScatter")
    assert p2g_cuda._register(name, "libp2g_scatter.so", "P2GScatter")
    assert len(calls) == 1
