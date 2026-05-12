from pathlib import Path
from types import SimpleNamespace

from mpm_jax.cuda import p2g_cuda


def test_compile_kernel_writes_shared_library_to_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "ffi-cache"
    captured = {}

    monkeypatch.setenv("MPM_FFI_CACHE", str(cache_dir))
    monkeypatch.setenv("NVCC", "/opt/cuda/bin/nvcc")
    monkeypatch.setattr(p2g_cuda.jax.ffi, "include_dir", lambda: "/jax/ffi/include")

    def fake_run(cmd, capture_output=False, text=False):
        if cmd == ["gcc", "-print-file-name=libstdc++.so"]:
            return SimpleNamespace(
                returncode=0,
                stdout="/usr/lib/gcc/libstdc++.so\n",
                stderr="",
            )
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(p2g_cuda.subprocess, "run", fake_run)

    assert p2g_cuda._compile_kernel("p2g_scatter.cu", "libp2g_scatter.so")

    cmd = captured["cmd"]
    output_path = Path(cmd[cmd.index("-o") + 1])
    assert output_path == cache_dir / "libp2g_scatter.so"
    assert Path(cmd[-1]) == Path(p2g_cuda._KERNEL_DIR) / "p2g_scatter.cu"


def test_register_uses_cached_library_and_explicit_ffi_api(monkeypatch, tmp_path):
    cache_dir = tmp_path / "ffi-cache"
    calls = {}
    loaded = {}

    class FakeLibrary:
        P2GScatter = object()

    monkeypatch.setenv("MPM_FFI_CACHE", str(cache_dir))
    monkeypatch.setattr(p2g_cuda, "_compile_kernel", lambda cu_name, so_name: True)
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
        "unit_test_p2g_scatter_cuda",
        "p2g_scatter.cu",
        "libp2g_scatter.so",
        "P2GScatter",
    )

    assert Path(loaded["path"]) == cache_dir / "libp2g_scatter.so"
    assert calls["name"] == "unit_test_p2g_scatter_cuda"
    assert calls["platform"] == "CUDA"
    assert calls["api_version"] == 1
