import builtins
import importlib


def test_flash_attn_falls_back_if_kernels_missing(monkeypatch, capsys):
    # Import is lazy; reloading keeps the test isolated from earlier state.
    import nanochat.gpt as gpt

    importlib.reload(gpt)
    gpt._FLASH_ATTN = None
    gpt._FLASH_ATTN_FALLBACK_WARNED = False
    monkeypatch.delenv("NANOCHAT_DISABLE_FA3", raising=False)

    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "kernels" or name.startswith("kernels."):
            raise ModuleNotFoundError("No module named 'kernels'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    flash_attn = gpt._get_flash_attn()
    assert hasattr(flash_attn, "flash_attn_func")
    assert hasattr(flash_attn, "flash_attn_with_kvcache")

    captured = capsys.readouterr()
    assert "falling back" in captured.out.lower()


def test_flash_attn_glibc_mismatch_prints_hint(monkeypatch, capsys):
    import nanochat.gpt as gpt

    importlib.reload(gpt)
    gpt._FLASH_ATTN = None
    gpt._FLASH_ATTN_FALLBACK_WARNED = False
    monkeypatch.delenv("NANOCHAT_DISABLE_FA3", raising=False)

    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "kernels" or name.startswith("kernels."):
            raise ImportError(
                "/usr/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found "
                "(required by .../_C.abi3.so)"
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    _ = gpt._get_flash_attn()
    captured = capsys.readouterr()
    out = captured.out.lower()
    assert "glibc" in out
    assert "ubuntu 22.04" in out
