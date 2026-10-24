"""Cross-check that pywwt + its Jupyter widget machinery actually works.

Run:  python scripts/check_pywwt.py
"""
import sys

ok = True


def step(name, fn):
    global ok
    try:
        result = fn()
        print(f"[PASS] {name}: {result}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def check_import():
    import pywwt
    return f"pywwt {pywwt.__version__}"


def check_widget_registry():
    # The Jupyter widget factory must be registered for pywwt's WWTView
    import pywwt.jupyter as pj
    return f"module OK: {[n for n in dir(pj) if 'WWT' in n or 'Jupyter' in n][:6]}"


def check_data_server():
    import pywwt.data_server as ds_mod
    names = [n for n in dir(ds_mod) if not n.startswith("_")]
    return f"data_server symbols: {names[:8]}"


def check_base_widget():
    from pywwt.core import BaseWWTWidget
    from traitlets import HasTraits
    assert issubclass(BaseWWTWidget, HasTraits)
    return "BaseWWTWidget is a valid traitlets widget class"


def check_layers():
    from pywwt.layers import TableLayer  # noqa: F401
    return "layers module OK"


def check_annotation():
    import pywwt.annotation as pa
    names = [n for n in dir(pa) if "Annotation" in n]
    assert names, "no annotation classes found"
    return f"annotation classes: {names}"


def check_labextension_installed():
    import json, os
    from pathlib import Path
    pkg = (Path(os.environ["APPDATA"]) / "Python" / "share" / "jupyter"
           / "labextensions" / "pywwt" / "package.json")
    d = json.loads(pkg.read_text(encoding="utf-8"))
    assert d["jupyterlab"]["_build"]["load"].endswith("remoteEntry.js")
    return "pywwt federated labextension installed with remoteEntry"


step("pywwt import", check_import)
step("jupyter widget module", check_widget_registry)
step("BaseWWTWidget", check_base_widget)
step("layers", check_layers)
step("annotation", check_annotation)
step("data server", check_data_server)
step("labextension installed", check_labextension_installed)

sys.exit(0 if ok else 1)
