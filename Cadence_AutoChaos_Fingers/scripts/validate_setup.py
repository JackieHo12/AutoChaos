#!/usr/bin/env python3
import sys

def check_package(name, import_name=None):
    import_name = import_name or name
    try:
        __import__(import_name)
        print(f"[OK]   {name} installed")
        return True
    except ImportError:
        print(f"[FAIL] {name} NOT installed")
        return False

print("Validating AutoChaos setup...")
print("-" * 40)

all_ok = True

if sys.version_info >= (3, 9):
    print(f"[OK]   Python {sys.version_info.major}.{sys.version_info.minor}")
else:
    print(f"[FAIL] Python {sys.version_info.major}.{sys.version_info.minor} "
          "(3.9 or newer required; the reported runs used 3.12)")
    all_ok = False

checks = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("ray[rllib]", "ray"),
    ("gymnasium", "gymnasium"),
    ("torch", "torch"),
]

for pkg, import_pkg in checks:
    if not check_package(pkg, import_pkg):
        all_ok = False

if all_ok:
    print("-" * 40)
    print("[OK]   All checks passed.")
    sys.exit(0)
else:
    print("-" * 40)
    print("[FAIL] Some checks failed. Run: pip install -r requirements.txt")
    sys.exit(1)
