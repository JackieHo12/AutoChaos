#!/usr/bin/env python3
import sys

def check_package(name, import_name=None):
    import_name = import_name or name
    try:
        __import__(import_name)
        print(f"✓ {name} installed")
        return True
    except ImportError:
        print(f"✗ {name} NOT installed")
        return False

print("Validating AutoChaos setup...")
print("-" * 40)

checks = [
    ("Python 3.8+", None),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("ray[rllib]", "ray"),
    ("gymnasium", "gymnasium"),
    ("torch", "torch"),
]

all_ok = True
for pkg, import_pkg in checks:
    if import_pkg and not check_package(pkg, import_pkg):
        all_ok = False

if all_ok:
    print("-" * 40)
    print("✓ All checks passed!")
    sys.exit(0)
else:
    print("-" * 40)
    print("✗ Some packages missing. Run: pip install -r requirements.txt")
    sys.exit(1)
