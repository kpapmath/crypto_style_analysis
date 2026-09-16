#!/usr/bin/env python
try:
    from cvxopt import matrix, solvers
    print("✓ SUCCESS: cvxopt imported successfully")
    print(f"cvxopt version: {matrix.__module__}")
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

