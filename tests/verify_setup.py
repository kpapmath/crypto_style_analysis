#!/usr/bin/env python
"""Verify that the virtual environment is set up correctly."""

import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("\nTesting imports:")

try:
    import numpy as np
    print(f"✅ numpy {np.__version__}")
except ImportError as e:
    print(f"❌ numpy: {e}")

try:
    import pandas as pd
    print(f"✅ pandas {pd.__version__}")
except ImportError as e:
    print(f"❌ pandas: {e}")

try:
    from cvxopt import matrix, solvers
    from cvxopt.coneprog import coneqp
    from cvxopt.solvers import qp
    print(f"✅ cvxopt (all modules)")
except ImportError as e:
    print(f"❌ cvxopt: {e}")

try:
    import matplotlib
    print(f"✅ matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ matplotlib: {e}")

try:
    import sklearn
    print(f"✅ scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"❌ scikit-learn: {e}")

try:
    import statsmodels
    print(f"✅ statsmodels {statsmodels.__version__}")
except ImportError as e:
    print(f"❌ statsmodels: {e}")

try:
    import ipykernel
    print(f"✅ ipykernel {ipykernel.__version__}")
except ImportError as e:
    print(f"❌ ipykernel: {e}")

print("\n🎉 All packages successfully imported!")

