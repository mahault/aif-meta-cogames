#!/usr/bin/env python3
"""Fit POMDP A/B matrices from trajectory data.

Thin CLI wrapper around aif_meta_cogames.aif_agent.fit_matrices.

Usage::

    python scripts/fit_pomdp.py \\
        --data ./trajectory_data_v3 \\
        --output ./fitted_pomdp

Or via module::

    python -m aif_meta_cogames.aif_agent.fit_matrices --data ... --output ...
"""

from aif_meta_cogames.aif_agent.fit_matrices import main

if __name__ == "__main__":
    main()
