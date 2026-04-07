#!/usr/bin/env python3
"""Default values for tunable parameters shared across modules.

These are the fallback values used when config.yaml omits a field.
Kept in a lightweight module (no heavy dependencies) so both
config/settings.py and training/ can import without circular deps.
"""

# --- Focal loss (validated over 25+ training iterations on IAB 2.2) ---
DEFAULT_FOCAL_ALPHA = 0.75        # Higher than paper default (0.25)
DEFAULT_FOCAL_GAMMA = 3.5         # Aggressive hard-example focus
DEFAULT_CLASS_WEIGHT_CAP = 7.0    # NEVER normalize after capping

# --- LLM token budgets ---
DEFAULT_ENRICHMENT_MAX_TOKENS = 300   # Per-category enrichment response
DEFAULT_STAGE1_MAX_TOKENS = 150       # Labeling tier-1 triage response
DEFAULT_STAGE2_MAX_TOKENS = 500       # Labeling subtree classification response

# --- Evaluation ---
MIN_SAMPLES_FOR_OPTIMIZATION = 5  # Below this, use global threshold
