#!/bin/bash
set -euo pipefail

# evaluate.py writes the Harbor verifier result to /logs/verifier/reward.json.
python3 /tests/evaluate.py
