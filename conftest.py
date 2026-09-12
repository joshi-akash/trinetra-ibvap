import sys
import os

# Add the monorepo root to sys.path so that `ai_behavior` is resolvable
# both by pytest (at runtime) and by Pylance (via pyrightconfig.json below).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
