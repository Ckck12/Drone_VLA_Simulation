import pathlib
import sys

# make `import dronevla` work however pytest is launched
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
