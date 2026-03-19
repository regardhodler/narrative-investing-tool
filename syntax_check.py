#!/usr/bin/env python
import ast
import sys

# Check engine
try:
    with open('services/elliott_wave_engine.py', encoding='utf-8') as f:
        ast.parse(f.read())
    print('engine OK')
except SyntaxError as e:
    print(f'engine SYNTAX ERROR: {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'engine ERROR: {e}', file=sys.stderr)
    sys.exit(1)

# Check module
try:
    with open('modules/elliott_wave.py', encoding='utf-8') as f:
        ast.parse(f.read())
    print('module OK')
except SyntaxError as e:
    print(f'module SYNTAX ERROR: {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'module ERROR: {e}', file=sys.stderr)
    sys.exit(1)
