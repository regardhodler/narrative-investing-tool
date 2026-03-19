import ast
for f in ['services/wyckoff_engine.py', 'modules/wyckoff.py']:
    try:
        with open(f, encoding='utf-8') as fh: ast.parse(fh.read())
        print(f'OK: {f}')
    except Exception as e: print(f'ERROR {f}: {e}')
