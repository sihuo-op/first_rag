import ragas
print('ragas version:', ragas.__version__)
print('\navailable attributes:')
for attr in dir(ragas):
    if not attr.startswith('_'):
        print(f'  {attr}')
