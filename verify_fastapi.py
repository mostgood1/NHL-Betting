import fastapi
from fastapi import FastAPI
print('fastapi version:', fastapi.__version__)
print('Has FastAPI:', hasattr(fastapi, 'FastAPI'))
