"""
json_loader.py - A utility to easily access JSON config entries.
"""
import json

class Loader:
    def __init__(self, d: dict):
        for key in d:
            if isinstance(d[key], dict):
                self.__setattr__(key, Loader(d[key]))
            else:
                self.__setattr__(key, d[key])

def json_loader(file: str):
    with open(file) as f:
        return Loader(json.load(f))