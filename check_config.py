import yaml
import os
import re

with open('config.yaml') as f:
    config = yaml.safe_load(f)

def flatten_keys(d, prefix=""):
    keys = []
    for k, v in d.items():
        keys.append(k)
        if isinstance(v, dict):
            keys.extend(flatten_keys(v, prefix + k + "."))
    return keys

keys_to_search = flatten_keys(config)
unused_keys = []

for key in keys_to_search:
    # Use grep to find the key in .py files
    ret = os.system(f"grep -r '{key}' . --include='*.py' > /dev/null")
    if ret != 0:
        unused_keys.append(key)

print("Unused keys:")
for key in unused_keys:
    print(key)
