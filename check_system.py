import platform
import sys

print(f"System: {platform.system()} {platform.machine()}")
print(f"Python: {sys.version.split()[0]}")
print(f"Processor: {platform.processor()}")
