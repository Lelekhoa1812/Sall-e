import os

CACHE_DIR = "/app/cache"
MODEL_DIR = "/app/model"

# Verify structures:

# Model Dir
def print_model():
    print("\n📂 Model Structure (Build Level):")
    for root, dirs, files in os.walk(MODEL_DIR):
        print(f"📁 {root}/")
        for file in files:
            print(f"  📄 {file}")

# Cache Dir
def print_cache()
    print("\n📂 Cache Structure (Build Level):")
    for root, dirs, files in os.walk(CACHE_DIR):
        print(f"📁 {root}/")
        for file in files:
            print(f"  📄 {file}")