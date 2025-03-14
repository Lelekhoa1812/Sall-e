import os

# Dir path
CACHE_DIR = "/home/user/app/cache"
MODEL_DIR = "/home/user/app/model"

# Verify structures:

# Model Dir
def print_model():
    print("\n📂 Model Structure (Build Level):")
    for root, dirs, files in os.walk(MODEL_DIR):
        print(f"📁 {root}/")
        for file in files:
            print(f"  📄 {file}")

# Cache Dir
def print_cache():
    print("\n📂 Cache Structure (Build Level):")
    for root, dirs, files in os.walk(CACHE_DIR):
        print(f"📁 {root}/")
        for file in files:
            print(f"  📄 {file}")


# Show
print_model()
print_cache()