"""
STEP 0: INSTALLATION SCRIPT
Run this first to install all dependencies
"""

# Save this as: install_dependencies.py
# Then run: python install_dependencies.py

import subprocess
import sys

def run_command(cmd):
    """Run a shell command and print output"""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠️  Warning: Command failed with return code {result.returncode}")
        return False
    return True

def main():
    print("🚀 Starting PDF RAG Setup...\n")
    
    # Step 1: Upgrade pip
    print("📦 Step 1/6: Upgrading pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Step 2: Install core dependencies
    print("\n📦 Step 2/6: Installing core dependencies...")
    core_packages = [
        "streamlit>=1.28.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "chromadb>=0.4.0",
        "docling>=2.0.0",
        "docling-core>=2.0.0",
        "docling-parse>=2.0.0",
    ]
    for package in core_packages:
        run_command(f"{sys.executable} -m pip install '{package}'")
    
    # Step 3: Install Ollama integration
    print("\n📦 Step 3/6: Installing Ollama integration...")
    run_command(f"{sys.executable} -m pip install langchain-ollama")
    
    # Step 4: Install optional dependencies
    print("\n📦 Step 4/6: Installing optional dependencies...")
    optional_packages = [
        "pypdf",
        "pdf2image",
        "pillow",
        "pydantic>=2.0",
    ]
    for package in optional_packages:
        run_command(f"{sys.executable} -m pip install '{package}'")
    
    # Step 5: Install docling tools
    print("\n📦 Step 5/6: Installing docling-tools...")
    run_command(f"{sys.executable} -m pip install docling-tools")
    
    # Step 6: Download models
    print("\n📦 Step 6/6: Downloading Docling models...")
    print("⏳ This may take 5-15 minutes on first run...")
    run_command("docling-tools models download")
    
    print("\n" + "="*60)
    print("✅ Installation complete!")
    print("="*60)
    print("\n📝 Next steps:")
    print("1. Make sure Ollama is running on your system")
    print("2. Run the RAG app: streamlit run rag_app.py")
    print("\n")

if __name__ == "__main__":
    main()