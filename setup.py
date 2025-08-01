#!/usr/bin/env python3
"""
Setup guide and environment configuration for the Investment News Analysis Pipeline
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install required packages"""
    requirements = [
        "spacy>=3.4.0",
        "pandas>=1.3.0",
        "openai>=1.0.0",
        "python-dotenv>=0.19.0",
        "transformers>=4.20.0"
    ]
    
    print("📦 Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {package}")
            return False
    return True

def download_spacy_models():
    """Download spaCy models"""
    models = [
        ("en_core_web_sm", "Small English model (~50MB)"),
        ("en_core_web_md", "Medium English model (~50MB)"),
        ("en_core_web_trf", "Transformer English model (~500MB, best accuracy)")
    ]
    
    print("\n🧠 Downloading spaCy models...")
    for model, description in models:
        print(f"\n📥 Downloading {model} - {description}")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            print(f"✅ Downloaded: {model}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to download: {model}")
            if model == "en_core_web_trf":
                print("💡 Note: The transformer model requires additional dependencies.")
                print("   Try installing with: pip install spacy[transformers]")

def setup_env_file():
    """Create .env file for OpenAI API key"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        with open(env_file, 'r') as f:
            content = f.read()
            if "OPENAI_API_KEY" in content:
                print("✅ OPENAI_API_KEY found in .env file")
                return True
    
    print("\n🔑 Setting up OpenAI API key...")
    print("You need an OpenAI API key to use the sentiment analysis features.")
    print("Get your API key from: https://platform.openai.com/api-keys")
    
    api_key = input("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 ").strip()
    
    if api_key:
        with open(env_file, 'a') as f:
            f.write(f"\nOPENAI_API_KEY={api_key}\n")
        print("✅ OpenAI API key saved to .env file")
        return True
    else:
        print("⚠️ Skipping OpenAI API key setup. Sentiment analysis will use fallback method.")
        return False

def test_setup():
    """Test the setup"""
    print("\n🧪 Testing setup...")
    
    # Test spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Goldman Sachs raises Apple target")
        print("✅ spaCy is working correctly")
    except Exception as e:
        print(f"❌ spaCy test failed: {e}")
        return False
    
    # Test pandas
    try:
        import pandas as pd
        df = pd.DataFrame({'test': [1, 2, 3]})
        print("✅ Pandas is working correctly")
    except Exception as e:
        print(f"❌ Pandas test failed: {e}")
        return False
    
    # Test OpenAI (if API key is available)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("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")
        if api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            print("✅ OpenAI client initialized successfully")
        else:
            print("⚠️ OpenAI API key not found. Sentiment analysis will use fallback method.")
    except Exception as e:
        print(f"⚠️ OpenAI test failed: {e}")
    
    return True

def main():
    """Main setup function"""
    print("🚀 Investment News Analysis Pipeline Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return False
    
    # Download spaCy models
    download_spacy_models()
    
    # Setup environment file
    setup_env_file()
    
    # Test setup
    if not test_setup():
        print("❌ Setup test failed")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Make sure your CSV file is in the same directory")
    print("2. Run the pipeline with:")
    print("   python pipeinvest.py                    # Default configuration")
    print("   python pipeinvest.py --compare          # Compare all models")
    print("   python pipeinvest.py --single en_core_web_trf true  # Single model config")
    print("\n💡 Tips:")
    print("- Use en_core_web_trf for best accuracy (if you have the resources)")
    print("- Use en_core_web_sm for fastest processing")
    print("- Set OPENAI_API_KEY in .env file for best sentiment analysis")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)