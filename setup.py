#!/usr/bin/env python3
"""
Setup script for Multi-Agent Research and Summarization System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Successfully installed all requirements")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🔍 Testing imports...")
    
    test_modules = [
        ("langchain.chat_models", "ChatOpenAI"),
        ("langchain.prompts", "ChatPromptTemplate"),
        ("langchain.schema.output_parser", "StrOutputParser"),
        ("chromadb", None),
        ("sentence_transformers", "SentenceTransformer"),
        ("requests", None),
        ("bs4", "BeautifulSoup"),
        ("pandas", None),
        ("numpy", None)
    ]
    
    all_good = True
    
    for module_name, class_name in test_modules:
        try:
            module = __import__(module_name, fromlist=[class_name] if class_name else [])
            if class_name:
                getattr(module, class_name)
            print(f"✅ {module_name}" + (f".{class_name}" if class_name else ""))
        except ImportError as e:
            print(f"❌ {module_name}" + (f".{class_name}" if class_name else "") + f": {e}")
            all_good = False
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name}: {e}")
            all_good = False
    
    return all_good

def test_workflow_import():
    """Test if the main workflow can be imported"""
    print("\n🚀 Testing workflow import...")
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        from workflow.multi_agent_workflow import MultiAgentWorkflow
        print("✅ Successfully imported MultiAgentWorkflow")
        return True
    except Exception as e:
        print(f"❌ Error importing workflow: {e}")
        return False

def check_environment():
    """Check if environment is properly configured"""
    print("\n🔧 Checking environment...")
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found. Creating from template...")
        if os.path.exists('.env.example'):
            with open('.env.example', 'r') as src:
                content = src.read()
            with open('.env', 'w') as dst:
                dst.write(content)
            print("✅ Created .env file from template")
            print("🔑 Please edit .env and add your OpenAI API key")
        else:
            print("❌ No .env.example found")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"⚠️  Python {python_version.major}.{python_version.minor} detected. Python 3.8+ recommended")

def main():
    """Main setup function"""
    print("🎯 Multi-Agent Research System Setup")
    print("=" * 50)
    
    # Check environment
    check_environment()
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed at package installation")
        return False
    
    # Test imports
    if not test_imports():
        print("\n❌ Setup failed at import testing")
        return False
    
    # Test workflow import
    if not test_workflow_import():
        print("\n❌ Setup failed at workflow import")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run: python main.py")
    print("3. Or open: multi_agent_notebook.ipynb")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)