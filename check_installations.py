import sys
import pkg_resources

def check_python_version():
    print(f"Python Version: {sys.version}")
    print(f"Python Location: {sys.executable}")

def check_required_packages():
    required_packages = [
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'opencv-python',
        'matplotlib',
        'seaborn',
        'jupyter'
    ]
    
    print("\nChecking required packages:")
    print("-" * 50)
    
    for package in required_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✓ {package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"✗ {package}: Not installed")
        except Exception as e:
            print(f"✗ {package}: Error checking version - {str(e)}")

if __name__ == "__main__":
    print("Checking Python and required libraries installation...")
    print("=" * 50)
    
    check_python_version()
    check_required_packages() 