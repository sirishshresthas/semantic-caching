import importlib
import subprocess


def install_package(package):
    subprocess.check_call(["pip", "install", package])

def check_and_install_package(package):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"The package '{package}' is not installed. Installing...")
        install_package(package)
        print(f"The package '{package}' has been installed.")

