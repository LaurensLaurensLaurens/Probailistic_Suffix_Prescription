import os
import pkgutil
import importlib

# Dynamically discover and import all subpackages
package_dir = os.path.dirname(__file__)
for module_info in pkgutil.iter_modules([package_dir]):
    if module_info.ispkg:  # Check if it's a package
        importlib.import_module(f"{__name__}.{module_info.name}")