# importer_test.py
print("Attempting to import vector_store module and function...")
try:
    # Option 1: Try importing the function directly
    from vector_store import vector_store 
    print("SUCCESS: Imported 'vector_store' function directly.")
    print("Type:", type(vector_store))

    # Option 2: Try importing the module
    import vector_store as vs_module
    print("SUCCESS: Imported 'vector_store' as module.")
    print("Type of module:", type(vs_module))
    print("Function defined in module?", hasattr(vs_module, 'vector_store'))
    if hasattr(vs_module, 'vector_store'):
         print("Type of function in module:", type(vs_module.vector_store))

except ImportError as ie:
    print(f"*** FAILED: ImportError: {ie}")
except Exception as e:
    print(f"*** FAILED: Other exception during import: {e}")
    import traceback
    traceback.print_exc()
print("Import test finished.")