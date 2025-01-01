import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
try:
    import truststore
    if not os.environ.get("NUNIF_TRUSTSTORE_INJECTED", False):
        truststore.inject_into_ssl()
        os.environ["NUNIF_TRUSTSTORE_INJECTED"] = "1"
except ModuleNotFoundError:
    pass
