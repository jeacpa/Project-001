from datetime import datetime, timezone
from psycopg2.extras import Json
import numpy as np

def utc_now():
    return datetime.now(timezone.utc)

# Converts numpy types to native Python types for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, dict):
        return Json({k: convert_numpy(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
