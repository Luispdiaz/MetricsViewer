import decimal

def convert_decimals(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_decimals(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    else:
        return obj