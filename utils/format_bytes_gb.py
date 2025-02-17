def format_bytes_to_gb(bytes_value):
    if isinstance(bytes_value, (int, float)):
        return f"{bytes_value / (1024**3):.2f} GB"
    return 'Not Available'