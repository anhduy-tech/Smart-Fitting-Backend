# app/querydict.py
def querydict_to_nested_dict(query_dict):
    """
    Convert QueryDict to nested dictionary.
    Useful for handling form data with nested keys like 'user[name]'.
    """
    data = {}
    for key, value in query_dict.items():
        # Handle nested keys like parent[child]
        if '[' in key and ']' in key:
            # Simple nested support
            outer_key = key.split('[')[0]
            inner_key = key.split('[')[1].split(']')[0]
            if outer_key not in data:
                data[outer_key] = {}
            data[outer_key][inner_key] = value
        else:
            data[key] = value
    return data