def _validate_int_tuple(*dims, func_name = ''): 
    dims = tuple(dims)
    if len(dims) == 1 and isinstance(dims, (tuple, list)): 
        dims = tuple(dims[0])

    if any((not isinstance(d, int) or d < 0) for d in dims): 
        raise ValueError('Expected Positive int in : {func_name}')
    return dims
