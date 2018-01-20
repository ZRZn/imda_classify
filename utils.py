def batch_generator(X, y, batch_size):
    size = len(X)
    i = 0
    while True:
        if i + batch_size <= size:
            yield X[i:i + batch_size], y[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            continue


