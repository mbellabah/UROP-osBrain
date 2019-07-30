import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print("#"*31)
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
            print("#"*31)
        return result
    return timed


def print_final(rounds):
    num_hashtags: int = (31 - len(str(rounds)) - len("FINAL: "))//2
    print("\n")
    print(f'{"#"*num_hashtags} FINAL: {rounds} {"#"*num_hashtags}')


def col_print(name, data):
    print("-"*31)
    print(f"\t{name}")
    print("-"*31)
    print(data)


