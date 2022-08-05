def generator_limit(generator, n):
    limit = 0
    for item in generator:
        if limit >= n:
            break
        yield item
        limit += 1
 
