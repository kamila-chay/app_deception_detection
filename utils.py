def rem_duplicates(orig):
    res = []
    res_set = set()
    for item in orig:
        if item not in res_set:
            res.append(item)
            res_set.add(item)
    return res