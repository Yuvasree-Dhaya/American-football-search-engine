with open('hierarchial_complete_clustering.txt', 'r') as f:
    lines = f.readlines()
    with open('hierarchial_clustering_complete.txt', 'w') as out:
        try:
            for line in lines:
                url, value = line.split(',', maxsplit=1)
                url = url.strip("[]'")
                out.write(url + ',' + value)
        except ValueError:
            pass
            