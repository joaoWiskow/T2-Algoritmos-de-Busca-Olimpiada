def cromossomo_aleatorio(n):
    c = list(range(n))
    random.shuffle(c)
    return c


def decodificar(cromossomo):
    return [(i, cromossomo[i]) for i in range(len(cromossomo))]


def fitness(cromossomo, n, rank_b, rank_a):
    total = 0
    for i, j in decodificar(cromossomo):
        score_a = (n - rank_a[i][j] + 1)   # Ai avalia Bj
        score_b = (n - rank_b[j][i] + 1)   # Bj avalia Ai
        total += score_a + score_b
    return total


def fitness_maximo(n):
    return n * 2 * n