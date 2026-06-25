# ══════════════════════════════════════════════════════════════════════════════
#  LEITURA DO ARQUIVO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

def ler_preferencias(caminho):
    """
    Lê o arquivo de preferências e retorna as estruturas de rank.

    Formato do arquivo:
      Linha 1      : N (número de alunos por escola)
      Linhas 2..N+1   : preferências de B — 1º número é índice (ignorado),
                        restante é lista ordenada de favoritos de A (1-based)
      Linhas N+2..2N+1: preferências de A — mesmo formato, lista de B (1-based)

    Retorna:
      n        (int)            — número de alunos por escola
      rank_b   (list[list[int]])— rank_b[i][j] = rank que Bi dá para Aj (1 = favorito)
      rank_a   (list[list[int]])— rank_a[i][j] = rank que Ai dá para Bj (1 = favorito)
      Índices internos são 0-based.
    """
    # Passo 1: abrir e limpar o arquivo
    with open(caminho) as f:
        linhas = [l.strip() for l in f if l.strip()]

    # Passo 2: ler N da primeira linha
    n = int(linhas[0])

    # Passo 3: função interna que descarta o índice inicial de cada linha
    def parse_linha(linha):
        nums = list(map(int, linha.split()))
        return nums[1:]   # descarta o índice inicial (1º número)

    # Passo 4: construir rank_b
    rank_b = [[0] * n for _ in range(n)]
    for i in range(n):
        prefs = parse_linha(linhas[1 + i])         
        for pos, aluno_a in enumerate(prefs):
            rank_b[i][aluno_a - 1] = pos + 1        # pos=0 → rank 1 (favorito)

    # Passo 5: construir rank_a 
    rank_a = [[0] * n for _ in range(n)]
    for i in range(n):
        prefs = parse_linha(linhas[1 + n + i])
        for pos, aluno_b in enumerate(prefs):
            rank_a[i][aluno_b - 1] = pos + 1
    return n, rank_b, rank_a


#testes

if __name__ == "__main__":
    import sys
    import os

    caminho = sys.argv[1] if len(sys.argv) > 1 else "arquivoDeTeste1.txt"

    if not os.path.exists(caminho):
        print(f"Arquivo não encontrado: {caminho}")
        sys.exit(1)

    n, rank_b, rank_a = ler_preferencias(caminho)

    print(f"N = {n}")
    print()

    # Verifica que nenhuma célula ficou zerada (todo rank foi preenchido)
    zeros_b = sum(rank_b[i][j] == 0 for i in range(n) for j in range(n))
    zeros_a = sum(rank_a[i][j] == 0 for i in range(n) for j in range(n))
    print(f"Células zeradas em rank_b: {zeros_b}  (esperado: 0)")
    print(f"Células zeradas em rank_a: {zeros_a}  (esperado: 0)")
    print()

    # Verifica que cada linha é uma permutação de 1..n (sem ranks duplicados)
    ok_b = all(sorted(rank_b[i]) == list(range(1, n + 1)) for i in range(n))
    ok_a = all(sorted(rank_a[i]) == list(range(1, n + 1)) for i in range(n))
    print(f"rank_b é permutação válida em todas as linhas: {ok_b}")
    print(f"rank_a é permutação válida em todas as linhas: {ok_a}")
    print()

    # Exibe rank_b[0] e rank_a[0] para conferência manual
    print(f"rank_b[0] (preferências de B1 sobre cada Aj): {rank_b[0]}")
    print(f"  → B1 prefere A{rank_b[0].index(1) + 1} em primeiro lugar")
    print()
    print(f"rank_a[0] (preferências de A1 sobre cada Bj): {rank_a[0]}")
    print(f"  → A1 prefere B{rank_a[0].index(1) + 1} em primeiro lugar")
