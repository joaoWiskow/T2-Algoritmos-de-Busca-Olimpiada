"""
Algoritmo Genético para Distribuição de Duplas Heterogêneas
===========================================================
Codificação : Vetor de permutação (índices da Escola B)
Seleção     : Torneio (k=3)
Crossover   : Ordem (OX)
Mutação     : Troca (Swap)
Elitismo    : k=1 (melhor indivíduo sempre preservado)
"""
#Arquivo baseado no seguinte:Vetor de Permutação (Escola B) + Seleção por Torneio + Crossover por Ordem (OX) 
# + Mutação por Troca (Swap) + Elitismo de 1.
import random
import sys
import os
import time
import copy

# ─── Parâmetros do AG ────────────────────────────────────────────────────────
TAM_POPULACAO  = 100
MAX_GERACOES   = 500
TAXA_CROSSOVER = 0.9
TAXA_MUTACAO   = 0.15
TAM_TORNEIO    = 3
ELITISMO       = 1
SEMENTE        = 42          # None para resultados diferentes a cada execução
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  LEITURA DO ARQUIVO
# ══════════════════════════════════════════════════════════════════════════════

def ler_preferencias(caminho):
    """
    Formato esperado:
      Linha 1 : N
      Linhas 2..N+1  : preferências de B  (1º número = índice, ignorado)
      Linhas N+2..2N+1 : preferências de A (1º número = índice, ignorado)

    Retorna:
      pref_b[i][j] = rank que Bi dá para Aj  (1 = favorito)
      pref_a[i][j] = rank que Ai dá para Bj  (1 = favorito)
      (índices 0-based internamente)
    """
    with open(caminho) as f:
        linhas = [l.strip() for l in f if l.strip()]

    n = int(linhas[0])

    def parse_linha(linha):
        nums = list(map(int, linha.split()))
        return nums[1:]            # descarta o índice inicial

    # pref_b[i] = lista ordenada por preferência de Bi sobre alunos de A
    # ou seja, pref_b[i][0] é o 1º preferido de Bi (valor 1-based → converte para 0-based)
    # Internamente vamos armazenar como RANK: rank_b[i][j] = posição que Bi dá para Aj
    rank_b = [[0] * n for _ in range(n)]
    for i in range(n):
        prefs = parse_linha(linhas[1 + i])          # lista de preferência (1-based índices de A)
        for pos, aluno_a in enumerate(prefs):
            rank_b[i][aluno_a - 1] = pos + 1        # rank começa em 1

    rank_a = [[0] * n for _ in range(n)]
    for i in range(n):
        prefs = parse_linha(linhas[1 + n + i])      # lista de preferência (1-based índices de B)
        for pos, aluno_b in enumerate(prefs):
            rank_a[i][aluno_b - 1] = pos + 1

    return n, rank_b, rank_a


# ══════════════════════════════════════════════════════════════════════════════
#  CODIFICAÇÃO / DECODIFICAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def cromossomo_aleatorio(n):
    """Permutação aleatória de 0..n-1 (índices de B)."""
    c = list(range(n))
    random.shuffle(c)
    return c


def decodificar(cromossomo):
    """
    Retorna lista de duplas: [(A0,B?), (A1,B?), ...]
    índices 0-based, exibição 1-based.
    """
    return [(i, cromossomo[i]) for i in range(len(cromossomo))]


# ══════════════════════════════════════════════════════════════════════════════
#  FUNÇÃO DE FITNESS
# ══════════════════════════════════════════════════════════════════════════════

def fitness(cromossomo, n, rank_b, rank_a):
    """
    Para cada dupla (Ai, Bj):
      score = (N - rank_A[i][j] + 1) + (N - rank_B[j][i] + 1)
    fitness = soma de todos os scores  (quanto maior, melhor)
    """
    total = 0
    for i, j in decodificar(cromossomo):
        score_a = (n - rank_a[i][j] + 1)   # Ai avalia Bj
        score_b = (n - rank_b[j][i] + 1)   # Bj avalia Ai
        total += score_a + score_b
    return total


def fitness_maximo(n):
    """Fitness máximo teórico: todos os pares com rank=1 em ambos os lados."""
    return n * 2 * n


# ══════════════════════════════════════════════════════════════════════════════
#  OPERADORES GENÉTICOS
# ══════════════════════════════════════════════════════════════════════════════

def selecao_torneio(populacao, fitnesses, k=TAM_TORNEIO):
    candidatos = random.sample(range(len(populacao)), k)
    melhor = max(candidatos, key=lambda i: fitnesses[i])
    return populacao[melhor][:]


def crossover_ox(p1, p2):
    """Order Crossover (OX) — gera 2 filhos."""
    n = len(p1)
    c1, c2 = [-1] * n, [-1] * n

    def ox_filho(pai1, pai2):
        filho = [-1] * n
        i1, i2 = sorted(random.sample(range(n), 2))
        # copia segmento do pai1
        filho[i1:i2+1] = pai1[i1:i2+1]
        segmento = set(pai1[i1:i2+1])
        # preenche com pai2 a partir de i2+1
        pos = (i2 + 1) % n
        for gene in pai2[i2+1:] + pai2[:i2+1]:
            if gene not in segmento:
                filho[pos] = gene
                segmento.add(gene)
                pos = (pos + 1) % n
        return filho

    return ox_filho(p1, p2), ox_filho(p2, p1)


def mutacao_swap(cromossomo, taxa=TAXA_MUTACAO):
    c = cromossomo[:]
    if random.random() < taxa:
        i, j = random.sample(range(len(c)), 2)
        c[i], c[j] = c[j], c[i]
    return c


# ══════════════════════════════════════════════════════════════════════════════
#  ALGORITMO GENÉTICO
# ══════════════════════════════════════════════════════════════════════════════

def inicializar_populacao(n, tam):
    return [cromossomo_aleatorio(n) for _ in range(tam)]


def proxima_geracao(populacao, fitnesses, n):
    tam = len(populacao)

    # Elitismo: preserva o melhor
    idx_elite = max(range(tam), key=lambda i: fitnesses[i])
    nova = [populacao[idx_elite][:]]

    # Gera filhos até completar a população
    while len(nova) < tam:
        p1 = selecao_torneio(populacao, fitnesses)
        p2 = selecao_torneio(populacao, fitnesses)

        if random.random() < TAXA_CROSSOVER:
            f1, f2 = crossover_ox(p1, p2)
        else:
            f1, f2 = p1[:], p2[:]

        nova.append(mutacao_swap(f1))
        if len(nova) < tam:
            nova.append(mutacao_swap(f2))

    return nova


def melhor_da_populacao(populacao, fitnesses):
    idx = max(range(len(populacao)), key=lambda i: fitnesses[i])
    return populacao[idx], fitnesses[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  EXIBIÇÃO
# ══════════════════════════════════════════════════════════════════════════════

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
GRAY   = "\033[90m"
MAGENTA= "\033[95m"

def limpar():
    os.system("cls" if os.name == "nt" else "clear")

def barra_progresso(valor, maximo, largura=30):
    preenchido = int(largura * valor / maximo)
    barra = "█" * preenchido + "░" * (largura - preenchido)
    pct = 100 * valor / maximo
    return f"[{barra}] {pct:.1f}%"

def grafico_ascii(historico, largura=50, altura=10):
    """Mini gráfico ASCII do fitness ao longo das gerações."""
    if len(historico) < 2:
        return ""
    minv, maxv = min(historico), max(historico)
    if minv == maxv:
        return ""
    linhas = []
    pontos = historico
    # sub-amostra se necessário
    if len(pontos) > largura:
        step = len(pontos) / largura
        pontos = [pontos[int(i * step)] for i in range(largura)]
    colunas = len(pontos)
    grade = [[" "] * colunas for _ in range(altura)]
    for col, val in enumerate(pontos):
        row = altura - 1 - int((val - minv) / (maxv - minv) * (altura - 1))
        row = max(0, min(altura - 1, row))
        grade[row][col] = "▪"
    linhas.append(f"{GRAY}  {maxv:>6.0f} ┤{''.join(grade[0])}{RESET}")
    for r in range(1, altura - 1):
        linhas.append(f"  {'':>6} │{''.join(grade[r])}")
    linhas.append(f"  {minv:>6.0f} ┤{''.join(grade[-1])}{RESET}")
    linhas.append(f"  {'':>6}  {'G=1':^{colunas//3}}{'G={len(historico)//2}':^{colunas//3}}{'G={len(historico)}':^{colunas//3}}")
    return "\n".join(linhas)

def exibir_solucao(cromossomo, n, rank_b, rank_a, fit, fit_max):
    print(f"\n{BOLD}{CYAN}══ SOLUÇÃO ENCONTRADA ══{RESET}")
    print(f"\n{BOLD}Codificada:{RESET}")
    print(f"  {cromossomo}")
    print(f"\n{BOLD}Decodificada:{RESET}")
    print(f"  {'Dupla':<8} {'Aluno A':<10} {'Aluno B':<10} {'Rank A→B':<10} {'Rank B→A':<10} {'Score'}")
    print(f"  {'-'*58}")
    total = 0
    for i, j in decodificar(cromossomo):
        ra = rank_a[i][j]
        rb = rank_b[j][i]
        score = (n - ra + 1) + (n - rb + 1)
        total += score
        cor = GREEN if score >= n else (YELLOW if score >= n // 2 else RESET)
        print(f"  {cor}#{i+1:<7} A{i+1:<9} B{j+1:<9} {ra:<10} {rb:<10} {score}{RESET}")
    print(f"  {'-'*58}")
    print(f"  {'Fitness total:':<28} {BOLD}{total}{RESET}")
    print(f"  {'Fitness máximo teórico:':<28} {fit_max}")
    print(f"  {'Aproveitamento:':<28} {BOLD}{GREEN}{100*total/fit_max:.1f}%{RESET}")


def exibir_geracao(geracao, historico, melhor_fit, fit_max, n_geracoes):
    limpar()
    print(f"{BOLD}{CYAN}╔══ AG — Duplas Heterogêneas ══╗{RESET}")
    print(f"  Geração : {BOLD}{geracao+1:>4}{RESET} / {n_geracoes}")
    print(f"  Fitness : {BOLD}{GREEN}{melhor_fit}{RESET}  (máx teórico: {fit_max})")
    print(f"  {barra_progresso(melhor_fit, fit_max)}")
    print()
    print(f"{BOLD}Evolução do fitness:{RESET}")
    print(grafico_ascii(historico))
    print()
    print(f"{GRAY}  [Enter] próxima geração   [q + Enter] pular para o final{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODO PASSO A PASSO
# ══════════════════════════════════════════════════════════════════════════════

def modo_passo_a_passo(n, rank_b, rank_a):
    if SEMENTE is not None:
        random.seed(SEMENTE)

    fit_max = fitness_maximo(n)
    populacao = inicializar_populacao(n, TAM_POPULACAO)
    fitnesses = [fitness(c, n, rank_b, rank_a) for c in populacao]
    historico = []
    melhor_cromo, melhor_fit = melhor_da_populacao(populacao, fitnesses)

    for geracao in range(MAX_GERACOES):
        historico.append(melhor_fit)
        exibir_geracao(geracao, historico, melhor_fit, fit_max, MAX_GERACOES)

        entrada = input().strip().lower()
        if entrada == "q":
            # executa o restante sem pausa
            for g in range(geracao + 1, MAX_GERACOES):
                populacao = proxima_geracao(populacao, fitnesses, n)
                fitnesses = [fitness(c, n, rank_b, rank_a) for c in populacao]
                cm, cf = melhor_da_populacao(populacao, fitnesses)
                if cf > melhor_fit:
                    melhor_cromo, melhor_fit = cm, cf
                historico.append(melhor_fit)
            break

        populacao = proxima_geracao(populacao, fitnesses, n)
        fitnesses = [fitness(c, n, rank_b, rank_a) for c in populacao]
        cm, cf = melhor_da_populacao(populacao, fitnesses)
        if cf > melhor_fit:
            melhor_cromo, melhor_fit = cm, cf

    limpar()
    print(f"{BOLD}{CYAN}╔══ AG — Evolução Completa ══╗{RESET}\n")
    print(f"{BOLD}Gráfico de fitness:{RESET}")
    print(grafico_ascii(historico, largura=60, altura=12))
    exibir_solucao(melhor_cromo, n, rank_b, rank_a, melhor_fit, fit_max)


# ══════════════════════════════════════════════════════════════════════════════
#  MODO AUTOMÁTICO
# ══════════════════════════════════════════════════════════════════════════════

def modo_automatico(n, rank_b, rank_a):
    if SEMENTE is not None:
        random.seed(SEMENTE)

    fit_max = fitness_maximo(n)
    populacao = inicializar_populacao(n, TAM_POPULACAO)
    fitnesses = [fitness(c, n, rank_b, rank_a) for c in populacao]
    historico = []
    melhor_cromo, melhor_fit = melhor_da_populacao(populacao, fitnesses)

    print(f"{BOLD}{CYAN}╔══ AG — Duplas Heterogêneas (Automático) ══╗{RESET}")
    print(f"  Populacao={TAM_POPULACAO}  Gerações={MAX_GERACOES}  "
          f"Pc={TAXA_CROSSOVER}  Pm={TAXA_MUTACAO}  Elite={ELITISMO}\n")

    intervalo = max(1, MAX_GERACOES // 20)   # exibe ~20 linhas de progresso

    for geracao in range(MAX_GERACOES):
        historico.append(melhor_fit)

        if geracao % intervalo == 0 or geracao == MAX_GERACOES - 1:
            pct = 100 * melhor_fit / fit_max
            print(f"  Gen {geracao+1:>4}/{MAX_GERACOES}  "
                  f"fitness={melhor_fit:>6}  {barra_progresso(melhor_fit, fit_max, 20)}")

        populacao = proxima_geracao(populacao, fitnesses, n)
        fitnesses = [fitness(c, n, rank_b, rank_a) for c in populacao]
        cm, cf = melhor_da_populacao(populacao, fitnesses)
        if cf > melhor_fit:
            melhor_cromo, melhor_fit = cm, cf

    print(f"\n{BOLD}Gráfico de evolução do fitness:{RESET}")
    print(grafico_ascii(historico, largura=60, altura=12))
    exibir_solucao(melhor_cromo, n, rank_b, rank_a, melhor_fit, fit_max)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRADA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def uso():
    print(f"""
{BOLD}Uso:{RESET}
  python ag_duplas.py <arquivo> <modo>

{BOLD}Modos:{RESET}
  passo   — executa geração a geração (pressione Enter para avançar, q para pular)
  auto    — executa tudo e exibe o resultado final

{BOLD}Exemplo:{RESET}
  python ag_duplas.py arquivoDeTeste1.txt auto
  python ag_duplas.py arquivoDeTeste1.txt passo
""")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        uso()
        sys.exit(1)

    caminho = sys.argv[1]
    modo    = sys.argv[2].lower()

    if not os.path.exists(caminho):
        print(f"Arquivo não encontrado: {caminho}")
        sys.exit(1)

    if modo not in ("passo", "auto"):
        print(f"Modo inválido: '{modo}'. Use 'passo' ou 'auto'.")
        uso()
        sys.exit(1)

    n, rank_b, rank_a = ler_preferencias(caminho)
    print(f"{GREEN}Arquivo carregado: {caminho}  |  N={n} duplas{RESET}\n")

    if modo == "passo":
        modo_passo_a_passo(n, rank_b, rank_a)
    else:
        modo_automatico(n, rank_b, rank_a)