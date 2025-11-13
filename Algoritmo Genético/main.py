# =============================================================================
# Gabriela Marculino -  RGM: 41431
# TRABALHO III de Inteligência Artificial
# Profº Drº Osvaldo Jacques
#
# Implementação do Algoritmo Genético (GA) Interativo
# com Semente Aleatória (Random Seed) para testes.
# =============================================================================

# =============================================================================
#  #  COMO TESTAR 
# =============================================================================
#
# Para comparar o impacto de diferentes parâmetros (ex: aumentar
# a 'População' ou a 'Elite'), é crucial usar a mesma "Semente".
#
# Exemplo de Teste Comparativo:
# 1. Execução 1:
#    - População: 100
#    - Elite: 20
#    - Semente: 42 (anote o resultado/distância final)
#
# 2. Execução 2:
#    - População: 500  <- (Valor alterado)
#    - Elite: 100     <- (Valor alterado)
#    - Semente: 42 (A semente DEVE ser a mesma)
#
# Ao usar a mesma semente (ex: 42), o mapa de cidades e toda a
# "sorte" aleatória serão idênticos. Isso garante que qualquer
# melhoria no resultado foi *realmente* por causa da mudança nos
# parâmetros (População/Elite), e não por "sorte".
#
# Deixar a semente em branco (apertar ENTER) gera um novo
# problema aleatório a cada execução.
# =============================================================================

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def calcular_distancia_total(cidades, rota):
    """Calcula a distância total de uma rota que passa por todas as cidades."""
    distancia_total = 0
    num_cidades = len(rota)
    for i in range(num_cidades):
        cidade_de_partida = rota[i]
        cidade_de_chegada = rota[(i + 1) % num_cidades]
        dist = np.linalg.norm(cidades[cidade_de_partida] - cidades[cidade_de_chegada])
        distancia_total += dist
    return distancia_total

def calcular_fitness(cidades, rota):
    """Calcula o fitness (aptidão) de uma rota."""
    distancia = calcular_distancia_total(cidades, rota)
    fitness = 1.0 / (distancia + 1e-6)
    return fitness

def criar_populacao_inicial(tam_populacao, num_cidades):
    """Cria a população inicial com rotas aleatórias."""
    populacao = []
    base_rota = list(range(num_cidades))
    for _ in range(tam_populacao):
        rota = random.sample(base_rota, len(base_rota))
        populacao.append(rota)
    return populacao

def classificar_rotas(populacao, cidades):
    """
    Avalia todas as rotas na população e as classifica pela fitness.
    Retorna uma lista de tuplas (fitness, rota), ordenada da melhor para a pior.
    """
    pop_com_fitness = []
    for rota in populacao:
        fitness = calcular_fitness(cidades, rota)
        pop_com_fitness.append((fitness, rota))
    pop_com_fitness.sort(key=lambda x: x[0], reverse=True)
    return pop_com_fitness

def selecao_roleta(pop_classificada):
    """
    Seleciona um "pai" usando o método da Roleta.
    (Conforme material do professor)
    """
    fitness_total = sum(fitness for fitness, rota in pop_classificada)
    ponto_sorteado = random.uniform(0, fitness_total)
    
    soma_fitness_atual = 0
    for fitness, rota in pop_classificada:
        soma_fitness_atual += fitness
        if soma_fitness_atual >= ponto_sorteado:
            return rota
            
    return pop_classificada[0][1]

def crossover_ordenado(pai1, pai2):
    """Implementa o Crossover Ordenado (OX), essencial para TSP."""
    tam = len(pai1)
    filho = [None] * tam
    p1, p2 = sorted(random.sample(range(tam), 2))
    
    segmento_pai1 = pai1[p1:p2+1]
    filho[p1:p2+1] = segmento_pai1
    
    genes_pai2 = []
    for i in range(tam):
        idx = (p2 + 1 + i) % tam
        if pai2[idx] not in segmento_pai1:
            genes_pai2.append(pai2[idx])
            
    idx_filho = (p2 + 1) % tam
    for gene in genes_pai2:
        while filho[idx_filho] is not None:
            idx_filho = (idx_filho + 1) % tam
        filho[idx_filho] = gene
        
    return filho

def mutacao(rota, taxa_mutacao):
    """Aplica uma mutação de "swap" (troca) com uma certa probabilidade."""
    if random.random() < taxa_mutacao:
        i, j = random.sample(range(len(rota)), 2)
        rota[i], rota[j] = rota[j], rota[i]
    return rota

def algoritmo_genetico_generator(cidades, tam_populacao, tam_elite, taxa_mutacao):
    """
    Esta função implementa o Algoritmo Genético como um gerador em Python.
    (Agora usando Seleção por Roleta)
    """
    num_cidades = len(cidades)
    
    populacao = criar_populacao_inicial(tam_populacao, num_cidades)
    pop_classificada = classificar_rotas(populacao, cidades)
    
    melhor_rota_geracao = pop_classificada[0][1]
    melhor_dist_geracao = calcular_distancia_total(cidades, melhor_rota_geracao)
    
    geracao_atual = 0
    
    yield melhor_rota_geracao, melhor_dist_geracao, geracao_atual
    
    while True:
        geracao_atual += 1
        
        # Passo 3: Criar Nova Geração
        nova_populacao = []
        
        # 3a. Elitismo
        for i in range(tam_elite):
            nova_populacao.append(pop_classificada[i][1])
        
        # 3b. Crossover e Mutação
        tam_nova_pop = tam_populacao - tam_elite
        for _ in range(tam_nova_pop):
            
            pai1 = selecao_roleta(pop_classificada)
            pai2 = selecao_roleta(pop_classificada)
            
            filho = crossover_ordenado(pai1, pai2)
            filho_mutado = mutacao(filho, taxa_mutacao)
            nova_populacao.append(filho_mutado)
        
        # A nova geração substitui a antiga
        populacao = nova_populacao
        
        # Classifica a nova população para o próximo loop e para o yield
        pop_classificada = classificar_rotas(populacao, cidades)
        melhor_rota_geracao = pop_classificada[0][1]
        melhor_dist_geracao = calcular_distancia_total(cidades, melhor_rota_geracao)
        
        if geracao_atual % 10 == 0:
             print(f"Geração {geracao_atual:03d} | Melhor Distância: {melhor_dist_geracao:.2f}")

        # Pausa e retorna o melhor da geração atual
        yield melhor_rota_geracao, melhor_dist_geracao, geracao_atual

if __name__ == "__main__":

    print("="*60)
    print("  Configuração dos Parâmetros do Algoritmo Genético (TSP)")
    print("="*60)
    print("Pressione ENTER para usar o valor padrão (ex: [25]).\n")

    # Parâmetro 1: Número de Cidades
    while True:
        try:
            val = input("1. Número de Cidades [padrão: 25]: ")
            NUM_CIDADES = int(val) if val else 25
            if NUM_CIDADES < 3:
                print("ERRO: O número de cidades deve ser pelo menos 3.")
            else:
                break
        except ValueError:
            print("ERRO: Por favor, digite um número inteiro.")

    # Parâmetro 2: Tamanho da População
    while True:
        try:
            val = input("2. Tamanho da População [padrão: 100]: ")
            TAM_POPULACAO = int(val) if val else 100
            if TAM_POPULACAO < 10:
                print("ERRO: A população deve ter pelo menos 10 indivíduos.")
            else:
                break
        except ValueError:
            print("ERRO: Por favor, digite um número inteiro.")

    # Parâmetro 3: Tamanho da Elite
    while True:
        try:
            val = input(f"3. Tamanho da Elite [padrão: 20]: ")
            TAM_ELITE = int(val) if val else 20
            if TAM_ELITE >= TAM_POPULACAO:
                print(f"ERRO: A elite ({TAM_ELITE}) não pode ser maior ou igual à população ({TAM_POPULACAO}).")
            elif TAM_ELITE < 0:
                print("ERRO: A elite não pode ser negativa.")
            else:
                break
        except ValueError:
            print("ERRO: Por favor, digite um número inteiro.")

    # Parâmetro 4: Taxa de Mutação
    while True:
        try:
            val = input("4. Taxa de Mutação (ex: 0.01) [padrão: 0.01]: ")
            TAXA_MUTACAO = float(val) if val else 0.01
            if not (0.0 <= TAXA_MUTACAO <= 1.0):
                print("ERRO: A taxa de mutação deve estar entre 0.0 e 1.0.")
            else:
                break
        except ValueError:
            print("ERRO: Por favor, digite um número (ex: 0.01).")
            
    # Parâmetro 5: Semente Aleatória
    while True:
        try:
            val = input("5. Semente Aleatória (ex: 42) [deixe em branco para aleatório]: ")
            # Se o usuário digitar algo, converte para int. Se deixar em branco, SEMENTE será None.
            SEMENTE = int(val) if val else None
            break
        except ValueError:
            print("ERRO: Por favor, digite um número inteiro (ou deixe em branco).")


    print("\n" + "-"*60)
    print("Parâmetros configurados. Iniciando o algoritmo...")
    print(f"Cidades: {NUM_CIDADES} | População: {TAM_POPULACAO} | Elite: {TAM_ELITE} | Mutação: {TAXA_MUTACAO}")

    if SEMENTE is not None:
        print(f"Usando semente aleatória fixa: {SEMENTE} (Resultados serão reprodutíveis)")
        random.seed(SEMENTE)         # Semente para a biblioteca 'random'
        np.random.seed(SEMENTE)  # Semente para a biblioteca 'numpy.random'
    else:
        print("Executando com semente aleatória (Resultados variarão a cada execução)")
    
    print("Pressione 'q' na janela da animação para fechar.")
    print("="*60 + "\n")
    

    # Gera coordenadas aleatórias para as cidades
    # Se a semente foi definida, np.random.rand() vai gerar o *mesmo* mapa sempre
    cidades = np.random.rand(NUM_CIDADES, 2) * 100

    # Configura os gráficos (plots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Problema do Caixeiro Viajante com Algoritmo Genético', fontsize=16)

    # Inicializa o gerador do GA
    ga_generator = algoritmo_genetico_generator(
        cidades=cidades,
        tam_populacao=TAM_POPULACAO,
        tam_elite=TAM_ELITE,
        taxa_mutacao=TAXA_MUTACAO
    )
    
    # Pega o primeiro estado (Geração 0)
    rota_inicial, dist_inicial, _ = next(ga_generator)

    # Plot 1: Rota Inicial
    ax1.set_title(f'Rota Inicial (Melhor da 1ª Geração)\nDistância: {dist_inicial:.2f}')
    ax1.plot(cidades[:, 0], cidades[:, 1], 'ro')
    
    x_coords_inicial = [cidades[i][0] for i in rota_inicial] + [cidades[rota_inicial[0]][0]]
    y_coords_inicial = [cidades[i][1] for i in rota_inicial] + [cidades[rota_inicial[0]][1]]
    ax1.plot(x_coords_inicial, y_coords_inicial, 'b-')

    # Plot 2: Animação
    ax2.set_title('Otimizando...')
    ax2.plot(cidades[:, 0], cidades[:, 1], 'ro')
    linha_rota, = ax2.plot([], [], 'g-') # Linha verde para o GA

    def atualizar_frame(frame_data):
        """Função chamada a cada novo 'yield' do gerador."""
        rota_atual, dist_atual, geracao = frame_data
        ax2.set_title(f'Otimizando (Algoritmo Genético)\nGeração: {geracao} | Melhor Dist.: {dist_atual:.2f}')
        
        x_coords = [cidades[i][0] for i in rota_atual] + [cidades[rota_atual[0]][0]]
        y_coords = [cidades[i][1] for i in rota_atual] + [cidades[rota_atual[0]][1]]
        linha_rota.set_data(x_coords, y_coords)
        
        return linha_rota,

    # Cria a animação
    ani = FuncAnimation(
        fig, 
        atualizar_frame, 
        frames=ga_generator, 
        interval=50, 
        blit=False, 
        repeat=False
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
