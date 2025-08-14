# =============================================================================
# Gabriela Marculino -  RGM: 41431
# TRABALHO II de Inteligência Artificial
# Profº Drº Osvaldo Jacques
#
# Implementação do Algoritmo Simulated Annealing para resolver o
# Problema do Caixeiro Viajante (TSP).
# =============================================================================

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def calcular_distancia_total(cidades, rota):
   
    distancia_total = 0
    num_cidades = len(rota)

    # Itera por todas as cidades na ordem da rota
    for i in range(num_cidades):
        cidade_de_partida = rota[i]

        # Pega a próxima cidade na lista. O operador de módulo (%) garante
        # que, ao chegar no final da lista (i = num_cidades - 1),
        # a próxima cidade seja a primeira (índice 0).
        cidade_de_chegada = rota[(i + 1) % num_cidades]
        
        dist = np.linalg.norm(cidades[cidade_de_partida] - cidades[cidade_de_chegada])
        distancia_total += dist

    return distancia_total

#  Aplicação do Algoritmo Simulated Annealing 

def simulated_annealing_generator(cidades, temp_inicial, fator_resfriamento, temp_final_min):
    """
    Esta função implementa o Simulated Annealing como um gerador em Python.
    Usar 'yield' permite que a função "pause" seu estado a cada iteração e
    retorne os valores atuais, o que é perfeito para criar a animação.
    """
    num_cidades = len(cidades)

    # Passo 1: Começa com uma solução aleatória.
    # A rota é uma lista de índices embaralhada.
    rota_atual = list(range(num_cidades))
    random.shuffle(rota_atual)
    distancia_atual = calcular_distancia_total(cidades, rota_atual)

    # Armazena a melhor rota encontrada até o momento.
    melhor_rota = list(rota_atual)
    melhor_distancia = distancia_atual
    rota_inicial_const = list(rota_atual) 
    distancia_inicial_const = distancia_atual

    temperatura = temp_inicial

    # Estado inicial do loop.
    yield rota_inicial_const, distancia_inicial_const, rota_atual, distancia_atual, temperatura, melhor_distancia

    # Passo 2: Loop principal do algoritmo, que continua enquanto o sistema está "quente".
    while temperatura > temp_final_min:
        nova_rota = list(rota_atual)
        i, j = random.sample(range(num_cidades), 2)
        cidade_movida = nova_rota.pop(i)
        nova_rota.insert(j, cidade_movida)

        distancia_nova = calcular_distancia_total(cidades, nova_rota)

        # Passo 3: Decidir se aceita a nova solução.
        diferenca_distancia = distancia_nova - distancia_atual

        # 1. Se a nova solução é melhor (diferença < 0), aceita.
        # 2. Se for pior, aceita com uma probabilidade que depende da temperatura.
        if diferenca_distancia < 0 or random.random() < math.exp(-diferenca_distancia / temperatura):
            rota_atual = list(nova_rota)
            distancia_atual = distancia_nova

        # Atualiza com a melhor solução encontrada.
        if distancia_atual < melhor_distancia:
            melhor_rota = list(rota_atual)
            melhor_distancia = distancia_atual

        # Passo 4: Resfriar o sistema.
        temperatura *= fator_resfriamento

        yield rota_inicial_const, distancia_inicial_const, rota_atual, distancia_atual, temperatura, melhor_distancia

if __name__ == "__main__":

    NUM_CIDADES = 20
    TEMP_INICIAL = 1000.0  # Temperatura alta para explorar bastante no início.
    FATOR_RESFRIAMENTO = 0.99 # Resfriamento lento (valores > 0.9) para gerar possíveis bons resultados.
    TEMP_FINAL_MIN = 0.01   # Temperatura mínima para parar o algoritmo.

    # Gera coordenadas aleatórias para as cidades num plano de 100x100.
    cidades = np.random.rand(NUM_CIDADES, 2) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Problema do Caixeiro Viajante com Simulated Annealing', fontsize=16)

    sa_generator = simulated_annealing_generator(cidades, TEMP_INICIAL, FATOR_RESFRIAMENTO, TEMP_FINAL_MIN)
    rota_inicial, dist_inicial, *_ = next(sa_generator)

    ax1.set_title(f'Rota Inicial (Embaralhada)\nDistância: {dist_inicial:.2f}')
    ax1.plot(cidades[:, 0], cidades[:, 1], 'ro')
    

    # Para fechar o ciclo, o último ponto do gráfico deve ser igual ao primeiro.
    x_coords_inicial = [cidades[i][0] for i in rota_inicial] + [cidades[rota_inicial[0]][0]]
    y_coords_inicial = [cidades[i][1] for i in rota_inicial] + [cidades[rota_inicial[0]][1]]
    ax1.plot(x_coords_inicial, y_coords_inicial, 'b-') 

    ax2.set_title('Otimizando...')
    ax2.plot(cidades[:, 0], cidades[:, 1], 'ro')
    linha_rota, = ax2.plot([], [], 'purple') 

    def atualizar_frame(frame_data):
        
        _, _, rota_atual, dist_atual, temperatura, melhor_distancia = frame_data

        ax2.set_title(f'Otimizando (Simulated Annealing)\nTemp: {temperatura:.2f} | Melhor Dist.: {melhor_distancia:.2f}')
        
        x_coords = [cidades[i][0] for i in rota_atual] + [cidades[rota_atual[0]][0]]
        y_coords = [cidades[i][1] for i in rota_atual] + [cidades[rota_atual[0]][1]]
        linha_rota.set_data(x_coords, y_coords)
        
        return linha_rota,

    ani = FuncAnimation(fig, atualizar_frame, frames=sa_generator, interval=20, blit=False, repeat=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()