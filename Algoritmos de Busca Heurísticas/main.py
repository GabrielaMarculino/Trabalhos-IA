from queue import PriorityQueue, LifoQueue, Queue

# Trabalho I - Inteligência Artificial
# Gabriela Marculino - RGM: 41431
# txt baseado no mapa da Romênia passado em sala de aula

class Grafo:
    def __init__(self):
        self.grafo = {}
        self.heuristica = {}

    def adicionar_aresta(self, origem, destino, custo):
        origem = origem.lower()
        destino = destino.lower()
        if origem not in self.grafo:
            self.grafo[origem] = {}
        if destino not in self.grafo:
            self.grafo[destino] = {}
        self.grafo[origem][destino] = custo
        self.grafo[destino][origem] = custo  # bidirecional

    def carregar_de_arquivo(self, nome_arquivo):
        secao = 'arestas'
        try:
            with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
                for linha in arquivo:
                    linha = linha.strip()
                    if not linha:
                        continue

                    if linha.lower() == 'heuristics':
                        secao = 'heuristica'
                        continue

                    if secao == 'arestas':
                        # Formato: cidade vizinho1:custo vizinho2:custo ...
                        partes = linha.split()
                        cidade = partes[0].lower()
                        if cidade not in self.grafo:
                            self.grafo[cidade] = {}
                        for viz_custo in partes[1:]:
                            try:
                                vizinho, custo = viz_custo.split(':')
                                vizinho = vizinho.lower()
                                custo = int(custo)
                            except ValueError:
                                print(f"Aviso: formato inválido na linha: {linha}")
                                continue
                            self.adicionar_aresta(cidade, vizinho, custo)
                    elif secao == 'heuristica':
                        partes = linha.split()
                        if len(partes) == 2:
                            cidade, valor = partes
                            self.heuristica[cidade.lower()] = int(valor)
        except FileNotFoundError:
            print(f"Arquivo '{nome_arquivo}' não encontrado.")
            exit(1)

    def vizinhos(self, cidade):
        return self.grafo.get(cidade.lower(), {})


def calcular_custo(grafo, caminho):
    custo = 0
    for i in range(len(caminho) - 1):
        origem = caminho[i].lower()
        destino = caminho[i + 1].lower()
        custo += grafo.grafo[origem][destino]
    return custo


def busca_largura(grafo, inicio, objetivo):
    inicio = inicio.lower()
    objetivo = objetivo.lower()

    fila = Queue()
    fila.put((inicio, [inicio]))

    while not fila.empty():
        atual, caminho = fila.get()
        if atual == objetivo:
            custo = calcular_custo(grafo, caminho)
            return caminho, custo

        for vizinho in grafo.vizinhos(atual):
            if vizinho not in caminho:
                fila.put((vizinho, caminho + [vizinho]))
    return None


def busca_profundidade(grafo, inicio, objetivo):
    inicio = inicio.lower()
    objetivo = objetivo.lower()

    pilha = LifoQueue()
    pilha.put((inicio, [inicio]))

    while not pilha.empty():
        atual, caminho = pilha.get()
        if atual == objetivo:
            custo = calcular_custo(grafo, caminho)
            return caminho, custo

        for vizinho in grafo.vizinhos(atual):
            if vizinho not in caminho:
                pilha.put((vizinho, caminho + [vizinho]))
    return None


def busca_profundidade_limitada(grafo, inicio, objetivo, limite=3):
    inicio = inicio.lower()
    objetivo = objetivo.lower()

    def recursao(atual, caminho, profundidade):
        if profundidade > limite:
            return None
        if atual == objetivo:
            custo = calcular_custo(grafo, caminho)
            return caminho, custo
        for vizinho in grafo.vizinhos(atual):
            if vizinho not in caminho:
                resultado = recursao(vizinho, caminho + [vizinho], profundidade + 1)
                if resultado:
                    return resultado
        return None

    return recursao(inicio, [inicio], 0)


def busca_custo_uniforme(grafo, inicio, objetivo):
    inicio = inicio.lower()
    objetivo = objetivo.lower()

    fila = PriorityQueue()
    fila.put((0, inicio, [inicio]))

    while not fila.empty():
        custo, atual, caminho = fila.get()

        if atual == objetivo:
            return caminho, custo

        for vizinho, custo_vizinho in grafo.vizinhos(atual).items():
            if vizinho not in caminho:
                fila.put((custo + custo_vizinho, vizinho, caminho + [vizinho]))
    return None


def busca_gulosa(grafo, inicio, objetivo):
    # Objetivo fixo em 'bucharest'
    inicio = inicio.lower()
    objetivo = 'bucharest'

    fila = PriorityQueue()
    fila.put((grafo.heuristica.get(inicio, float('inf')), inicio, [inicio]))

    while not fila.empty():
        _, atual, caminho = fila.get()

        if atual == objetivo:
            custo = calcular_custo(grafo, caminho)
            return caminho, custo

        for vizinho in grafo.vizinhos(atual):
            if vizinho not in caminho:
                heuristica = grafo.heuristica.get(vizinho, float('inf'))
                fila.put((heuristica, vizinho, caminho + [vizinho]))
    return None


def busca_a_estrela(grafo, inicio, objetivo):
    # Objetivo fixo em 'bucharest'
    inicio = inicio.lower()
    objetivo = 'bucharest'

    fila = PriorityQueue()
    fila.put((grafo.heuristica.get(inicio, float('inf')), 0, inicio, [inicio]))

    while not fila.empty():
        prioridade, custo, atual, caminho = fila.get()

        if atual == objetivo:
            return caminho, custo

        for vizinho, custo_vizinho in grafo.vizinhos(atual).items():
            if vizinho not in caminho:
                novo_custo = custo + custo_vizinho
                heuristica = grafo.heuristica.get(vizinho, float('inf'))
                prioridade = novo_custo + heuristica
                fila.put((prioridade, novo_custo, vizinho, caminho + [vizinho]))
    return None


def busca_ida(grafo, inicio, objetivo):
    # Objetivo fixo em 'bucharest'
    inicio = inicio.lower()
    objetivo = 'bucharest'

    def dfs_f_limitado(atual, caminho, g, limite):
        f = g + grafo.heuristica.get(atual, float('inf'))
        if f > limite:
            return None, f
        if atual == objetivo:
            return caminho, g

        minimo = float('inf')
        for vizinho, custo_vizinho in grafo.vizinhos(atual).items():
            if vizinho not in caminho:
                resultado, temp = dfs_f_limitado(vizinho, caminho + [vizinho], g + custo_vizinho, limite)
                if isinstance(resultado, list):
                    return resultado, temp
                if temp < minimo:
                    minimo = temp
        return None, minimo

    limite = grafo.heuristica.get(inicio, 0)
    while True:
        resultado, temp = dfs_f_limitado(inicio, [inicio], 0, limite)
        if isinstance(resultado, list):
            return resultado, temp
        if temp == float('inf'):
            return None
        limite = temp


def main():
    grafo = Grafo()
    grafo.carregar_de_arquivo('romenia.txt')

    print("\n=== Algoritmos de Busca na Romênia ===")

    while True:
        inicio = input("Digite a cidade de origem: ").strip().lower()
        if inicio in grafo.grafo:
            break
        print("Cidade de origem não encontrada no grafo. Tente novamente.")

    while True:
        objetivo = input("Digite a cidade de destino: ").strip().lower()
        if objetivo in grafo.grafo:
            break
        print("Cidade de destino não encontrada no grafo. Tente novamente.")

    print("\n\n Gabriela Marculino - RGM: 41431")
    print("\n\n Escolha o algoritmo:")
    
    print("\n\n BUSCA INFORMADA")
    print("1 - Busca em Largura")
    print("2 - Busca em Profundidade")
    print("3 - Busca em Profundidade Limitada")
    print("4 - Busca de Custo Uniforme")
    
    print("\n\n BUSCA NÃO INFORMADA")
    print("5 - Busca Gulosa (destino fixo em Bucharest)")
    print("6 - Busca A* (destino fixo em Bucharest)")
    print("7 - Busca IDA* (destino fixo em Bucharest)")

    opcao = input("Digite sua opção: ").strip()

    if opcao == '1':
        resultado = busca_largura(grafo, inicio, objetivo)
    elif opcao == '2':
        resultado = busca_profundidade(grafo, inicio, objetivo)
    elif opcao == '3':
        while True:
            try:
                limite = int(input("Digite o limite de profundidade: "))
                break
            except ValueError:
                print("Por favor, digite um número inteiro válido para o limite.")
        resultado = busca_profundidade_limitada(grafo, inicio, objetivo, limite)
    elif opcao == '4':
        resultado = busca_custo_uniforme(grafo, inicio, objetivo)
    elif opcao == '5':
        resultado = busca_gulosa(grafo, inicio, 'bucharest')
    elif opcao == '6':
        resultado = busca_a_estrela(grafo, inicio, 'bucharest')
    elif opcao == '7':
        resultado = busca_ida(grafo, inicio, 'bucharest')
    else:
        print("Opção inválida")
        return

    if resultado:
        caminho, custo = resultado
        caminho_formatado = ' -> '.join([cidade.title() for cidade in caminho])
        print(f"\n>>> Caminho encontrado: {caminho_formatado}")
        print(f">>> Custo total: {custo} KM")
    else:
        print("Caminho não encontrado.")


if __name__ == "__main__":
    main()
