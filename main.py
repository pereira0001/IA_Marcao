import re  # Importa o módulo para expressões regulares
from graph import Graph  # Importa a classe Graph do módulo graph
from genetic_algorithm_tsp import GeneticAlgorithmTSP  # Importa a classe GeneticAlgorithmTSP do módulo genetic_algorithm_tsp
from plot import plot_tsp_path, plot_genetic_diversity  # Importa as funções plot_tsp_path e plot_genetic_diversity do módulo plot
from collections import OrderedDict  # Importa OrderedDict, uma estrutura de dados que mantém a ordem de inserção
import time  # Importa o módulo time para manipulação de tempo

inicio = time.time()  # Marca o tempo inicial do programa (não usado, mas estava presente)

def main():
    # Marca o tempo inicial da execução
    start_time = time.time()  # Marca o momento em que o código começa a rodar

    # Lê os dados do arquivo 'cities.txt'
    with open('cities.txt', 'r') as file:
        cities_data = file.read()  # Lê todo o conteúdo do arquivo

    # Expressão regular para extrair o nome da cidade e suas coordenadas (x, y)
    city_pattern = re.compile(r'(\w+):\s\((\d+),\s(\d+)\)')

    # Extrai as cidades e suas coordenadas (nome, x, y) usando a expressão regular
    cities = city_pattern.findall(cities_data)

    # Verifica se o número de cidades excede o limite de 94
    if len(cities) > 94:
        raise ValueError("Cannot accept more cities.")  # Lança um erro se houver mais de 94 cidades

    # Cria um mapeamento de nomes de cidades para caracteres usando OrderedDict
    # A chave é o nome da cidade, o valor é um caractere a partir de 33 no código ASCII
    city_mapping = OrderedDict((city[0], chr(i + 33)) for i, city in enumerate(cities))

    # Cria uma instância da classe Graph, representando o grafo das cidades
    germany_graph = Graph(len(cities), False)

    # Adiciona cada cidade ao grafo com seu nome (mapeado para um caractere) e coordenadas (x, y)
    for city, x, y in cities:
        germany_graph.add_node(city_mapping[city], int(x), int(y))

    # Define a cidade inicial para o algoritmo TSP
    germany_graph.start_city = city_mapping['Academic_aeroport']

    # Cria uma instância da classe GeneticAlgorithmTSP com os parâmetros definidos
    ga_tsp_germany = GeneticAlgorithmTSP(
        graph=germany_graph,  # Passa o grafo das cidades
        city_names=[city for city, _, _ in cities],  # Lista de nomes das cidades
        generations=30,  # Número de gerações do algoritmo genético
        population_size=10,  # Tamanho da população na geração
        tournament_size=5,  # Tamanho do torneio para seleção
        mutationRate=0.1,  # Taxa de mutação
        fitness_selection_rate=0.5,  # Taxa de seleção por fitness
    )

    # Encontra o caminho mais apto usando o algoritmo genético
    fittest_path, path_cost = ga_tsp_germany.find_fittest_path(germany_graph)

    # Obtém os valores de diversidade genética de cada geração
    genetic_diversity_values = ga_tsp_germany.get_genetic_diversity_values()

    # Exibe o caminho mais apto e seu custo
    formatted_path = ' -> '.join(fittest_path)  # Formata o caminho para exibição
    print('\nPath: {0}\nCost: {1}'.format(formatted_path, path_cost))  # Exibe o caminho e o custo

    # Cria um dicionário com as coordenadas das cidades
    coordinates_dict = {city: (int(x), int(y)) for city, x, y in cities}

    # Cria uma lista de coordenadas do caminho mais apto
    coordinates_list = [coordinates_dict[city] for city in fittest_path]

    # Marca o tempo final e calcula a duração da execução
    end_time = time.time()  # Marca o tempo de término
    execution_time = end_time - start_time  # Calcula o tempo total de execução
    print(f"\nTempo de execução: {execution_time:.4f} segundos")  # Exibe o tempo de execução

    # Plota o caminho do TSP, incluindo os nós, as arestas e a anotação do custo
    plot_tsp_path(fittest_path, coordinates_list, 'Ucrania.jpg', path_cost)

    # Plota a diversidade genética ao longo das gerações
    plot_genetic_diversity(genetic_diversity_values)


if __name__ == "__main__":
    main()  # Executa a função main quando o script é executado
