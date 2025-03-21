import random as rd  # Importa a biblioteca random para gerar números aleatórios
import math  # Importa a biblioteca math para operações matemáticas
import numpy as np  # Importa a biblioteca numpy para manipulação de arrays e cálculos
from collections import OrderedDict  # Importa OrderedDict para manter a ordem de inserção dos itens em um dicionário

class GeneticAlgorithmTSP:

    def __init__(self, graph, city_names, generations=20, population_size=10, tournament_size=4, mutationRate=0.1, fitness_selection_rate=0.1):
        """
        Inicializa o objeto GeneticAlgorithmTSP (Algoritmo Genético para TSP).

        Parâmetros:
        - graph: O objeto Graph que representa o grafo do TSP (Cidades e Conexões).
        - city_names: Lista com os nomes das cidades.
        - generations: Número de gerações a serem executadas no algoritmo.
        - population_size: Tamanho da população de rotas.
        - tournament_size: Número de rotas a serem selecionadas para crossover.
        - mutationRate: Probabilidade de mutação de um cromossomo.
        - fitness_selection_rate: Percentual de rotas mais aptas para serem mantidas para a próxima geração.
        """
        self.graph = graph  # Grafo do TSP
        self.population_size = population_size  # Tamanho da população
        self.generations = generations  # Número de gerações
        self.tournament_size = tournament_size  # Tamanho do torneio para seleção de pais
        self.mutationRate = mutationRate  # Taxa de mutação
        self.fitness_selection_rate = fitness_selection_rate  # Taxa de seleção dos indivíduos mais aptos

        # Lista para armazenar os valores de diversidade genética
        self.genetic_diversity_values = []

        # Mapeamento entre as cidades e caracteres
        self.city_map = OrderedDict((char, city) for char, city in zip(range(32, 127), graph.vertices()))
        self.city_mapping = {char: city for char, city in zip(range(32, 127), city_names)}
        self.city_map[32], self.city_map[33] = self.city_map[33], self.city_map[32]  # Troca as posições das cidades 32 e 33 no mapeamento

    def calculate_genetic_diversity(self, population):
        """
        Calcula a diversidade genética dentro da população.

        Parâmetros:
        - population: Lista de rotas na população atual.

        Retorna:
        - A diversidade genética como um valor float.
        """
        # Converte as rotas para uma matriz para facilitar o cálculo das distâncias
        routes_matrix = np.array([list(route) for route in population])

        # Calcula as distâncias genéticas entre as rotas
        pairwise_distances = np.sum(routes_matrix[:, None, :] != routes_matrix[None, :, :], axis=2)

        # Calcula a distância genética média
        total_distance = np.sum(pairwise_distances)
        num_pairs = len(population) * (len(population) - 1)
        average_distance = total_distance / num_pairs

        # A diversidade genética é o inverso da distância média
        genetic_diversity = 1 / (1 + average_distance)

        return genetic_diversity

    def get_genetic_diversity_values(self):
        """
        Obtém os valores de diversidade genética para cada geração.

        Retorna:
        - Lista com os valores de diversidade genética.
        """
        return self.genetic_diversity_values

    def minCostIndex(self, costs):
        """
        Retorna o índice do menor custo na lista de custos.

        Parâmetros:
        - costs: Lista de custos das rotas.

        Retorna:
        - O índice da rota com o menor custo.
        """
        return min(range(len(costs)), key=costs.__getitem__)

    def find_fittest_path(self, graph):
        """
        Encontra o caminho mais apto através do grafo TSP usando o algoritmo genético.

        Parâmetros:
        - graph: O objeto Graph que representa o grafo TSP.

        Retorna:
        - fittest_route: A lista de cidades que representa o caminho mais apto.
        - fittest_fitness: O valor de fitness (custo mínimo) do caminho mais apto.
        """
        population = self.randomizeCities(graph.vertices())  # Gera uma população inicial de rotas aleatórias
        number_of_fits_to_carryover = math.ceil(self.population_size * self.fitness_selection_rate)  # Número de rotas mais aptas a serem mantidas

        if number_of_fits_to_carryover > self.population_size:
            raise ValueError('Fitness rate must be in [0, 1].')  # Verifica se a taxa de fitness está entre 0 e 1

        print('Optimizing TSP Route for Graph:')

        for generation in range(1, self.generations + 1):
            new_population = self.create_next_generation(graph, population, number_of_fits_to_carryover)  # Cria a próxima geração de rotas
            population = new_population

            fittest_index, fittest_route, fittest_fitness = self.get_fittest_route(graph, population)  # Obtém a rota mais apta
            fittest_route = [list(OrderedDict(self.city_mapping).values())[list(OrderedDict(self.city_map).values()).index(char)] if char in list(OrderedDict(self.city_map).values()) else char for char in fittest_route]

            genetic_diversity = self.calculate_genetic_diversity(population)  # Calcula a diversidade genética da população
            self.genetic_diversity_values.append(round(genetic_diversity, 4))  # Adiciona a diversidade genética à lista

            if self.converged(population):  # Verifica se a população convergiu
                print("Converged", population)
                print('\nConverged to a local minima.')
                break

        return fittest_route, fittest_fitness

    def create_next_generation(self, graph, population, number_of_fits_to_carryover):
        """
        Cria a próxima geração de rotas com base na população atual.

        Parâmetros:
        - graph: O objeto Graph que representa o grafo TSP.
        - population: Lista de rotas da população atual.
        - number_of_fits_to_carryover: Número de rotas mais aptas a serem mantidas para a próxima geração.

        Retorna:
        - new_population: Lista de rotas da nova geração.
        """
        new_population = self.add_fittest_routes(graph, population, number_of_fits_to_carryover)  # Adiciona as rotas mais aptas à nova população
        new_population += [self.mutate(self.crossover(*self.select_parents(graph, population))) for _ in range(self.population_size - number_of_fits_to_carryover)]  # Aplica mutação nas novas rotas

        return new_population

    def add_fittest_routes(self, graph, population, number_of_fits_to_carryover):
        """
        Adiciona as rotas mais aptas à próxima geração.

        Parâmetros:
        - graph: O objeto Graph que representa o grafo TSP.
        - population: Lista de rotas da população atual.
        - number_of_fits_to_carryover: Número de rotas mais aptas a serem mantidas.

        Retorna:
        - sorted_population: Lista de rotas ordenadas por fitness (custo).
        """
        sorted_population = [x for _, x in sorted(zip(self.computeFitness(graph, population), population))]  # Ordena as rotas pela aptidão
        return sorted_population[:number_of_fits_to_carryover]

    def get_fittest_route(self, graph, population):
        """
        Obtém a rota mais apta da população.

        Parâmetros:
        - graph: O objeto Graph que representa o grafo TSP.
        - population: Lista de rotas da população.

        Retorna:
        - fittest_index: Índice da rota mais apta na população.
        - fittest_route: A rota mais apta.
        - fittest_fitness: O valor de fitness (custo) da rota mais apta.
        """
        fitness = self.computeFitness(graph, population)  # Calcula a aptidão (custo) de cada rota
        fittest_index = self.minCostIndex(fitness)  # Encontra o índice da rota com o menor custo
        return fittest_index, population[fittest_index], fitness[fittest_index]

    def select_parents(self, graph, population):
        """
        Seleciona dois pais para crossover usando seleção por torneio.

        Parâmetros:
        - graph: O objeto Graph que representa o grafo TSP.
        - population: Lista de rotas da população.

        Retorna:
        - parent1: O primeiro pai selecionado.
        - parent2: O segundo pai selecionado.
        """
        return self.tournamentSelection(graph, population), self.tournamentSelection(graph, population)

    def randomizeCities(self, graph_nodes):
        """
        Gera rotas aleatórias para a população inicial.

        Parâmetros:
        - graph_nodes: Lista de nós do grafo TSP (cidades).

        Retorna:
        - Lista de rotas aleatórias.
        """
        nodes = [node for node in graph_nodes if node != self.graph.start_city]  # Remove a cidade inicial

        return [
            self.graph.start_city + ''.join(rd.sample(nodes, len(nodes))) + self.graph.start_city
            for _ in range(self.population_size)  # Gera rotas aleatórias
        ]

    def computeFitness(self, graph, population):
        """
        Calcula a aptidão (custo) de cada rota na população.

        Parâmetros:
        - graph: O objeto Graph que representa o grafo TSP.
        - population: Lista de rotas.

        Retorna:
        - Lista de valores de fitness (custos).
        """
        return [graph.getPathCost(path) for path in population]

    def tournamentSelection(self, graph, population):
        """
        Realiza a seleção por torneio para escolher um pai para crossover.

        Parâmetros:
        - graph: O objeto Graph que representa o grafo TSP.
        - population: Lista de rotas na população.

        Retorna:
        - O pai selecionado para o crossover.
        """
        tournament_contestants = rd.choices(population, k=self.tournament_size)  # Seleciona candidatos aleatórios para o torneio
        return min(tournament_contestants, key=lambda path: graph.getPathCost(path))  # Retorna o vencedor (rota de menor custo)

    def crossover(self, parent1, parent2):
        """
        Realiza o crossover entre dois pais para gerar um filho.

        Parâmetros:
        - parent1: Primeiro pai.
        - parent2: Segundo pai.

        Retorna:
        - O filho gerado a partir do crossover.
        """
        offspring_length = len(parent1) - 2  # Exclui as cidades inicial e final

        offspring = ['' for _ in range(offspring_length)]

        index_low, index_high = self.computeTwoPointIndexes(parent1)  # Gera os pontos de crossover

        # Copia os genes do pai 1 para o filho
        offspring[index_low: index_high + 1] = list(parent1)[index_low: index_high + 1]

        # Copia os genes restantes do pai 2 para o filho
        empty_place_indexes = [i for i in range(offspring_length) if offspring[i] == '']
        for i in parent2[1: -1]:  # Exclui as cidades inicial e final
            if '' not in offspring or not empty_place_indexes:
                break
            if i not in offspring:
                offspring[empty_place_indexes.pop(0)] = i

        offspring = [self.graph.start_city] + offspring + [self.graph.start_city]  # Reinsere a cidade inicial e final
        return ''.join(offspring)

    def mutate(self, genome):
        """
        Aplica uma mutação no cromossomo com a probabilidade especificada por mutationRate.

        Parâmetros:
        - genome: O cromossomo a ser mutado.

        Retorna:
        - O cromossomo mutado.
        """
        if rd.random() < self.mutationRate:
            index_low, index_high = self.computeTwoPointIndexes(genome)  # Gera os pontos de mutação
            return self.swap(index_low, index_high, genome)  # Aplica a troca nos pontos gerados
        else:
            return genome  # Se não houver mutação, retorna o cromossomo original

    def computeTwoPointIndexes(self, parent):
        """
        Gera dois pontos de crossover para o processo de mutação.

        Parâmetros:
        - parent: O cromossomo (rota) de onde serão gerados os pontos.

        Retorna:
        - Um tupla com os dois pontos de mutação (index_low, index_high).
        """
        index_low = rd.randint(1, len(parent) - 3)  # Gera um índice baixo
        index_high = rd.randint(index_low+1, len(parent) - 2)  # Gera um índice alto

        # Verifica se a diferença entre os índices é menor que metade do comprimento do pai
        if index_high - index_low > math.ceil(len(parent) // 2):
            return self.computeTwoPointIndexes(parent)
        else:
            return index_low, index_high

    def swap(self, index_low, index_high, string):
        """
        Troca dois elementos em uma string.

        Parâmetros:
        - index_low: Índice do primeiro elemento a ser trocado.
        - index_high: Índice do segundo elemento a ser trocado.
        - string: A string onde os elementos serão trocados.

        Retorna:
        - A string com os elementos trocados.
        """
        string = list(string)
        string[index_low], string[index_high] = string[index_high], string[index_low]  # Realiza a troca
        return ''.join(string)

    def converged(self, population):
        """
        Verifica se todos os cromossomos na população são iguais.

        Parâmetros:
        - population: Lista de cromossomos.

        Retorna:
        - True se todos os cromossomos forem iguais, False caso contrário.
        """
        return all(genome == population[0] for genome in population)  # Verifica se todos os cromossomos são iguais
