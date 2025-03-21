import matplotlib.pyplot as plt  # Importa o módulo pyplot do matplotlib para criação de gráficos
import matplotlib.image as mpimg  # Importa o módulo image do matplotlib para ler imagens

def plot_tsp_path(path, coordinates, image_path, cost):
    """
    Função para plotar o caminho do problema do caixeiro viajante (TSP) em um mapa.
    
    Parâmetros:
    - path: lista de cidades que representam o caminho mais eficiente (ordem das cidades).
    - coordinates: lista de tuplas com as coordenadas (x, y) de cada cidade no caminho.
    - image_path: caminho para a imagem do mapa onde as cidades serão plotadas.
    - cost: custo total do caminho (distância total percorrida).
    """
    
    # Lê a imagem do mapa a partir do caminho especificado
    img = mpimg.imread(image_path)
    
    # Cria uma figura e um eixo para o gráfico
    fig, ax = plt.subplots()

    # Plota a imagem do mapa no fundo
    ax.imshow(img)

    # Plota os nós (as cidades) como pontos azuis no gráfico
    for city, (x, y) in zip(path, coordinates):
        ax.plot(x, y, 'bo')  # 'bo' significa 'blue o', ou seja, pontos azuis

    # Plota as arestas (linhas) conectando os nós (cidades)
    for i in range(len(path) - 1):
        city1 = path[i]
        city2 = path[i + 1]
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]
        ax.plot([x1, x2], [y1, y2], 'r-')  # 'r-' significa linha vermelha

    # Conecta a última cidade com a primeira para formar um ciclo completo
    x1, y1 = coordinates[-1]
    x2, y2 = coordinates[0]
    ax.plot([x1, x2], [y1, y2], 'r-')  # Linha vermelha conectando a última e a primeira cidade

    # Anota o custo total do caminho no gráfico
    cost_text = 'Total Cost: {:.2f}'.format(cost)  # Formata o custo para mostrar com 2 casas decimais
    ax.text(0.5, -0.1, cost_text, ha='center', va='center', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='white', boxstyle='round4'))  # Exibe o texto com fundo branco e bordas arredondadas

    # Título do gráfico
    plt.title('TSP Path')
    
    # Exibe o gráfico
    plt.show()


def plot_genetic_diversity(genetic_diversity_values):
    """
    Função para plotar a diversidade genética ao longo das gerações do algoritmo genético.
    
    Parâmetros:
    - genetic_diversity_values: lista com os valores de diversidade genética em cada geração.
    """
    
    # Gera uma sequência de gerações (1, 2, 3, ..., n)
    generations = range(1, len(genetic_diversity_values) + 1)
    
    # Plota a diversidade genética ao longo das gerações
    plt.plot(generations, genetic_diversity_values, marker='o')  # 'o' significa marcar os pontos com círculos
    
    # Título do gráfico
    plt.title('Genetic Diversity Over Generations')
    
    # Rótulos dos eixos
    plt.xlabel('Generation')
    plt.ylabel('Genetic Diversity')
    
    # Define os valores de 'x' (gerações) como rótulos do eixo x
    plt.xticks(generations)
    
    # Exibe o gráfico
    plt.show()
