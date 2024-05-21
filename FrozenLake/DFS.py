def dfs(matrix, start, end):
    stack = [(start, [start])]
    visited = set()

    while stack:
        (vertex, path) = stack.pop()
        if vertex not in visited:
            if vertex == end:
                return path
            visited.add(vertex)
            for neighbor in get_neighbors(matrix, vertex):
                stack.append((neighbor, path + [neighbor]))
    return None

def get_neighbors(matrix, vertex):
    neighbors = []
    rows = len(matrix)
    cols = len(matrix[0])

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    for direction in directions:
        neighbor_row = vertex[0] + direction[0]
        neighbor_col = vertex[1] + direction[1]

        if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
            if matrix[neighbor_row][neighbor_col] != 'H':  # Se agregan solo si no es 'H'
                neighbors.append((neighbor_row, neighbor_col))

    return neighbors


def get_instructions (path:list[tuple[int,int]]):
    steps = []
    for i in range(1, len(path)):
        # Obtener el punto actual y el punto anterior
        punto_anterior = path[i-1]
        punto_actual = path[i]
        
        # Comparar las coordenadas para determinar la dirección del movimiento
        if punto_actual[0] < punto_anterior[0]:
            steps.append(3)
        elif punto_actual[0] > punto_anterior[0]:
            steps.append(1)
        elif punto_actual[1] < punto_anterior[1]:
            steps.append(0)
        elif punto_actual[1] > punto_anterior[1]:
            steps.append(2)
    return steps