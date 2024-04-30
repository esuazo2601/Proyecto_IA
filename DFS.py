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
