class Celda:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.h = 0
        self.g = 0
        self.padre = (0,0)

def A_star(matrix, start, end):
    porVisitar = []
    visitados = []

    rows = len(matrix)
    cols = len(matrix[0])
    
    grilla = [[Celda() for x in range(rows)] for y in range(cols)]
    
    for i, row in enumerate(matrix):
        for j, cell in enumerate(row):
            
            grilla[i][j].x = i
            grilla[i][j].y = j
            
            grilla[i][j].h = abs(i - end[0]) + abs(j - end[1])

    porVisitar.append(grilla[start[0]][start[1]])
    while len(porVisitar) > 0:
        currentCell = porVisitar[0]
        
        for celda in porVisitar:
            minF = currentCell.g + currentCell.h
            f = celda.g + celda.h
            if f < minF or (f == minF and celda.h < currentCell.h):
                currentCell = celda
        
        porVisitar.remove(currentCell)
        visitados.append(currentCell)
        
        if currentCell.x == end[0] and currentCell.y == end[1]:
            break
        
        vecinos = get_neighbors(matrix, grilla, currentCell)
        for vecino in vecinos:
            if visitados.__contains__(vecino):
                continue
            
            costeMovimiento = currentCell.g + 1
            if costeMovimiento < vecino.g or not porVisitar.__contains__(vecino):
                vecino.g = costeMovimiento
                vecino.padre = (currentCell.x, currentCell.y)
                
                if not porVisitar.__contains__(vecino):
                    porVisitar.append(vecino)
                    
    path = []
    aux = grilla[end[0]][end[1]]
    while True:
        path.append((aux.x, aux.y))
        aux = grilla[aux.padre[0]][aux.padre[1]]
        
        if aux.x == start[0] and aux.y == start[1]:
            path.append((aux.x, aux.y))
            break
    
    path.reverse()
    
    return path

def get_neighbors(matrix, grilla, celda):
    neighbors = []
    rows = len(matrix)
    cols = len(matrix[0])

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    for direction in directions:
        neighbor_row = celda.x + direction[0]
        neighbor_col = celda.y + direction[1]

        if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
            if matrix[neighbor_row][neighbor_col] != 'H':  # Se agregan solo si no es 'H'
                neighbors.append(grilla[neighbor_row][neighbor_col])

    return neighbors