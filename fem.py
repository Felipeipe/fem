import numpy as np
import pandas as pd

nodal_coordinates : pd.DataFrame = pd.read_excel("nodal_coordinates.xlsx")
element_node_id : pd.DataFrame = pd.read_excel("element_node_ID.xlsx")
fixed_node_potential: pd.DataFrame = pd.read_excel("Potential_at_fixed_nodes.xlsx")

ND : int = len(nodal_coordinates)    # number of total nodes of a mesh
NE : int = len(element_node_id)      # number of total elements (triangles) of a mesh
NF : int = len(fixed_node_potential) # number of fixed nodes (nodes where potential is known)


i : int = 0

C : np.ndarray = np.zeros((3,3)) 
P = pd.DataFrame(columns = ['P1', 'P2', 'P3'])
Q = pd.DataFrame(columns = ['Q1', 'Q2', 'Q3'])

while i < NE:
    # Se obtiene los nodos por cada elemento
    N1 : int = element_node_id.iloc[i, 1] 
    N2 : int = element_node_id.iloc[i, 2]
    N3 : int = element_node_id.iloc[i, 3]
    
    # Luego se obtiene la coordenada de cada nodo perteneciente al i-ésimo elemento
    x1, y1 = nodal_coordinates.iloc[N1-1, 1], nodal_coordinates.iloc[N1-1, 2]
    x2, y2 = nodal_coordinates.iloc[N2-1, 1], nodal_coordinates.iloc[N2-1, 2]
    x3, y3 = nodal_coordinates.iloc[N3-1, 1], nodal_coordinates.iloc[N3-1, 2]

    P.loc[i] = [(y2 - y3), (y3 - y1), (y1 - y2)]
    Q.loc[i] = [(x3 - x2), (x1 - x3), (x2 - x1)]

    i += 1 

def Ce(A : float, 
       num_element : int, 
       P : pd.DataFrame, 
       Q : pd.DataFrame
       )-> np.ndarray:
    
    i : int = 0
    j : int = 0
    Ce = np.zeros((3,3))
    Ce = np.zeros((3,3))
    for i in range(3):
        for j in range(i, 3):  # Iterating over half of the matrix
            Ce[i, j] = (P.iloc[num_element, i] * P.iloc[num_element, j] 
                        + Q.iloc[num_element, i] * Q.iloc[num_element, j]) / (4*A)
            
            Ce[j, i] = Ce[i, j]
    
    return Ce
print(NE)
i : int = 0
C_element : list[np.ndarray] = [] 
while i < NE:
    A : float = (P.iloc[i,1] * Q.iloc[i,2] - P.iloc[i,2] * Q.iloc[i,1]) / 2
    C_element.append(Ce(A, i, P, Q))
    i += 1

def ensamblar_matriz_global(matrices_elementos : list[np.ndarray],
                            df_nodos : pd.DataFrame,
                            N_nodos : int
                            )-> np.ndarray:
    # Obtener el número total de nodos
    n = N_nodos 

    # Inicializar la matriz global con ceros
    matriz_global = np.zeros((n, n))

    # Recorrer las matrices de elementos y ensamblar en la matriz global
    for matriz_elemento, idx_elemento in zip(matrices_elementos, df_nodos.index):
        nodos_elemento = df_nodos.loc[idx_elemento]
        # Ensamblar la matriz de elemento en la matriz global
        for i, ni in enumerate(nodos_elemento):
            for j, nj in enumerate(nodos_elemento):
                matriz_global[ni, nj] += matriz_elemento[i, j]

    return matriz_global

print(f"Matriz global de coeficientes:\n{ensamblar_matriz_global(C_element, element_node_id, ND)}")
