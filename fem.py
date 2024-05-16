import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nodal_coordinates : pd.DataFrame = pd.read_excel("nodal_coordinates.xlsx")
element_node_id : pd.DataFrame = pd.read_excel("element_node_ID.xlsx")
fixed_node_potential: pd.DataFrame = pd.read_excel("Potential_at_fixed_nodes.xlsx")

print(element_node_id)

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



i : int = 0
C_element : list[np.ndarray] = [] 

while i < NE:
    A : float = (P.iloc[i,1] * Q.iloc[i,2] - P.iloc[i,2] * Q.iloc[i,1]) / 2
    C_element.append(Ce(A, i, P, Q))
    i += 1


def row_to_list(node_id : pd.DataFrame,
        element_number : int) -> list[int]:
    """ Transforma la fila numero element_number del dataframe node_id a una lista.
    """
    node_loc : list[int] = list(node_id.iloc[element_number])
    return node_loc[1:]


C : np.ndarray = np.array([[1,  0     , 0,  0     ],
                           [0,  1.25  , 0, -0.0143],
                           [0,  0     , 1,  0     ],
                           [0, -0.0143, 0,  0.8381]])

B : np.array = np.array([0, 4.571, 10, 3.6667])


V = np.dot(np.linalg.inv(C),B)

L = row_to_list(element_node_id, 0)

Ve1 : list = []
for x in L:
    Ve1.append(V[x - 1]) 

print(Ve1)
L = row_to_list(element_node_id, 1)

Ve2 : list = []
for x in L:
    Ve2.append(V[x - 1]) 

print(Ve2)


Me1 : np.ndarray = np.array([[1, 0.8, 1.8],
                             [1, 1.4, 1.4],
                             [1, 1.2, 2.7]])

Me2 : np.ndarray = np.array([[1, 1.4, 1.4],
                             [1, 2.1, 2.1],
                             [1, 1.2, 2.7]]) 


coef_elemento_1 : np.array = np.dot(np.linalg.inv(Me1),Ve1)
coef_elemento_2 : np.array = np.dot(np.linalg.inv(Me2),Ve2)
coef_elemento_3 = (coef_elemento_1 + coef_elemento_2) / 2

def V_elemento(x : float,
                 y : float,
                 param : np.array
                 ) -> float:
    a, b, c = param
    return a + b * x + c * y

# Generar datos para graficar
x = np.linspace(0.5, 2.5, 100)
y = np.linspace(1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = V_elemento(X, Y,coef_elemento_3)

# Graficar la función con un mapa de calor
plt.figure()
heatmap = plt.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='viridis')
plt.colorbar(heatmap)  # Añadir barra de color

# Graficar el triángulo
element1_vertices = np.array([[0.8, 1.8], [1.4, 1.4], [1.2, 2.7], [0.8, 1.8]])
element2_vertices = np.array([[1.4, 1.4], [2.1, 2.1], [1.2, 2.7], [1.4, 1.4]])

plt.plot(element1_vertices[:, 0], element1_vertices[:, 1], 'r-')
plt.plot(element2_vertices[:, 0], element2_vertices[:, 1], 'r-')

# Configuraciones adicionales (opcional)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Potencial en función de la posición')

# Mostrar la gráfica
plt.show()
