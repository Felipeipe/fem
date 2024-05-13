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
P = pd.DataFrame(columns = ['P1','P2','P3'])
Q = pd.DataFrame(columns = ['Q1','Q2','Q3'])

while i < NE:
    N1 : int = element_node_id.iloc[i][1]
    N2 : int = element_node_id.iloc[i][2]
    N3 : int = element_node_id.iloc[i][3]

    x1, y1 = [nodal_coordinates.iloc[N1-1][1],nodal_coordinates.iloc[N1-1][2]]
    x2, y2 = [nodal_coordinates.iloc[N2-1][1],nodal_coordinates.iloc[N2-1][2]]
    x3, y3 = [nodal_coordinates.iloc[N3-1][1],nodal_coordinates.iloc[N3-1][2]]

    P.loc[i] = [(y2 - y3), (y3 - y1), (y1 - y2)]
    Q.loc[i] = [(x3 - x2), (x1 - x3), (x2 - x1)]

    i += 1 
print(P)
print(Q)

