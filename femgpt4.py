import numpy as np
import pandas as pd

def main():
    # Load data from Excel files
    coordinates = pd.read_excel('nodal_coordinates.xlsx', header=None).values
    elements = pd.read_excel('element_node_ID.xlsx', header=None).values.astype(int) - 1
    fixed_nodes = pd.read_excel('Potential_at_fixed_nodes.xlsx', header=None).values

    num_nodes = coordinates.shape[0]
    num_elements = elements.shape[0]

    # Assembly of the global stiffness matrix
    A = np.zeros((num_nodes, num_nodes))
    for element in elements:
        local_stiffness = local_stiffness_matrix(coordinates[element])
        for i in range(3):
            for j in range(3):
                A[element[i], element[j]] += local_stiffness[i, j]

    # Apply boundary conditions
    b = np.zeros(num_nodes)
    for fixed_node in fixed_nodes:
        node_index = int(fixed_node[0]) - 1  # Adjusting index to be zero-based
        node_value = fixed_node[1]
        A[node_index, :] = 0
        A[:, node_index] = 0
        A[node_index, node_index] = 1
        b[node_index] = node_value

    # Solve the linear system
    potentials = np.linalg.solve(A, b)
    print("Node Potentials:")
    for i, p in enumerate(potentials):
        print(f"Node {i+1}: {p:.2f}")

def local_stiffness_matrix(coords):
    # Compute the local stiffness matrix for triangular elements
    area = 0.5 * np.abs(np.linalg.det(np.array([
        [1, coords[0][0], coords[0][1]],
        [1, coords[1][0], coords[1][1]],
        [1, coords[2][0], coords[2][1]]
    ])))
    k = (1/(4*area))
    return np.array([
        [2*k, -1*k, -1*k],
        [-1*k, 2*k, -1*k],
        [-1*k, -1*k, 2*k]
    ])

if __name__ == "__main__":
    main()