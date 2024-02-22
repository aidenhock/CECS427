"""
Assignment Social and Large-Scale Networks
"""
import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import dwave_networkx as dnx
from neal import SimulatedAnnealingSampler

def read_graph(file_name):
    """
    Reads a graph from a file with adjacency list format. Nodes are expected to be integers.
    """
    try:
        with open(file_name, "r") as f:
            G = nx.parse_adjlist(f, nodetype=int)
            return G
    except FileNotFoundError:
        print("File not found. Please check the file name and try again.")
        return None
    except Exception as e:
        print(f"Unexpected error reading the graph: {e}")
        return None


def write_graph(G, file_name):
    """
    Writes the graph to a file using the adjacency list format.
    """
    try:
        with open(file_name, "w") as f:
            for line in nx.generate_adjlist(G):
                f.write(line + "\n")
    except Exception as e:
        print(f"An error occurred while writing to file: {e}")


def create_random_erdos_reny_graph(n, c):
    """
    Creates an Erdős-Rényi graph with n nodes and probability p calculated using the provided c parameter.
    """
    if n < 1:
        print("Number of nodes must be a positive integer.")
        return None
    p = (c * math.log(n)) / n if n > 1 else 0
    G = nx.erdos_renyi_graph(n, p)
    return G


def create_karate_club_graph():
    """
    Creates the Karate Club graph using NetworkX's built-in function.
    """
    G = nx.karate_club_graph()
    return G


def compute_shortest_path(G, source, target):
    """
    Computes the shortest path between two nodes in the graph and prints the path.
    """
    try:
        path = nx.shortest_path(G, source=int(source), target=int(target))
        print("Shortest path from", source, "to", target, "is:", " -> ".join(map(str, path)))
        return path
    except ValueError:
        print("Source and target must be integers.")
        return None
    except nx.NodeNotFound as e:
        print(f"Node not found: {e}")
        return None
    except nx.NetworkXNoPath:
        print(f"No path between {source} and {target}.")
        return None
    

def partition_graph(G, num_components):
    """
    Partitions the graph G by removing edges to achieve a target number of connected components.

    Parameters:
    G (networkx.Graph): The graph to be partitioned.
    num_components (int): The target number of connected components.

    Returns:
    networkx.Graph: The partitioned graph with the desired number of connected components.
    """
    if num_components < 1 or num_components > len(G.nodes):
        raise ValueError("Number of components must be at least 1 and at most equal to the number of nodes in G.")
    
    # Make a copy of the graph so as not to alter the original graph
    H = G.copy()
    
    # Continue removing edges until we achieve the desired number of connected components
    while nx.number_connected_components(H) < num_components:
        # Calculate all edge betweenness centrality values
        edge_betweenness = nx.edge_betweenness_centrality(H)
        # Find the edge with the maximum betweenness centrality
        edge = max(edge_betweenness, key=edge_betweenness.get)
        # Remove the edge with the highest betweenness centrality
        H.remove_edge(*edge)
    
    return H


def plot_graph(G, path=None, plot_shortest_path=True, plot_cluster_coefficient=False, plot_neighborhood_overlap=False, max_pixel=500, min_pixel=100):
    """
    Plots the graph using Matplotlib with options to highlight the shortest path,
    display cluster coefficients, and show neighborhood overlaps.
    """
    if G is None:
        print("No graph to plot.")
        return

    # Check if the user has requested to plot the shortest path without computing it
    if plot_shortest_path and path is None:
        print("Shortest path has not been computed. Cannot plot, please navigate to algorithm menu to compute first.")
        plot_shortest_path = False  # Disable plotting the shortest path

    pos = nx.spring_layout(G)  # Compute layout for visualisation

    # Base graph drawing
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=300, edge_color='gray')

    # Highlight the shortest path if the option is enabled
    if plot_shortest_path and path:
        edge_list = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color='r', width=2, style='dotted')
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='r', node_size=500)

    # Plot cluster coefficients if the option is enabled
    if plot_cluster_coefficient:
        clustering = nx.clustering(G)
        cluster_min = min(clustering.values())
        cluster_max = max(clustering.values())
        for node in G.nodes():
            cv = clustering[node]
            pv = (cv - cluster_min) / (cluster_max - cluster_min) if cluster_max > cluster_min else 0
            size = int(min_pixel + pv * (max_pixel - min_pixel))
            color = (pv, pv, 0)  # Blend from black to yellow
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=size, node_color=[color])

    # Highlight neighborhood overlaps if the option is enabled
    if plot_neighborhood_overlap:
        overlap = {edge: len(set(G.neighbors(edge[0])) & set(G.neighbors(edge[1]))) / 
                          len(set(G.neighbors(edge[0])) | set(G.neighbors(edge[1]))) for edge in G.edges()}
        edge_colors = [overlap[edge] for edge in G.edges()]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=plt.cm.Blues, width=2)

    plt.show()


def assign_homophily(G, p):
    """
    Assigns a color (either red or blue) to each node in the graph based on probability p
    and calculates the assortativity coefficient to check for homophily.

    Parameters:
    G (networkx.Graph): The graph on which to assign colors.
    p (float): Probability of a node being assigned the color red.

    Returns:
    networkx.Graph: The graph with node colors assigned.
    """
    # Ensure the input graph is valid
    if not isinstance(G, nx.Graph):
        raise ValueError("The provided graph is not a valid NetworkX graph.")
    
    # Ensure the probability is within the correct range
    if not (0 <= p <= 1):
        raise ValueError("Probability p must be between 0 and 1.")
    
    # Assign a color to each node based on probability p
    for node in G.nodes():
        G.nodes[node]['color'] = 'red' if random.random() < p else 'blue'
    
    # Calculate the assortativity coefficient for node colors
    assortativity_coeff = nx.attribute_assortativity_coefficient(G, 'color')
    print(f"Assortativity coefficient for homophily: {assortativity_coeff}")
    
    # Check if the assortativity coefficient is similar to p
    if abs(assortativity_coeff - p) < 0.1:  # Threshold 
        print("The graph shows evidence of homophily.")
    else:
        print("The graph does not show strong evidence of homophily.")

    return G


def assign_balanced_signs(G, p):
    """
    Assigns a sign (1 or -1) to each edge in the graph based on probability p.

    Parameters:
    G (networkx.Graph): The graph on which to assign signs.
    p (float): Probability of an edge being assigned a positive sign (1).

    Returns:
    networkx.Graph: The graph with edge signs assigned.
    """
    if not isinstance(G, nx.Graph):
        raise ValueError("The provided graph is not a valid NetworkX graph.")
    
    if not (0 <= p <= 1):
        raise ValueError("Probability p must be between 0 and 1.")
    
    for u, v in G.edges():
        G[u][v]['sign'] = 1 if random.random() < p else -1

    return G


def validate_balanced_graph(G):
    """
    Validates whether the graph is structurally balanced based on the signs of the edges.

    Parameters:
    G (networkx.Graph): The graph to validate.

    Returns:
    bool: True if the graph is balanced, False otherwise.
    """
    SG = nx.Graph()
    for u, v, data in G.edges(data=True):
        SG.add_edge(u, v, sign=data['sign'])

    # Initialize the SimulatedAnnealingSampler
    sampler = SimulatedAnnealingSampler()

    # Check for structural imbalance using the provided sampler
    imbalance, _ = dnx.structural_imbalance(SG, sampler)
    is_balanced = len(imbalance) == 0
    print(f"The graph is {'balanced' if is_balanced else 'not balanced'}.")
    return is_balanced


def get_yes_no_input(prompt):
    """
    Helper function to get a 'y' or 'n' input from the user.
    """
    valid_responses = {'y', 'n'}
    response = input(prompt).strip().lower()
    while response not in valid_responses:
        print("Invalid input. Please enter 'y' for yes or 'n' for no.")
        response = input(prompt).strip().lower()
    return response == 'y'


def get_probability_input(prompt):
    """
    Helper function for getting menu option #6 Assign and validate Attributes probability p.
    """
    while True:
        try:
            p = float(input(prompt))
            if not 0 <= p <= 1:
                raise ValueError("Probability must be between 0 and 1.")
            return p
        except ValueError as e:
            print(f"Invalid input: {e}")


def main():
    """
    The main function that runs the menu-driven program for graph operations.
    """
    G = None
    shortest_path = None
    
    while True:
        print("\nMenu:")
        print("1 - Read a Graph")
        print("2 - Save the Graph")
        print("3 - Create a Graph")
        print("4 - Algorithms")
        print("5 - Plot Graph")
        print("6 - Assign and Validate Attributes")
        print("x - Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            file_name = input("Enter the file name: ")
            G = read_graph(file_name)
            shortest_path = None  # Clear previous shortest path

        elif choice == '2':
            if G is None:
                print("No graph to save. Please create or read a graph first.")
            else:
                file_name = input("Enter the file name: ")
                write_graph(G, file_name)

        elif choice == '3':
            print("\nCreate Graph Submenu:")
            print("A - Random Erdos-Reny Graph")
            print("B - Karate-Club Graph")
            graph_type = input("Choose the type of graph to create (A/B): ").strip().upper()

            try:
            # Submenu for create graph options
                if graph_type == 'A':
                    # Create random Erdős-Rényi graph
                    n = int(input("Enter the number of nodes: "))
                    c = float(input("Enter the constant c: "))
                    G = create_random_erdos_reny_graph(n, c)
                    if G is None:
                        raise ValueError("Failed to create a Random Erdos-Reny Graph.")
                    else:
                        print(f"Random Erdos-Reny Graph with {n} nodes created.")
                elif graph_type == 'B':
                    # Create Karate Club graph
                    G = create_karate_club_graph()
                    print("Karate Club graph created.")
                else:
                    print("Invalid choice. Please enter 'A' or 'B'.")
            except ValueError as e:
                print(f"Invalid input: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            shortest_path = None  # Clear previous shortest path

        elif choice == '4':
            if G is None:
                print("No graph to operate on. Please create or read a graph first.")
            else:
                # Submenu for algorithm options
                print("\nAlgorithms Submenu:")
                print("A - Shortest Path")
                print("B - Partition Graph")
                algorithm_choice = input("Enter your choice (A/B): ").strip().upper()

                if algorithm_choice == 'A':
                # Compute shortest path
                    try:
                        source = int(input("Enter the source node: "))
                        target = int(input("Enter the target node: "))
                        shortest_path = compute_shortest_path(G, source, target)
                    except Exception as e:
                        print(f"An error occurred: {e}")
                
                elif algorithm_choice == 'B':
                # Partition
                    try:
                        num_components = int(input("Enter the number of components: "))
                        if num_components < 1 or num_components > len(G.nodes):
                            raise ValueError("Number of components must be at least 1 and at most equal to the number of nodes in G.")
                        G = partition_graph(G, num_components)
                        print(f"The graph has been partitioned into {nx.number_connected_components(G)} components.")
                    except ValueError as e:
                        print(f"Invalid input: {e}")
                    except Exception as e:
                        print(f"An error occurred: {e}")
                else:
                    print("Invalid choice. Please enter 'A' or 'B'.")

        elif choice == '5':
            # Submenu for plotting options
            if G is None:
                print("No graph to plot. Please create or read a graph first.")
            else:
                try:
                    plot_shortest_path = get_yes_no_input("Plot the shortest path? (y/n): ")
                    plot_cluster_coefficient = get_yes_no_input("Plot cluster coefficients? (y/n): ")
                    plot_neighborhood_overlap = get_yes_no_input("Plot neighborhood overlaps? (y/n): ")

                    # Call the plot_graph function with user inputs
                    plot_graph(G, shortest_path, plot_shortest_path, plot_cluster_coefficient, plot_neighborhood_overlap)

                except Exception as e:
                    print(f"An error occurred while plotting the graph: {e}")
                
        elif choice == '6':
            # Submenu for assign and validate attributes options
            print("\nAssign and Validate Attributes Submenu:")
            print("A - Homophily")
            print("B - Balanced Graph")
            attr_choice = input("Choose an attribute to assign (A/B): ").strip().upper()

            if attr_choice in ['A', 'B']:
                p = get_probability_input("Enter the probability (0 to 1): ")
                if attr_choice == 'A':
                    G = assign_homophily(G, p)
                elif attr_choice == 'B':
                    G = assign_balanced_signs(G, p)
                    validate_balanced_graph(G)
            else:
                print("Invalid choice. Please enter 'A' or 'B'.")

        elif choice == 'x':
            break

        else:
            print("Invalid choice. Please enter a number between 1-5 or x to exit.")


if __name__ == "__main__":
    main()
