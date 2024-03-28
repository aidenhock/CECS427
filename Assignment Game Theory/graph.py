"""
Assignment Game Theory
"""
import networkx as nx
from networkx import DiGraph 
import matplotlib.pyplot as plt
import math
import random
import dwave_networkx as dnx
from neal import SimulatedAnnealingSampler
from itertools import product


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
    

def read_digraph(file_name):
    """
    Reads a weighted directed graph (digraph) from a file.
    The file is expected to contain lines with 'source target a, b',
    where 'a' and 'b' are the polynomial factors of the edge weight (a * x + b).
    """
    try:
        G = DiGraph()
        with open(file_name, "r") as file:
            for line in file:
                parts = line.split()
                if len(parts) == 4:  # Check if line has exactly 4 elements
                    source, target, a, b = parts
                    G.add_edge(int(source), int(target), weight=(int(a), int(b)))
        return G
    except FileNotFoundError:
        print("File not found. Please check the file name and try again.")
        return None
    except Exception as e:
        print(f"Unexpected error reading the digraph: {e}")
        return None


def write_graph(G, file_name):
    """
    Writes the graph to a file using the adjacency list format.
    Checks if the graph is directed or undirected and saves accordingly.
    """
    try:
        # Check if the graph is directed or undirected
        if G.is_directed():
            # For directed graphs (digraphs)
            with open(file_name, "w") as f:
                for source, target, data in G.edges(data=True):
                    # Assuming weight is a tuple (a, b)
                    a, b = data.get('weight', (0, 0))
                    f.write(f"{source} {target} {a} {b}\n")
            print(f"Digraph saved to {file_name}")
        else:
            # For undirected graphs
            with open(file_name, "w") as f:
                for line in nx.generate_adjlist(G):
                    f.write(line + "\n")
            print(f"Graph saved to {file_name}")
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


def calculate_nash_equilibrium(G, n, source, destination):
    """
    Computes the Nash equilibrium where no driver benefits from changing their path in isolation. It iterates 
    over drivers, letting them switch to beneficial paths, factoring in the resulting congestion, until no cost 
    improvements are possible.

    Parameters:
    G (networkx.DiGraph): A directed graph with edge weights (a, b) for cost 'a*x + b' per edge flow 'x'.
    n (int): Number of drivers from 'source' to 'destination'.
    source (int): Start node.
    destination (int): End node.

    Returns:
    total_nash_cost (float): Total travel cost at Nash equilibrium.
    driver_counts (dict): Edge-wise driver distribution at equilibrium.
    """
    # Function to calculate the cost for an edge given the number of drivers
    def edge_cost(u, v, drivers):
        a, b = G[u][v]['weight']
        return a * drivers + b  

    # Function to calculate the cost of a path for a given number of drivers on it
    def path_cost(path, driver_counts):
        cost = 0
        for i in range(len(path) - 1):
            cost += edge_cost(path[i], path[i+1], driver_counts.get((path[i], path[i+1]), 0))
        return cost

    # Initialize the driver count on each edge to zero
    driver_counts = {(u, v): 0 for u, v in G.edges()}

    # Get all simple paths from the source to the destination
    all_paths = list(nx.all_simple_paths(G, source, destination))

    # Initially distribute drivers evenly across all paths
    for i in range(n):
        path = all_paths[i % len(all_paths)]
        for j in range(len(path) - 1):
            driver_counts[path[j], path[j+1]] += 1

    # Function to find a better path for each driver, if it exists
    def find_better_path(current_path, driver_counts):
        current_cost = path_cost(current_path, driver_counts)
        for path in all_paths:
            if path != current_path:
                # Calculate the cost if the driver took an alternative path
                new_cost = path_cost(path, driver_counts)
                if new_cost < current_cost:
                    # If the cost is less, it's a better path
                    return path
        return current_path

    stable = False
    while not stable:
        stable = True
        for path in all_paths:
            for _ in range(driver_counts.get((path[0], path[1]), 0)):
                better_path = find_better_path(path, driver_counts)
                if better_path != path:
                    # Move a driver to the better path
                    driver_counts[path[0], path[1]] -= 1
                    driver_counts[better_path[0], better_path[1]] += 1
                    stable = False
                    break  # Break to restart the search after the driver count has been updated

    # Calculate the total cost at the Nash equilibrium
    nash_total_cost = sum(path_cost(path, driver_counts) * driver_counts.get((path[0], path[1]), 0)
                          for path in all_paths)
    return nash_total_cost, driver_counts


def calculate_social_optimum(G, n, source, destination):
    """
    Computes the social optimum distribution of 'n' drivers in a graph 'G' from 'source' to 'destination'.
    The social optimum minimizes the total travel cost, with edge costs given by a quadratic function.
    
    Parameters:
    G (networkx.DiGraph): Directed graph with edge weights (a, b) for cost function 'a * x^2 + b * x'.
    n (int): Number of drivers to route.
    source (int): Starting node.
    destination (int): Destination node.
    
    Returns:
    min_social_cost (float): Minimum total travel cost for all drivers.
    best_distribution (tuple): Optimum number of drivers on each path.
    """
    # Function to calculate the cost for an edge given the number of drivers
    def edge_cost(u, v, drivers):
        a, b = G[u][v]['weight']
        return a * drivers * drivers + b * drivers

    # Initialize the social optimum cost to a high number
    min_social_cost = float('inf')
    best_distribution = None

    # We check every possible distribution of drivers on the two paths
    for drivers_on_path_0_1_2 in range(n + 1):  # This is the number of drivers taking the path 0->1->2
        drivers_on_path_0_2 = n - drivers_on_path_0_1_2  # The rest go directly from 0 to 2

        # Calculate the total cost for this distribution
        total_cost = edge_cost(0, 2, drivers_on_path_0_2)
        if drivers_on_path_0_1_2 > 0:  # Only calculate the cost of 0->1->2 if there are drivers on this path
            total_cost += edge_cost(0, 1, drivers_on_path_0_1_2)
            total_cost += edge_cost(1, 2, drivers_on_path_0_1_2)

        # If the total cost for this distribution is less than the current best, update the social optimum
        if total_cost < min_social_cost:
            min_social_cost = total_cost
            best_distribution = (drivers_on_path_0_2, drivers_on_path_0_1_2)

    return min_social_cost, best_distribution


def plot_combined_graphs(G, total_nash, total_social):
    """
    Plots a bar graph comparing total Nash and social optimum costs alongside the directed graph G.
    
    Parameters:
    - G: The directed graph object (networkx.DiGraph) with polynomial edge weights.
    - total_nash: Total cost based on Nash equilibrium.
    - total_social: Total cost based on the social optimum scenario.
    """
    
    # Create a copy of the graph for visualization to use simplified edge weights
    G_vis = G.copy()
    for u, v, data in G_vis.edges(data=True):
        # Use the 'a' coefficient from the polynomial weight for the layout algorithm
        G_vis[u][v]['weight'] = data['weight'][0]

    # Setup the figure and axes for two subplots: bar chart and graph layout
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart plotting for comparing costs
    bars = axes[0].bar(['Nash Equilibrium', 'Social Optimum'], [total_nash, total_social], color=['cyan', 'purple'])
    axes[0].set_title('Comparison of Total Costs')
    axes[0].set_ylabel('Total Cost')
    
    # Add text labels above the bars
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(f'{height}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')
    
    # Set the ylim for the bar chart to ensure all bars and labels fit within the plot
    axes[0].set_ylim(0, max(total_nash, total_social) * 1.15)  # 15% more than the max value

    # Graph layout plotting for the directed graph with polynomial weights
    pos = nx.circular_layout(G_vis)  # Positions for nodes in a circular layout
    nx.draw(G_vis, pos, with_labels=True, node_color='lightblue', ax=axes[1], arrows=True)
    
    # Polynomial edge labels mapping from G
    edge_labels = {(u, v): f'{a}x + {b}' for u, v, (a, b) in G.edges(data='weight')}
    # Draw edge labels on the graph
    nx.draw_networkx_edge_labels(G_vis, pos, edge_labels=edge_labels, ax=axes[1], label_pos=0.3)
    axes[1].set_title('Directed Graph G with Polynomial Weights')

    # Layout adjustment to prevent overlap and ensure all elements fit within the plot area
    plt.tight_layout()
    plt.show()


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


def assign_plot_homophily(G, p):
    """
    Assigns a color (either red or blue) to each node in the graph based on probability p
    and calculates the assortativity coefficient to check for homophily.
    Plots the graph with the assigned node colors.

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
    
    # Handle edge cases where all nodes will have the same color
    if p in [0, 1]:  
        print("All nodes will have the same color, making assortativity coefficient undefined.")
        return G
    
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

    # Plot the graph with assigned colors
    color_map = [G.nodes[node]['color'] for node in G.nodes()]
    plt.figure(figsize=(8, 8))  # Set the figure size for better visibility
    nx.draw(G, node_color=color_map, with_labels=True, node_size=500, font_size=10, edge_color=".5")
    plt.title("Network Homophily")
    plt.show()

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


def validate_plot_balanced_graph(G):
    """
    Validates whether the graph is structurally balanced based on the signs of the edges.
    Plots the graph with '+' and '-' symbols on edges to indicate their signs.

    Parameters:
    G (networkx.Graph): The graph to validate.

    Returns:
    bool: True if the graph is balanced, False otherwise.
    """
    # Make a copy of G to work with
    SG = G.copy()

    # Initialize the SimulatedAnnealingSampler
    sampler = SimulatedAnnealingSampler()

    # Check for structural imbalance using the provided sampler
    imbalance, _ = dnx.structural_imbalance(SG, sampler)
    is_balanced = len(imbalance) == 0
    print(f"The graph is {'balanced' if is_balanced else 'not balanced'}.")

    # Prepare for plotting: convert numeric signs to symbolic for visualization
    edge_labels = {(u, v): '+' if SG[u][v]['sign'] == 1 else '-' for u, v in SG.edges()}
    edge_colors = ['green' if SG[u][v]['sign'] == 1 else 'red' for u, v in SG.edges()]

    # Plot the graph
    pos = nx.spring_layout(SG)  # Generate a layout for the nodes
    plt.figure(figsize=(8, 8))  # Set the figure size for better visibility
    nx.draw(SG, pos, node_color='lightblue', with_labels=True, node_size=500, font_size=10, edge_color=edge_colors)
    
    # Add edge labels ('+' or '-')
    nx.draw_networkx_edge_labels(SG, pos, edge_labels=edge_labels)

    plt.title("Structurally Balanced Graph" if is_balanced else "Structurally Unbalanced Graph")
    plt.show()

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
            print("\nRead Graph Submenu:")
            print("A - Read a Graph")
            print("B - Read a Digraph")
            read_choice = input("Choose an option (A/B): ").strip().upper()
            
            if read_choice == 'A':
                file_name = input("Enter the file name for the graph: ")
                G = read_graph(file_name)
            elif read_choice == 'B':
                file_name = input("Enter the file name for the digraph: ")
                G = read_digraph(file_name)
            else:
                print("Invalid choice. Please enter 'A' or 'B'.")
            
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
                print("\nAlgorithms Submenu:")
                print("A - Shortest Path")
                print("B - Partition Graph")
                print("C - Travel Equilibrium and Social Optimality")
                algorithm_choice = input("Enter your choice (A/B/C): ").strip().upper()

                if algorithm_choice == 'A':
                    if isinstance(G, nx.DiGraph):
                        print("Option 'A - Shortest Path' is not applicable for directed graphs.")
                    else:
                        try:
                            source = int(input("Enter the source node: "))
                            target = int(input("Enter the target node: "))
                            shortest_path = compute_shortest_path(G, source, target)
                        except Exception as e:
                            print(f"An error occurred: {e}")

                elif algorithm_choice == 'B':
                    if isinstance(G, nx.DiGraph):
                        print("Option 'B - Partition Graph' is not applicable for directed graphs.")
                    else:
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

                elif algorithm_choice == 'C':
                    if not isinstance(G, nx.DiGraph):
                        print("Option 'C - Travel Equilibrium and Social Optimality' can only be performed on a directed graph.")
                    else:
                        try:
                            n = int(input("Enter the number of drivers: "))
                            source = int(input("Enter the source node: "))
                            destination = int(input("Enter the destination node: "))
                            total_nash, best_nash_distribution = calculate_nash_equilibrium(G, n, source, destination)
                            total_social, best_social_distribution = calculate_social_optimum(G, n, source, destination)
                            print(f"Nash Equilibrium: {total_nash}")
                            print(f"Social Optimum: {total_social}")
                            # print(f"Best Nash Distribution: {best_nash_distribution}")
                            # print(f"Best Social Distribution: {best_social_distribution}")
                            # Option to plot the graph with the results here but will opt to do it in menu 5 instead
                        except ValueError as e:
                            print(f"Invalid input: {e}")
                        except Exception as e:
                            print(f"An error occurred: {e}")
                else:
                    print("Invalid choice. Please enter 'A', 'B', or 'C'.")

        elif choice == '5':
            # Submenu for plotting options
            if G is None:
                print("No graph to plot. Please create or read a graph first.")
            else:
                try:
                    plot_shortest_path = get_yes_no_input("Plot the shortest path? (y/n): ")
                    plot_cluster_coefficient = get_yes_no_input("Plot cluster coefficients? (y/n): ")
                    plot_neighborhood_overlap = get_yes_no_input("Plot neighborhood overlaps? (y/n): ")
                    plot_digraph = get_yes_no_input("Plot the digraph with Nash Equilibrium and Social Optimum flows? (y/n): ")

                    # Plot the digraph if requested
                    if plot_digraph:
                        plot_combined_graphs(G, total_nash, total_social)
                    else:
                        # Call the plot_graph function with user inputs for shortest path, cluster coefficient, and neighborhood overlap
                        plot_graph(G, plot_shortest_path, plot_cluster_coefficient, plot_neighborhood_overlap)

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
                    G = assign_plot_homophily(G, p)
                elif attr_choice == 'B':
                    G = assign_balanced_signs(G, p)
                    validate_plot_balanced_graph(G)
            else:
                print("Invalid choice. Please enter 'A' or 'B'.")

        elif choice == 'x':
            break

        else:
            print("Invalid choice. Please enter a number between 1-5 or x to exit.")


if __name__ == "__main__":
    main()
