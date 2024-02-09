"""
Version 2 only uses integer nodes instead of letters.
"""
import networkx as nx
import matplotlib.pyplot as plt
import math

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

def create_random_graph(n, c):
    """
    Creates an Erdős-Rényi graph with n nodes and probability p calculated using the provided c parameter.
    """
    if n < 1:
        print("Number of nodes must be a positive integer.")
        return None
    p = (c * math.log(n)) / n if n > 1 else 0
    G = nx.erdos_renyi_graph(n, p)
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

def plot_graph(G, path=None):
    """
    Plots the graph using Matplotlib, highlighting the shortest path if provided.
    """
    if G is None:
        print("No graph to plot.")
        return
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue')
    
    if path:
        edge_list = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color='r', width=2, style='dotted')
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='r', node_size=500)
    
    plt.show()

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
        print("3 - Create a Random Graph")
        print("4 - Shortest Path")
        print("5 - Plot Graph")
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
            n = int(input("Enter the number of nodes: "))
            c = float(input("Enter the constant c: "))
            G = create_random_graph(n, c)
            shortest_path = None  # Clear previous shortest path

        elif choice == '4':
            if G is None:
                print("No graph to compute the shortest path. Please create or read a graph first.")
            else:
                source = input("Enter the source node: ")
                target = input("Enter the target node: ")
                shortest_path = compute_shortest_path(G, source, target)

        elif choice == '5':
            plot_graph(G, shortest_path)

        elif choice == 'x':
            break

        else:
            print("Invalid choice. Please enter a number between 1-5 or x to exit.")

if __name__ == "__main__":
    main()
