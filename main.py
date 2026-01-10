import os

from graph_generation import *
from graph_generation.graph_generation_binaryninja import generate_and_output_networkx_graph
from pytorch_geometric_data_converter import *


def main():
    # generate_all_juliet_graphs('/home/aegis/Projects/Datasets/src_bin/', '/home/aegis/Projects/Datasets/Graphs/', 12, reduce_functions=True)
    # graph = get_graph('../Datasets/Graphs/ASTCFGDDG/CWE23/CWE23_Relative_Path_Traversal__char_file_ifstream_13-bad_graph')
    # graph = get_graph('../Datasets/Graphs/ASTCFGDDG/CWE23/CWE23_Relative_Path_Traversal__char_file_ifstream_13-bad')
    # visualize_graph_html(graph, 'EXAMPLEGRAPHASTCFGDDG.html')
    generate_graphs('PaperExample/test_prgm', ['AST', 'CFG', 'DDG'])

    # convert_directory_to_pyg_data('/home/aegis/Projects/Datasets/Graphs/', '/home/aegis/Projects/Datasets/PYGDATAREDUCED/', num_subprocesses=9)

main()
