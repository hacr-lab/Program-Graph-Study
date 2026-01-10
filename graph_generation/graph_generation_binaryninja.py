import binaryninja as bn
import networkx as nx
from networkx.classes import edges
from pyvis.network import Network
import pickle
import os
import gc
import concurrent.futures

cfg_edges = ['UnconditionalBranch', 'FalseBranch', 'TrueBranch', 'CallDestination', 'FunctionReturn',
             'SystemCall', 'IndirectCall', 'ExceptionBranch', 'UnresolvedBranch', 'UserDefinedBranch',
             'Epsilon']


class BinaryFunctions:
    def __init__(self):
        self.function_statements = []
        self.function_map = {}
        self.function_id = 0
        self.node_id = 0

    def add_function(self, function_statements):
        function_statements.set_function_id(self.function_id)
        self.function_map[self.function_id] = function_statements.function_name
        self.function_statements.append(function_statements)
        self.function_id += 1

    def get_function_statements(self, function_name):
        for function in self.function_statements:
            if function.function_name == function_name:
                return function

    def get_node_id(self):
        node_id = self.node_id
        self.node_id += 1
        return node_id

    def print_statement_attr(self, name=None):
        if name is None:
            for function in self.function_statements:
                for statement in function.statement_nodes:
                    statement.print_attr()
        else:
            function_statements = self.get_function_statements(name)
            if function_statements is not None:
                for statement in function_statements.statement_nodes:
                    statement.print_attr()
            else:
                print('Function Not Found!')


class FunctionStatements:
    def __init__(self, function):
        self.function_name = function.name
        self.function_id = 0  # use the name to map to a number
        self.function_object = function
        self.statement_nodes = []
        # CFG helper attributes
        self.start_node = None
        self.end_node = None

    def set_function_id(self, new_function_id):
        self.function_id = new_function_id

    def add_statement_node(self, statement_node):
        self.statement_nodes.append(statement_node)

    def find_statement_node(self, statement):
        if len(self.statement_nodes) > 0:
            for statement_node in reversed(self.statement_nodes):
                if statement_node.statement == statement:
                    return statement_node
        return None

    def set_start_node(self, start_node):
        self.start_node = start_node

    def set_end_node(self, end_node):
        self.end_node = end_node

    def get_end_nodes(self):
        end_nodes = []
        edges = []
        for statement_node in self.statement_nodes:
            if statement_node.has_cfg_edges:
                end_nodes.append(statement_node)
                for edge in statement_node.incoming_edges:
                    if edge[1] in cfg_edges:
                        edges.append(edge[0])

        for statement_node in edges:
            if statement_node in end_nodes:
                end_nodes.remove(statement_node)

        self.set_end_node(end_nodes)


    def set_var_locations(self):
        for node in self.statement_nodes:
            if node.abstract_statement_name == 'HLIL_VAR':
                parent = node.parent
                while parent is not None:
                    parent_node = self.find_statement_node(parent)
                    if parent_node.has_cfg_edges and len(node.vars_read) > 0:
                        if node.operand_name == 'dest':
                            parent_node.var_written_statements[str(node.statement)] = node
                        else:
                            parent_node.var_statements[str(node.statement)] = node
                        break
                    else:
                        parent = parent.parent


class StatementNode:
    def __init__(self, statement):
        self.statement = statement
        self.function_id = 0
        self.incoming_edges = []
        self.outgoing_edges = []
        self.basic_block = None
        self.basic_block_dominator = None
        self.abstract_statement_name = None  # Statement HLIL information
        self.tokens = None  # Replace with the tokens
        self.block_index = None
        self.function_name = None
        self.statement_index = None
        self.operand_name = None
        self.statement_type = None
        # parent and parent_node are exclusively ast information
        self.parent = None
        self.parent_node = None
        self.original_statement = None
        self.var_statements = {}
        self.var_written_statements = {}
        # vars read and written are data dependence information
        self.vars_written = []
        self.vars_read = []
        self.has_cfg_edges = False
        self.alternative_nodes = []

        self.node_id = None

    def print_attr(self):
        print("~~~~~~~~~~~~~")
        print(f"Object: {self}")
        print(f"Statement: {self.statement}")
        print(f"Basic Block Dominator: {self.basic_block_dominator}")
        print(f"Function ID: {self.function_id}")
        print(f"Incoming edges: {self.incoming_edges}")
        # print(f"Outgoing edges: {self.outgoing_edges}")
        print(f"Abstract Name: {self.abstract_statement_name}")
        print(f"Tokens: {self.tokens}")
        print(f"Block Index: {self.block_index}")
        print(f"Function name: {self.function_name}")
        print(f"Statement index: {self.statement_index}")
        print(f"Operand name: {self.operand_name}")
        print(f"Statement type: {self.statement_type}")
        print(f"Parent: {self.parent}")
        print(f"Parent node: {self.parent_node}")
        print(f"Original Statement: {self.original_statement}")
        print(f"Var Read Locations: {self.var_statements}")
        print(f"Var Write Locations: {self.var_written_statements}")

        print(f"Variables written: {self.vars_written}")
        print(f"Variables read: {self.vars_read}")
        print(f"Has CFG edges: {self.has_cfg_edges}")
        print("~~~~~~~~~~~~~~")

    def add_edge(self, incoming_edge, label):
        self.incoming_edges.append([incoming_edge, label])
        if label in cfg_edges:
            self.has_cfg_edges = True

    def set_cfg_edge(self, cfg_edge):
        self.has_cfg_edges = cfg_edge

    def get_cfg_edge_count(self):
        count = 0
        for edge in self.incoming_edges:
            if edge[1] in cfg_edges:
                count += 1
        return count

    def get_ddg_statement_read(self, key):
        if key in self.var_statements:
            return self.var_statements[key]
        else:
            return self

    def get_ddg_statement_write(self, key):
        if key in self.var_written_statements:
            return self.var_written_statements[key]
        else:
            return self

    def set_node_id(self, node_id):
        self.node_id = node_id

    def get_node_id(self):
        return self.node_id


def ast_visitor(operand_name, inst, instr_type_name, parent, basicblock, function_statements, statement, bn_func):
    statement_node = StatementNode(inst)
    statement_node.basic_block = basicblock
    statement_node.basic_block_dominator = basicblock.immediate_dominator
    statement_node.abstract_statement_name = inst.operation.name
    statement_node.tokens = inst.tokens
    statement_node.block_index = basicblock.index
    statement_node.function_name = basicblock.function.name
    statement_node.statement_index = inst.instr_index
    statement_node.operand_name = operand_name
    statement_node.statement_type = instr_type_name
    statement_node.parent = parent
    statement_node.parent_node = function_statements.find_statement_node(parent)
    statement_node.original_statement = function_statements.find_statement_node(statement)

    statement_node.set_node_id(bn_func.get_node_id())

    # HLILDeclare reads the variable they declare
    if statement_node.abstract_statement_name == 'HLIL_VAR_DECLARE':
        statement_node.vars_written = inst.vars_read
    elif statement_node.abstract_statement_name == 'HLIL_IF':
        statement_node.vars_written = inst.condition.vars_written
        statement_node.vars_read = inst.condition.vars_read
    else:
        for var_read in inst.vars_read:
            if str(var_read) in str(inst):
                if var_read not in statement_node.vars_read:
                    statement_node.vars_read.append(var_read)
                if statement_node.original_statement is not None:
                    if var_read not in statement_node.original_statement.vars_read and str(var_read) in str(
                            statement_node.original_statement.statement) and statement_node.abstract_statement_name == 'HLIL_VAR' and statement_node.operand_name != 'dest':
                        statement_node.original_statement.vars_read.append(var_read)
        for var_written in inst.vars_written:
            if str(var_written) in str(inst) and var_written not in statement_node.vars_written:
                statement_node.vars_written.append(var_written)

    if statement_node.parent_node is not None:
        statement_node.add_edge(statement_node.parent_node, 'AST')
    # removes duplicate statements, some instr are duplicated with different dominators messes up graph
    if function_statements.find_statement_node(inst) is None:
        function_statements.add_statement_node(statement_node)
    else:
        function_statements.find_statement_node(inst).alternative_nodes.append(statement_node)


def generate_ast_statements(binary_ninja_project) -> BinaryFunctions:
    binary_functions = BinaryFunctions()
    for function in binary_ninja_project.functions:
        function_statements = FunctionStatements(function)
        if function.hlil:
            function_il = function.hlil
            for block in function_il.basic_blocks:
                for statement in block:
                    statement.visit(
                        lambda op_name, inst, inst_type, parent: ast_visitor(op_name, inst, inst_type, parent,
                                                                             block, function_statements, statement,
                                                                             binary_functions))
        binary_functions.add_function(function_statements)
    return binary_functions


def get_cfg_edge_label(prev_statement, statement):
    # only UnconditionalBranch, FalseBranch, TrueBranch have been found in the edge encoder
    bn_edge_map = {0: 'UnconditionalBranch',
                   1: 'FalseBranch',
                   2: 'TrueBranch',
                   3: 'CallDestination',
                   4: 'FunctionReturn',
                   5: 'SystemCall',
                   6: 'IndirectCall',
                   7: 'ExceptionBranch',
                   127: 'UnresolvedBranch',
                   128: 'UserDefinedBranch'}  # from bn api
    prev_block = prev_statement.il_basic_block
    current_block = statement.il_basic_block
    for outgoing_edge in prev_block.outgoing_edges:
        if outgoing_edge.target == current_block:
            return bn_edge_map[outgoing_edge.type]
    return 'Epsilon'


def generate_cfg_statements_with_ast(binary_ninja_project, binary_functions):
    for function in binary_ninja_project.functions:
        if not function.hlil:
            continue

        function_il = function.hlil
        function_object = binary_functions.get_function_statements(function.name)
        for block in function_il.basic_blocks:
            prev_statement = None
            for i, statement in enumerate(block):
                if i == 0 and prev_statement is None:
                    prev_statement = statement
                    statement_obj = function_object.find_statement_node(statement)
                    if function_object.start_node is None:
                        function_object.set_start_node(statement_obj)
                        statement_obj.set_cfg_edge(True)
                    continue
                statement_object = function_object.find_statement_node(statement)
                if statement_object is None:
                    continue
                label = get_cfg_edge_label(prev_statement, statement)
                statement_object.add_edge(function_object.find_statement_node(prev_statement), label)
                prev_statement = statement
            for edge in block.incoming_edges:
                source_statement = edge.source[-1]
                target_statement = block[0]
                label = get_cfg_edge_label(source_statement, target_statement)
                statement_object = function_object.find_statement_node(target_statement)
                if statement_object is None:
                    continue
                statement_object.add_edge(function_object.find_statement_node(source_statement), label)
        function_object.get_end_nodes()


def find_var_write(var_to_find, node):
    var_locations = []
    edges_searched = []
    find_var_helper(var_to_find, node, var_locations, edges_searched, node)
    return var_locations


def find_var_helper(var_to_find, node, var_locations, edges_searched, original_node):
    if node == original_node and len(var_locations) > 0:
        if var_to_find in node.vars_written and node not in var_locations:
            var_locations.append(node)
    if var_to_find in node.vars_written:
        if node not in var_locations:
            var_locations.append(node)

    if len(node.incoming_edges) == 0:
        return
    for edge in node.incoming_edges:
        if edge[1] in cfg_edges and edge[0] not in edges_searched:
            edges_searched.append(edge[0])
            find_var_helper(var_to_find, edge[0], var_locations, edges_searched, original_node)
    return


def bottom_up_data_flow(end_node, dataflow_edges, statements_completed):
    current_node = end_node
    if current_node in statements_completed:
        return

    if len(current_node.vars_read) == 0:
        for edge in current_node.incoming_edges:
            if edge[1] in cfg_edges:
                statements_completed.append(current_node)
                bottom_up_data_flow(edge[0], dataflow_edges, statements_completed)
    else:
        for var_read in current_node.vars_read:
            if current_node.get_cfg_edge_count() > 1:  # if statement
                val_set = set()
                for edge in current_node.incoming_edges:
                    if edge[1] in cfg_edges:
                        loc = find_var_write(var_read, edge[0])
                        statements_completed.append(current_node)
                        if len(loc) > 0:
                            for stmt in loc:
                                val_set.add(stmt)
                        bottom_up_data_flow(edge[0], dataflow_edges, statements_completed)
                for item in val_set:
                    if item != current_node and [item, current_node, str(var_read)] not in dataflow_edges:
                        dataflow_edges.append([item, current_node, str(var_read)])
            elif current_node.get_cfg_edge_count() == 1:
                nodes = find_var_write(var_read, current_node)  # This function should add to the dataflow edges list
                statements_completed.append(current_node)
                if len(nodes) > 0:
                    for node in nodes:
                        if [node, current_node, str(var_read)] not in dataflow_edges:
                            dataflow_edges.append([node, current_node, str(var_read)])
                for edge in current_node.incoming_edges:
                    if edge[1] in cfg_edges:
                        bottom_up_data_flow(edge[0], dataflow_edges, statements_completed)


def generate_ddg_statements_with_ast(binary_ninja_project, binary_functions, verbose=False):
    for functionstatement in binary_functions.function_statements:
        dataflow_edges = []  # [[edge1, edge2, variable], ...]
        statements_completed = []
        if functionstatement.end_node is not None:
            for end_statement in functionstatement.end_node:
                bottom_up_data_flow(end_statement, dataflow_edges, statements_completed)

            functionstatement.set_var_locations()

            for values in dataflow_edges:
                statement1 = values[0].get_ddg_statement_write(values[2])
                statement2 = values[1].get_ddg_statement_read(values[2])
                statement2.add_edge(statement1, 'DDG')


def open_binary_ninja_project(filename):
    project = None
    try:
        project = bn.load(filename)
    except Exception as e:
        print(f"Error opening {filename}: {e}")
    finally:
        return project


def generate_graphs(filename, edges_to_add, specific_function=None):
    bn_project = open_binary_ninja_project(filename)
    binaryfunc = generate_ast_statements(bn_project)
    generate_cfg_statements_with_ast(bn_project, binaryfunc)
    generate_ddg_statements_with_ast(bn_project, binaryfunc)
    visualize_graph_html(convert_binaryfunction_to_networkx(binaryfunc, edges_to_add, specific_function=specific_function), f'{filename}_graphs')


def generate_all_graphs(filename, edges_to_add):
    bn_project = open_binary_ninja_project(filename)
    binaryfunc = generate_ast_statements(bn_project)
    generate_cfg_statements_with_ast(bn_project, binaryfunc)
    generate_ddg_statements_with_ast(bn_project, binaryfunc)
    for function_stmt in binaryfunc.function_statements:
        visualize_graph_html(convert_binaryfunction_to_networkx(binaryfunc, edges_to_add, specific_function=function_stmt.function_name), f'{filename}_{function_stmt.function_name}')


def process_file(input_file_path, input_directory, output_directory, all_edges, edge_output_names, overwrite, reduce_functions):
    bn_project = open_binary_ninja_project(input_file_path)
    binaryfunc = generate_ast_statements(bn_project)
    generate_cfg_statements_with_ast(bn_project, binaryfunc)
    generate_ddg_statements_with_ast(bn_project, binaryfunc)

    for index, edges in enumerate(all_edges):
        new_output = output_directory + f'{edge_output_names[index]}/'
        save_dir = input_file_path.replace(input_directory, new_output)
        graph_object = convert_binaryfunction_to_networkx(binaryfunc, edges, reduce_functions=True)
        save_graph_to_file_full_path(graph_object, save_dir)

    # For memory management
    bn_project.file.close()
    del bn_project
    gc.collect()
    print(f"Generated all graphs for {input_file_path}")


def generate_all_juliet_graphs(input_directory, output_directory, num_subprocesses, overwrite=False,
                               reduce_functions=True):
    all_edges = [['AST'], ['CFG'], ['DDG'], ['AST', 'CFG'], ['AST', 'DDG'], ['CFG', 'DDG'], ['AST', 'CFG', 'DDG']]
    edge_output_names = ['AST', 'CFG', 'DDG', 'ASTCFG', 'ASTDDG', 'CFGDDG', 'ASTCFGDDG']
    directory_files = []

    # Create the required directories if they don't exist
    for root, dirs, files in os.walk(output_directory):
        for edge_output_name in edge_output_names:
            if os.path.exists(os.path.join(output_directory, edge_output_name)):
                pass
            else:
                os.mkdir(os.path.join(output_directory, edge_output_name))

    # Collect all input files
    for root, dirs, files in os.walk(input_directory):
        for edge in edge_output_names:
            for dir in dirs:
                if os.path.exists(os.path.join(output_directory, edge, dir)):
                    pass
                else:
                    os.mkdir(os.path.join(output_directory, edge, dir))

        for file_name in files:
            input_file_path = os.path.join(root, file_name)
            relative_path = os.path.join(root, input_directory)
            if file_name not in directory_files or overwrite == True:
                directory_files.append(input_file_path)

    # Split the work into subprocesses
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_subprocesses) as executor:
        # Submit each file to the pool of subprocesses
        futures = [
            executor.submit(process_file, input_file, input_directory, output_directory, all_edges, edge_output_names,
                            overwrite, reduce_functions)
            for input_file in directory_files]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")

    print("Completed generating all graphs.")


def generate_and_output_networkx_graph(input_path, input_file_name, output_directory, edges_to_add, reduce_functions=False):
    bn_project = open_binary_ninja_project(input_path)
    binaryfunc = generate_ast_statements(bn_project)
    generate_cfg_statements_with_ast(bn_project, binaryfunc)
    generate_ddg_statements_with_ast(bn_project, binaryfunc)
    graph_object = convert_binaryfunction_to_networkx(binaryfunc, edges_to_add, reduce_functions=reduce_functions)
    save_graph_to_file(graph_object, output_directory, f'{input_file_name}_graph')

    # For memory management
    bn_project.file.close()
    del bn_project
    gc.collect()


def generate_graph_from_directory(input_directory, output_directory, edges_to_add, overwrite=False, reduce_functions=False):
    directory_files = []
    if not os.path.isdir(input_directory):
        print("Input directory does not exist")
        return

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    for root, dirs, files in os.walk(output_directory):
        for file_name in files:
            file_to_check = file_name.replace('_graph', '')
            directory_files.append(file_to_check)

    for root, dirs, files in os.walk(input_directory):
        for file_name in files:
            input_file_path = os.path.join(root, file_name)
            relative_path = os.path.join(root, input_directory)
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            if file_name not in directory_files or overwrite == True:
                print(f"Processing {input_file_path}")
                generate_and_output_networkx_graph(input_file_path, file_name, output_directory, edges_to_add, reduce_functions=reduce_functions)
                print(f"Generated graph for {input_file_path}")
            else:
                print(f"Skipping {input_file_path} because it exists")


def convert_binaryfunction_to_networkx(binary_functions, edges_to_add, add_start_edge=True, specific_function=None, reduce_functions=True):
    graph = nx.MultiDiGraph()
    nodes = []
    edges = []

    total_count = 0
    function_count = 0

    for function in binary_functions.function_statements:
        if specific_function:
            if function.function_name != specific_function:
                continue
        if len(edges_to_add) < 1:
            print('No graph type specified')
            break

        if reduce_functions:
            # skip if the function doesn't match the pattern
            total_count += 1
            if not (
                    function.function_name.lower() == "main" or
                    'cwe' in function.function_name.lower() or
                    'good' in function.function_name.lower() or
                    'bad' in function.function_name.lower()
            ):
                continue


        function_count += 1

        function_nodes, function_edges = generate_function_graph_nodes_edges(function, binary_functions, edges_to_add, add_start_edge)
        for node in function_nodes:
            nodes.append(node)
        for edge in function_edges:
            edges.append(edge)

    generate_function_graph(nodes, edges, graph)
    # print(f'Generated {function_count} functions out of a total: {total_count} functions')

    return graph


def generate_function_graph(nodes, edges, graph):
    for node in nodes:
        graph.add_node(node[0], label=node[1])
    for edge in edges:
        graph.add_edge(edge[0], edge[1], label=edge[2])


def generate_function_graph_nodes_edges(function, binary_functions, edges_to_add, add_start_edge):
    ## For cfgs
    node_indexes = []
    nodes = []
    edges = []
    function_start = False
    function_node_id = binary_functions.get_node_id()
    nodes.append([function_node_id, f"FUNCTION"])
    node_indexes.append(function_node_id)
    for statement in function.statement_nodes:
        if statement.has_cfg_edges and 'CFG' in edges_to_add:
            nodes.append([statement.get_node_id(), f'{str(statement.abstract_statement_name)}'])
            node_indexes.append(statement.get_node_id())
            if statement.incoming_edges is not None:
                for incoming_edge in statement.incoming_edges:
                    incoming_edge_node = incoming_edge[0]
                    incoming_edge_node_index = incoming_edge_node.get_node_id()
                    if incoming_edge[1] in cfg_edges:
                        if incoming_edge_node_index not in node_indexes:
                            nodes.append([incoming_edge_node_index,
                                           f'{str(incoming_edge_node.abstract_statement_name)}'])
                        edges.append([incoming_edge_node_index, statement.get_node_id(), incoming_edge[1]])
            if not function_start and add_start_edge:
                edges.append([function_node_id, statement.get_node_id(), 'FunctionStart'])
                function_start = True
        if 'DDG' in edges_to_add or 'AST' in edges_to_add:
            if statement.get_node_id() not in node_indexes:
                nodes.append([statement.get_node_id(),
                           f'{str(statement.abstract_statement_name)}'])
            if statement.incoming_edges is not None:
                for incoming_edge in statement.incoming_edges:
                    incoming_edge_node = incoming_edge[0]
                    incoming_edge_node_index = incoming_edge_node.get_node_id()
                    if incoming_edge[1] == 'AST' and 'AST' in edges_to_add:
                        if incoming_edge_node_index not in node_indexes:
                            nodes.append([incoming_edge_node_index,
                                       f'{str(incoming_edge_node.abstract_statement_name)}'])
                    elif incoming_edge[1] != 'AST' and incoming_edge[1] not in cfg_edges and 'DDG' in edges_to_add:
                        if incoming_edge_node_index not in node_indexes:
                            nodes.append([incoming_edge_node_index,
                                       f'{str(incoming_edge_node.abstract_statement_name)}'])
                    if 'AST' in edges_to_add and incoming_edge[1] == 'AST':
                        edges.append([incoming_edge_node_index, statement.get_node_id(), incoming_edge[1]])
                    elif 'DDG' in edges_to_add and incoming_edge[1] != 'AST' and incoming_edge[1] not in cfg_edges:
                        edges.append([incoming_edge_node_index, statement.get_node_id(), incoming_edge[1]])
            if statement.parent_node is None and statement.operand_name == 'root' and 'AST' in edges_to_add:
                edges.append([function_node_id, statement.get_node_id(), 'AST'])

    return nodes, edges


def save_graph_to_file(graph, save_directory, file_name):
    """

    :param graph: graph object extracted from the above functions, should be a networkx DiGraph
    :param save_directory: file directory to save the graph object to
    :param file_name: file name for the saved graph
    :return:
    """
    with open('{}/{}'.format(save_directory, file_name), 'wb') as f:
        pickle.dump(graph, f)
    f.close()


def save_graph_to_file_full_path(graph, save_directory):
    with open(save_directory, 'wb') as f:
        pickle.dump(graph, f)
    f.close()

def visualize_graph_html(graph, filename):
    """

    :param graph: graph object extracted from the above functions, should be a networkx DiGraph
    :param filename: should contain .html at the end, but if not specified it will be added
    :return:
    """
    if not filename.__contains__(".html"):
        filename += ".html"
    net = Network(notebook=True, directed=True)
    net.from_nx(graph)
    for node in net.nodes:
        node['title'] = node['label']
    for edge in net.edges:
        edge['title'] = edge['label']

    net.show(filename)


def get_graph(file_name):
    with open(file_name, 'rb') as f:
        graph = pickle.load(f)
        return graph