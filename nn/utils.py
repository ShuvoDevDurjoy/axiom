from graphviz import Digraph

def visualize_computation_graph(output_tensor, filename='computation_graph', format='png', view = True, c_view_options = []):
    dot = Digraph(format=format, graph_attr={'rankdir': 'LR'}, )
    visited = set()
    
    #build topological order of nodes
    def add_nodes(tensor):
        if tensor in visited:
            return
        visited.add(tensor)
        show_data = ('data' in c_view_options) and tensor.ndim == 0
        show_grad = ('grad' in c_view_options) and tensor.ndim == 0
        dot.node(str(id(tensor)), label=f'Tensor\n shape: {tensor.shape}' + (f'\n data: {tensor.data}' if show_data else '') + (f'\n grad: {tensor.grad}' if show_grad else ''), shape='box')
        if hasattr(tensor, '_op') and tensor._op:
            dot.node(str(id(tensor)) + '_op', label=f'Op: {tensor._op}', shape='ellipse')
            dot.edge(str(id(tensor)) + '_op', str(id(tensor)))

        for parent in tensor._prevs:
            add_nodes(parent)
            if hasattr(tensor, '_op') and tensor._op:
                dot.edge(str(id(parent)), str(id(tensor)) + '_op')
            else: 
                dot.edge(str(id(parent)), str(id(tensor)))

    add_nodes(output_tensor)
    if view is False:
        dot.render(filename, view=view)
    return dot
