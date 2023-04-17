import dash
from dash import dcc
from dash import html
import networkx as nx
import plotly.graph_objs as go
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
# Create a graph using the networkx library

import random
import pickle
import time
import pandas as pd


oG = nx.Graph()
pos= 0
node_counter = 0
nnodes = []
iG = nx.Graph()
gc_map = []
added_nodes = []

all_colors = [i for i in range(1,27)]

random.seed(123)

colormap = {
    1:"blue",2:"red",3:"green",4:"orange",5:"purple",6:"brown",7:"pink",8:"gray",9:"olive",10:"cyan",11:"black",
    12:"yellow",13:"lightcoral",14:"maroon",15:"sienna",16:"peru",17:"tan",18:"gold",19:"darkkhaki",20:"olive",
    21:"lime",22:"turquoise",23:"teal",24:"midnightblue",25:"darkviolet",26:"fuchsia"
}


def generate_map(n,p):
    # create an empty graph
    G = nx.Graph()

    # add nodes to the graph
    for i in range(n):
        G.add_node(i)

    # add edges to the graph
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p and not G.has_edge(i, j):
                # add the edge if the probability is met and it doesn't create a triangle
                common_neighbors = set(G.neighbors(i)).intersection(set(G.neighbors(j)))
                if len(common_neighbors) == 0:
                    G.add_edge(i, j)
    return G



# Create a Dash app
app = dash.Dash(__name__)

# Create a NetworkX graph
G = nx.Graph()

# Define the app layout
app.layout = html.Div([
    html.H1('GRAPH COLORING BY FIRSTFIT - COMP6551'),
    dcc.Input(id='input-nodes', type='text', placeholder='Enter no of nodes'),
    dcc.Input(id='input-edges', type='text', placeholder='Enter probability of edges'),
    html.Button(id='generate-button', n_clicks=0, children='generate'),   
    dcc.Graph(id='graph'),
    html.Button(id='submit-button', n_clicks=0, children='Add next Node'), 
    dcc.Graph(id='graph2')
])




# Define a callback to update the graph based on user input
@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [dash.dependencies.Input('generate-button', 'n_clicks')],
    [dash.dependencies.State('input-nodes', 'value'),
     dash.dependencies.State('input-edges', 'value')]
)
def update_graph(n_clicks, input_nodes, input_edges):
    if n_clicks > 0:
        # Convert the user input into lists of nodes and edges
        
        n = int(input_nodes)
        p = float(input_edges)
#         nodes = [int(x.strip()) for x in input_nodes.split(',')]
#         edges = [(int(x.strip().split('-')[0]), int(x.strip().split('-')[1])) for x in input_edges.split(',')]
        
#         # Add the nodes and edges to the graph
#         G.add_nodes_from(nodes)
#         G.add_edges_from(edges)
#         print(G.nodes())
#         print(G.edges())
#         output = Firstfit(G)
#         print(output)

        global oG 
        oG = generate_map(n,p)
        
        global nnodes 
        nnodes = list(oG.nodes())
        print('length of node list:')
        print(len(nnodes))
        
        global gc_map
        gc_map = [-1 for i in range(0,len(nnodes))]
    
        import random

#         def generate_hex_color(num_colors):
#             # Generate a list of random hex color values
#             hex_colors = []
#             for i in range(num_colors):
#                 hex_colors.append('#{:06x}'.format(random.randint(0, 256**3-1)))
#             return hex_colors

#         # Generate a list of 5 random hex color values
#         num_colors = len(set(list(output.values())[:-1]))
#         hex_colors = generate_hex_color(num_colors)
#         colors = []
#         for a in list(output.values())[:-1]:
#             colors.append(hex_colors[a])
        # print(hex_colors)
        # Create a Plotly figure from the NetworkX graph
        global pos
        pos = nx.spring_layout(oG)
        
        
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5,color='#888'),
            hoverinfo='none',
            mode='lines')
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(                
                size=30,
                line_width=2))

        for node in oG.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            
        for edge in oG.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        # Set the colors and text for each node
        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(oG.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
#             node_text.append('Node ' + str(node) + ' has ' + str(len(adjacencies[1])) + ' connections' + " with color: " + str(output[node]))
            
        node_trace['marker']['color'] = node_adjacencies
        node_trace['text'] = node_text
        
        # Create the Plotly figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='<br>Network graph made with Python',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[dict(
                            text="GRAPH",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 )],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

@app.callback(
    dash.dependencies.Output('graph2', 'figure'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-nodes', 'value'),
     dash.dependencies.State('input-edges', 'value')]
)
def update_graph_2(n_clicks, input_nodes, input_edges):
    if n_clicks > 0:
        global oG
        global pos 
        global iG
        global nnodes
        global gc_map
        global added_nodes
        global all_colors
        print(nnodes)
        if len(nnodes)>0:
            vert = nnodes.pop(0)
            edgeso = iG.edges()
    #         for e in edgeso:
    #             print(e)
    #         print("Processing vertex "+str(vert))
            vert_edges = oG.edges(vert)
            n_edges =[]
            non_adjacent_colors = all_colors.copy() 
    #         print(non_adjacent_colors)
            for edge in vert_edges:
                source, end = edge
                if end in added_nodes:
                    n_edges.append(edge)
    #                 print(gc_map[end])
                    if gc_map[end] in non_adjacent_colors:
                        non_adjacent_colors.remove(gc_map[end])
            vcolor = min(non_adjacent_colors)
            node_color = {vert: colormap[vcolor]}
            iG.add_node(vert)
            nx.set_node_attributes(iG, node_color, 'color')
            gc_map[vert]=vcolor
            nc_map = gc_map[0:vert+1]
            print("color map is:")
            print(nc_map)
            for edge in n_edges:
                start, end = edge
                iG.add_edge(start, end)
            added_nodes.append(vert)
            
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5,color='#888'),
                hoverinfo='none',
                mode='lines')
            node_trace = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode='markers',
                hoverinfo='text',
                marker=dict(size=30,
                    line_width=2,
                    color=nc_map)
            )

            for node in iG.nodes():
                x, y = pos[node]
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])

            for edge in iG.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += tuple([x0, x1, None])
                edge_trace['y'] += tuple([y0, y1, None])

            # Set the colors and text for each node
            node_adjacencies = []
            node_text = []
            for node, adjacencies in enumerate(iG.adjacency()):
                node_adjacencies.append(len(adjacencies[1]))
    #             node_text.append('Node ' + str(node) + ' has ' + str(len(adjacencies[1])) + ' connections' + " with color: " + str(output[node]))

    #         node_trace['marker']['color'] = node_adjacencies
    #         node_trace['text'] = node_text

            # Create the Plotly figure
            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title='<br>Network graph made with Python',
                                titlefont_size=16,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                annotations=[dict(
                                text="GRAPH",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002 )],
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
            print("Current nodes in the graph are:")
            print(list(iG.nodes))
        else :
             
            nnodes = list(oG.nodes())
            print('length of node list:')
            print(len(nnodes))
        
            
            gc_map = [-1 for i in range(0,len(nnodes))]
            print('global var not updated')
            iG = nx.Graph()
            added_nodes = []
        

    return fig
            
            
            
    
    


if __name__ == '__main__':
    app.run_server(debug=True)