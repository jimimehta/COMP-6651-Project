import networkx as nx
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.models import GraphRenderer, Circle, MultiLine
from bokeh.palettes import Spectral4

# Create a graph using the networkx library

import random
import pickle
import time

import pandas as pd

#Color map for each color assigned to specific value
colormap = {
    1:"blue",2:"red",3:"green",4:"orange",5:"purple",6:"brown",7:"pink",8:"gray",9:"olive",10:"cyan",11:"black",
    12:"yellow",13:"lightcoral",14:"maroon",15:"sienna",16:"peru",17:"tan",18:"gold",19:"darkkhaki",20:"olive",
    21:"lime",22:"turquoise",23:"teal",24:"midnightblue",25:"darkviolet",26:"fuchsia"
}

#Fuction to sort Graph Vertices on the basis of Degree of Vertex
def sort_by_func(lst, func):
    tuples = [(val, func(val)) for val in lst]
    # Sort the tuples based on the second element (i.e. the function output).
    sorted_tuples = sorted(tuples, key=lambda tup: tup[1])
    # Return a new list containing the first element of each tuple (i.e. the original values) in the sorted order.
    return [tup[0] for tup in sorted_tuples]

#First-Fit algorithm to color traingle-free graph in an online fashion
def firstFit(graph, colormap):
    og_nodes = graph.nodes()
    #nodes = sort_by_func(og_nodes, graph.degree)
    nodes = list(og_nodes)
    nG = nx.Graph()
    all_colors = [i for i in range(1,27)]
    gc_map = [-1 for i in range(0,len(nodes))]
    #print(gc_map)
    added_nodes = []
    vert = nodes.pop(0)
    nG.add_node(vert)
    added_nodes.append(vert)
    
    node_color = {vert: colormap[1]}  # Node 3 will be red
    nx.set_node_attributes(nG, node_color, 'color')
    gc_map[vert]= 1
<<<<<<< HEAD
=======
    ggid=10
>>>>>>> 2d4a972aba475850ca3e02b8a7f6a4b9daff3448
    while(len(nodes)>0):
        vert = nodes.pop(0)
        edgeso = nG.edges()
#         for e in edgeso:
#             print(e)
#         print("Processing vertex "+str(vert))
        vert_edges = graph.edges(vert)
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
        nG.add_node(vert)
        nx.set_node_attributes(nG, node_color, 'color')
        gc_map[vert]=vcolor
        for edge in n_edges:
            start, end = edge
            nG.add_edge(start, end)
        added_nodes.append(vert)
        plot_colored_v2(nG, ggid,gc_map)
        ggid+=1
        #print nG graph here - to show the progress vertex by vertex
#         print("----------------------------")
    return nG, gc_map 
	
	
#First-Fit algorithm to color traingle-free graph in an online fashion step by step
def firstFit_v2(graph, colormap):
    og_nodes = graph.nodes()
    #nodes = sort_by_func(og_nodes, graph.degree)
    nodes = list(og_nodes)
    nG = nx.Graph()
    all_colors = [i for i in range(1,27)]
    gc_map = [-1 for i in range(0,len(nodes))]
    #print(gc_map)
    added_nodes = []
    vert = nodes.pop(0)
    nG.add_node(vert)
    added_nodes.append(vert)
    
    node_color = {vert: colormap[1]}  # Node 3 will be red
    nx.set_node_attributes(nG, node_color, 'color')
    gc_map[vert]= 1
    ggid=10
    while(len(nodes)>0):
        vert = nodes.pop(0)
        edgeso = nG.edges()
#         for e in edgeso:
#             print(e)
#         print("Processing vertex "+str(vert))
        vert_edges = graph.edges(vert)
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
        nG.add_node(vert)
        nx.set_node_attributes(nG, node_color, 'color')
        gc_map[vert]=vcolor
        for edge in n_edges:
            start, end = edge
            nG.add_edge(start, end)
        added_nodes.append(vert)
        plot_colored_v2(nG, ggid,gc_map)
        ggid+=1
        #print nG graph here - to show the progress vertex by vertex
#         print("----------------------------")
    return nG, gc_map 

#Generate Traingle-Free Graph
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
	

#Save Graph as Pickle
def save_graph(G, gid):
    with open('Graph_'+str(gid)+'.pickle', 'wb') as f:
        pickle.dump(G, f)
    return 'Graph_'+str(gid)+'.pickle'
	
	
    
#Load Graph from Generate Pickle    
def load_graph(pfile):
    with open(pfile, 'rb') as f:
        G = pickle.load(f)
    gid = pfile.split('_')[1].split(".")[0]
    return G, gid
	

    
#Plot Normal Graph without coloring
def plot_normal(G, gid):
    pos = nx.circular_layout(G)
#     fig, ax = plt.subplots()
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(G, pos, width=1)
    nx.draw_networkx_labels(G, pos, font_size=10,  font_family="sans-serif")
    
#     plt.axis("off")
#     plt.show()
    plt.savefig('Initial_'+str(gid)+'.png', dpi=300, bbox_inches='tight')
    plt.clf()
	

#Plot Colored Graph using First-Fit Algorithm
def plot_colored(G, gid):
    
    nG, gc_map = firstFit(G, colormap)
    pos = nx.circular_layout(nG)
    nx.draw_networkx_nodes(nG, pos, node_size=500)
    nx.draw_networkx_edges(nG, pos, width=1)
    
    #colors = ["r" if color_map[v] == 0 else "b" for v in G]
#     print(gc_map)
    ncolors = [colormap[gc_map[i]] for i in nG.nodes()]

    nx.draw(nG, pos=nx.circular_layout(nG), node_color=ncolors)
    #nx.draw_networkx_labels(nG, pos, font_size=20, font_family="sans-serif")
#     plt.axis("off")
#     plt.show()
    plt.savefig('Colored_'+str(gid)+'.png', dpi=300, bbox_inches='tight')
    plt.clf()
    return nG, gc_map
    
	

    
#Use this Function to Generate First-Fit Colored Triangle-Free Graphs in Bulk
#Change parameters as per your desire
#default is 100 Graphs of 15 vertices
#Create FirstFitStats.xlsx with column names as Graph Id, vertex_count, color_count or use the empty file given with code
<<<<<<< HEAD
def run_experiment(n):
    df = pd.read_excel('FirstFitStats.xlsx')
    #gid is id number of generated graph
    gid = 1000
    #range(n,n+1), n=number of vertices 
    for i in range(n,n+1):
=======
def run_experiment():
    df = pd.read_excel('FirstFitStats.xlsx')
    #gid is id number of generated graph
    gid = 2200
    #range(n,n+1), n=number of vertices 
    for i in range(15,16):
>>>>>>> 2d4a972aba475850ca3e02b8a7f6a4b9daff3448
        #Do not make changes to p if you want to generate 100 graphs of n vertices
        p=0.25
        n = i
        while p <= 0.40:
            for j in tqdm(range(0,25)):
                G = generate_map(n,p)
                gid = int(gid)
                gid+=1
                filepath = save_graph(G, gid)
                time.sleep(1)
                G, gid = load_graph(filepath)
                plot_normal(G, gid)
                nG, gc_map = plot_colored(G, gid)
                colorset = set(gc_map)
                colorCount = len(colorset)
                new_row = { 'Graph Id':filepath, 'vertex_count':n, 'color_count':colorCount }
                df.loc[len(df)] = new_row
            p+=0.05
            
    df.to_excel('FirstFitStats.xlsx', index=False)    
#Stat file of First-Fit is generated and written in FirstFitStats.xlsx    


<<<<<<< HEAD
#Plot Colored Graph using First-Fit Step by Step Coloring Algorithm
=======

>>>>>>> 2d4a972aba475850ca3e02b8a7f6a4b9daff3448
def plot_colored_v2(nG, ggid, gc_map):
    
    pos = nx.circular_layout(nG)
    nx.draw_networkx_nodes(nG, pos, node_size=500)
    nx.draw_networkx_edges(nG, pos, width=1)
    ng_nodes = list( nG.nodes() )
    nGcolors = {i:colormap[gc_map[i]] for i in ng_nodes if gc_map[i]!=-1}
    nx.draw(nG, pos=nx.circular_layout(nG), node_color=[nGcolors.get(node, 'k') for node in nG.nodes()])
    plt.savefig('StepByStep_'+str(ggid)+'.png', dpi=300, bbox_inches='tight')
    plt.clf()
    
<<<<<<< HEAD
#Plot Colored Graph using First-Fit Step by Step Coloring Algorithm
def plot_colored_v3(G, gid):
    
    nG, gc_map = firstFit_v2(G, colormap)
    pos = nx.circular_layout(nG)
    nx.draw_networkx_nodes(nG, pos, node_size=500)
    nx.draw_networkx_edges(nG, pos, width=1)
    
    #colors = ["r" if color_map[v] == 0 else "b" for v in G]
#     print(gc_map)
    ncolors = [colormap[gc_map[i]] for i in nG.nodes()]

    nx.draw(nG, pos=nx.circular_layout(nG), node_color=ncolors)
    #nx.draw_networkx_labels(nG, pos, font_size=20, font_family="sans-serif")
#     plt.axis("off")
#     plt.show()
    plt.savefig('Colored_'+str(gid)+'.png', dpi=300, bbox_inches='tight')
    plt.clf()
    return nG, gc_map

#To plot Single Graph images and pickle Step by Step    
def run_experiment_v2(n):
    gid = 100
    p=0.35
    #n = Number of Vertices
=======


#To plot Single Graph images and pickle Step by Step    
def run_experiment_v2():
    gid = 100
    p=0.35
    #n = Number of Vertices
    n = 10
>>>>>>> 2d4a972aba475850ca3e02b8a7f6a4b9daff3448
    G = generate_map(n,p)
    gid = int(gid)
    gid+=1
    filepath = save_graph(G, gid)
    time.sleep(1)
    G, gid = load_graph(filepath)
    plot_normal(G, gid)
<<<<<<< HEAD
    nG, gc_map = plot_colored_v3(G, gid) 
=======
    nG, gc_map = plot_colored(G, gid) 
>>>>>>> 2d4a972aba475850ca3e02b8a7f6a4b9daff3448
    
    
#Use this Function to Duplicated experiment from the pickle files that are genrated using function run_experiment() or run_experiment_v2()
def duplicate_experiment(folder_name):
    # replace with the name of the folder you want to list
    folder_path = os.path.join(os.getcwd(), folder_name)
    extension = ".pickle"  # replace with the extension you want to filter
    files = os.listdir(folder_path)
    files = [f for f in os.listdir(folder_path) if f.endswith(extension)]
    column_names = ['Graph Id', 'vertex_count', 'color_count']  
    df = pd.DataFrame({}, columns=column_names)
    for f in files:
        file_path = os.path.join(folder_path, f)
        G, gid = load_graph(file_path)
        n=len(list(G.nodes()))
        plot_normal(G, gid)
        nG, gc_map = plot_colored(G, gid)
        colorset = set(gc_map)
        colorCount = len(colorset)
        new_row = { 'Graph Id':f, 'vertex_count':n, 'color_count':colorCount }
        df.loc[len(df)] = new_row
    df.to_excel(folder_name+'-FirstFitStats.xlsx', index=False)   
<<<<<<< HEAD
    
=======
    
>>>>>>> 2d4a972aba475850ca3e02b8a7f6a4b9daff3448
