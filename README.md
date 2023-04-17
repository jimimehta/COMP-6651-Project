# COMP-6651-Project

https://github.com/jimimehta/COMP-6651-Project/blob/main/

you can run the functions defined in graph.py python file in Jupyter notebook cells and run the ui.py file through the command line.

For running the UI - input the number of nodes in the graph in the first input box and in the second input box, input probability of an edge between two vertices. THis value needs to be between 0 to 1 ( preferable 0.3 to 0.4 ). Click Generate button to generate the graph. 

For seeing the coloring of graph vertex by vertex in an online fashion, click on Add Next Node button - the graph will be generated and colored by adding one vertex at a time in the lower frame of the screen below 'Add Next Node' button.

(Please note, while generation of graph node by node, the position of vertices change in each step ( due to change in layout done by underlying python graphing tool) but the actual subgraph is the same i.e nodes and the edges between the nodes in the current subgraph remain same as in the original graph. When all vertices are added to the graph, the result graph will match the generated graph in layout. )

Dependencies - Install following modules using pip command -
pip install networkx

pip install matplotlib

pip install tqdm

pip install os

pip install pandas

pip install bokeh

FirstFitStats.xlxs saves stats data to the excel

Open graph.py in Jupyter Notebook
run "graph.py"

#Use this Function to first generate Triangle-Free Graphs, save the generated graphs in pickle folder, load back the pickled graph objects, save a picture of the graph's plot and then apply FirstFit algorithm to the graphs and save the pictured of the colored graph.

#The function does this whole process for the given number of vertices about 100 times generating 100 graphs and saving data into an excel file for those graphs.

#Change parameters as per your desire by changing value of n
#default is 100 Graphs of 15 vertices
#Create FirstFitStats.xlsx with column names as Graph Id, vertex_count, color_count or use the empty file given with code
run "run_experiment(n)"

#Use this to generate and then plot Single Graph images and a single pickle and then generate colored graphs for each step of the FirstFit algorithm.
    
run "run_experiment_v2()"

#Use this Function to load all the graph objects generated in the original experiment, stored in their pickle files and then again run the First Fit algorithm to regenerate the data in an excel file which is generate in the current folder of the code file

#folder_name is the name of the immediate folder where the files are saved, pickle as well as graph.py
run "duplicate_experiment(folder_name)"


