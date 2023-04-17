# COMP-6651-Project

pip install python
pip install anaconda
pip install jupyter notebook
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

#Use this Function to load all the graph objects generated in the original experiment, stored in their pickle files and then again run the First Fit algorithm to #regenerate the data in an excel file which is generate in the current folder of the code file
#folder_name is the name of the immediate folder where the files are saved, pickle as well as graph.py
run "duplicate_experiment(folder_name)"


