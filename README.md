# COMP-6651-Project

pip install python
pip install anaconda
pip install jupyter notebook
pip install networkx
pip install matplotlib
pip install tqdm
pip install pandas
pip install bokeh

FirstFitStats.xlxs saves stats data to the excel

Open graph.py in Jupyter Notebook
run "graph.py"

#Use this Function to Generate First-Fit Colored Triangle-Free Graphs in Bulk
#Change parameters as per your desire
#default is 100 Graphs of 15 vertices
#Create FirstFitStats.xlsx with column names as Graph Id, vertex_count, color_count or use the empty file given with code
run "run_experiment()"

#Use this to plot Single Graph images and a single pickle for Step by Step graph coloring    
run "run_experiment_v2()"

#Use this Function to Duplicated experiment from the pickle files that are genrated using function run_experiment() or run_experiment_v2()
#pickle are already generated graphs
#folder_name is the immediate folder where the files are saved, pickle as well as graph.py
run "duplicate_experiment(folder_name)"


