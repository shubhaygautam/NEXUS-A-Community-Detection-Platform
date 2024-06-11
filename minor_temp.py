import time
import pandas as pd
import networkx as nx
import community as com
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from flask import Flask, render_template, request, send_file
from flask import Blueprint
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import os

image_dir = 'static/images'

def perform_c_d(sort,algo,budget,data):
        
        def flatten_community_dict(community_dict):
            flattened_dict = {} 
            for community_id, nodes in community_dict.items():
                for node in nodes:
                    flattened_dict[node] = community_id
            return flattened_dict

#
        def graph_to_adjacency_matrix(G):
            return nx.to_numpy_array(G)

        def initialize_population(num_bees, num_nodes, num_communities):
            return np.random.randint(num_communities, size=(num_bees, num_nodes))

        def evaluate_fitness(bees, adjacency_matrix):
            num_bees = len(bees)
            #print(bees)
            fitness = np.zeros(num_bees)
            for i in range(num_bees):
                community_matrix = np.eye(len(np.unique(bees[i])), dtype=int)
                for j, community in enumerate(np.unique(bees[i])):
                    community_nodes = np.where(bees[i] == community)[0]               
                    community_adjacency = adjacency_matrix[community_nodes][:, community_nodes]
                    community_fitness += np.sum(community_adjacency)
                fitness[i] = community_fitness
            return fitness

        def scout_bees(bees, fitness, limit):
            num_bees = len(bees)
            for i in range(num_bees):
                if fitness[i] < limit:
                    bees[i] = np.random.randint(num_communities, size=len(bees[i]))
            return bees

        def employed_bees(bees, fitness, limit, adjacency_matrix):
            num_bees = len(bees)
            for i in range(num_bees):
                trial_bee = bees[i].copy()
                j = np.random.randint(len(trial_bee))
                k = np.random.randint(len(trial_bee))
                while j == k:
                    k = np.random.randint(len(trial_bee))
                trial_bee[j] = np.random.choice(np.setdiff1d(np.unique(trial_bee), trial_bee[j]))
                trial_bee[k] = np.random.choice(np.setdiff1d(np.unique(trial_bee), trial_bee[k]))
                trial_fitness = np.sum(trial_bee @ adjacency_matrix @ trial_bee.T)
                if trial_fitness > fitness[i]:
                    bees[i] = trial_bee
            return bees

        def onlooker_bees(bees, fitness, limit, adjacency_matrix):
            num_bees = len(bees)
            probabilities = fitness / np.sum(fitness)
            for i in range(num_bees):
                if np.random.rand() < probabilities[i]:
                    trial_bee = bees[i].copy()
                    j = np.random.randint(len(trial_bee))
                    k = np.random.randint(len(trial_bee))
                    while j == k:
                        k = np.random.randint(len(trial_bee))
                    trial_bee[j] = np.random.choice(np.setdiff1d(np.unique(trial_bee), trial_bee[j]))
                    trial_bee[k] = np.random.choice(np.setdiff1d(np.unique(trial_bee), trial_bee[k]))
                    trial_fitness = np.sum(trial_bee @ adjacency_matrix @ trial_bee.T)
                    if trial_fitness > fitness[i]:
                        bees[i] = trial_bee
            return bees

        def abc_algorithm(G, num_bees, num_communities, max_iterations=100, limit=5):
            adjacency_matrix = graph_to_adjacency_matrix(G)
            num_nodes = adjacency_matrix.shape[0]
            bees = initialize_population(num_bees, num_nodes, num_communities)
            best_solution = None
            best_fitness = float('-inf')
            for _ in range(max_iterations):
                fitness = evaluate_fitness(bees, adjacency_matrix)
                if np.max(fitness) > best_fitness:
                    best_fitness = np.max(fitness)
                    best_solution = bees[np.argmax(fitness)]    
                bees = employed_bees(bees, fitness, limit, adjacency_matrix)
                bees = onlooker_bees(bees, fitness, limit, adjacency_matrix)
                bees = scout_bees(bees, fitness, limit)  
            best_partition = {node: best_solution[i] for i, node in enumerate(G.nodes())}
            return best_partition

#
        def allocate_budget(community_density_ratio, budget, st):
            allocated_budget = {}
            if st=="density":
                    total_density = sum(community_density_ratio.values())
                    #allocated_budget = {}
                    for community_id, density_ratio in community_density_ratio.items():
                        allocated_budget[community_id] = int((density_ratio / total_density) * budget)
                    
                
                    allocation_list = [allocated_budget.get(community_id, 0)+1 for community_id in range(1, len(community_density_ratio) + 1)]

            if st=="size":
                    total_size = sum(community_density_ratio.values())
                    for community_id, size_ratio in community_density_ratio.items():
                        allocated_budget[community_id] = int((size_ratio / total_size) * budget)
                    
                
                    allocation_list = [allocated_budget.get(community_id, 0)+1 for community_id in range(1, len(community_density_ratio) + 1)]    
            return allocation_list
        
        def create_graph_from_csv(df):
            G = nx.from_pandas_edgelist(df, 'Source', 'Target')
            return G

        def sort_partition_by_density(G, partition, budget):
            community_dict = {}
            for node, community_id in partition.items():
                if community_id not in community_dict:
                    community_dict[community_id] = [node]
                else:
                    community_dict[community_id].append(node)

            community_den = {}
            for community_id, nodes in community_dict.items():
                subgraph = G.subgraph(nodes)
                community_den[community_id] = nx.density(subgraph)

            
            sorted_partition = {k: v for k, v in sorted(community_den.items(), key=lambda item: item[1])}
            
            
            b = budget
            sorted_partition = partition

            return sorted_partition, b


        def sort_partition_by_size(partition,budget):
            #
            community_dict = {}
            for node, community_id in partition.items():
                if community_id not in community_dict:
                    community_dict[community_id] = [node]
                else:
                    community_dict[community_id].append(node)

            community_s= {}
            for community_id, nodes in community_dict.items():
                subgraph = G.subgraph(nodes)
                community_s[community_id] = subgraph.number_of_nodes()
            #print(community_s)

            #
            #community_size = {}
            #for community_id, nodes in partition.items():
             #   subgraph = G.subgraph(nodes)
             #   community_size[community_id] = subgraph.number_of_nodes()

            #print(community_s)
            b=budget
            sorted_partition=partition 
            return sorted_partition, b

        def find_communities(G, algorithm, sort,budget):
            start_time = time.time()
            if algorithm == "Louvain":
                init_partition = com.best_partition(G)
                #print(init_partition)
                if sort=="density":
                    partition,bud = sort_partition_by_density(G, init_partition,budget)
                elif sort=="size":
                    partition,bud= sort_partition_by_size(init_partition,budget)
                else:
                    print("Wrong input metrics")
                    exit()
            elif algorithm == "Bee":
                alpha = 0.2
                start=time.time()
                init_partition = abc_algorithm(G, num_bees=10, num_communities=4)
                #init_partition = {node: idx for idx, part in enumerate(init_partition) for node in part}
                if sort=="density":
                    partition,bud = sort_partition_by_density(G, init_partition,budget)
                elif sort=="size":
                    partition,bud= sort_partition_by_size(init_partition,budget)
                else:
                    print("Wrong input metrics")
                    exit()
            elif algorithm == "labelpropgation":
                init_partition = list(nx.algorithms.community.greedy_modularity_communities(G))
                init_partition = {node: idx for idx, part in enumerate(init_partition) for node in part}
                #init_partition = com.best_partition(G)
                if sort=="density":
                    partition,bud = sort_partition_by_density(G, init_partition,budget)
                elif sort=="size":
                    partition,bud= sort_partition_by_size(init_partition,budget)
                else:
                    print("Wrong input metrics")
                    exit()
            elif algorithm == "Maxmin":
                init_partition = list(nx.algorithms.community.greedy_modularity_communities(G))
                init_partition = {node: idx for idx, part in enumerate(init_partition) for node in part}
                #init_partition = com.best_partition(G)
                if sort=="density":
                    partition,bud = sort_partition_by_density(G, init_partition,budget)
                elif sort=="size":
                    partition,bud= sort_partition_by_size(init_partition,budget)
                else:
                    print("Wrong input metrics")
                    exit()
            
            end_time = time.time()
            time_by_all = {}
            if(algorithm=="Bee"):
                time_by_all.append((end_time-start)/10)
            else:
                time_by_all.append(end_time - start_time)
            
#            print(f"Time taken by {algorithm} algorithm: {end_time - start_time:.4f} seconds")
            community_dict = {}
            for node, community_id in partition.items():
                if community_id not in community_dict:
                    community_dict[community_id] = [node]
                else:
                    community_dict[community_id].append(node)
#            for community_id, nodes in community_dict.items():
#              print(f"Community {community_id+1}: {nodes}")
            return partition, community_dict, bud
            

        def create_subgraphs_by_community(G, partition):
            subgraphs = {}
            if isinstance(partition, dict):  
                for community_id in set(partition.values()):
                    nodes_in_community = [node for node, com in partition.items() if com == community_id]
                    subgraph = G.subgraph(nodes_in_community)
                    subgraphs[community_id] = subgraph
            else:  
                for idx, communities in enumerate(partition):
                    for community_id, nodes_in_community in enumerate(communities):
                        subgraph = G.subgraph(nodes_in_community)
                        subgraphs[community_id] = subgraph
            return subgraphs

        def degree_centrality_algorithm(graph, k):
            influence_nodes = set()
            seed_nodes = []
            image_counter = 0
            for _ in range(k):
                max_node = None
                max_degree = -1
                for node, degree in nx.degree_centrality(graph).items():
                    if node not in seed_nodes and degree > max_degree:
                        max_degree = degree
                        max_node = node
                seed_nodes.append(max_node)
                neighbors = set(graph.neighbors(max_node))
                influence_nodes.update(neighbors) 
                image_counter += 1
            return seed_nodes, influence_nodes

        def visualize_graph_with_communities(G, community_dict):
            colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
            random.shuffle(colors) 
            community_color_map = {community_id: colors[i % len(colors)] for i, community_id in enumerate(community_dict.keys())}

          
            node_colors = []
            for node in G.nodes():
                for community_id, nodes_in_community in community_dict.items():
                    if node in nodes_in_community:
                        node_colors.append(community_color_map[community_id])
                        break

            color_positions = {}
            for color in set(node_colors):
                color_positions[color] = []

           
            for node, color in zip(G.nodes(), node_colors):
                color_positions[color].append(node)

            pos = {}
            offset = 0
            for color, nodes_in_color in color_positions.items():
                subgraph = G.subgraph(nodes_in_color)
                color_pos = nx.spring_layout(subgraph, seed=42)
                for node, position in color_pos.items():
                    pos[node] = (position[0] + offset, position[1])
                offset += 2  

        
            plt.figure(figsize=(10, 8))
            nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=300, font_size=10, edge_color='grey', alpha=0.7)
            legend_labels = {str(community_id): color for community_id, color in community_color_map.items()}
            legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for label, color in legend_labels.items()]
            plt.legend(handles=legend_handles, loc='best', title="Community ID", fontsize=10)
            plt.savefig("static/images/graphin/totalgraphwithcom.jpg")   #2


        G = create_graph_from_csv(data)
        for i in range(1):
#        
            partition, community_det, bud = find_communities(G, algo,sort,budget)
            subgraphs = create_subgraphs_by_community(G, partition)
            if(flag):
                plt.figure(figsize=(12, 10))
                pos = nx.spring_layout(G)
                #plt.title('Total Graph without communities')
                nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=200, edge_color='gray', font_size=8)
                plt.savefig("static/images/graphin/totalgraph.jpg") #1st
                
                visualize_graph_with_communities(G,community_det)
            flag=False
            com_to_seed={}
            seed_to_inf={}
            cmt=0
            try:
                num_communities = len(set(partition.values()))
                community_colors = dict(zip(sorted(set(partition.values())), mcolors.CSS4_COLORS.keys()))
                for community_id, subgraph in subgraphs.items():
                    seed_nodes, inf = degree_centrality_algorithm(subgraph, bud[cmt])
                    visualize_graph(subgraph, seed_nodes, inf, (community_id+1), community_id=community_id)
                    #print(f"\nCommunity {community_id+1}:")
                    com_to_seed[community_id+1]=seed_nodes
                    seed_to_inf[community_id+1]=inf
                    #print("Seed Nodes:", seed_nodes)
                    #print("Nodes in Community:")
                    #print(subgraph.nodes())
                    cmt=cmt+1
            except:
                num_communities = len(partition)
                community_colors = dict(zip(range(num_communities), mcolors.CSS4_COLORS.keys()))  
                for idx, (community_id, subgraph) in enumerate(subgraphs.items()):
                    seed_nodes, inf = degree_centrality_algorithm(subgraph, bud)
                    com_to_seed[community_id+1]=seed_nodes
                    seed_to_inf[community_id+1]=inf
                    visualize_graph(subgraph, seed_nodes,inf,(community_id+1), community_id=community_id)
                    #print(f"Community {community_id+1}: Seed Nodes - {seed_nodes}")
                    cmt=cmt+1
        cmt=0
        #print("\n\n")
        #print("Time taken by your choosen algo: ")
            
        #for i in range(len(time_by_all)):
            #print(f"It took total of {time_by_all[0]} seconds")
        return community_det, time_by_all,com_to_seed,seed_to_inf

app = Flask(__name__)

site = Blueprint('site', __name__, template_folder='templates')
app.register_blueprint(site)

result=[]
directory = "static/images"

files = os.listdir(directory)


for file in files:
    if file.endswith(".png"):
        os.remove(os.path.join(directory, file))

file_to_delete = "static/images/graphin/totalgraph.jpg"
if os.path.exists(file_to_delete):
    os.remove(file_to_delete)

@app.route('/')
def index():

    return render_template('free1.html')

@app.route('/process', methods=['POST'])
def process():
    def_csv_file=pd.read_csv("facebook.csv")
    budget = int(request.form['required-influencers'])
    csv_file = request.files['csv-file']
    algo=request.form.get('algorithmSelect')
    sort=request.form.get('commSelect')
    directory = "static/images"
    
    files = os.listdir(directory)


    for file in files:
        if file.endswith(".png"):
            os.remove(os.path.join(directory, file))

    file_to_delete = "static/images/graphin/totalgraph.jpg"
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)
    
    folder_path = "static/images"

    image_extensions = [".png",".jpg"]


    image_files = []

  
    for file in image_files:
        os.remove(file) 
    
    
    csv_file.save('temp.csv')
    
    try:
        data = pd.read_csv('temp.csv')
    except:
        data=def_csv_file
    connection_counts = data['Source'].value_counts().reset_index()
    connection_counts.columns = ['Person id', 'Followers']
    df=pd.DataFrame(connection_counts,index=None)
    
    df=df.head(10)
    new=df.to_csv('newtemp.csv')
    ok=pd.read_csv('newtemp.csv', usecols=[1, 2])
    comm_det, time,com_to_seed,seed_to_inf = perform_c_d(sort,algo,budget,data)
    #ok=pd.read_csv('newtemp.csv')
    
    

    folder_path = "static/images"
    image_paths = [os.path.join("static", "images", filename) for filename in os.listdir(folder_path) if filename.endswith((".png", ".jpg"))]


    table2_html = ok.to_html(classes='table table-bordered', index=False, escape=False)
    image2_url = 'static/images/graphin/totalgraph.jpg'
    image3_url = 'static/images/graphin/totalgraphwithcom.jpg'
    #image4_url = 'static/images/Scatter.jpg'
    #image5_url = 'static/images/line.jpg'
    #image6_url = 'static/images/heatmap.jpg'
    

    table_html = data.to_html(classes='table table-bordered', index=False)
    #numinf=len(influenced_nodes)
    return render_template('free1.html',graph=image2_url,graph2=image3_url,table=table_html,table2_html=table2_html,image_paths=image_paths,time=time,community_dict=comm_det,comm_to_inf=seed_to_inf,comm_to_seed=com_to_seed)

if __name__ == '_main_':
    app.run(debug=True)