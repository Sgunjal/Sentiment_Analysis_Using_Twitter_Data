from collections import defaultdict
import networkx as nx

def girvan_newman(G, depth=0):
    
    if G.order() == 1:
        return [G.nodes()]
    
    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)
        return sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0]
    components = [c for c in nx.connected_component_subgraphs(G)]

    indent = '   ' * depth 
    
    while len(components) == 1:
        edge_to_remove = find_best_edge(G)
        
        G.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(G)]
    
    result = [c.nodes() for c in components]
    
    return result

def read_data_from_file(file_name):
    
    user_follwer_dict=defaultdict(list)
    f=open(file_name,"r")
    line=f.readline()
    while(line !=''):
        #print(line)
        index=line.find(" $|$ ")
        scree_name=line[0:index]
        #print(scree_name)
        cut_str=line[index+5:]
        #print(cut_str)
        cut_str=cut_str.strip(" ")
        cut_str=cut_str.replace("[","")
        cut_str=cut_str.replace("]","")
        cut_str=cut_str.replace(" ","")
        cut_str=cut_str.split(",")
        
        user_follwer_dict[scree_name]=cut_str
        line=f.readline()
    
    list1=[]
    x=0
    graph = nx.Graph()
    for key,val in user_follwer_dict.items():        
        graph.add_node(key) 
        list1=val
        len1=len(list1)        
        for x in range(len1):                      
            graph.add_node(list1[x])
            graph.add_edge(list1[x], key)            

    result=girvan_newman(graph)
    f3=open('cluster_data.txt', 'a')
    f3.write("Number of communities discovered: "+str(len(result))+"\n\n")
    total_len=0
    for x in range(len(result)):
        total_len=total_len+len(result[x])
    f3.write("Average number of users per community: "+str(total_len/len(result))+"\n\n")
    f3.close()
    
    f.close()

def main():
    f3=open('cluster_data.txt', 'w+')
    f3.close()
    read_data_from_file('followers_list.txt')
    
if __name__ == '__main__':
    main()