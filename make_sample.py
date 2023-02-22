import pandas as pd
import numpy as np
import networkx as nx
import pickle

def get_rank(rank):
	rank = rank.sort_values(ascending=False)
	rank = rank.index.tolist()

	tmp_rank = {}
	for i,gene_name in enumerate(rank):
		tmp_rank[gene_name] = i+1

	rank = pd.Series(tmp_rank)
	rank = rank.loc[genes]
	return rank
    
def create_adj_matrix(edges, genes, num_of_genes) :
    G = nx.DiGraph()
    G.add_edges_from(edges)
    A = nx.to_numpy_array(G, nodelist=genes)
    A = A.T # equation 1 A, reversed direction gene network
    I = np.identity(len(genes))
    A = A * (1-I) 

    tel = A.sum(axis=1).reshape(-1)
    tel = (tel==0).astype(float)
    tel = tel.reshape(-1,1) * np.ones(num_of_genes)
    A = A + tel
    return A

def damping_factor(A) :
    A_sum = A.sum(axis=1).reshape(-1)
    D = A_sum/(A_sum+3) # Damping factor
    D = D.reshape(-1)
    return D
    
def calculate_f(eset_T, eset_N) :
    all_barcode = set(eset_T.index) & set(eset_N.index) # pair sample-barcode
    only_barcode = set(eset_T.index) - set(eset_N.index) # only Tumor sample-barcode

    T_all = eset_T.loc[all_barcode]
    T_only = eset_T.loc[only_barcode]

    f_all = abs(T_all[genes] - eset_N.loc[all_barcode])
    f_only = abs(T_only[genes] - eset_N.mean(axis=0))

    f = pd.concat([f_all,f_only])
    return f
    
def calculate_score(eset_T, eset_N, SM, genes, num_of_genes, A, D, f) :
    R = []
    SM_w = SM + 1
    for samp in eset_T.index:
        p_sm = SM_w.loc[samp].values.reshape(-1,1)

        exp_T = eset_T.loc[samp]
        exp_T = get_rank(exp_T).values.reshape(-1)

        W = np.zeros([num_of_genes,num_of_genes])

        for samp2 in eset_N.index:
            exp_N = eset_N.loc[samp2]
            exp_N = get_rank(exp_N).values.reshape(-1)
            
            tmp_w = abs(exp_T - exp_N).reshape(-1)	# equation 3
            tmp_w2 = np.repeat(tmp_w,num_of_genes).reshape(num_of_genes,num_of_genes)
            tmp_w2 = np.minimum(tmp_w2,tmp_w2.T)
            tmp_w3 = tmp_w + tmp_w.reshape(-1,1) #d(A) + d(B)
            W += tmp_w3 * tmp_w2 # (d(A) + d(B)) * min(d(A),d(B)) # equation 4

        W = W/len(eset_N.index)
        W = A * W

        W_denominator = W.sum(axis=1).reshape(-1)
        W_denominator[W_denominator==0] = 1.
        W = W.T / W_denominator	

        W = W * p_sm # mutation weight
        W = W.T
        W_denominator = W.sum(axis=1).reshape(-1)
        W_denominator[W_denominator==0] = 1. 
        W = W.T / W_denominator

        rt = f.loc[samp]
        S0 = (1-D) * rt
        for _ in range(5):
            rt = S0 + D*np.dot(W,rt)
        R.append(rt)
    R = pd.DataFrame(R,index=f.index,columns=genes)
    return R
    
def calculate_winning_rate(SM, genes, R,good_or_bad, path) :
    win_per = []
    for samp in R.index:
        all = np.zeros(len(genes))
        win = np.zeros(len(genes))
        
        rt = R.loc[samp].values
        mut = SM.loc[P_barcode].values
        rt[mut==0] = rt[mut==0] * 0.85
        rt = pd.DataFrame(rt,index=genes)
        
        mut_genes = genes[mut==1]
        not_mut_genes = genes[mut==0]
        
        all[mut==1] = len(genes)-1
        all[mut==0] = len(mut_genes)
        
        mut_rt = rt.loc[mut_genes]
        
        for gene in mut_genes:
             gene_idx = np.where(gene == genes)[0][0]
             win[gene_idx] = np.sum(mut_rt.loc[gene].values[0]>rt)
             
            
        for gene in not_mut_genes:
            gene_idx = np.where(gene == genes)[0][0]
            win[gene_idx] = np.sum(rt.loc[gene].values[0]>mut_rt)
        
        win_per.append(win/all)
    win_per = pd.DataFrame(win_per,index=R.index,columns=genes)

    with open(path + f"{good_or_bad}_samples.pkl","wb") as f:
        pickle.dump(win_per,f)
        
#####################
PATH = "./data/"
edges = pd.read_csv(PATH + "network.csv", index_col=0) #network edges

eset = pd.read_csv(PATH + "eset_T.csv",index_col=0) # good or bad gene expression dataframe
eset_G = eset[eset['label']==0].drop(['label'], axis=1)
eset_B = eset[eset['label']==1].drop(['label'], axis=1)
eset_N = pd.read_csv(PATH + "eset_N.csv", index_col=0) #nomal gene expression dataframe

genes = np.array(eset_G.columns)
num_of_genes = len(genes)

SM = pd.read_csv(PATH + "SM.csv", index_col=0) #somatic mutation dataframe

#######################

A = create_adj_matrix(edges,genes,num_of_genes)
D = damping_factor(A)

######################

f_Good = calculate_f(eset_G,eset_N)
R_Good = calculate_score(eset_G, eset_N, SM, genes, num_of_genes, A, D, f_Good)
calculate_winning_rate(SM, genes, R_Good, 'Good', PATH)
print("** Good sample Generation Completed **")

f_Bad = calculate_f(eset_B,eset_N)
R_Bad = calculate_score(eset_B, eset_N, SM, genes, num_of_genes, A, D, f_Bad)
calculate_winning_rate(SM, genes, R_Bad, 'Bad', PATH)
print("** Bad sample Generation Completed **")

######################



