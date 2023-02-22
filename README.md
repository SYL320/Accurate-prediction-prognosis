# Accurate Prediction Cancer Prognosis
This method focuses on the potential use of genes as a driver gene to predict cancer prognoses considering that it is very likely that patients with different prognoses will have different driver genes. To identify patient-specific cancer driver genes, we first gen-erated patient-specific gene networks before using modified PageRank to generate feature vectors that represented the impacts genes had on the patient-specific gene network. Subsequently, the feature vectors of the good and poor prognosis groups were used to train the deep feedforward network. 

## Data formats
__network.csv__  
A DataFrame file with the number of edges.
It consists of two genes per row.  
  
__eset_T.csv__  
A DataFrame file n+1 rows (index) and s+1 columns. s is the number of genes and n is the number of cancer samples.
The additional column at first is the 0/1 binary prognosis labels.  
  
__eset_N.csv__  
A DataFrame file k rows and s columns. s is the number of genes and k is the number of normal samples.
  
__SM__  
Table data representing somatic cell mutations.  
It has a value of 1 if there is a variation and 0 if there is no variation.
```
         gene_1  gene_2  gene_3  gene_4 ... gene_s
sample_1    0       0       1       0         0
sample_2
sample_3
...
sample_n
```
## Run
In the terminal,  
```
$ python make_sample.py
$ python DNN.py
```

## Result
After DNN.py is executed, the AUC value for 10 tests, the average of the values, and the standard deviation value are shown on the terminal screen.
The same data will then be stored in the DNN_AUC_result.csv file.
