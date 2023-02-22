# Accurate Prediction Cancer Prognosis
This method focuses on the potential use of genes as a driver gene to predict cancer prognoses considering that it is very likely that patients with different prognoses will have different driver genes. To identify patient-specific cancer driver genes, we first gen-erated patient-specific gene networks before using modified PageRank to generate feature vectors that represented the impacts genes had on the patient-specific gene network. Subsequently, the feature vectors of the good and poor prognosis groups were used to train the deep feedforward network. 

## Data formats
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
