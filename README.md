
#Graph-Based Pharmacovigilance for Adverse Drug Event Prediction
#Overview

This project explores the application of Graph Neural Networks (GNNs) to pharmacovigilance and adverse drug event (ADE) prediction. It models relationships between drugs and adverse events within a heterogeneous biomedical graph and applies graph-based representation learning to identify potentially important drug–event associations.

The project combines biomedical data processing, heterogeneous graph construction, graph neural networks, and link prediction using Python, PyTorch, and PyTorch Geometric.

#Objective

The primary objective is to investigate a graph-based machine learning framework for predicting drug–adverse event associations.

The project focuses on:

Processing and integrating biomedical datasets
Representing biomedical entities and their relationships as a heterogeneous graph
Learning entity representations using Graph Neural Networks
Applying graph attention-based message passing
Predicting drug–adverse event associations through link prediction
Evaluating predictions using ROC-AUC and AUPRC
Research Focus

This project lies at the intersection of:

Pharmacovigilance
Biomedical Data Analysis
Graph Machine Learning
Adverse Drug Event Prediction
Heterogeneous Graph Representation Learning
Graph Neural Networks
Link Prediction
Healthcare AI
Biomedical Graph Representation

The project represents biomedical information as a heterogeneous graph containing multiple entity types and relationships.

Node Types
Drug
Side Effect
Disease
Gene
Relationship Types

The graph incorporates relationships including:

Drug → Side Effect
Drug → Disease
Drug → Gene
Reverse relationships where applicable

This heterogeneous representation enables the model to learn from interconnected biomedical entities rather than treating drug–event associations as independent observations.

## 📁 Project Structure

```
├── main.ipynb
├── model.ipynb
├── requirments.txt
└── README.md
```

## 📦 Dependencies

- Python 3.8+
- PyTorch
- PyTorch Geometric
- RDKit
- pandas, numpy, scikit-learn

Install dependencies using:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:
```txt
torch>=2.0.0
torch-geometric
scikit-learn
rdkit
pandas
numpy
```

## 📥 Data Preparation

The following datasets are required in the `data/` directory:

| File | Description | Download Link |
|------|-------------|----------------|
| `meddra_all_se.tsv` | SIDER side effect mappings | [Download from SIDER](http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz) |
| `drug_names.tsv` | STITCH drug names | [Download from SIDER](http://sideeffects.embl.de/media/download/drug_names.tsv.gz) |
| `CTD_chemicals_diseases.csv.gz` | Chemical-disease relationships | [Download from CTD](http://ctdbase.org/reports/CTD_chemicals_diseases.csv.gz) |
| `CTD_chem_gene_ixns.csv.gz` | Chemical-gene interactions | [Download from CTD](http://ctdbase.org/reports/CTD_chem_gene_ixns.csv.gz) |
| `CTD_chemicals.csv.gz` | Chemical metadata | [Download from CTD](http://ctdbase.org/reports/CTD_chemicals.csv.gz) |
| `CTD_genes.csv.gz` | Gene metadata | [Download from CTD](http://ctdbase.org/reports/CTD_genes.csv.gz) |

Place all files in the `data/` folder. You may need to decompress the `.gz` files if not handled automatically in code.

## 📊 Model Architecture

A two-layer heterogeneous GNN with `GATConv` layers processes:

- Drug ↔ Side Effect
- Drug ↔ Disease
- Drug ↔ Gene

Each node type has separate initial features (random embeddings in this baseline), and edge-specific convolutions are aggregated using `HeteroConv`.

Link prediction is performed on the `drug-causes-side_effect` relation using a simple `MLP` predictor head.

## 🧪 Training & Evaluation

Training is done using:

- Binary cross-entropy loss
- Random negative sampling
- AUC and AUPRC as metrics

Early stopping is used based on validation AUC.
Evaluation

Model performance is evaluated using:

#ROC-AUC

Measures the model's ability to distinguish positive drug–side effect associations from negative associations.

#AUPRC

Measures the precision–recall trade-off and is particularly informative for link-prediction tasks where positive associations may be relatively sparse.

Sample output:
```
Epoch: 001, Train Loss: 0.7351, Val AUC: 0.9780, Val AUPRC: 0.9562
Epoch: 002, Train Loss: 0.3883, Val Loss: 0.1951, Val AUC: 0.9735, Val AUPRC: 0.9339
...
Epoch: 203, Train Loss: 0.0587, Val Loss: 0.0786, Val AUC: 0.9909, Val AUPRC: 0.9857
Epoch: 204, Train Loss: 0.0561, Val Loss: 0.0803, Val AUC: 0.9911, Val AUPRC: 0.9862
Early stopping triggered after 204 epochs due to no improvement.
Test AUC: 0.9911, Test AUPRC: 0.9857
```

This project investigates how heterogeneous Graph Neural Networks can model biomedical relationships and predict drug–side effect associations, providing a computational foundation for graph-based pharmacovigilance research.



