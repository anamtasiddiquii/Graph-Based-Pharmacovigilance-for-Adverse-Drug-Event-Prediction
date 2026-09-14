
# ⚕️ Personalized Adverse Drug Reaction Prediction using Heterogeneous GNNs

This project implements a heterogeneous GNN model using PyTorch Geometric for **drug-side effect link prediction**, leveraging biomedical relationships from **SIDER** and **CTD** datasets.

## 🔬 Objective

Predict links between drugs and side effects by modeling a heterogeneous graph with the following node types:

- **Drug**
- **Side Effect**
- **Disease**
- **Gene**

And the following edge types:

- Drug → Side Effect (`causes`)
- Drug → Disease (`associates`)
- Drug → Gene (`interacts`)
- Reverse relations for all of the above

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

## 📈 Results

High validation and test AUC/AUPRC indicate the heterogeneous GNN effectively captures drug-side effect relationships in the biomedical graph.

## 🚀 Future Work

- Use molecular fingerprints or SMILES embeddings for drugs
- Pretrained embeddings for genes and diseases
- Extend to other relation prediction tasks (e.g., drug-gene, drug-disease)
- Use larger subsets of CTD/SIDER
- use drig bank's dataset 

## 👨‍🔬 Author

Project by [Arnav Goyal]

---

For any questions, feel free to raise an issue or reach out.

