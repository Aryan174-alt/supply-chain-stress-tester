# Supply Chain Stress Tester

This repository scaffolds a pipeline to train models, simulate disruptions, run survival analysis, cluster suppliers, and explain model outputs.

Folder structure:
- data/: place `DataCoSupplyChainDataset.csv` and `DescriptionDataCoSupplyChain.csv` here (user to provide)
- models/: trained models and results
- notebooks/: exploration notebooks
- src/: pipeline modules

Next steps:
1. Put CSV files into `data/`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Implement feature engineering and training code in `src/`.
