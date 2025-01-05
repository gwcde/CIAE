# CIAE
### Quick Start

```
git clone https://github.com/gwcde/CIAE.git
```

### Acquire Original Metagenome Data

EW-T2D: https://www.ebi.ac.uk/ena/browser/view/PRJEB1786

C-T2D: https://www.ncbi.nlm.nih.gov/sra/?term=SRA045646

### Data Pre-Processing

```
conda env create -f environment.yaml  #create a new enviroment 

conda activate grkoa  #activate current enviroment

bash generate_KO_relative_abundance.sh  #generate KO relative abundance
```

### Training Models

##### create a new enviroment

```
conda create -n CIAE python=3.8
conda activate CIAE
conda install --yes --file requirements.txt
```

##### train

```
#EW-T2D-species
python main.py -d EW-T2D -f species -in ./data/EW_species_abundance.csv --gpu 0 -m CatBoost -ot ./result/species/EW_T2D_KO_Cat.csv
python main.py -d EW-T2D -f species -in ./data/EW_species_abundance.csv --gpu 0 -m FT -lr 0.0001 -bs 8 -nb 2 -ot ./result/species/EW_T2D_KO_FT.csv
```

##### generate explanation

```
#six explainers generate each explanations and acquire average explanations
python generate_explanations.py --KO_EW_file ./data/EW_species_abundance.csv --KO_C_file ./data/C_species_abundance.csv --config_file ./configs/KO_config.yaml --EW_T2D_out_file ./result/KO/EW-T2D/ --C_T2D_out_file ./result/KO/C-T2D/
```

##### train using explanation features
```
#EW-T2D-species
python main.py -d EW-T2D -f species -in ./data/EW_species_abundance.csv --gpu 0 -m CatBoost -ot ./result/species/EW_T2D_KO_Cat.csv -uc -ex ./result/KO/EW-T2D/EW***.csv
```

##### generate explanation weights by CIAE
```
python compute_weight.py -d EW-T2D -in ./data/EW_species_abundance.csv --config_file ./configs/KO_config.yaml --root_files ./result/KO/EW-T2D/ -wk ./result/KO/EW_CIAE_weighted_species_Cat.csv -m CatBoost
```

#### transfer train
```
#EW-T2D-species --> C-T2D-species
python transfer_main.py -d EW-T2D -f species -in ./data/C_species_abundance.csv --gpu 0 -m CatBoost -ot ./result/species/EW_T2D_KO_Cat.csv -uc -ex ./result/KO/EW-T2D/EW***.csv
```