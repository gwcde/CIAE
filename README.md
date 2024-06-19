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
python main.py -d EW-T2D -f species -uc --gpu 0 -m CatBoost
python main.py -d EW-T2D -f species -uc --gpu 0 -m FT-Transformer -lr 0.0001 -bs 8 -nb 2
```

##### generate explanation

```
#six explainers

```

