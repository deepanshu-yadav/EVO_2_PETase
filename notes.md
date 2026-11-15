

# data 

1. The diversity of PET degrading enzymes: A systematic review of sequence, structure, and function
https://onlinelibrary.wiley.com/doi/full/10.1002/pro.70282
(PANDA dataset)

2. Pazy 

https://www.pazy.eu/plastics/pet

3. Plastic db
https://plasticdb.org/static/degraders_list.tsv
https://plasticdb.org/static/PlasticDB.fasta

Optional  https://plasticdb.org/static/structures.tar.gz


# enzyme dicover  engine

https://github.com/msysbio/PlasticEnz

# evo 2 repositories 

https://github.com/evo-design/proto-tools

https://github.com/evo-design/evo2_flash

# commands 

for preparing a pool of proteins 
python phase1_build_pool.py --panda data\panda.xlsx --pazy data\pazy_proteins.fasta --plasticdb data\PlasticDB.fasta

for converting the fasta pool to dna sequences using codon transformer 

python .\phase2_codon_cache.py






