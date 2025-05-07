```diff
- Readme re-writting in progress...
```

### Dataset Notes:
1. This folder contains the raw dataset for serving the website **PePr2DS**, check the original `README.md` [here](https://github.com/reuter-group/peprmint-web/blob/main/web-client/src/datasets/README.md)
- `PePr2DS.csv`([source data](https://github.com/reuter-group/pepr2ds/blob/main/Ressources/datasets/PePr2DS.csv.zip)): the complete dataset 
    - `domain_<domain>.csv`: generated from the complete dataset by running `dataset_preprocess.sh`

2. Information for each dataset column:

| Column                      | Meaning                                                       | Additional info.                                                                                |
| --------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| domain                      | domain name                                                   | \-                                                                                              |
| cathpdb                     | CATH ID                                                       | \-                                                                                              |
| pdb                         | PDB ID                                                        | \-                                                                                              |
| uniprot\_acc                | Uniprot Accession number                                      | Uniprot accession number (eg: Q9ULH1)                                                           |
| uniprot\_id                 | Uniprot ID                                                    | Uniprot ID (eg: ASAP1\_HUMAN)                                                                   |
| residue\_name               | residue name                                                  | \-                                                                                              |
| IBS                         | binding sites?                                                | True if residue part of the IBS, False otherwise                                                |
| chain\_id                   | chain name                                                    | PDB Chain ID                                                                                    |
| residue\_number             | residue id/number                                             | \-                                                                                              |
| b\_factor                   | Bfactor of each atom                                          | \-                                                                                              |
| sec\_struc                  | secondary structure                                           | secondary structures simplified                                                                 |
| sec\_struc\_full            | Secondary structure                                           | Secondary structures detailed                                                                   |
| prot\_block                 | Protein Block                                                 | [see https://github.com/pierrepo/PBxplore for more info.](https://github.com/pierrepo/PBxplore) |
| data\_type                  | source database                                               | If data are experimentale (cathpdb) or models (alphafold)                                       |
| Experimental Method         | experiment method for obtaining the structure                 | \-                                                                                              |
| resolution                  | strcuture resolution                                          | 999 if the structure is NMR                                                                     |
| RSA\_total\_freesasa\_tien  | Exposition calculated by the definition of Tien et al 2013    |                                                                                                 |
| convhull\_vertex            | convex hull flag                                              | residue part of the Convex Hull                                                                 |
| protrusion                  | protrusion flag                                               | residue is a protrusion                                                                         |
| is\_hydrophobic\_protrusion | hydrophobic protrusion flag                                   | residue is a hydrophobic protrusion                                                             |
| is\_co\_insertable          | co-insertable flag                                            | residue is a co-insertable                                                                      |
| neighboursList              | neighbour residue number list                                 | Neighbours list of residue (if residue convexhull)                                              |
| density                     | protein density                                               | Number of CA/CB in a radius of 1nm                                                              |
| exposed                     | exposition flag                                               | if Residue is exposed (RSA >20%) or not (RSA <= 20%)                                            |
| S35                         | Cath Cluster number at 35% of idendity                        | Cath cluster id at 35% of seq id.                                                               |
| S60                         | Cath Cluster number at 60% of idendity                        | Cath cluster id at 60% of seq id.                                                               |
| S95                         | Cath Cluster number at 95% of idendity                        | Cath cluster id at 95% of seq id.                                                               |
| S100                        | Cath Cluster number at 100% of idendity                       | Cath cluster id at 100% of seq id.                                                              |
| uniref50                    | Representative uniprot\_acc for cluster with 50% of idendity  | Representative sequence for protein sequence at 50% of seq id.                                  |
| uniref90                    | Representative uniprot\_acc for cluster with 90% of idendity  | Representative sequence for protein sequence at 90% of seq id.                                  |
| uniref100                   | Representative uniprot\_acc for cluster with 100% of idendity | Representative sequence for protein sequence at 100% of seq id.                                 |
| origin                      | Specie of origin                                              | Origin of the protein (eg, HUMAN, MOUSE...)                                                     |
| location                    | Location of the protein in the cell                           | Localisation of the protein in the cell                                                         |
| taxon                       | Taxon of the protein                                          | Taxon at level 0 and 1 of the protein (eucaryote/procaryote etc..)                              |
