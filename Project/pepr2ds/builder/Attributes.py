import pandas as pd

class Attributes():
    def __init__(self, SETUP: dict, recalculate=True, update=True, notebook=True, core=4):
        """

        Args:
            SETUP: Dictionnary containing all needed keys (PEPRMINT_FOLDER, WORKDIR, CATHFOLDER, PROSITEFOLDER,
            UNIPROTFOLDER, FIGURESFOLDER, DOMAIN_PROSITE, PROSITE_DOMAIN, DOMAIN_CATH, CATH_DOMAIN, SUPERFAMILY
            recalculate (bool): Recalculate all dataset without using checkpoint
            execution (bool): if execution is within a notebook

        """
        try:
            from pandarallel import pandarallel
            if core == 1:
                self.PARALLEL = False
            else:
                pandarallel.initialize(nb_workers=core, progress_bar=True)
                self.PARALLEL = True
        except:
            self.PARALLEL = False

        if notebook:
            from tqdm.notebook import tnrange, tqdm
            tqdm.pandas()  # activate tqdm progressbar for pandas apply
        else:
            #print("notebook = False")
            from tqdm import tnrange, tqdm
            tqdm.pandas()  # activate tqdm progressbar for pandas apply

        # Define global varaibles
        self.RECALCULATE = recalculate  # Recalculate everything
        self.UPDATE = update
        self.map3to1 = None
        self.map1to3 = None
        self.AATYPE = None
        self.DOMAIN_SEQ = None

        # convert SETUP Dict to variable
        for key in SETUP:
            exec(f"self.{key} = SETUP['{key}']")

        self._init_dictionnaries()


    def _init_paths(self, peprmint_folder):
        """

        Args:
            peprmint_folder: Path to the PEPRMINT_FOLDER

        Returns:

        """
        self.PEPRMINT_FOLDER = peprmint_folder
        self.WORKDIR = f"{self.PEPRMINT_FOLDER}/dataset/"
        self.CATHFOLDER = f"{self.PEPRMINT_FOLDER}/databases/cath/"
        self.ALPHAFOLDFOLDER = f"{self.PEPRMINT_FOLDER}/databases/alphafold/"
        self.PROSITEFOLDER = f"{self.PEPRMINT_FOLDER}/databases/prosite/"
        self.UNIPROTFOLDER = f"{self.PEPRMINT_FOLDER}/databases/uniprot/"
        self.FIGURESFOLDER = f"{self.PEPRMINT_FOLDER}/figures/"


    def _init_dictionnaries(self):
        # Mapping 3letters code to 1 and vice et versa
        self.map3to1 = {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLU": "E",
            "GLN": "Q",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
            "UNK": "X"
        }
        self.map1to3 = {value: key for key, value in self.map3to1.items()}
        # MAPPING AMINO ACIDS #4 to define
        self.AATYPE = {
            "LEU": "Hydrophobic,H-non-aromatic",
            "ILE": "Hydrophobic,H-non-aromatic",
            "CYS": "Hydrophobic,H-non-aromatic",
            "MET": "Hydrophobic,H-non-aromatic",
            "TYR": "Hydrophobic,H-aromatic",
            "TRP": "Hydrophobic,H-aromatic",
            "PHE": "Hydrophobic,H-aromatic",
            "HIS": "Polar",
            "LYS": "Positive",
            "ARG": "Positive",
            "ASP": "Negative",
            "GLU": "Negative",
            "VAL": "Non-polar",
            "ALA": "Non-polar",
            "SER": "Polar",
            "ASN": "Polar",
            "GLY": "Non-polar",
            "PRO": "Non-polar",
            "GLN": "Polar",
            "THR": "Polar",
            "UNK": "none"
        }


        # GENERATE DOMAIN SEQ
        self.DOMAIN_SEQ = {}
        if self.DOMAIN_PROSITE != None:
            for domain in self.DOMAIN_PROSITE.keys():
                prositeid = self.DOMAIN_PROSITE[domain]
                if type(prositeid) == type(""):
                    self.DOMAIN_SEQ[domain] = f"{self.PROSITEFOLDER}msa/{prositeid}.msa"
                else:
                    dom = []
                    for i, ps in enumerate(prositeid):
                        dom.append(f"{self.PROSITEFOLDER}msa/{ps}.msa")

                    self.DOMAIN_SEQ[domain] = dom

    def save_checkpoint_dataset(self, DATASET, name="checkpoint_structure", path=None):
        if path== None:
            path = self.WORKDIR
        DATASET.to_pickle(f"{path}/{name}.pkl")

    def load_dataset(self, name, path=None):
        if path== None:
            path = self.WORKDIR
        try:
            DATASET = pd.read_pickle(f"{path}/{name}.pkl")
            return DATASET
        except:
            print("File not found")
            return None

    def selectUniquePerCluster(self, df, cathCluster, Uniref, withAlignment=True, pdbreference=None,
                               removeStrand=False):
        """
        Return a datasert with only 1 data per choosed clusters.
        """

        if cathCluster not in ["S35", "S60", "S95", "S100"]:
            raise ValueError('CathCluster given not in ["S35","S60","S95","S100"]')

        if Uniref not in ["uniref50", "uniref90", "uniref100"]:
            raise ValueError('CathCluster given not in ["uniref50","uniref90","uniref100"]')

        if withAlignment:
            df = df[~df.alignment_position.isnull()]

        cathdf = df.query("data_type == 'cathpdb'")
        seqdf = df.query("data_type == 'prosite'")

        def selectUniqueCath(group):
            uniqueNames = group.cathpdb.unique()
            if pdbreference:
                if pdbreference in uniqueNames:
                    select = pdbreference
                else:
                    select = uniqueNames[0]
            else:
                select = uniqueNames[0]

            # return group.query("cathpdb == @select")
            return select

        def selectUniqueUniref(group, exclusion):
            uniqueNames = group.uniprot_acc.unique()
            select = uniqueNames[0]
            # return group.query("uniprot_acc == @select")
            if select not in exclusion:
                return select

        dfReprCathNames = cathdf.groupby(["domain", cathCluster]).apply(selectUniqueCath).to_numpy()

        excludeUniref = df.query(
            "cathpdb in @dfReprCathNames").uniprot_acc.unique()  # Structures are prior to sequences.
        dfReprUnirefNames = seqdf.groupby(["domain", Uniref]).apply(selectUniqueUniref,
                                                                    exclusion=excludeUniref).to_numpy()
        dfReprCath = cathdf.query("cathpdb in @dfReprCathNames")
        dfReprUniref = seqdf.query("uniprot_acc in @dfReprUnirefNames")

        return (pd.concat([dfReprCath, dfReprUniref]))