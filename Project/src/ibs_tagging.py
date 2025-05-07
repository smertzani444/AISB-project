#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
# 
# Copyright (c) 2022 Reuter Group
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


""" Tagging the IBS in the dataset (previously on notebooks under "tools")

Methods to tag the interfacial binding sites (IBS) of proteins in the dataset, 
with or without using the alphafold models.

__author__ = ["Thibault Tubiana", "Phillippe Samer"]
__organization__ = "Computational Biology Unit, Universitetet i Bergen"
__copyright__ = "Copyright (c) 2022 Reuter Group"
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Phillippe Samer"
__email__ = "samer@uib.no"
__status__ = "Prototype"
"""

import pandas as pd
import numpy as np
import math
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import ConvexHull
import MDAnalysis as mda

from tqdm import tnrange, tqdm

import os
import shutil
from pathlib import Path
import urllib
import glob
from urllib.error import HTTPError

import importlib
import pepr2ds
from pepr2ds.dataset.tagibs import Dataset

from src.settings import Settings
from src.notebook_handle import NotebookHandle

class IBSTagging:

    def __init__(self, global_settings: Settings, data_type: str):
        self.settings = global_settings

        # data_type can be cath, alphafold, or cath+af (NB! do not change this as they are pepr2ds settings)
        self.data_type = data_type
        if data_type not in ["cath", "alphafold", "cath+af"]:
            self.data_type = "cath+af"
            print(f"> Warning: did not recognize option '{data_type}' to IBSTagging constructor")
            print(f"           - defaulting to 'cath+af'")

        self.cluster_level = self.settings.config_file['IBS_TAGGING']['cluster_level']
        self.uniref_level = self.settings.config_file['IBS_TAGGING']['uniref_level']
        self.z_axis_level = self.settings.config_file.getint(
            'IBS_TAGGING','z_axis_level')
        self.comparison_mode = self.settings.config_file.getboolean(
            'IBS_TAGGING', 'comparison_mode')

        self.pepr2ds_dataset = None          # the resulting Dataset (from pepr2ds.dataset.tagibs) instance, after merging
        self._df = None                      # the input pandas dataframe
        self._superfamily_to_pepr2ds = {}    # temporary map (pre-merging) Dataset instances from each superfamily

        self._libs_setup()
        self._directories_setup()

    def _libs_setup(self):
        # IPython
        if self.settings.USING_NOTEBOOK:
            self.settings.NOTEBOOK_HANDLE.ibs_options()
        
        # Pandas
        pd.options.mode.chained_assignment = (
            None  # remove warning when adding a new column; default='warn'
        )
        pd.set_option("display.max_columns", None)
        tqdm.pandas()   # activate tqdm progressbar for pandas

        # Numpy
        np.seterr(divide='ignore', invalid='ignore')

        # Seaborn
        sns.set_style("darkgrid")

    def _directories_setup(self):
        running_in_overwrite_mode = self.settings.config_file.getboolean('PREPROCESSING', 'overwrite_original_pdbs')

        # pepr2ds.Dataset.tag_ibs() needs either 'cath' (and it appends 'domains/' to the path) or the parent folder containing superfamily folders
        self.base_folder = self.data_type if self.data_type in ["cath", "alphafold"] else self.settings.ALIGNED_CATH_AND_AF

        # pepr2ds.Dataset.tag_ibs() will look for (superimposed, reoriented) pdbs in a subfolder with this name
        self.coordinates_folder = self.settings.ALIGNED_SUBDIR
        if running_in_overwrite_mode:
            print(f"> Warning: preprocessing option 'overwrite_original_pdbs' is active in the config file")
            if self.data_type == "cath":
                print(f"           - assuming aligned files are under 'raw'")
                self.coordinates_folder = 'raw'
            elif self.data_type == "alphafold":
                print(f"           - assuming aligned files are under 'extracted'")
                self.coordinates_folder = 'extracted'
            else:   # cath+af
                print(f"           - assuming aligned files are under 'cath/domains/raw' and 'alphafold/extracted'")
                print(f"           - will now copy them to {self.settings.ALIGNED_CATH_AND_AF}/.../{self.settings.ALIGNED_SUBDIR}")

        # pepr2ds.Dataset.tag_ibs() needs an extra directory containing both CATH and AF oriented files
        if self.data_type == "cath+af":
            parent_folder = f"{self.settings.PEPRMINT_FOLDER}/databases/{self.settings.ALIGNED_CATH_AND_AF}/"
            if os.path.exists(parent_folder):
                shutil.rmtree(parent_folder)
            for superfamily in self.settings.active_superfamilies:
                new_folder = parent_folder + superfamily + "/" + self.settings.ALIGNED_SUBDIR

                src_cath = self.settings.CATHFOLDER + "domains/" + superfamily + "/"
                src_af = self.settings.ALPHAFOLDFOLDER + superfamily + "/"
                if running_in_overwrite_mode:
                    src_cath = src_cath + "raw"
                    src_af = src_af + "extracted"
                else:
                    src_cath = src_cath + self.settings.ALIGNED_SUBDIR
                    src_af = src_af + self.settings.ALIGNED_SUBDIR

                # creates all necessary intermediate nodes before a leaf (:
                shutil.copytree(src=src_cath, dst=new_folder)

                # param dirs_exist_ok did not exist before python3.8, so we need to copy the second folder files one by one
                for f in [str(x) for x in Path(src_af).glob("*.pdb")]:
                    shutil.copy(src=f, dst=new_folder)

    def run(self, df: pd.DataFrame):
        self._df = df

        if self.settings.use_PH:
            self._PH_domain_preprocessing()

        print("> Tagging IBS")
        for superfamily in self.settings.active_superfamilies:
            print(f"  {superfamily}... ", end='')
            self._tag_superfamily(superfamily)
            print("done")
        
        self.pepr2ds_dataset = self._merge_data()

    def make_analysis_report(self):
        if self.pepr2ds_dataset is not None:
            self.pepr2ds_dataset.analysis.report(displayHTML=False)
        else:
            print("Error: cannot make analysis report without completing IBSTagging.run() successfully")

    
    """
    ### All methods below just encapsulate the steps in the original notebook
    """

    def _PH_domain_preprocessing(self):
        """
        Prepare Exclusion of PTB and other domains
        Procedure
        1. Get protein template from Mark Lemmon review (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1221219/pdf/10926821.pdf) or from the PDB
        2. search from this protein in rcsb PDB -> Annotation -> Click on the PFAM familly 
        3. In the results, display only PDB IDs 
        4. For each PDB, get S60 Cluster IDS. 
        5. Remove those S60 Cluster IDs from the PH superfamilly.

        LIST OF PFAM
        PTB = PF00640
        RANDB = PF00638
        EVH/WH1 = PF00568
        """
        pdb_PTB = "2YT8_1,2YT7_1,6DM4_2,3O17_2,3D8F_1,2E45_1,2YT1_1,6O5O_1,1NU2_1,3O2M_2,3H8D_2,1NTV_1,5LW1_3,6FUZ_1,2G01_2,6LNM_2,5YI7_1,5YI8_1,4DBB_1,6F5E_3,2NMB_1,2MYQ_1,1N3H_1,3F0W_1,3SUZ_1,3SV1_1,1SHC_1,2L1C_1,1AQC_1,2YSC_1,3D8D_1,2YSZ_1,3D8E_1,1OY2_1,3VUG_2,4JMH_2,5NQH_1,3VUH_2,3VUK_2,3VUI_2,4E73_2,2FPD_1,3VUL_2,2EJ8_1,6ITU_1,1WJ1_1,3OXI_2,5C5B_2,5C5B_1,6F7O_1,2FPF_1,2LSW_1,1WGU_1,4Z88_2,2ELA_1,2ELB_1,2DYQ_1,2FPE_1,3VUM_2,5UWS_4,2GMX_2,6OVF_1,2IDH_1,2OEI_1,4H8S_1,2EAM_1,1OQN_1,1TCE_1,4G1W_2,4XWX_1,5YQG_2,1X11_1,1M7E_1,1MIL_1,4HYS_2,1P3R_1,2Z0N_1,2LMR_1,3QYB_1,5NJK_1,2Q12_1,5NJJ_1,3PTG_2,3DXC_1,3QYE_1,1X45_1,1UKI_2,3DXE_1,1Y7N_1,3SO6_1,2Z0O_1,2KIV_1,2Q13_1,4NC6_1,3DXD_1,1UKH_2,2HO2_1,2YT0_1,2M38_1,2H96_2,5ZRY_1,4HYU_2,1DDM_1,1QG1_2,1U39_1,4IZY_2,3V3V_2,3VUD_2,6KMH_2,2KE7_1,1U3B_1,2ROZ_2,5CZI_2,1U37_1,4H39_2,1U38_1"
        pdb_RANDB = "5UWW_2,6KFT_2,1K5G_2,5UWT_2,3UIO_4,5UWI_2,6CIT_2,4HAU_2,4HB2_2,4HAT_2,4HB3_2,4HB4_2,4HAX_3,4HAV_2,4HAW_2,3TJ3_2,4HB0_2,5DIF_2,1XKE_1,5DH9_2,4GMX_2,2CRF_1,6XJT_2,6XJU_2,6XJR_2,6XJS_2,5DHF_2,5DHA_2,4LQW_1,5UWJ_2,3UIP_4,5UWH_2,3UIN_4,5JLJ_2,5ZPU_2,2C1M_2,4GA0_1,6A38_2,2C1T_2,5DI9_2,6A3E_2,6A3C_2,6A3B_2,6A3A_2,4HAZ_2,4HAY_2,4GPT_2,2LAS_2,7L5E_2,4GA1_1,1UN0_2,5UWU_2,1RRP_2,5UWS_2,3OAN_1,5CLL_2,5YRO_2,6XJP_2,4L6E_1,6M6X_2,5CLQ_2,6X2M_2,6X2O_2,5YSU_2,6X2S_2,6X2U_2,5YST_2,6X2V_2,6X2W_2,6X2X_2,6X2Y_2,6X2R_2,6X2P_2,4WVF_2,5YTB_2,2EC1_1,5UWQ_2,1K5D_2,5UWR_2,5UWO_2,5UWP_2,2Y8G_1,3N7C_1,4I9Y_1,3M1I_2,5XZX_2,1Z5S_4,2Y8F_1,3WYF_2,4GA2_1"
        pdb_EVH = "1TJ6_1,5N91_1,5ZZ9_1,5N9C_1,1CEE_2,2XQN_2,3RSE_8,6RCJ_1,6RCF_1,3CHW_3,2V8C_2,2PBD_3,6RD2_1,2PAV_3,2IYB_1,6XXR_1,7A5M_1,2OT0_2,1XOD_1,1I2H_1,1EVH_1,6V65_1,1USD_1,1USE_1,7AKI_1,1ZUK_2,4MY6_1,2IFS_1,3SYX_1,1QC6_1,2JP2_1,3CVF_1,1T84_1,5N9P_1,2FF3_2,6UHC_8,1I7A_1,2P8V_1,5NDU_1,2LNH_1,3M3N_2,5NCG_1,5NCF_1,5NEG_1,2A3Z_3,6XVT_1,1MKE_1,2VCP_2,2HO2_2,3CVE_1,1EJ5_1,5NCP_1,5ND0_1,1DDW_1,5NC7_1,1DDV_1,2K42_1,4CC7_2,4CC3_2,1EGX_1,4CC2_2,5NBF_1,5NAJ_1,6V6F_1,5NC2_1,5NBX_1"
        # pdb_DCP1="5JP4_1,2QKL_1,4B6H_1,5J3Q_1,5N2V_1,1Q67_1,6Y3Z_2,2QKM_1,5J3Y_1,5LOP_2,5LON_2,5KQ1_1,6AM0_2,5J3T_1,2LYD_1,5KQ4_1"
        pdb_PTB = [x[:4] for x in pdb_PTB.split(',')]
        pdb_RANDB = [x[:4] for x in pdb_RANDB.split(',')]
        pdb_EVH = [x[:4] for x in pdb_EVH.split(',')]
        # pdb_DCP1 = [x[:4] for x in pdb_DCP1.split(',')]

        pdbs_to_remove = pdb_PTB+pdb_RANDB+pdb_EVH

        removeS60 = (self._df
                     .query("domain == 'PH' and pdb in @pdbs_to_remove")
                     ["S60"]
                     .unique())
        cathpdbs_to_remove = (self._df
                              .query("domain == 'PH' and S60 in @removeS60")
                              .cathpdb
                              .unique())

        print(f"{len(cathpdbs_to_remove)} 'fake' PH entries removed; csv saved under figures folder")
        self._df = self._df.query("cathpdb not in @cathpdbs_to_remove")

        pd.DataFrame(cathpdbs_to_remove).to_csv(f"{self.settings.FIGURESFOLDER}/Fake_PH.csv",
                                                index = False,
                                                header = False)

    def _tag_superfamily(self, superfamily: str):
        ref_pdb = self.settings.config_file['PREPROCESSING']["ref_"+superfamily+"_pdb"]
        if ref_pdb is None:
            print(f"  Warning: no reference protein for '{domain}' defined on .config file - skipping IBS tagging")
            return

        self._superfamily_to_pepr2ds[superfamily] = Dataset(self._df, self.settings.PEPRMINT_FOLDER)
        
        filter_uniprot_acc = self._get_uniprot_in_common(superfamily) if self.comparison_mode else None

        self._superfamily_to_pepr2ds[superfamily].tag_ibs(
                self._df, 
                domain = superfamily,
                pdbreference = ref_pdb,
                includeResidueRange = [],                          # not used when overide_axis_mode = True
                excludeResidueRange=[],                            # not used when overide_axis_mode = True
                extendSS=False,                                    # extend the secondary structures
                withAlignment=False,                               # restrict the results with pdb that have a sequences
                onlyC=False,                                       # get only COIL in the IBS
                cathCluster=self.cluster_level,                    # structure redundancy filter
                Uniref=self.uniref_level,                          # sequence redundancy filter
                addSequence=False,                                 # add the non structural data in the IBS/NONIBS dataset
                extendAlign=False,                                 # extend the secondary structure instead of a raw "cut" based on the alignment position
                excludeStrand=False,                               # exclude "strand" From secondary structure
                overide_axis_mode = True,                          # use the z axis instead of the alignment to tag the IBS
                zaxis=self.z_axis_level,                           # z axis plane to define "IBS" or not IBS
                extendCoilOnly = False,                            # extend coil only
                coordinates_folder_name = self.coordinates_folder,
                data_type = self.data_type,
                base_folder= self.base_folder,
                silent=True,
                filter_uniprot_acc = filter_uniprot_acc)           # Give a set of Uniref structure to take for comparison test (between AF and Cath)

    # Set of Uniref structures to take for comparison test (between AF and Cath)
    def _get_uniprot_in_common(self, domain, unique_per_cluster=True):
        cath_data = self._df.query("domain == @domain and data_type == 'cathpdb'")
        af_data = self._df.query("domain == @domain and data_type == 'alphafold'")

        if unique_per_cluster:
            cath_data = self._superfamily_to_pepr2ds[domain].selectUniquePerCluster(cath_data, self.cluster_level, self.uniref_level)
            af_data = self._superfamily_to_pepr2ds[domain].selectUniquePerCluster(af_data, self.cluster_level, self.uniref_level)

        cathcluster_uniprot = cath_data.uniprot_acc.unique()
        afcluster_uniprot = af_data.uniprot_acc.unique()

        return list(set(afcluster_uniprot).intersection(cathcluster_uniprot))

    def _merge_data(self):
        importlib.reload(pepr2ds)
        from pepr2ds.dataset.tagibs import Dataset
        merged = Dataset(self._df, self.settings.PEPRMINT_FOLDER)

        merged.ibs      = pd.concat([self._superfamily_to_pepr2ds[x].ibs for x in self.settings.active_superfamilies])
        merged.nonibs   = pd.concat([self._superfamily_to_pepr2ds[x].nonibs for x in self.settings.active_superfamilies])
        merged.domainDf = pd.concat([self._superfamily_to_pepr2ds[x].domainDf for x in self.settings.active_superfamilies])

        merged.domainLabel = "+".join(self.settings.active_superfamilies)

        # Thibault: "Just in case..."
        merged.domainDf.reset_index(drop=True, inplace=True)
        # Thibault: "clean the resname type"
        merged.domainDf.residue_name = merged.domainDf.residue_name.astype(str)
        merged.domainDf.domain = merged.domainDf.domain.astype(str)
        
        return(merged)

    def _test_num_protrusions(self, ibs_only=False):
        target = self.pepr2ds_dataset.domainDf
        if ibs_only:
            target = target.query("IBS == True")
        target = target.groupby('uniprot_acc')

        num_protrusions = target.apply(lambda x: self.__query_protrusions(x))
        return num_protrusions

    def __query_protrusions(self, group):
        seeking = "is_hydrophobic_protrusion == True and atom_name == 'CB'"
        return len(group.query(seeking))
