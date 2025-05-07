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


""" Global settings (previously on Notebook #0)

__author__ = ["Thibault Tubiana", "Phillippe Samer"]
__organization__ = "Computational Biology Unit, Universitetet i Bergen"
__copyright__ = "Copyright (c) 2022 Reuter Group"
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Phillippe Samer"
__email__ = "samer@uib.no"
__status__ = "Prototype"
"""

import os
import sys
import platform
from typing import Optional, Union, Sequence
#from collections.abc import Sequence
import configparser

import pandas as pd
from pandarallel import pandarallel
from tqdm.auto import trange, tqdm

from src.notebook_handle import NotebookHandle

class Settings:

    def __init__(self, path: Optional[str] = None):
        self._DEFAULT_CONFIG_FILE = '{0}/{1}'.format(os.getcwd(),
                                                     'peprmint_default.config')
        self._get_platform()
        self._run_config(path)

        # experimental mode
        self.XP_MODE = self.config_file.getboolean('EXPERIMENTAL_MODE','xp_mode')
        if self.XP_MODE:
            print("\n*** User option: experimental mode on")
            print("*** Warning! This limits the dataset size for experimenting purposes")
            print("***          Use results obtained in this mode with caution")
            print("***          To turn it off, change 'xp_mode' to False in the config file\n")

            self.xp_cath_limit = self.config_file.getint(
                'EXPERIMENTAL_MODE','xp_mode_max_cath_entries_per_domain')
            self.xp_alphafold_limit = self.config_file.getint(
                'EXPERIMENTAL_MODE','xp_mode_max_alphafold_entries_per_domain')
            self.PARALLEL = self.config_file.getboolean(
                'EXPERIMENTAL_MODE', 'xp_mode_allow_parallel')
        else:
            self.xp_cath_limit = None
            self.xp_alphafold_limit = None
            self.PARALLEL = True

        self.num_threads = 1 if not self.PARALLEL else self.config_file.getint(
                'GENERAL','number_of_threads_when_parallel')

        self._libs_setup()
        self._set_active_superfamilies()

        # create directory structure for peprmint
        cwd = self.config_file['GENERAL']['working_folder_path']
        try:
            print('Working directory: {0}'.format(cwd))
            os.makedirs(cwd)
            self.FORMER_WORKING_DIR = False
        except FileExistsError:
            self.FORMER_WORKING_DIR = True

        self.PEPRMINT_FOLDER = cwd
        self.SETUP = {}   # dictionary with ALL parameters
        self.define_folders()
        self.create_directories()
        self.map_cath_and_prosite()

    
    def _is_notebook(self) -> bool:
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # jupyter notebook
            elif shell == 'TerminalInteractiveShell':
                return False  # terminal running IPython
            else:
                return False  # other type (?)
        except NameError:
            return False      # probably standard Python interpreter

    
    def _get_platform(self):
        # OS
        if platform.system() == "Linux":
            self.OS = "linux"
        elif platform.system() == "Darwin":
            self.OS = "macos"
        elif platform.system() == "Windows":
            self.OS = "windows"
        else:
            self.OS = None
        
        # Jupyter notebook or not
        self.USING_NOTEBOOK = self._is_notebook()
        if self.USING_NOTEBOOK:
            self.NOTEBOOK_HANDLE = NotebookHandle()
        else:
            self.NOTEBOOK_HANDLE = None

    
    def _run_config(self, path: Optional[str] = None):
        """ Read configuration file
        First, try to read the user configurations, either at the standard file
        or at the location given in the optional constructor parameter.
        If neither of these work, genereate the original config file.
        """
        config_ready = False

        if path is not None:
            config_ready = self._read_config_file(path)
            if config_ready:
                print("Using configuration file '{0}'".format(path))
            else:
                print("Could not find configuration file '{0}'".format(path))

        if not config_ready:
            print("Reading standard configuration file... ", end = '')
            config_ready = self._read_config_file(self._DEFAULT_CONFIG_FILE)
            if config_ready:
                print("done")
            else:
                print("not found")
                print("Using factory configuration, saved locally at '{0}'".format(self._DEFAULT_CONFIG_FILE))
                self._write_default_config_file(self._DEFAULT_CONFIG_FILE)

    
    def _read_config_file(self, path: str) -> bool:
        if not os.path.isfile(path):
            return False

        # TO DO: maybe check if given file indeed has all of the expected sections and fields?
        self.config_file = configparser.ConfigParser(allow_no_value=True)
        self.config_file.optionxform = str
        self.config_file.read(path)
        return True

    
    def _write_default_config_file(self, path: str):
        self.config_file = configparser.ConfigParser(allow_no_value=True)
        self.config_file.optionxform = str

        self.config_file['GENERAL'] = {}
        # default folder: a new one in the current working directory
        self.config_file['GENERAL']['working_folder_path'] = '{0}/data'.format(os.getcwd())
        self.config_file['GENERAL']['number_of_threads_when_parallel'] = "4"
        self.config_file['GENERAL']['include_PH'] = str(True)
        self.config_file['GENERAL']['include_C2'] = str(True)
        self.config_file['GENERAL']['include_C1'] = str(True)
        self.config_file['GENERAL']['include_PX'] = str(False)
        self.config_file['GENERAL']['include_FYVE'] = str(False)
        self.config_file['GENERAL']['include_BAR'] = str(False)
        self.config_file['GENERAL']['include_ENTH'] = str(False)
        self.config_file['GENERAL']['include_SH2'] = str(False)
        self.config_file['GENERAL']['include_SEC14'] = str(False)
        self.config_file['GENERAL']['include_START'] = str(False)
        self.config_file['GENERAL']['include_C2DIS'] = str(False)
        self.config_file['GENERAL']['include_GLA'] = str(False)
        self.config_file['GENERAL']['include_PLD'] = str(False)
        self.config_file['GENERAL']['include_PLA'] = str(False)
        self.config_file['GENERAL']['include_ANNEXIN'] = str(False)

        self.config_file['EXPERIMENTAL_MODE'] = {}
        self.config_file['EXPERIMENTAL_MODE']['xp_mode'] = str(True)
        self.config_file['EXPERIMENTAL_MODE']['xp_mode_allow_parallel'] = "False"
        self.config_file['EXPERIMENTAL_MODE']['xp_mode_max_cath_entries_per_domain'] = "3"
        self.config_file['EXPERIMENTAL_MODE']['xp_mode_max_alphafold_entries_per_domain'] = "3"

        self.config_file['PREPROCESSING'] = {}
        self.config_file['PREPROCESSING']['overwrite_original_pdbs'] = str(False)
        self.config_file['PREPROCESSING']['keep_ssap_alignment_files'] = str(False)
        self.config_file['PREPROCESSING']['ref_PH_pdb'] = "2da0A00"
        self.config_file['PREPROCESSING']['ref_PH_res1'] = "19"
        self.config_file['PREPROCESSING']['ref_PH_res2'] = "42"
        self.config_file['PREPROCESSING']['ref_PH_res3'] = "50"
        self.config_file['PREPROCESSING']['ref_C2_pdb'] = "1rsyA00"
        self.config_file['PREPROCESSING']['ref_C2_res1'] = "169"
        self.config_file['PREPROCESSING']['ref_C2_res2'] = "178"
        self.config_file['PREPROCESSING']['ref_C2_res3'] = "237"
        self.config_file['PREPROCESSING']['ref_C1_pdb'] = "1ptrA00"
        self.config_file['PREPROCESSING']['ref_C1_res1'] = "243"
        self.config_file['PREPROCESSING']['ref_C1_res2'] = "257"
        self.config_file['PREPROCESSING']['ref_C1_res3'] = "237"
        self.config_file['PREPROCESSING']['ref_PX_pdb'] = "1h6hA00"
        self.config_file['PREPROCESSING']['ref_PX_res1'] = "33"
        self.config_file['PREPROCESSING']['ref_PX_res2'] = "74"
        self.config_file['PREPROCESSING']['ref_PX_res3'] = "100"
        self.config_file['PREPROCESSING']['ref_FYVE_pdb'] = "1jocA02"
        self.config_file['PREPROCESSING']['ref_FYVE_res1'] = "1373"
        self.config_file['PREPROCESSING']['ref_FYVE_res2'] = "1392"
        self.config_file['PREPROCESSING']['ref_FYVE_res3'] = "1382"
        # TO DO: set a representative for the BAR domain
        self.config_file.set('PREPROCESSING', 'ref_BAR_pdb') 
        self.config_file.set('PREPROCESSING', 'ref_BAR_res1')  
        self.config_file.set('PREPROCESSING', 'ref_BAR_res2')  
        self.config_file.set('PREPROCESSING', 'ref_BAR_res3')
        self.config_file['PREPROCESSING']['ref_ENTH_pdb'] = "1h0aA00"
        self.config_file['PREPROCESSING']['ref_ENTH_res1'] = "17"
        self.config_file['PREPROCESSING']['ref_ENTH_res2'] = "70"
        self.config_file['PREPROCESSING']['ref_ENTH_res3'] = "116"
        self.config_file['PREPROCESSING']['ref_SH2_pdb'] = "2oq1A03"
        self.config_file['PREPROCESSING']['ref_SH2_res1'] = "180"
        self.config_file['PREPROCESSING']['ref_SH2_res2'] = "209"
        self.config_file['PREPROCESSING']['ref_SH2_res3'] = "243"
        # TO DO: set a representative for the SEC14 domain
        self.config_file.set('PREPROCESSING', 'ref_SEC14_pdb') 
        self.config_file.set('PREPROCESSING', 'ref_SEC14_res1')  
        self.config_file.set('PREPROCESSING', 'ref_SEC14_res2')  
        self.config_file.set('PREPROCESSING', 'ref_SEC14_res3')
        comment_on_alignment_START = "# alternative orientation for START: 567 470 509"
        self.config_file.set('PREPROCESSING', comment_on_alignment_START)
        self.config_file['PREPROCESSING']['ref_START_pdb'] = "2e3mA00"
        self.config_file['PREPROCESSING']['ref_START_res1'] = "412"
        self.config_file['PREPROCESSING']['ref_START_res2'] = "448"
        self.config_file['PREPROCESSING']['ref_START_res3'] = "515"
        self.config_file['PREPROCESSING']['ref_C2DIS_pdb'] = "1czsA00"
        self.config_file['PREPROCESSING']['ref_C2DIS_res1'] = "23"
        self.config_file['PREPROCESSING']['ref_C2DIS_res2'] = "76"
        self.config_file['PREPROCESSING']['ref_C2DIS_res3'] = "45"
        # TO DO: set a representative for the GLA domain
        self.config_file.set('PREPROCESSING', 'ref_GLA_pdb') 
        self.config_file.set('PREPROCESSING', 'ref_GLA_res1')  
        self.config_file.set('PREPROCESSING', 'ref_GLA_res2')  
        self.config_file.set('PREPROCESSING', 'ref_GLA_res3')
        comment_on_alignment_PLD = "# alternative orientation (CAGE instead of OPM) for PLD: 53 41 99"
        self.config_file.set('PREPROCESSING', comment_on_alignment_PLD)
        self.config_file['PREPROCESSING']['ref_PLD_pdb'] = "3rlhA00"
        self.config_file['PREPROCESSING']['ref_PLD_res1'] = "59"
        self.config_file['PREPROCESSING']['ref_PLD_res2'] = "205"
        self.config_file['PREPROCESSING']['ref_PLD_res3'] = "198"
        self.config_file['PREPROCESSING']['ref_PLA_pdb'] = "1pocA00"
        self.config_file['PREPROCESSING']['ref_PLA_res1'] = "7"
        self.config_file['PREPROCESSING']['ref_PLA_res2'] = "92"
        self.config_file['PREPROCESSING']['ref_PLA_res3'] = "76"
        self.config_file['PREPROCESSING']['ref_ANNEXIN_pdb'] = "1a8aA01"
        self.config_file['PREPROCESSING']['ref_ANNEXIN_res1'] = "25"
        self.config_file['PREPROCESSING']['ref_ANNEXIN_res2'] = "68"
        self.config_file['PREPROCESSING']['ref_ANNEXIN_res3'] = "77"

        self.config_file['CATH'] = {}
        self.config_file['CATH']['version'] = 'v4_2_0'
        self.config_file['CATH']['domain_list_url'] = "http://download.cathdb.info/cath/releases/all-releases/{0}/cath-classification-data/cath-domain-list-{0}.txt".format(self.config_file['CATH']['version'])
        self.config_file['CATH']['fetch_pdb_url'] = "http://www.cathdb.info/version/{0}/api/rest/id/".format(self.config_file['CATH']['version'])
        comment_on_superpose = "# check https://github.com/UCLOrengoGroup/cath-tools/releases/latest"
        self.config_file.set('CATH', comment_on_superpose)
        self.config_file['CATH']['superpose_url_linux'] = "https://github.com/UCLOrengoGroup/cath-tools/releases/download/v0.16.10/cath-superpose.ubuntu-20.04"
        self.config_file['CATH']['superpose_url_macos'] = "https://github.com/UCLOrengoGroup/cath-tools/releases/download/v0.16.10/cath-superpose.macos-10.15"

        self.config_file['UNIPROT'] = {}
        self.config_file['UNIPROT']['url'] = "ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz"

        self.config_file['PROSITE'] = {}
        self.config_file['PROSITE']['url'] = "ftp://ftp.expasy.org/databases/prosite/prosite_alignments.tar.gz"

        self.config_file['DATASET_MANAGER'] = {}
        self.config_file['DATASET_MANAGER']['recalculate'] = str(False)

        self.config_file['ALPHAFOLD_UTILS'] = {}
        self.config_file['ALPHAFOLD_UTILS']['rebuild'] = str(True)
        self.config_file['ALPHAFOLD_UTILS']['use_uniprot_boundaries'] = str(True)
        self.config_file['ALPHAFOLD_UTILS']['use_all_AFmodels'] = str(True)
        self.config_file['ALPHAFOLD_UTILS']['AF_pdbs_url_prefix'] = "https://alphafold.ebi.ac.uk/files/AF-"
        self.config_file['ALPHAFOLD_UTILS']['AF_pdbs_url_suffix'] = "-F1-model_v1.pdb"
        self.config_file['ALPHAFOLD_UTILS']['AF_interpro_url_prefix'] = "https://www.ebi.ac.uk/interpro/api/entry/"
        self.config_file['ALPHAFOLD_UTILS']['AF_interpro_url_middle'] = "/protein/reviewed/"
        self.config_file['ALPHAFOLD_UTILS']['AF_interpro_url_suffix'] = "/?page_size=200"

        self.config_file['IBS_TAGGING'] = {}
        self.config_file['IBS_TAGGING']['comparison_mode'] = str(False)
        self.config_file['IBS_TAGGING']['cluster_level'] = "S100"
        self.config_file['IBS_TAGGING']['uniref_level'] = "uniref100"
        self.config_file['IBS_TAGGING']['z_axis_level'] = "0"

        with open(path, 'w') as configfile:
            self.config_file.write(configfile)

    
    def _libs_setup(self):
        # Place any additional settings for imported libraries here

        # Pandas
        tqdm.pandas()   # activate tqdm progressbar for pandas

        # try to use pandarallel (only on GNU/Linux and OS X?)
        if self.PARALLEL:
            try:
                pandarallel.initialize(nb_workers = self.num_threads,
                                       progress_bar = True)
            except:
                self.PARALLEL = False
                self.num_threads = 1

    
    def _set_active_superfamilies(self):
        self.use_PH = self.config_file.getboolean('GENERAL','include_PH')
        self.use_C2 = self.config_file.getboolean('GENERAL','include_C2')
        self.use_C1 = self.config_file.getboolean('GENERAL','include_C1')
        self.use_PX = self.config_file.getboolean('GENERAL','include_PX')
        self.use_FYVE = self.config_file.getboolean('GENERAL','include_FYVE')
        self.use_BAR = self.config_file.getboolean('GENERAL','include_BAR')
        self.use_ENTH = self.config_file.getboolean('GENERAL','include_ENTH')
        self.use_SH2 = self.config_file.getboolean('GENERAL','include_SH2')
        self.use_SEC14 = self.config_file.getboolean('GENERAL','include_SEC14')
        self.use_START = self.config_file.getboolean('GENERAL','include_START')
        self.use_C2DIS = self.config_file.getboolean('GENERAL','include_C2DIS')
        self.use_GLA = self.config_file.getboolean('GENERAL','include_GLA')
        self.use_PLD = self.config_file.getboolean('GENERAL','include_PLD')
        self.use_PLA = self.config_file.getboolean('GENERAL','include_PLA')
        self.use_ANNEXIN = self.config_file.getboolean('GENERAL','include_ANNEXIN')

        self.active_superfamilies = []

        if self.use_PH:
            self.active_superfamilies.append("PH")
        if self.use_C2:
            self.active_superfamilies.append("C2")
        if self.use_C1:
            self.active_superfamilies.append("C1")
        if self.use_PX:
            self.active_superfamilies.append("PX")
        if self.use_FYVE:
            self.active_superfamilies.append("FYVE")
        if self.use_BAR:
            self.active_superfamilies.append("BAR")
        if self.use_ENTH:
            self.active_superfamilies.append("ENTH")
        if self.use_SH2:
            self.active_superfamilies.append("SH2")
        if self.use_SEC14:
            self.active_superfamilies.append("SEC14")
        if self.use_START:
            self.active_superfamilies.append("START")
        if self.use_C2DIS:
            self.active_superfamilies.append("C2DIS")
        if self.use_GLA:
            self.active_superfamilies.append("GLA")
        if self.use_PLD:
            self.active_superfamilies.append("PLD")
        if self.use_PLA:
            self.active_superfamilies.append("PLA")
        if self.use_ANNEXIN:
            self.active_superfamilies.append("ANNEXIN")

    
    def define_folders(self):
        self.WORKDIR = f"{self.PEPRMINT_FOLDER}/dataset/"
        self.CATHFOLDER = f"{self.PEPRMINT_FOLDER}/databases/cath/"
        self.REF_FOLDER = f"{self.PEPRMINT_FOLDER}/databases/cath/ref/"
        self.ALPHAFOLDFOLDER = f"{self.PEPRMINT_FOLDER}/databases/alphafold/"
        self.PROSITEFOLDER = f"{self.PEPRMINT_FOLDER}/databases/prosite/"
        self.UNIPROTFOLDER = f"{self.PEPRMINT_FOLDER}/databases/uniprot/"
        self.FIGURESFOLDER = f"{self.PEPRMINT_FOLDER}/figures/"

        # used by default; ignored when "overwrite_original_pdbs" (under preprocessing in the config file) is true
        self.ALIGNED_SUBDIR = "zaligned"

        # used by default; ignored when tagging IBS on CATH or AF data separately
        self.ALIGNED_CATH_AND_AF = "merged_aligned_cath_and_af"

        self.SETUP["PEPRMINT_FOLDER"] = self.PEPRMINT_FOLDER
        self.SETUP["WORKDIR"] = self.WORKDIR
        self.SETUP["CATHFOLDER"] = self.CATHFOLDER
        self.SETUP["PROSITEFOLDER"] = self.PROSITEFOLDER
        self.SETUP["ALPHAFOLDFOLDER"] = self.ALPHAFOLDFOLDER
        self.SETUP["UNIPROTFOLDER"] = self.UNIPROTFOLDER
        self.SETUP["FIGURESFOLDER"] = self.FIGURESFOLDER

        for k in self.SETUP:
            exec(f"self.{k}2 = self.SETUP['{k}']")

    
    def create_directories(self):
        if not os.path.exists(self.PEPRMINT_FOLDER):
            os.makedirs(self.PEPRMINT_FOLDER)
        if not os.path.exists(self.WORKDIR):
            os.makedirs(self.WORKDIR)
        if not os.path.exists(self.FIGURESFOLDER):
            os.makedirs(self.FIGURESFOLDER)
        if not os.path.exists(self.ALPHAFOLDFOLDER):
            os.makedirs(self.ALPHAFOLDFOLDER)
        if not os.path.exists(self.UNIPROTFOLDER):
            os.makedirs(self.UNIPROTFOLDER)
        if not os.path.exists(self.PROSITEFOLDER):
            #MSA will contains the alignments in "msa" format (FASTA)
            os.makedirs(self.PROSITEFOLDER)
        if not os.path.exists(self.CATHFOLDER):
            os.makedirs(self.CATHFOLDER)
        if not os.path.exists(self.REF_FOLDER):
            os.makedirs(self.REF_FOLDER)


    def map_cath_and_prosite(self):
        self.CATHVERSION = self.config_file['CATH']['version']

        self.DOMAIN_PROSITE = {
            "PH": "PS50003",
            "C2": ["PS50004","PS51547"],
            "C1": "PS50081",  # Note : no C1 prosite on SMART but 2 C1 ProSite on Interprot (PS50081,PS00479), I took PS50081 since the data in PS00479 are in PS50081.
            "PX": "PS50195",
            # "FYVE":"PS50178",
            "FYVE": ["PS50178",'PS50089', 'PS00518','PS50016','PS01359','PS50014','PS00633','PS50119'],  # FYVE CAN BE THIS ONE TOO....
            # "PPASE_MYOTUBULARIN":"PS51339",# no GRAM domain found on prosite. Has to do this manually. Go on http://smart.embl-heidelberg.de/smart/do_annotation.pl?DOMAIN=GRAM&BLAST=DUMMY
            "BAR": "PS51021",  # 1URU is missing on prosite
            # "GLA":"PS50963",
            "ENTH": "PS50942",
            "SH2": "PS50001",
            "SEC14": "PS50191",
            "START": "PS50848",
            "C2DIS":"PS50022",
            "GLA": "PS50998",
            "PLD":"PS50035",
            "PLA":"PS00118",
            "ANNEXIN":"PS00223",
        }

        self._filter_active_domains(self.DOMAIN_PROSITE)

        # Invert keys and values to have PROSITEID ==> DOMAIN
        self.PROSITE_DOMAIN = {}
        for key, value in self.DOMAIN_PROSITE.items():
            if type(value) == type([]):
                for subvalues in value:
                    self.PROSITE_DOMAIN[subvalues] = key
            else:
                self.PROSITE_DOMAIN[value] = key
        # self.PROSITE_DOMAIN = {v: k for k, v in self.DOMAIN_PROSITE.items()}

        self.DOMAIN_CATH = {
            "PH": "2.30.29.30",
            "C2": "2.60.40.150",
            "C1": "3.30.60.20",
            "PX": "3.30.1520.10",
            "FYVE": "3.30.40.10",
            "BAR": "1.20.1270.60",
            "ENTH": "1.25.40.90",
            "SH2": "3.30.505.10",
            "SEC14": "3.40.525.10",
            "START": "3.30.530.20",
            "C2DIS": "2.60.120.260",
            "GLA":"2.40.20.10",
            "PLD":"3.20.20.190",
            "PLA":"1.20.90.10",
            "ANNEXIN":"1.10.220.10",
        }

        self._filter_active_domains(self.DOMAIN_CATH)

        self.DOMAIN_INTERPRO = {
            "PH": "SSF50729",
            "C2": "SSF49562",
            "C1": None,
            "PX": "SSF64268",
            "FYVE": "SSF57903", #badly classified it looks like...
            "BAR": "SSF103657",
            "ENTH": "SSF48464",
            "SH2": "SSF55550",
            "SEC14": ["SSF52087","SSF46938"], #the CRAL TRIO domain is truncated in SSF.
            "START": "SSF55961",
            "C2DIS": "SSF49785",
            "GLA":None,
            "PLD":"SSF51695",
            "PLA":"G3DSA:1.20.90.10",
            "ANNEXIN":"SSF47874",
        }

        self._filter_active_domains(self.DOMAIN_INTERPRO)

        self.DOMAIN_INTERPRO_REFINE = {
            "PH": True,
            "C2": False,
            "C1": False,
            "PX": True,
            "FYVE": False,
            "BAR": False,
            "ENTH": False,
            "SH2": False,
            "SEC14": False,
            "START": True,
            "C2DIS": False,
            "GLA":False,
            "PLD":False,
            "PLA":True,
            "ANNEXIN":False,
        }

        self._filter_active_domains(self.DOMAIN_INTERPRO_REFINE)
        self._update_pepr2ds_setup_indices()


    def _filter_active_domains(self, d: dict):
        # keep only the entries for the active domains
        inactive_keys = [k for k in d.keys() if k not in self.active_superfamilies]
        for key in inactive_keys:
            del d[key]


    def _update_pepr2ds_setup_indices(self):
        # keys used by the constructors of pepr2ds.Builder and pepr2ds.Dataset
        self.CATH_DOMAIN = {}   # inverted index cath id -> superfamily
        for k, v in self.DOMAIN_CATH.items():
            if type(v) == str:
                self.CATH_DOMAIN[v] = k
            else:
                for entry in v:
                    self.CATH_DOMAIN[entry] = k

        self.SUPERFAMILY = self.CATH_DOMAIN
        self.SETUP["DOMAIN_PROSITE"] = self.DOMAIN_PROSITE
        self.SETUP["PROSITE_DOMAIN"] = self.PROSITE_DOMAIN
        self.SETUP["DOMAIN_CATH"] = self.DOMAIN_CATH
        self.SETUP["CATH_DOMAIN"] = self.CATH_DOMAIN
        self.SETUP["SUPERFAMILY"] = self.SUPERFAMILY


    def add_new_superfamily(self,
                            name: str,
                            ref_pdb: str,   # representative for reorientation
                            ref_res1: str,
                            ref_res2: str,  # 3 residues for reorientation along the z axis
                            ref_res3: str,
                            cath_domain: Union[str, Sequence[str]],
                            prosite_domain: Union[str, Sequence[str]],
                            interpro_domain: Union[str, Sequence[str]],
                            refine_AF_data_with_interpro: bool):
        """
        NB! None of the data structures initialized with the current Settings
        object reflect the new superfamily added! It might suffice to execute
        Preprocessing.run(database="cath")
        DatasetManager.build(recalculate=True)
        DatasetManager.fetch_alphafold_data()
        Preprocessing.run(database="alphafold")
        DatasetManager.build(recalculate=True)
        DatasetManager.add_IBS_data(db="cath+af")
        """
        self.active_superfamilies.append(name)

        self.config_file['PREPROCESSING']['ref_'+name+'_pdb'] = ref_pdb
        self.config_file['PREPROCESSING']['ref_'+name+'_res1'] = ref_res1
        self.config_file['PREPROCESSING']['ref_'+name+'_res2'] = ref_res2
        self.config_file['PREPROCESSING']['ref_'+name+'_res3'] = ref_res3

        self.DOMAIN_CATH[name] = cath_domain
        self.DOMAIN_INTERPRO[name] = interpro_domain
        self.DOMAIN_PROSITE[name] = prosite_domain

        if type(prosite_domain) == str:
            self.PROSITE_DOMAIN[prosite_domain] = name
        else:
            for entry in prosite_domain:
                self.PROSITE_DOMAIN[entry] = name

        self.DOMAIN_INTERPRO_REFINE[name] = refine_AF_data_with_interpro

        self._update_pepr2ds_setup_indices()
