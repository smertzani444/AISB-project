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


""" Methods to download/use AlphaFold structures (previously on Notebook #3)

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
import re

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import MDAnalysis as mda
from scipy.spatial import ConvexHull

from biopandas.pdb import PandasPdb
from Bio import AlignIO
from Bio.PDB import PDBParser

import os
import glob
import json
import urllib
from urllib.error import HTTPError, URLError
import requests

from tqdm import tnrange, tqdm
from typing import Optional

from src.settings import Settings
from src.notebook_handle import NotebookHandle

class AlphaFoldUtils:

    def __init__(self, global_settings: Settings):
        self.settings = global_settings
        self.dataset = None

        self._libs_setup()

        self.REBUILD = self.settings.config_file.getboolean(
            'ALPHAFOLD_UTILS', 'rebuild')
        self.use_uniprot_boundaries = self.settings.config_file.getboolean(
            'ALPHAFOLD_UTILS', 'use_uniprot_boundaries')
        self.use_all_AFmodels = self.settings.config_file.getboolean(
            'ALPHAFOLD_UTILS', 'use_all_AFmodels')

        self.REGEX = re.compile("^(\w+)\|(\w+)\/(\d+)-(\d+)")

    def _libs_setup(self):
        # Seaborn
        sns.set_style("darkgrid")

        # Pandas
        pd.options.mode.chained_assignment = (
            None  # remove warning when adding a new column; default='warn'
        )
        pd.set_option("display.max_columns", None)
        tqdm.pandas()   # activate tqdm progressbar for pandas

        # Numpy
        np.seterr(divide='ignore', invalid='ignore')

        # IPython
        if self.settings.USING_NOTEBOOK:
            self.settings.NOTEBOOK_HANDLE.alphafold_utils_options()

    def run(self,
            df: pd.DataFrame,
            EXCLUDE_SEQS: Optional[list] = None,
            EXCLUDE_DOMAIN: Optional[list] = None):

        # fetch and prepare alphafold data (as of Notebook #3)
        print(f"Preparing AlphaFold data")

        self.dataset = df

        if EXCLUDE_SEQS is not None:
            print("User option: excluding list ", end='')
            print(EXCLUDE_SEQS)
        else:
            EXCLUDE_SEQS = []

        if EXCLUDE_DOMAIN is not None:
            print("User option: excluding domain ", end='')
            print(EXCLUDE_DOMAIN)
        else:
            EXCLUDE_DOMAIN = []

        self.extract_domains_seqs_from_alphafold(EXCLUDE_SEQS, EXCLUDE_DOMAIN)
        
        # TO DO: move these to a proper method
        #self._test_printing_msa_dir()
        #self._test_query()
        #self.dataset.groupby("cathpdb")


    """
    ### All methods below just encapsulate the steps in Notebook #3
    """

    def extract_domains_seqs_from_alphafold(self, EXCLUDE_SEQS, EXCLUDE_DOMAIN):
        domains = self.dataset.domain.unique()

        for domain in domains:
            if domain in EXCLUDE_DOMAIN:
                continue

            print(f"----- PROCESSING DOMAIN {domain} -----")

            group = self.dataset.query("domain == @domain")

            uniprot_acc_cathpdb = group.query("data_type == 'cathpdb'").uniprot_acc.unique()

            boundaries_prosite = self._get_prosite_boundaries_dict(domain)

            # 1. fetch PDB files from AlphaFold
            if self.use_all_AFmodels:
                prosite_uniprot_acc = list(boundaries_prosite.keys()) 
                uniprot_acc_cathpdb = [acc for acc in uniprot_acc_cathpdb if acc in prosite_uniprot_acc]

                uniprot_acc_list = prosite_uniprot_acc + uniprot_acc_cathpdb

                seqs_with_model, seqs_without_model = self._fetch_pdb_alphafold(uniprot_acc_list, 
                                                                                domain)
            else:
                seqs_no_pdb = group[group["pdb"].isnull()].uniprot_acc.unique()

                seqs_with_model, seqs_without_model = self._fetch_pdb_alphafold(seqs_no_pdb, 
                                                                                domain)

            # 2. extract the corresponding domain into a separate PDB
            for uniprot_id in tqdm(seqs_with_model, desc = "processing"):
                if uniprot_id in EXCLUDE_SEQS:
                    continue

                raw_pdb_path = f"{self.settings.ALPHAFOLDFOLDER}/{domain}/raw/{uniprot_id}.pdb"
                extracted_pdb_path = f"{self.settings.ALPHAFOLDFOLDER}/{domain}/extracted/{uniprot_id}.pdb"
                # structure = PDBParser().get_structure('uniprot_id',)

                # TO DO: I think this should be extracted_pdb_path - check later
                if os.path.isfile(raw_pdb_path) and self.REBUILD == False:
                    continue   # skip the file if it already exists

                query = self._get_domain_fragment_query(uniprot_id, domain, boundaries_prosite)
                if query == None:
                    continue

                ppdb = PandasPdb().read_pdb(raw_pdb_path)
                ppdb.df["ATOM"] = ppdb.df["ATOM"].query(f"{query}")
                ppdb.to_pdb(extracted_pdb_path)

    def _get_prosite_boundaries_dict(self, domain):
        boundaries = {}
        prosite_ids = self.settings.DOMAIN_PROSITE[domain]

        if type(prosite_ids) != type([]):
            prosite_ids = [prosite_ids]

        for prosite_id in prosite_ids:
            msafilepath = f"{self.settings.PROSITEFOLDER}/msa/{prosite_id}.msa"
            msa = AlignIO.read(msafilepath,'fasta')
            for record in msa:
                seqid = record.id
                match = self.REGEX.match(seqid)
                if match:
                    uniprot_id = match.group(2)
                    start = match.group(3)
                    end = match.group(4)
                    boundaries[uniprot_id] = (int(start),int(end))
        return boundaries

    def _fetch_pdb_alphafold(self, uniprotids, domain):
        nomodels = []
        withmodels = []

        outfolder = f"{self.settings.ALPHAFOLDFOLDER}/{domain}/raw"
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        
        extractedfolder = f"{self.settings.ALPHAFOLDFOLDER}/{domain}/extracted"
        if not os.path.exists(extractedfolder):
            os.makedirs(extractedfolder)
        else:
            if self.REBUILD == True:   #delete extracted files
                files = glob.glob(f"{extractedfolder}/*.pdb")
                for f in files:
                    os.remove(f)
        
        jsonfolder = f"{self.settings.ALPHAFOLDFOLDER}/{domain}/json"
        if not os.path.exists(jsonfolder):
            os.makedirs(jsonfolder)

        prefix = self.settings.config_file['ALPHAFOLD_UTILS']['AF_pdbs_url_prefix']
        suffix = self.settings.config_file['ALPHAFOLD_UTILS']['AF_pdbs_url_suffix']

        ids_iterator = tqdm(uniprotids, desc="Downloading ")

        if self.settings.XP_MODE:
            # the tail of the list contains the ids for entries in cath, so we keep those
            uniprotids.reverse()
            ids_iterator = uniprotids

        for uniprot_id in ids_iterator:
            url = prefix + uniprot_id + suffix
            destination = f"{outfolder}/{uniprot_id}.pdb"
            if not os.path.isfile(destination): 
                try:
                    urllib.request.urlretrieve(url, destination)
                    withmodels.append(uniprot_id)
                    if self.settings.XP_MODE and len(withmodels) >= self.settings.xp_alphafold_limit:
                        break
                except urllib.error.HTTPError as err:
                    nomodels.append(uniprot_id)

        total = len(nomodels) + len(withmodels)
        rate = len(nomodels)/total if len(uniprotids) > 0 else 0
        print(f"{len(nomodels)} out of {total} without AlphaFold models ({rate*100:.2f}%)")

        return withmodels, nomodels

    def _get_domain_fragment_query(self, uniprot_acc, domain, boundaries_prosite):
        start_PS, end_PS = boundaries_prosite[uniprot_acc]
        starts_ends = [boundaries_prosite[uniprot_acc]]

        # TO DO: this looks wrong... starts_ends is REPLACED in each iteration... shouldn't it be EXTENDED?
        # why bother if nothing changes between iterations (e.g. method returns)
        if self.settings.DOMAIN_INTERPRO_REFINE[domain] == True:
            source = 'cathgene3d' if domain == "PLA" else 'ssf'

            interpro = self._get_json(uniprot_acc, domain, source)
            if interpro is None:
                return None

            for result in interpro["results"]:
                if result["metadata"]["accession"] == self.settings.DOMAIN_INTERPRO[domain]:
                    entry_protein_locations = result["proteins"][0]["entry_protein_locations"]
                    for entry in entry_protein_locations:
                        nfrag = len(entry['fragments'])   # number of truncations in the domain
                        
                        if domain == 'PLA':
                            # special case for PLA, we will ignore PROSITE annotation that are actually wrong
                            frag = entry['fragments'][0]   # get first monomer only
                            s = entry['fragments'][0].get('start')
                            e = entry['fragments'][0].get('end')
                            starts_ends = [[s,e]]
                        else:
                            if nfrag >= 2 and ( entry['fragments'][0].get('start') - 50 <= start_PS <= entry['fragments'][0].get('start')+50):
                                # if truncated domain AND correspond to the prosite domain
                                #print(f"splitting {domain}-{uniprot_acc}")
                                queries = []
                                starts_ends = []
                                for frag in entry['fragments']:
                                    s = int(frag.get("start"))
                                    e = int(frag.get("end"))
                                    starts_ends.append([s,e])
                                if self.use_uniprot_boundaries == True:
                                    starts_ends[0][0] = start_PS
                                    starts_ends[-1][-1] = end_PS
                            else:
                                # use prosite fragment
                                starts_ends = [[start_PS, end_PS]]
        
        QueryString = " or ".join([f"({x} <= residue_number <= {y})" for x,y in starts_ends])
        return QueryString

    def _get_json(self, uniprot_acc, domain, source='ssf'):
        jsonfolder = f"{self.settings.ALPHAFOLDFOLDER}/{domain}/json"
        if not os.path.exists(jsonfolder):
            os.makedirs(jsonfolder)
        
        jsonfile = f"{jsonfolder}/{uniprot_acc}.json"
        if os.path.isfile(jsonfile):
            f = open(jsonfile)
            interpro = json.load(f)
        else:
            #make the query on ebi/interpro
            url_prefix = self.settings.config_file['ALPHAFOLD_UTILS']['AF_interpro_url_prefix']
            url_middle = self.settings.config_file['ALPHAFOLD_UTILS']['AF_interpro_url_middle']
            url_suffix = self.settings.config_file['ALPHAFOLD_UTILS']['AF_interpro_url_suffix']
            url = url_prefix + source + url_middle + uniprot_acc + url_suffix

            response = self.__request_URL(url)
            if response == None:
                return None
            try:
                interpro = json.loads(response)
            except:
                print(f"no data for {uniprot_acc}")
                return None
            with open(jsonfile,'w') as out:
                json.dump(interpro, out, indent=2)
                
        return interpro

    def __request_URL(self, link, trial=1):
        try:
            response = requests.get(link).text
            return response
        except URLError as e:
            print(e, link)
            if trial > 3 :
                print('3rd fail, skipping this one')
                return None
            else:
                print(f"Trial {trial}, waiting 10s and trying again")
                sleep(10)
                return self.__request_URL(link, trial=trial+1)

    # TO DO: remove this if OK with the two missing columns: 'add_sequence' and 'format'
    def _test_printing_msa_dir(self):
        for domain in self.dataset.domain.unique():
            prosite_ids = self.settings.DOMAIN_PROSITE[domain]
            if type(prosite_ids) != type([]):
                prosite_ids = [prosite_ids]
            for prosite_id in prosite_ids:
                msafilepath = f"{self.settings.PROSITEFOLDER}/msa/{prosite_id}.msa"
                msa = AlignIO.read(msafilepath, 'fasta')
                print(dir(msa))

    def _test_query(self):
        boundaries = self._get_prosite_boundaries_dict("PLA")
        query = self._get_domain_fragment_query("Q9Z0L3", 'PLA', boundaries)
        assert(query == "(314 <= residue_number <= 435)")

        boundaries = self._get_prosite_boundaries_dict("PH")
        query = self._get_domain_fragment_query('Q55E26', "PH", boundaries)
        assert(query == "(876 <= residue_number <= 942) or (994 <= residue_number <= 1026)")
        #query = self._get_domain_fragment_query('F1LXF1', "PH", boundaries)
        query = self._get_domain_fragment_query("Q54C71", "PH", boundaries)
        assert(query == "(601 <= residue_number <= 822)")
