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


""" Methods to generate figures for data analyses (previously on Notebook #4)

In particular, this implementation was used to generate figures in the 2022 
paper "Dissecting peripheral protein-membrane interfaces" by Tubiana, Sillitoe, 
Orengo, Reuter: https://doi.org/10.1371/journal.pcbi.1010346

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
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
import MDAnalysis as mda
from decimal import Decimal
import scipy.stats as stats

from tqdm import tnrange, tqdm

import urllib
import glob
from urllib.error import HTTPError
from pathlib import Path

from src.settings import Settings
from src.notebook_handle import NotebookHandle
from pepr2ds.dataset.tagibs import Dataset

class FigureGenerator:

    def __init__(self, global_settings: Settings, tagged_dataset: Dataset):
        self.settings = global_settings
        self.pepr2ds = tagged_dataset
        
        self._silent_eq_test = False

        # Warning
        if self.settings.use_ENTH:
            print("\n***")
            print("\n*** Warning: as of 2022, the authors did not include")
            print("\n***          entries in the ENTH superfamilly in the")
            print("\n***          figures; consider removing them before")
            print("\n***          proceeding with analysis/figure generation")
            print("\n***")

        self._libs_setup()
        self._palette_setup()
        self._data_setup()

    def _libs_setup(self):
        # IPython
        if self.settings.USING_NOTEBOOK:
            self.settings.NOTEBOOK_HANDLE.figure_generator_options()
        
        # Pandas
        pd.options.mode.chained_assignment = (
            None  # remove warning when adding a new column; default='warn'
        )
        pd.set_option("display.max_columns", None)
        tqdm.pandas()   # activate tqdm progressbar for pandas

        # Numpy
        np.seterr(divide='ignore', invalid='ignore')

        # Seaborn
        sns.set_style("whitegrid")

        # Matplotlib
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams["font.family"] = "Arial"

    def _palette_setup(self):
        self.dict_palette_IBS = {"IBS":"#A5CCE2",'nonIBS':"#DFA3AE"}
        self.palette_IBS = [self.dict_palette_IBS["nonIBS"],self.dict_palette_IBS["IBS"]]
        self.FORMAT="tif"
        
        self.colorsPerType = {"Positive":"tab:blue",
                 "Negative":"tab:red",
                 "Non-polar":"tab:gray",
                 "Hydrophobic,H-non-aromatic":"tab:brown",
                 "Hydrophobic,H-aromatic":"tab:pink",
                 "Polar":"tab:green"}

        #From http://acces.ens-lyon.fr/biotic/rastop/help/colour.htm#shapelycolours
        self.COLORS_taylor = {
            "LEU": "#33FF00",
            "ILE": "#66FF00",
            "CYS": "#FFFF00",
            "MET": "#00FF00",
            "TYR": "#00FFCC",
            "TRP": "#00CCFF",
            "PHE": "#00FF66",
            "HIS": "#0066FF",
            "LYS": "#6600FF",
            "ARG": "#0000FF",
            "ASP": "#FF0000",
            "GLU": "#FF0066",
            "VAL": "#99FF00",
            "ALA": "#CCFF00",
            "GLY": "#FF9900",
            "PRO": "#FFCC00",
            "SER": "#FF3300",
            "ASN": "#CC00FF",
            "GLN": "#FF00CC",
            "THR": "#FF6600",
            "UNK": "#000000"
        }

    def _data_setup(self):
        # Thibault: "Temporary fix: redefine type to have HIS as polar"
        self._AATYPE = {
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
        self.pepr2ds.domainDf["type"] = self.pepr2ds.domainDf.residue_name.apply(
            lambda x: self._AATYPE[x])

        # backup info
        self._backup_all = self.pepr2ds.domainDf.copy()

        self._data_type = list(self.pepr2ds.domainDf.data_type.unique())

        number_of_HP_per_pdbs = self.pepr2ds.domainDf.query('IBS == True and atom_name == "CB"').groupby("cathpdb").progress_apply(lambda group: self._count_hydr_protr_perpdb(group))
        nohydrprotrpdbs = list(number_of_HP_per_pdbs[number_of_HP_per_pdbs==0].index)
        self._backup_prot_HPIBS = self.pepr2ds.domainDf.query("cathpdb not in @nohydrprotrpdbs")
        self._backup_prot_noHPIBS = self.pepr2ds.domainDf.query("cathpdb in @nohydrprotrpdbs")
    
    """
    ### AUXILIARY METHODS
    """

    def _count_hydr_protr_perpdb(self, group):
        g = group.query("protrusion == True and atom_name == 'CB' and type.str.contains('Hydrophobic')",
                        engine="python")
        return (len(g))

    def _save_fig(self,
                  figname, 
                  folder="article", 
                  format="png", 
                  dpi=300, 
                  bbox_inches='tight', 
                  transparent=False, ):

        Path(f"{self.settings.FIGURESFOLDER}/{folder}").mkdir(parents=True, exist_ok=True)

        if format in ["tiff",'tif']:
            plt.savefig(f"{self.settings.FIGURESFOLDER}/{folder}/{figname}.{format}", dpi=dpi, bbox_inches=bbox_inches, transparent=transparent, pil_kwargs={"compression": "tiff_lzw"})
        else:
            plt.savefig(f"{self.settings.FIGURESFOLDER}/{folder}/{figname}.{format}", dpi=dpi, bbox_inches=bbox_inches, transparent=transparent)

    def _get_palette_OR(self, oddsratio):
        palette = []
        for index, row in oddsratio.iterrows():
            if row["pvalue"] < 0.05:
                if row.oddsratio > 0:
                    palette.append(self.dict_palette_IBS["IBS"])
                else:
                    palette.append(self.dict_palette_IBS["nonIBS"])
            else:
                palette.append("gray")
        return(palette)

    def _significance(self, ortable):
        def __add_symbol(row):
            if row.pvalue > 0.05:
                return "ns"
            elif row.pvalue <= 0.0001:
                return '****'
            elif row.pvalue <= 0.001:
                return '***'
            elif row.pvalue <= 0.01:
                return '**'
            elif row.pvalue <= 0.05:
                return '*'
            else:
                return 'NA'
            
        ortable["significance"] = ortable.apply(lambda x: __add_symbol(x), axis=1)
        ortable["labels"] ="("+ortable["significance"]+") " + ortable.iloc[:,0].astype(str)
        return ortable
    
    def _printmd(self, xstring, color=None):
        if self._silent_eq_test:
            return
        else:
            if self.settings.USING_NOTEBOOK:
                colorstr = "<span style='color:{}'>{}</span>".format(color, xstring)
                display(Markdown(colorstr))
            else:
                print(xstring)
                print("")

    def _equality_test(self,
                       pop1,                         # array with continuous or discrete data
                       pop2,                         # array with continuous or discrete data
                       ALTERNATIVE = "two-sided",    # 'two-sided', 'less' or 'greater'
                       pairwised = False,
                       silent = False,
                       returnPval = False):

        self._silent_eq_test = silent

        print("\n----------------------------------------")
        self._printmd("**STATISTICAL TEST BETWEEN TWO SAMPLES**")
        self._printmd(f" - ALTERNATIVE HYPOTHESIS = {ALTERNATIVE}")
        
        sign = {"two-sided":"≠",
           "less":"<",
           "greater":">"}

        self._printmd("**NORMALITY TEST (shapiro)**")
        normality = True
        self._printmd("*The two samples should follow a normal law to use a standard t.test*")

        normpop1 = stats.shapiro(pop1).pvalue
        normpop2 = stats.shapiro(pop2).pvalue
        if normpop1 < 0.05:
            self._printmd(f"---- Sample 1 shapiro test pvalue = {normpop1:.2E}, <= 0.05. This sample does NOT follow a normal law", color='red')
            normality = False
        else: 
            self._printmd(f"---- Sample 1 shapiro test pvalue = {normpop1:.2E}, > 0.05. This sample follows a normal law", color='blue')
        if normpop2 < 0.05:
            self._printmd(f"---- Sample 1 shapiro test pvalue = {normpop2:.2E}, <= 0.05. This sample does NOT follow a normal law", color='red')
            normality = False
        else: 
            self._printmd(f"---- Sample 1 shapiro test pvalue = {normpop2:.2E}, > 0.05. This sample follows a normal law", color='blue')

        if normality == True:
            self._printmd("Both samples follow a normal law")

            if pairwised == True:
                self._printmd("**TTest_REL Pairwise test **")
                equalstat, equalpval = stats.ttest_rel(pop1,pop2)
                if returnPval:
                    print("-- DONE\n\n")
                    return equalpval
            else: 
                print("Performing variance equality test")
                varstat, varpval = stats.levene(pop1,pop2)
                #Levene, pval < 0.05 --> Variances not equal.
                #Levene, pval > 0.05 --> Not significative and the hypothesis H0 is not rejected

                self._printmd("-- Null hypothesis : the variance of both samples are equal")
                print(f"---- Variance test --> stat={varstat:.2E}, p-value={varpval:.3E}")
                if varpval < 0.05:
                    self._printmd("P value <= 0.05, H0 rejected. The variances are not equal. Performing Welch’s t-test", color="red")
                    equal_var = False
                else:
                    self._printmd("Pvalue > 0.05, the variances are not equal. Performing a standard independent 2-sample test", color="blue")
                    equal_var = True
                equalstat, equalpval = stats.ttest_ind(pop1,
                                           pop2,
                                           equal_var=equal_var,)
                
            print(f"t-test --> stat={equalstat:.2E}, p-value={equalpval:.3E}")
            print(f"  Null hypothesis: the averages of both samples are equal")
            print(f"  Alternative hypothesis: average(sample2) {sign[ALTERNATIVE]} average(sample2)")
            if equalpval > 0.05:
                self._printmd("pvalue > 0.05, we cannot reject the null hypothesis of identical averages between both populations", color="blue")
            else:
                self._printmd("pvalue <= 0.05, the null hypothesis is rejected, the two samples are different", color="red")
        else:
            self._printmd("At least one sample does not follow a normal law")
            if pairwised==True:
                self._printmd("**WILCOXON SIGNED-RANK TEST**")
                self._printmd(f"  Null hypothesis: the two distributions are equal")
                self._printmd(f"  Alternative hypothesis: pop1 {sign[ALTERNATIVE]} pop2")
                stat, pval = stats.wilcoxon(pop1,pop2, alternative=ALTERNATIVE)
            else:
                self._printmd("Performing a Wilcoxon rank sum test with continuity correction")
                self._printmd("**WILCOXON RANK SUM TEST WITH CONTINUITY CORRECTION (or Mann-Whitney test)**")
                self._printmd(f"  Null hypothesis: the two distributions are equal")
                self._printmd(f"  Alternative hypothesis: pop1 {sign[ALTERNATIVE]} pop2")
                stat, pval = stats.mannwhitneyu(pop1,pop2, alternative=ALTERNATIVE)
            if pval < 0.05:
                self._printmd(f"  pvalue = {pval:.2E} which is <= 0.05. The null hypothesis is rejected. The alternative hypothesis (f{ALTERNATIVE}) is valid and the two distributions are statistically different", color="red")
            else:
                self._printmd(f"  pvalue = {pval:.2E} which is > 0.05. Null hypothesis not rejected. Both distributions are NOT statistically different", color="blue")
            if returnPval:
                print("-- DONE\n\n")
                return pval
        
        print("-- DONE\n")

    def _move_seaborn_legend(self, ax, new_loc, title=None, invert=True, order=None, **kws):
        # from https://github.com/mwaskom/seaborn/issues/2280#issuecomment-692350136
        old_legend = ax.legend_
        handles = old_legend.legendHandles
        labels = [t.get_text() for t in old_legend.get_texts()]

        labels_handles = dict(zip(labels,handles))

        if invert:
            handles=handles[::-1]
            labels=labels[::-1]
        if title == None:
            title = old_legend.get_title().get_text()

        if order:
            handles = [labels_handles[x] for x in order]
            labels = order

        ax.legend(handles, labels, loc=new_loc, title=title, **kws)

    """
    ### All methods below just encapsulate the steps in Notebook #4
    """

    #############################################################################
    # Figure 2 in the 2022 paper: composition of the exposed IBS for all proteins
    def make_figure_composition_of_exposed_IBS(self, fig_filename="Fig 2"):
        dataCathIBS = self.pepr2ds.domainDf.query("IBS == True and exposed == True").drop_duplicates(['residue_name', 'residue_number', 'cathpdb']).residue_name.value_counts(normalize=True)*100
        dataCathnoIBS = self.pepr2ds.domainDf.query("IBS == False and exposed == True").drop_duplicates(['residue_name', 'residue_number', 'cathpdb']).residue_name.value_counts(normalize=True)*100

        plt.subplots(figsize=(14, 4), )
        gs = gridspec.GridSpec(ncols=2, nrows=2)
        gs.update(hspace=0.3,)
        ax0 = plt.subplot(gs[0,0])
        ax1 = plt.subplot(gs[1,0])
        ax2 = plt.subplot(gs[0,1])
        ax3 = plt.subplot(gs[1,1])
        self._composition_of_exposed_IBS(dataCathIBS, ax=ax0, PERTYPE=True, putLegend=False, xlabel="")
        self._composition_of_exposed_IBS(dataCathnoIBS, ax=ax1, PERTYPE=True, putLegend=True, )

        self._composition_of_exposed_IBS(dataCathIBS, ax=ax2, PERTYPE=False, putLegend=False, xlabel="")
        self._composition_of_exposed_IBS(dataCathnoIBS, ax=ax3, PERTYPE=False, putLegend=True)
        _ = ax0.text(-0.07,0.15, "IBS Exposed", transform=ax0.transAxes, fontsize=10, rotation=90,weight="bold")
        _ = ax1.text(-0.07,0, "non-IBS Exposed", transform=ax1.transAxes, fontsize=10, rotation=90, weight="bold")
        _ = ax1.text(0.5,1.1,"Physicochemical properties", transform=ax0.transAxes, fontsize=12,  ha="center", weight="bold")
        _ = ax1.text(0.5,1.1,"Amino acid", transform=ax2.transAxes, fontsize=12, ha="center", weight="bold")

        _= ax0.text(-0.1,1.02, "A",transform=ax0.transAxes, fontsize=20)
        _= ax2.text(-0.1,1.02, "B",transform=ax2.transAxes, fontsize=20)
        _= ax1.text(-0.1,1.02, "C",transform=ax1.transAxes, fontsize=20)
        _= ax3.text(-0.1,1.02, "D",transform=ax3.transAxes, fontsize=20)

        self._save_fig(fig_filename,format=self.FORMAT)

    def _composition_of_exposed_IBS(self, data, ax=None, PERTYPE=False, putLegend=True, xlabel="Composition (%)"):
        graph_data = data.to_frame()
        if PERTYPE:
            order_legend=["Positive",
                          "Negative",
                          "Polar",
                          "Non-polar",
                          "Hydrophobic,H-aromatic",
                          "Hydrophobic,H-non-aromatic",
                         ]
            color_palette=self.colorsPerType
            hue="type"
            weights="Percentage_Type"
            # graph_res_data=graph_res_data.drop_duplicates(["domain","type","Percentage_Type"])
        else:
            order_legend=["LYS","ARG",
                          "ASP","GLU",
                          "HIS","ASN","GLN","THR","SER",
                          "PRO","ALA","VAL","GLY",
                           "TYR","TRP","PHE",
                           "LEU","ILE","CYS","MET",
                         ]
            color_palette = {x:self.COLORS_taylor[x] for x in list(self.COLORS_taylor.keys())}
            hue='residue_name'
            weights='Percentage'
        
        graph_data.reset_index(inplace=True)
        graph_data.columns = ["residue_name","Percentage"]

        graph_data["type"] = graph_data["residue_name"].apply(lambda x: self._AATYPE[x])

        graph_data = graph_data.set_index(["type"])
        graph_data["Percentage_Type"] = graph_data.groupby("type").Percentage.sum()
        graph_data.reset_index(inplace=True)

        graph_data["data"] = "data"

        if PERTYPE:
            graph_data = graph_data.drop_duplicates("type")

        graph = sns.histplot(graph_data,
                             y="data",
                             hue=hue,
                             weights=weights,
                             multiple='stack',
                             hue_order=order_legend[::-1],
                             edgecolor='k',
                             linewidth=0.1,
                             palette=color_palette,
                             legend=putLegend,
                             ax=ax)
        graph.set(ylabel="", xlabel=xlabel)
        graph.set_yticklabels("")
        graph.set(xlim=[-1,101])

        if PERTYPE:
            for rec, label in zip(graph.patches,graph_data['Percentage_Type'].round(1).astype(str)):
                        height = rec.get_height()
                        width = rec.get_width()
                        val = f"{rec.get_width():.1f} "
                        size=12
                        """
                        if PERTYPE:
                            size = 8
                        else:
                            size = 4
                        """
                        ax.text( (rec.xy[0]+rec.get_width()/2), (rec.xy[1]+rec.get_height()/2), val, size=size, color="#383838", ha = 'center', va='center',)

        if putLegend==True:
            self._move_seaborn_legend(graph, 
                                      new_loc='center', 
                                      title="",
                                      order=order_legend,
                                      ncol=math.ceil(len(order_legend)/4), 
                                      bbox_to_anchor=(0.5,-0.7))


    ####################################################################################
    # Figure 3 in the 2022 paper: protrusions and hydrophobic protrusions in the dataset
    def make_figure_protrusions(self, fig_filename="Fig 3", show_equality_test=True, show_percentage_without_HP=True):
        count_protr_whole = self.pepr2ds.domainDf.query('atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_protrusion_per_pdb(group)).to_frame("count") 
        count_hydr_protr_whole = self.pepr2ds.domainDf.query('atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_hydr_protrusion_per_pdb(group)).to_frame("count") 
        # count_polar_protr_whole = self.pepr2ds.domainDf.query('atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_polar_protrusion_per_pdb(group)).to_frame("count") 
        # count_nonpolar_protr_whole = self.pepr2ds.domainDf.query('atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_nonpolar_protrusion_per_pdb(group)).to_frame("count") 
        # count_positive_protr_whole = self.pepr2ds.domainDf.query('atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_positive_protrusion_per_pdb(group)).to_frame("count") 
        # count_negative_protr_whole = self.pepr2ds.domainDf.query('atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_negative_protrusion_per_pdb(group)).to_frame("count") 

        count_protr_IBS = self.pepr2ds.domainDf.query('IBS == True and atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_protrusion_per_pdb(group)).to_frame("count") 
        count_hydr_protr_IBS = self.pepr2ds.domainDf.query('IBS == True and atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_hydr_protrusion_per_pdb(group)).to_frame("count") 
        # count_polar_protr_IBS = self.pepr2ds.domainDf.query('IBS == True and atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_polar_protrusion_per_pdb(group)).to_frame("count") 
        # count_nonpolar_protr_IBS = self.pepr2ds.domainDf.query('IBS == True and atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_nonpolar_protrusion_per_pdb(group)).to_frame("count") 
        # count_positive_protr_IBS = self.pepr2ds.domainDf.query('IBS == True and atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_positive_protrusion_per_pdb(group)).to_frame("count") 
        # count_negative_protr_IBS = self.pepr2ds.domainDf.query('IBS == True and atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_negative_protrusion_per_pdb(group)).to_frame("count") 

        count_protr_nonIBS = self.pepr2ds.domainDf.query('IBS == False and atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_protrusion_per_pdb(group)).to_frame("count") 
        count_hydr_protr_nonIBS = self.pepr2ds.domainDf.query('IBS == False and atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_hydr_protrusion_per_pdb(group)).to_frame("count") 
        # count_polar_protr_nonIBS = self.pepr2ds.domainDf.query('IBS == False and atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_polar_protrusion_per_pdb(group)).to_frame("count") 
        # count_nonpolar_protr_nonIBS = self.pepr2ds.domainDf.query('IBS == False and atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_nonpolar_protrusion_per_pdb(group)).to_frame("count") 
        # count_positive_protr_nonIBS = self.pepr2ds.domainDf.query('IBS == False and atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_positive_protrusion_per_pdb(group)).to_frame("count") 
        # count_negative_protr_nonIBS = self.pepr2ds.domainDf.query('IBS == False and atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_negative_protrusion_per_pdb(group)).to_frame("count") 

        freqs_hydro_protrusion_whole = self.pepr2ds.domainDf.query('atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._calc_fract(group,"type.str.contains('Hydrophobic')", "protrusion == True")).to_frame("count") 
        freqs_hydro_protrusion_IBS = self.pepr2ds.domainDf.query('atom_name == "CB" and IBS == True').groupby('cathpdb').progress_apply(lambda group: self._calc_fract(group,"type.str.contains('Hydrophobic')", "protrusion == True")).to_frame("count") 
        freqs_hydro_protrusion_nonIBS = self.pepr2ds.domainDf.query('atom_name == "CB" and IBS == False').groupby('cathpdb').progress_apply(lambda group: self._calc_fract(group,"type.str.contains('Hydrophobic')", "protrusion == True")).to_frame("count")

        ###
        sns.set(font_scale=1.2)
        sns.set_style("whitegrid") #Seaborn style

        count_protr_whole["surface"] = "whole"
        count_hydr_protr_whole["surface"] = "whole"
        count_hydr_protr_IBS["surface"] = "IBS"
        count_protr_IBS["surface"] = "IBS"
        count_hydr_protr_nonIBS["surface"] = "nonIBS"
        count_protr_nonIBS["surface"] = "nonIBS"

        #WHOLE
        count_protr_whole["Type"] = "All"
        count_hydr_protr_whole["Type"] = "Hydrophobic"
        # count_positive_protr_whole["Type"] = "positive"
        # count_polar_protr_whole["Type"] = "polar"
        # count_nonpolar_protr_whole["Type"] = "nonpolar"
        # count_negative_protr_whole["Type"] = "negative"

        #IBS
        count_protr_IBS["Type"] = "All"
        count_hydr_protr_IBS["Type"] = "Hydrophobic"
        # count_positive_protr_IBS["Type"] = "positive"
        # count_polar_protr_IBS["Type"] = "polar"
        # count_nonpolar_protr_IBS["Type"] = "nonpolar"
        # count_negative_protr_IBS["Type"] = "negative"

        #non IBS
        count_protr_nonIBS["Type"] = "All"
        count_hydr_protr_nonIBS["Type"] = "Hydrophobic"
        # count_positive_protr_nonIBS["Type"] = "positive"
        # count_polar_protr_nonIBS["Type"] = "polar"
        # count_nonpolar_protr_nonIBS["Type"] = "nonpolar"
        # count_negative_protr_nonIBS["Type"] = "negative"

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        ylim = [0,900] if 'alphafold' in self._data_type else [0,575]
        
        #GRAPH A
        count_whole = pd.concat([count_protr_whole,
                                 count_hydr_protr_whole,
                                #count_positive_protr_whole,
                                #count_polar_protr_whole,
                                #count_nonpolar_protr_whole,
                                #count_negative_protr_whole,
                                ], axis=0).reset_index()
        max_x_whole = count_whole["count"].max()

        graph_whole = sns.histplot(count_whole, 
                                   x="count",
                                   hue="Type",
                                   #stat="probability",
                                   #kind="kde",
                                   palette=["green","#ff7F0E"],
                                   alpha=0.5,
                                   bins=list(range(0,max_x_whole)),
                                   edgecolor="gray", linewidth=0.2,
                                   common_bins=False,
                                   common_norm=False,
                                   ax=axs[0],)

        _ = sns.despine(left=False, bottom=False, top=False, right=False) #All 4 borders

        graph_whole.set(xlabel="Number of protrusions",
                        ylabel="Number of structures",
                        #ylim=[0,0.42],
                        #ylim=[0,575]
                        ylim=ylim,
                        xlim=[0,48],)
        # _ = graph_whole.set_title("Whole surface", fontsize=11)

        #Graph B
        count_IBS = pd.concat([count_protr_IBS,
                               count_hydr_protr_IBS,
                               #count_positive_protr_IBS,
                               #count_polar_protr_IBS,
                               #count_nonpolar_protr_IBS,
                               #count_negative_protr_IBS,
                              ], axis=0).reset_index()
        max_x_IBS = count_IBS["count"].max()

        graph_IBS = sns.histplot(count_IBS, 
                                 x="count",
                                 hue="Type",
                                 #stat="probability",
                                 bins=list(range(0,max_x_IBS)),
                                 #kind="kde",
                                 alpha=0.5,
                                 palette=["green","#ff7F0E"],
                                 edgecolor="gray", linewidth=0.2,
                                 common_norm=False,
                                 ax=axs[1])

        _ = graph_IBS.set(xlabel="Number of protrusions",
                          ylabel="",
                          ylim=ylim,
                          #ylim=[0,575],
                          xlim=[0,48],)
        #_ = graph_IBS.set_title("IBS surface", fontsize=11)
        _ = sns.despine(left=False, bottom=False, top=False, right=False) #All 4 borders

        #Write letters
        freqs_hydro_protrusion_whole["Surface"] = "whole"
        freqs_hydro_protrusion_nonIBS["Surface"] = "nonIBS"
        freqs_hydro_protrusion_IBS["Surface"] = "IBS"
        ratio_graph_data = pd.concat([freqs_hydro_protrusion_nonIBS,freqs_hydro_protrusion_IBS], axis=0).reset_index()
        max_x = ratio_graph_data["count"].max()

        #graph1 = sns.histplot(test.query("Surface == 'whole'"), x="count", bins=np.arange(0,0.8,0.05), alpha=0.6,color="#8de5A1", stat="probability",  ax=axs[2], linewidth=0.2,edgecolor="gray")
        #graph2 = sns.histplot(test.query("Surface == 'IBS'"), x="count", bins=np.arange(0,0.8,0.05), alpha=0.6, color="#cb6679",stat="probability",ax=axs[2], linewidth=0.2,edgecolor="gray")
        graph_ratio = sns.histplot(ratio_graph_data, 
                                  hue="Surface", 
                                  x="count", 
                                  bins=np.arange(0,0.8,0.05), 
                                  alpha=0.6,
                                  #stat="probability", 
                                  palette=["#cb6679","#69aacf"],  
                                  ax=axs[2],
                                  linewidth=0.2,
                                  kde=True,
                                  common_norm=False,
                                  edgecolor="gray")
        #old color = #cb6679 / #69aacf
        _ = sns.despine(left=False, bottom=False, top=False, right=False) #All 4 borders

        #graph.set(xlabel="Number of protrusions", ylabel="Number of protein",title="Number of hydrophobic protrusions per protein (Whole)")

        # ---- Legend ----
        #top_bar = mpatches.Patch(color='#8de5a1', label='Whole surface', alpha=0.6, linewidth=0)
        #bottom_bar = mpatches.Patch(color='#cb6679', label='IBS surface', alpha=0.6, linewidth=0)
        #axs[2].legend(handles=[top_bar, bottom_bar])

        _ = axs[2].set(xlabel="Ratio of protrusions being hydropobic",
                       ylabel="",
                       xlim=[0,0.7],
                       #ylim=[0,575],
                       ylim=ylim,)
        #_ = axs[2].text(-0.1,1.1, "Ratio of protrusions being hydrophobic", transform=axs[2].transAxes, fontsize=15)
        _ = axs[2].tick_params(labelsize=10)

        _ = axs[0].text(-0.1,1.02, "A", transform=axs[0].transAxes, fontsize=20)
        #_ = axs[0].text(0.5,1.1, "Number of protrusions per structure", transform=axs[0].transAxes, fontsize=15)
        _ = axs[1].text(-0.1,1.02, "B", transform=axs[1].transAxes, fontsize=20)
        _ = axs[2].text(-0.1,1.02, "C", transform=axs[2].transAxes, fontsize=20)

        self._save_fig(fig_filename, transparent=False, format=self.FORMAT)

        if show_equality_test:
            pop1 = ratio_graph_data.query("Surface == 'IBS'")["count"]
            pop2 = ratio_graph_data.query("Surface == 'nonIBS'")["count"]
            self._equality_test(pop1,pop2, "greater")

        if show_percentage_without_HP:
            print("----------------------------------------")
            print("**PERCENTAGE OF THE DATASET WITH/WITHOUT HP**")
            print("-- nonIBS SURFACE --")
            print(f" protrusions .............. {count_protr_nonIBS['count'].mean():.2f} ± {count_protr_nonIBS['count'].std():.2f}") 
            print(f" hydrophobic protrusions .. {count_hydr_protr_nonIBS['count'].mean():.2f} ± {count_hydr_protr_nonIBS['count'].std():.2f}") 

            print("-- IBS SURFACE --")
            print(f" protrusions .............. {count_protr_IBS['count'].mean():.2f} ± {count_protr_IBS['count'].std():.2f}") 
            print(f" hydrophobic protrusions .. {count_hydr_protr_IBS['count'].mean():.2f} ± {count_hydr_protr_IBS['count'].std():.2f}") 

            print("-- whole surface --")
            print(f" protrusions .............. {count_protr_whole['count'].mean():.2f} ± {count_protr_whole['count'].std():.2f}") 
            print(f" hydrophobic protrusions .. {count_hydr_protr_whole['count'].mean():.2f} ± {count_hydr_protr_whole['count'].std():.2f}")

            ###
            print("\nPercentage of the dataset without hydrophobic protrusions")
            perc = ratio_graph_data.groupby("Surface").apply(lambda x: len(x.query("count <0.05")) / len(x) * 100)
            vals = ratio_graph_data.groupby("Surface").apply(lambda x: len(x.query("count <0.05")))
            print("      percentage")
            print(perc)
            print("      values")
            print(vals)

            ratio_graph_data.groupby("Surface").apply(lambda x: len(x.query("count >=0.05")))

            ###
            res = self.pepr2ds.domainDf.groupby("cathpdb", as_index=False).progress_apply(lambda x: self._count_percentage_of_the_convexhull(x))
            print(f"Percentage of the convexhull being part of the IBS: {res.loc[(slice(None), 'fraction_IBS'), :].mean()[0]:.2f}% ± {res.loc[(slice(None), 'fraction_IBS'), :].std()[0]:.2f}%")
            print(f"Percentage of the convexhull NOT being part of the IBS: {res.loc[(slice(None), 'fraction_nonIBS'), :].mean()[0]:.2f}% ± {res.loc[(slice(None), 'fraction_nonIBS'), :].std()[0]:.2f}%")

            ###
            res = self._backup_prot_noHPIBS.groupby("cathpdb", as_index=False).progress_apply(lambda x: self._count_fraction_of_IBS(x))
            print(f"Percentage of the IBS covered by protrusion for protreins without hydrophobic protrusions at their IBS: {res.iloc[:,1].mean():.2f}±{res.iloc[:,1].std():.2f}%")
            print("\n-- DONE\n\n")

    def _count_protrusion_per_pdb(self, group):
        N = group.query("protrusion == True")
        return len(N)

    def _count_hydr_protrusion_per_pdb(self, group):
        N = group.query("protrusion == True and type.str.contains('Hydrophobic')", engine='python')
        return len(N)

    def _count_polar_protrusion_per_pdb(self, group):
        N = group.query("protrusion == True and type.str.contains('Polar')", engine='python')
        return len(N)

    def _count_nonpolar_protrusion_per_pdb(self, group):
        N = group.query("protrusion == True and type.str.contains('Non-polar')", engine='python')
        return len(N)

    def _count_negative_protrusion_per_pdb(self, group):
        N = group.query("protrusion == True and type.str.contains('Negative')", engine='python')
        return len(N)

    def _count_positive_protrusion_per_pdb(self, group):
        N = group.query("protrusion == True and type.str.contains('Positive')", engine='python')
        return len(N)

    def _calc_fract(self, group, property1, property2):
        # determine the fraction to get property1 IN property2
        total = len(group.query(property2, engine='python'))
        pr1 = len(group.query(f"{property1} and {property2}", engine='python'))
        if total == 0:
            return 0
        return pr1/total

    def _count_percentage_of_the_convexhull(self, pdb):
        # evaluate the percentage of the convexhull to be part of the IBS and not IBS
        df = pdb.query("atom_name == 'CB' and convhull_vertex == True")
        n_vertices = len(df)
        n_vertices_IBS = len(df.query("IBS == True"))
        n_vertices_nonIBS = len(df.query("IBS == False"))

        ret = [n_vertices_IBS/n_vertices*100,n_vertices_nonIBS/n_vertices*100]

        return(pd.DataFrame(ret, index=["fraction_IBS","fraction_nonIBS"]))
    
    def _count_fraction_of_IBS(self, pdb):
        #evaluate the percentage of the convexhull to be part of the IBS and not IBS.
        df = pdb.query("atom_name == 'CB' and convhull_vertex == True")
        n_vertices = len(df)
        n_vertices_IBS = len(df.query("IBS == True"))
        n_protrusion = len(df.query("protrusion == True"))

        ret = [n_vertices_IBS/n_vertices*100]

        return(ret)

    ######################################################################################
    # Figure 4 in the 2022 paper: composition of the IBS/nonIBS for protein with HP at IBS
    def make_figure_composition_for_proteins_with_HP_at_IBS(self, fig_filename="Fig 4"):
        sns.set_style("whitegrid") #Seaborn style
        sns.set(font_scale=1.2, style="whitegrid")

        self.pepr2ds.domainDf = self._backup_prot_HPIBS.copy()
        plt.rcParams["font.family"] = "DejaVu Sans"

        fig = plt.figure(figsize=(15,5))
        gs = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=[1,2], height_ratios=[8,3])
        gs.update(hspace=0.05)
        ax0 = plt.subplot(gs[0,0])
        ax1 = plt.subplot(gs[0,1])
        ax2 = plt.subplot(gs[1,0])
        ax3 = plt.subplot(gs[1,1])

        ### Protrusion Secondary Structure
        tc_sec_struc = self.pepr2ds.domainDf.query("protrusion == True and type in ['Hydrophobic,H-non-aromatic', 'Hydrophobic,H-aromatic']").groupby("IBS").sec_struc.value_counts(normalize=True).to_frame('Percentage').reset_index()
        tc_sec_struc["Surface"] = tc_sec_struc["IBS"].apply(lambda x: "IBS" if x == True else "nonIBS")
        labelsdict = {"C":"Loop", "E":"β-strand","H":"α-helix"}

        print(tc_sec_struc)
        try: #Just pandas update change in output index.
           tc_sec_struc["Secondary Structure"] = tc_sec_struc["sec_struc"].apply(lambda x: labelsdict[x])
        except: 
           tc_sec_struc["Secondary Structure"] = tc_sec_struc["level_1"].apply(lambda x: labelsdict[x])

        self.pepr2ds.domainDf["Secondary Structure"] = self.pepr2ds.domainDf["sec_struc"].apply(lambda x: labelsdict[x])
        tc_sec_struc["Percentage"] = tc_sec_struc["Percentage"]*100

        graph = sns.barplot(data=tc_sec_struc,
                            x="Secondary Structure",
                            y="Percentage",
                            hue="Surface",
                            palette=self.palette_IBS,
                            ax=ax0)
        
        _= graph.set(title="",#Secondary structure element",
                     xlabel="",
                     xticklabels=[])

        #### Amino acid composition
        tc_resname = self.pepr2ds.domainDf.query("protrusion == True ").groupby("IBS")["residue_name"].value_counts(normalize=True).to_frame('Percentage').reset_index()
        tc_resname["Surface"] = tc_resname["IBS"].apply(lambda x: "IBS" if x == True else "nonIBS")
        tc_resname["Percentage"] = tc_resname["Percentage"]*100
        residue_name_order = tc_resname.query("Surface == 'IBS'").sort_values("Percentage", ascending=False).residue_name
        graph2 = sns.barplot(data=tc_resname,
                             x="residue_name",
                             y="Percentage",
                             hue="Surface",
                             palette=self.palette_IBS,  
                             ax=ax1,
                             order = residue_name_order)
        
        _= graph2.set(title="", #Amino acid", 
                      xlabel="",
                     xticklabels=[],)
        #_= graph2.set_xticklabels(graph2.get_xticklabels(),rotation=30, size=8)

        ### ODDS RATIO SSE
        oddratios_SSE = self.pepr2ds.analysis.oddsratio_graph(among="HP", feature="sec_struc_full", axs=ax0, title="Log(OR) - Secondary structures", xlim=(-1.5,1.0), return_dataset=True)
        print(oddratios_SSE)

        oddratios_SSE = self._significance(oddratios_SSE)

        palette = self._get_palette_OR(oddratios_SSE)
        graphOR1 = sns.barplot(data=oddratios_SSE, x="sec_struc_full",y="oddsratio", orient="v", palette=palette, linewidth=0, yerr=oddratios_SSE["err"], ax=ax2)
        _= graphOR1.set(ylabel="log(OR)",
                        xlabel="",
                        ylim=[-1.90,1.6],)

        #_= graphOR1.set_xticklabels(graphOR1.get_xticklabels(),rotation=30, size=10)
        _= graphOR1.set_xticklabels(oddratios_SSE["labels"],rotation=30, size=10)

        ### ODDS RATIO Amino acids

        oddratios_AA = self.pepr2ds.analysis.oddsratio_graph(among="protrusions", feature="residue_name", axs=ax1,title="Log(OR) per amino acid",return_dataset=True)
        #Reorder the oddratio dataframe according the residue_name_order
        oddratios_AA = self._significance(oddratios_AA)

        oddratios_AA = oddratios_AA.set_index("residue_name").loc[residue_name_order].reset_index()
        palette = self._get_palette_OR(oddratios_AA)
        graphOR2 = sns.barplot(data=oddratios_AA, x="residue_name",y="oddsratio", orient="v", palette=palette, linewidth=0, yerr=oddratios_AA["err"], ax=ax3)
        _= graphOR2.set(ylabel="log(OR)",
                        xlabel="",
                        ylim=[-1.6,1.6],)

        #_= graphOR2.set_xticklabels(graphOR2.get_xticklabels(),rotation=30, size=10)
        _= graphOR2.set_xticklabels(oddratios_AA["labels"],rotation=30, size=10, horizontalalignment='right')

        #Adding labels
        #_= ax0.text(1,1.15, "Protrusions composition",transform=ax0.transAxes, fontsize=20)
        #_= ax0.text(2,1.15, "(Protreins with HP at IBS)",transform=ax0.transAxes, fontsize=8)

        _= ax0.text(-0.1,1.02, "A",transform=ax0.transAxes, fontsize=20)
        _= ax1.text(-0.1,1.02, "B",transform=ax1.transAxes, fontsize=20)
        # _= ax2.text(-0.1,0.8, "C",transform=ax2.transAxes, fontsize=20)
        # _= ax3.text(-0.1,0.8, "D",transform=ax3.transAxes, fontsize=20)

        self._save_fig(fig_filename, transparent=False, format=self.FORMAT)

        self.pepr2ds.domainDf = self._backup_all.copy()

    #######################################################
    # Figure 5 in the 2022 paper: neighbourhood composition
    def make_figure_neighbourhood_composition(self,
                                              fig_filename = "Fig 5",
                                              drop_AF_below_b_factor = None):

        # NB! The standard use (with CATH data, as indeed in Figure 5) assumes no AF entries are in the tagged dataset
        if drop_AF_below_b_factor is None:
            self.pepr2ds.domainDf = self._backup_prot_HPIBS.copy()
        else:
            # this case (with >= 70) was used in earlier Figure 8 in Notebook #4, but it was removed in the end
            self.pepr2ds.domainDf = self._backup_prot_HPIBS.copy().query(f"(data_type == 'cathpdb') or (data_type == 'alphafold' and b_factor >= {drop_AF_below_b_factor})")

        tableCount, oddratios_AA = self.pepr2ds.analysis.oddsratio_graph(among="is_hydrophobic_protrusion", feature="residue_name", envir=True, title="Log(OR) per amino acid",return_dataset=True, return_tablecount=True, condition="exposed", exclude_protrusion=True)
        tableCount.reset_index(inplace=True)
        tableCount["Percentage"] = tableCount.groupby(["IBS","residue_name"], as_index=False).apply(lambda x: x.Count/x.total*100).droplevel(0)
        tableCount["Surface"] = tableCount["IBS"].apply(lambda x: "IBS" if x == True else "nonIBS")

        #tableCount, oddratios_AA = self.pepr2ds.analysis.oddsratio_graph(among="protrusions", feature="residue_name", envir=True, title="Log(OR) per amino acid",return_dataset=True, return_tablecount=True, condition="exposed", exclude_protrusion=True)

        tableCountSSE, oddratios_SSE = self.pepr2ds.analysis.oddsratio_graph(among="is_hydrophobic_protrusion", feature="sec_struc_full", envir=True, title="Log(OR) per amino acid",return_dataset=True, return_tablecount=True, condition="exposed", exclude_protrusion=True)
        tableCountSSE.reset_index(inplace=True)
        tableCountSSE["Percentage"] = tableCountSSE.groupby(["IBS","sec_struc_full"], as_index=False).apply(lambda x: x.Count/x.total*100).droplevel(0)
        tableCountSSE["Surface"] = tableCountSSE["IBS"].apply(lambda x: "IBS" if x == True else "nonIBS")

        #labelsdict = {"C":"Loop", "E":"β-strand","H":"α-helix"}
        # TO DO: the map above generates a KeyError on the next calls below upon entries like "-", "B"
        # Apparently, this evolved from a "sec_struc" feature on the call to oddsratio_graph to "sec_struc_full" 
        # BUT, it seems safe to ignore, as both structures on the left-hand side are not used later (!)
        labelsdict = {"H":'α-helix',
                      "G":'α-helix',
                      "I":'α-helix',
                      "B":'β-strand',
                      "E":'β-strand',
                      "T":'Bend',
                      "S":'Turn',
                      "-":'Coil',}
        tableCountSSE["Secondary Structure"] = tableCountSSE["sec_struc_full"].apply(lambda x: labelsdict[x])
        oddratios_SSE.insert(0, "Secondary Structure", oddratios_SSE["sec_struc_full"].apply(lambda x: labelsdict[x]))

        self.pepr2ds.domainDf = self._backup_all.copy()

        ###
        sns.set_style("whitegrid",{'legend.frameon':True}) #Seaborn style
        sns.set(font_scale=1.2)
        self.pepr2ds.domainDf = self._backup_prot_noHPIBS.copy()   #!?

        fig = plt.figure(figsize=(10,5))
        gs = gridspec.GridSpec(ncols=1, nrows=2, width_ratios=[1], height_ratios=[8,3])
        gs.update(hspace=0.05)
        ax0 = plt.subplot(gs[0,0])
        ax1 = plt.subplot(gs[1,0])

        #### Amino acid composition
        tc_resname = tableCount
        residue_name_order = tc_resname.query("Surface == 'IBS'").sort_values("Percentage", ascending=False).residue_name
        graph2 = sns.barplot(data=tc_resname,
                             x="residue_name",
                             y="Percentage",
                             hue="Surface",
                             palette=self.palette_IBS,  
                             ax=ax0,
                             order = residue_name_order,)

        _= graph2.set(title="", #"Hydrophobic protrusions exposed environment composition", 
                      xlabel="",
                      xticklabels=[])

        ### ODDS RATIO Amino acids

        #Reorder the oddratio dataframe according the residue_name_order
        oddratios_AA = oddratios_AA.set_index("residue_name").loc[residue_name_order].reset_index()
        palette = self._get_palette_OR(oddratios_AA)
        graphOR2 = sns.barplot(data=oddratios_AA, x="residue_name",y="oddsratio", orient="v", palette=palette, linewidth=0, yerr=oddratios_AA["err"], ax=ax1)
        _= graphOR2.set(ylabel="log(OR)",
                        xlabel="",
                        ylim=(-1,1.5),)

        _= graphOR2.set_xticklabels(graphOR2.get_xticklabels(),rotation=30, size=12)
        oddratios_AA = self._significance(oddratios_AA)
        _= graphOR2.set_xticklabels(oddratios_AA["labels"], horizontalalignment="right",rotation=30, size=12)

        #Adding labels
        # _= ax0.text(0.1,1.15, "Hydrophobic protrusions exposed environment composition",transform=ax0.transAxes, fontsize=20)
        # _= ax0.text(-0.1,1.02, "A",transform=ax0.transAxes, fontsize=20)
        # _= ax1.text(-0.1,1.02, "B",transform=ax1.transAxes, fontsize=20)
        # _= ax2.text(-0.1,0.8, "C",transform=ax2.transAxes, fontsize=20)
        # _= ax3.text(-0.1,0.8, "D",transform=ax3.transAxes, fontsize=20)

        self._save_fig(fig_filename, transparent=False, format=self.FORMAT)
        self.pepr2ds.domainDf = self._backup_all.copy()

    #########################################################################################################
    # Figure 6 in the 2022 paper: number of structures with/without HP at IBS and comparison of both datasets
    def make_figure_number_of_structures_w_and_wo_HP_at_IBS(self, fig_filename="Fig 6"):
        count_protr_ibs_HPIBS = self._backup_prot_HPIBS.query('IBS == True and atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_protrusion_per_pdb(group)).to_frame("count")
        count_protr_ibs_noHPIBS = self._backup_prot_noHPIBS.query('IBS == True and atom_name == "CB"').groupby('cathpdb').progress_apply(lambda group: self._count_protrusion_per_pdb(group)).to_frame("count")

        sns.set(font_scale=1.2)
        sns.set_style("whitegrid",{'legend.frameon':True}) #Seaborn style
        plt.rcParams["font.family"] = "DejaVu Sans" #Old = "DejaVu Sans" / "DejaVu Serif"

        fig = plt.figure(figsize=(15,5))
        gs = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1,1], height_ratios=[1])
        ax0 = plt.subplot(gs[0,0])
        ax1 = plt.subplot(gs[0,1])

        #Distribution two populations
        count_protr_ibs_HPIBS["dataset"] = "with hydr. protr. (IBS)"
        count_protr_ibs_noHPIBS["dataset"] = "without hydr. protr. (IBS)"
        count_IBS = pd.concat([count_protr_ibs_HPIBS,
                               count_protr_ibs_noHPIBS,
                              ], axis=0)

        max_x_IBS = math.ceil(count_IBS["count"].max())

        graph_noHPIBS = sns.histplot(count_IBS, 
                                     x="count",
                                     hue="dataset",
                                     stat="probability",
                                     bins=list(range(0,max_x_IBS)),
                                     #kind="kde",
                                     alpha=0.5,
                                     palette=["#FFC20A","#0C7BDC"],
                                     edgecolor="gray", linewidth=0.2,
                                     common_norm=False,
                                     kde=True,
                                     ax=ax0,)

        _ = graph_noHPIBS.set(xlabel="Number of protrusions",
                              ylabel="Frequency",
                              #ylim=[0,575],
                              #xlim=[0,48],f
                              )

        _ = graph_noHPIBS.set_xticks(range(0,max_x_IBS,2))
        # graph_noHPIBS.set_title("Number of protrusion in the IBS per dataset", fontsize=11)
        sns.despine(left=False, bottom=False, top=False, right=False) #All 4 borders

        ### Barplot

        count_no_hydr = self._backup_prot_noHPIBS[["cathpdb","domain"]].drop_duplicates().domain.value_counts().to_frame().reset_index().rename(columns={"index":"domain","domain":"without hydrophobic protrusions"})

        # count_no_hydr.set_index('domain')
        number_of_structures = self.pepr2ds.domainDf[["cathpdb","domain"]].drop_duplicates().domain.value_counts()
        #number_of_structures.name = 'Number of PDBS'
        number_of_structures = number_of_structures.to_frame().reset_index().rename(columns={'domain':'Number of PDBS','index':'domain'})

        count_no_hydr = count_no_hydr.merge(number_of_structures)

        count_no_hydr["percentage"] = count_no_hydr["without hydrophobic protrusions"] / count_no_hydr["Number of PDBS"]

        #count_no_hydr.sort_values(by="Number of PDBS", inplace=True, ascending=False)

        order = sorted(list(count_no_hydr.domain.unique()))
        #order = ['ANNEXIN', 'C1', 'C2', 'C2DIS', 'PH', 'PLA', 'PLD', 'PX', 'START']

        if 'alphafold' in self._data_type:
            count_no_hydr = count_no_hydr.set_index("domain")
            count_no_hydr = count_no_hydr.loc[order]
            count_no_hydr = count_no_hydr.reset_index()
            
        count_no_hydr.domain = count_no_hydr.domain.astype(str)

        graphdata_nohydr = count_no_hydr.melt(value_vars=["without hydrophobic protrusions","Number of PDBS"],
                                              id_vars=["domain"],
                                              value_name="count",
                                              var_name="Observation")

        bar1 = sns.barplot(x="domain",
                           y="count",
                           data=graphdata_nohydr.query("Observation == 'Number of PDBS'"),
                           color='royalblue',
                           order=order,
                           ax=ax1) 

        bar2 = sns.barplot(x="domain",
                           y="count",
                           data=graphdata_nohydr.query("Observation == 'without hydrophobic protrusions'"),
                           color='#85BDED',
                           errorbar=None,
                           order=order,
                           ax=ax1)

        new_labels = []
        for dom in order:
            if dom == "C2DIS":
                new_labels.append("DIS-C2")
            elif dom == "PLD":
                new_labels.append("PLC/PLD")
            else:
                new_labels.append(dom)
        _ = bar2.set_xticklabels(new_labels)

        groupedvalues=graphdata_nohydr.groupby('domain').sum().reset_index()

        #Adding barplot
        count_no_hydr = count_no_hydr.reset_index(drop=True)
        count_no_hydr = count_no_hydr.set_index("domain").loc[order].reset_index()
        for index, row in count_no_hydr.iterrows():
            _ = bar2.text(row.name,
                          row["without hydrophobic protrusions"]+0.1, # TO DO: experiment with up to +5 depending on the max. number of structures
                          f"{row['percentage']*100:.1f}%",
                          color="black",
                          ha="center",
                          size=12)

        #Legend
        top_bar = mpatches.Patch(color='royalblue', label='Total')
        bottom_bar = mpatches.Patch(color='#85BDED', label='Without hydr. protr. (IBS)')
        _ = plt.legend(handles=[top_bar, bottom_bar])

        _ = ax1.set(title ="",# "Number of structures per superfamilly",
                    xlabel="Superfamilly", 
                    ylabel="Number of structures", 
                    ylim=None,)
        _ = ax1.tick_params(labelsize=7)

        _= ax0.text(-0.1,1.02, "A",transform=ax0.transAxes, fontsize=20)
        _= ax1.text(-0.1,1.02, "B",transform=ax1.transAxes, fontsize=20)

        # Increase fontsize
        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            _ = label.set_fontsize(14)

        for label in (ax0.get_xticklabels() + ax0.get_yticklabels()):
            _ = label.set_fontsize(14)

        _ = ax1.xaxis.label.set_size(14)
        _ = ax0.xaxis.label.set_size(14)
        _ = ax0.yaxis.label.set_size(14)
        _ = ax1.yaxis.label.set_size(14)

        for label in ax1.get_xticklabels():
            _ = label.set_rotation(45)
            _ = label.set_ha('right')
            _ = label.set_rotation_mode("anchor")

        _ = plt.setp(ax1.get_legend().get_texts(), fontsize='14')
        _ = plt.setp(ax0.get_legend().get_texts(), fontsize='14')
        _ = plt.setp(ax0.get_legend().get_title(), fontsize='15')

        self._save_fig(fig_filename, transparent=False, format=self.FORMAT)
        
        # raw values and statistical test
        #print(count_no_hydr.query("domain not in ['C2DIS','START','ENTH','PLA','PX']").percentage.mean())
        print(f"\n{count_no_hydr}")
        #print(count_no_hydr.percentage.mean())
        
        print("\nSee below for difference between the two subsets (HPIBS, noHPIBS)")
        pop1 = count_protr_ibs_HPIBS["count"]
        pop2 = count_protr_ibs_noHPIBS["count"]
        self._equality_test(pop1,pop2, "greater")

    ####################################################################################################################
    # Figure 7 in the 2022 paper: composition of the IBS and the protrusion neighbourhood for proteins WITHOUT HP AT IBS
    def make_figure_composition_for_proteins_without_HP_at_IBS(self, fig_filename="Fig 7"):
        self.pepr2ds.domainDf = self._backup_prot_noHPIBS.copy()

        tableCount_noHPIBS, oddratios_AA_noHPIBS_envir = self.pepr2ds.analysis.oddsratio_graph(among="protrusions", feature="residue_name", envir=True, title="Log(OR) per amino acid",return_dataset=True, return_tablecount=True, condition="exposed", exclude_protrusion=True)
        tableCount_noHPIBS.reset_index(inplace=True)
        tableCount_noHPIBS["Percentage"] = tableCount_noHPIBS.groupby(["IBS","residue_name"], as_index=False).apply(lambda x: x.Count/x.total*100).droplevel(0)
        tableCount_noHPIBS["Surface"] = tableCount_noHPIBS["IBS"].apply(lambda x: "IBS" if x == True else "nonIBS")

        #tableCount_noHPIBS, oddratios_AA_noHPIBS_envir = self.pepr2ds.analysis.oddsratio_graph(among="protrusions", feature="residue_name", envir=True, title="Log(OR) per amino acid",return_dataset=True, return_tablecount=True, condition="exposed", exclude_protrusion=True)

        tableCountSSE_noHPIBS, oddratios_SSE = self.pepr2ds.analysis.oddsratio_graph(among="is_hydrophobic_protrusion", feature="sec_struc_full", envir=True, title="Log(OR) per amino acid",return_dataset=True, return_tablecount=True, condition="exposed", exclude_protrusion=True)
        tableCountSSE_noHPIBS.reset_index(inplace=True)
        tableCountSSE_noHPIBS["Percentage"] = tableCountSSE_noHPIBS.groupby(["IBS","sec_struc_full"], as_index=False).apply(lambda x: x.Count/x.total*100).droplevel(0)
        tableCountSSE_noHPIBS["Surface"] = tableCountSSE_noHPIBS["IBS"].apply(lambda x: "IBS" if x == True else "nonIBS")

        #labelsdict = {"C":"Loop", "E":"β-strand","H":"α-helix"}
        # TO DO: the map above generates a KeyError on the next calls below upon entries like "-", "B"
        # Apparently, this evolved from a "sec_struc" feature on the call to oddsratio_graph to "sec_struc_full" 
        # BUT, it seems safe to ignore, as both structures on the left-hand side are not used later (!)
        labelsdict = {"H":'α-helix',
                      "G":'α-helix',
                      "I":'α-helix',
                      "B":'β-strand',
                      "E":'β-strand',
                      "T":'Bend',
                      "S":'Turn',
                      "-":'Coil',}
        tableCountSSE_noHPIBS["Secondary Structure"] = tableCountSSE_noHPIBS["sec_struc_full"].apply(lambda x: labelsdict[x])
        oddratios_SSE.insert(0, "Secondary Structure", oddratios_SSE["sec_struc_full"].apply(lambda x: labelsdict[x]))
        
        self.pepr2ds.domainDf = self._backup_prot_noHPIBS.copy()

        ###
        plt.clf()
        sns.set_style("whitegrid") #Seaborn style
        fig = plt.figure(figsize=(15,5))
        gs = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=[1,1], height_ratios=[8,3])
        gs.update(hspace=0.05)
        ax0 = plt.subplot(gs[0,0])
        ax1 = plt.subplot(gs[0,1])
        ax2 = plt.subplot(gs[1,0])
        ax3 = plt.subplot(gs[1,1])

        #1. Protrusions AA composition
        tc_resname_protrusions = self.pepr2ds.domainDf.query("protrusion == True").groupby("IBS")["residue_name"].value_counts(normalize=True).to_frame('Percentage').reset_index()
        tc_resname_protrusions["Surface"] = tc_resname_protrusions["IBS"].apply(lambda x: "IBS" if x == True else "nonIBS")
        tc_resname_protrusions["Percentage"] = tc_resname_protrusions["Percentage"]*100
        residue_name_order = tc_resname_protrusions.query("Surface == 'IBS'").sort_values("Percentage", ascending=False).residue_name
        graph_protrusions = sns.barplot(data=tc_resname_protrusions,
                                        x="residue_name",
                                        y="Percentage",
                                        hue="Surface",
                                        palette=self.palette_IBS,  
                                        ax=ax0,
                                        order = residue_name_order)
        _= graph_protrusions.set(title="", #Protrusion composition", 
                                 xlabel="",
                                 xticklabels=[],)

        #2.1 OR values
        oddratios_AA_protrusions = self.pepr2ds.analysis.oddsratio_graph(among="protrusions", feature="residue_name", axs=ax1,title="Log(OR) per amino acid",return_dataset=True)

        #Reorder the oddratio dataframe according the residue_name_order
        oddratios_AA_protrusions = oddratios_AA_protrusions.set_index("residue_name").loc[residue_name_order].reset_index()
        palette = self._get_palette_OR(oddratios_AA_protrusions)
        graphOR2 = sns.barplot(data=oddratios_AA_protrusions, x="residue_name",y="oddsratio", orient="v", palette=palette, linewidth=0, yerr=oddratios_AA_protrusions["err"], ax=ax2)
        _= graphOR2.set(ylabel="log(OR)",
                        xlabel="",
                        ylim=(-1,1.5),)
        _= graphOR2.set_xticklabels(graphOR2.get_xticklabels(),rotation=30, size=10)
        oddratios_AA_protrusions = self._significance(oddratios_AA_protrusions)
        _= graphOR2.set_xticklabels(oddratios_AA_protrusions["labels"], horizontalalignment="right",rotation=30, size=11)

        #2.2 Environment protrusions graph
        tc_resname_envir = tableCount_noHPIBS
        residue_name_order = tc_resname_envir.query("Surface == 'IBS'").sort_values("Percentage", ascending=False).residue_name
        graph2 = sns.barplot(data=tc_resname_envir,
                             x="residue_name",
                             y="Percentage",
                             hue="Surface",
                             palette=self.palette_IBS,  
                             ax=ax1,
                             order = residue_name_order,)

        _= graph2.set(title="", #Protrusion environment composition", 
                      xlabel="",
                      xticklabels=[])
                                 
        oddratios_AA_noHPIBS_envir = oddratios_AA_noHPIBS_envir.set_index("residue_name").loc[residue_name_order].reset_index()
        palette = self._get_palette_OR(oddratios_AA_noHPIBS_envir)
        graphOR2 = sns.barplot(data=oddratios_AA_noHPIBS_envir, x="residue_name",y="oddsratio", orient="v", palette=palette, linewidth=0, yerr=oddratios_AA_noHPIBS_envir["err"], ax=ax3)
        _= graphOR2.set(ylabel="log(OR)",
                        xlabel="",
                        ylim=(-1,1.5),)

        _= graphOR2.set_xticklabels(graphOR2.get_xticklabels(),rotation=30, size=11)
        oddratios_AA_noHPIBS_envir = self._significance(oddratios_AA_noHPIBS_envir)

        newlabels = []
        for label in oddratios_AA_noHPIBS_envir["labels"]:
            sig=label[:-3]
            AA = label[-3:]
            newAA = AA[0] + AA[1].lower() + AA[2].lower()
            newlabel = sig+newAA
            newlabels.append(newlabel)
        _= graphOR2.set_xticklabels(newlabels, horizontalalignment="right",rotation=30, size=11)

        #Adding labels
        #_= ax0.text(0.6,1.15, "Protrusions composition (protein without HP at IBS)",transform=ax0.transAxes, fontsize=20)
        _= ax0.text(-0.1, 1.02, "A", transform=ax0.transAxes, fontsize=20)
        _= ax1.text(-0.1, 1.02, "B", transform=ax1.transAxes, fontsize=20)
        # _= ax2.text(-0.1,0.8, "C",transform=ax2.transAxes, fontsize=20)
        # _= ax3.text(-0.1,0.8, "D",transform=ax3.transAxes, fontsize=20)

        self._save_fig(fig_filename, transparent=False, format=self.FORMAT)
        self.pepr2ds.domainDf = self._backup_all.copy()

    ####################################################################################################################
    # Figure 8 in the 2022 paper: superfamily decomposition of exposed environment of hydrophobic protrusions at the IBS
    # NB! On Notebook #4, this is listed as Figure 9 (the 8th one seems to be moved to supplement material)
    def make_figure_superfamily_decomposition_exposed_env_HP(self,
                                                             fig_filename="Fig 8",
                                                             xlsx_filename="Fib8B_data"):
        if 'alphafold' not in self._data_type:
            print("\nmake_figure_superfamily_decomposition_exposed_env_HP() depends on data from")
            print("AlphaFold entries, which are not in the tagged dataset provided to the")
            print("current object. Check the 'db' argument given to load_IBS_data() or")
            print("add_IBS_data() methods of DatasetManager\n")
            return

        sns.set(font_scale=1, style="whitegrid")
        plt.subplots(figsize=(14, 5))
        gs = gridspec.GridSpec(ncols=2, nrows=1)
        ax0 = plt.subplot(gs[0,0])
        ax1 = plt.subplot(gs[0,1])

        graph_superfamilies_type = self._make_ibs_perdomain_graph(data = self._backup_all.query("(data_type == 'cathpdb') or (data_type == 'alphafold' and b_factor >= 70)"),
                                                                  PERTYPE = True,
                                                                  nrow = 3,
                                                                  title = "",#Hydrophobic protrusions exposed environment at the IBS",
                                                                  outputname = "IBS_protrusions_composition_HYDRO_perdomain2",
                                                                  ax = ax0,
                                                                  legend = True,
                                                                  envir = False,
                                                                  #showstat = ["Negative","Positive", "Polar","Non-polar", "Hydrophobic,H-aromatic","Hydrophobic,H-non-aromatic"],
                                                                  subset = "IBS",   # NB! Also tried subset = "protrusion" on earlier, removed figure (under NB #4 Figure 9)
                                                                  fontsize = 10)

        graph_superfamilies_aa = self._make_ibs_perdomain_graph(data = self._backup_all.query("(data_type == 'cathpdb') or (data_type == 'alphafold' and b_factor >= 70)"),
                                                                PERTYPE = False,
                                                                nrow = 3,
                                                                title = "",#Hydrophobic protrusions exposed environment at the IBS",
                                                                outputname = "IBS_protrusions_composition_HYDRO_perdomain2",
                                                                ax = ax1,
                                                                legend = True,
                                                                envir = False,
                                                                #showstat = ["Negative","Positive", "Polar","Non-polar", "Hydrophobic,H-aromatic","Hydrophobic,H-non-aromatic"],
                                                                subset = "IBS",
                                                                savedata = xlsx_filename)

        _= ax0.text(-0.1,1.02, "A",transform=ax0.transAxes, fontsize=20)
        _= ax1.text(-0.1,1.02, "B",transform=ax1.transAxes, fontsize=20)

        self._save_fig(fig_filename, transparent=False, format=self.FORMAT)

    def _make_ibs_perdomain_graph(self,
                                  data: pd.DataFrame,
                                  PERTYPE,
                                  nrow,
                                  title,
                                  outputname,
                                  ax = None,
                                  legend = True,
                                  showstat = None,
                                  return_data = False,
                                  envir = False,
                                  legend_loc = 'center',
                                  among = "protrusions",
                                  subset = "IBS",
                                  fontsize = 8,
                                  savedata = None):
        # Fetch the dataset
        domlist = self.settings.active_superfamilies
        if subset == "IBS":
            dfibs = data.query("IBS == True and exposed == True and domain in @domlist")
        if subset == "protrusion":
            dfibs = data.query("IBS == True and protrusion == True and domain in @domlist")
        dfibs.residue_name = dfibs.residue_name.astype(str)

        #Set colors
        sns.set_style("whitegrid")
        plt.rcParams["font.family"] = "DejaVu Sans" #Old = "DejaVu Sans" / "DejaVu Serif"
        graph_res_data = dfibs.residue_name.value_counts(normalize=True).to_frame()*100
        colors_per_type = {x:self.colorsPerType[self._AATYPE[x]] for x in list(graph_res_data.index)}
        colors_per_type_and_aa = {x:self.COLORS_taylor[x] for x in list(self.COLORS_taylor.keys())}

        if ax == None:
            fig, ax = plt.subplots(figsize=(10, 5))

        if envir==False:
            graph_res_data = dfibs.groupby("domain").residue_name.value_counts(normalize=True).to_frame("Percentage")*100
        else:
            local_backup = self.pepr2ds.domainDf.copy()
            self.pepr2ds.domainDf = data
            graph_res_data = self.pepr2ds.analysis.oddsratio_graph(among = among,
                                                                   feature = "residue_name",
                                                                   envir = True,
                                                                   title = "Log(OR) per amino acid",
                                                                   return_dataset = True,
                                                                   return_tablecount = True,
                                                                   condition = "exposed",
                                                                   exclude_protrusion = True,
                                                                   envirPerDomain = True)

        graph_res_data.reset_index(inplace=True)
        graph_res_data["type"] = graph_res_data["residue_name"].apply(lambda x: self._AATYPE[x])
        graph_res_data = graph_res_data.set_index(["domain","type"])
        graph_res_data["Percentage_Type"] = graph_res_data.groupby(["domain","type"]).Percentage.sum()
        graph_res_data.reset_index(inplace=True)

        if PERTYPE:
            order_legend = ["Positive",
                            "Negative",
                            "Polar",
                            "Non-polar",
                            "Hydrophobic,H-aromatic",
                            "Hydrophobic,H-non-aromatic"]
            color_palette = self.colorsPerType
            hue = "type"
            weights = "Percentage_Type"
            graph_res_data = graph_res_data.drop_duplicates(["domain","type","Percentage_Type"])
            outputname += "_pertype"
        else:
            order_legend = ["LYS","ARG",
                            "ASP","GLU",
                            "HIS","ASN","GLN","THR","SER",
                            "PRO","ALA","VAL","GLY",
                            "TYR","TRP","PHE",
                            "LEU","ILE","CYS","MET"]

            color_palette = colors_per_type_and_aa
            hue = 'residue_name'
            weights = 'Percentage'
            outputname += "_perres"

        #Remove what's not in the dataset
        Labels = graph_res_data[hue].unique()
        order_legend = [x for x in order_legend if x in Labels]
        color_palette =  { key: color_palette[key] for key in Labels}

        #print(graph_res_data)
        # NB! Would just use self.settings.active_superfamilies, but on smaller XP data some superfamilies might have no entry in 'graph_res_data' at this point
        #order_domains = sorted(self.settings.active_superfamilies)
        order_domains = sorted(graph_res_data.domain.unique())
        graph_res_data = graph_res_data.set_index("domain").loc[order_domains].reset_index()

        graph_res = sns.histplot(graph_res_data, 
                                 y = 'domain',
                                 hue = hue,
                                 weights = weights,
                                 multiple = 'stack',
                                 #palette = 'tab20c_r',
                                 shrink = 0.8,
                                 #alpha = 1,
                                 ax = ax,
                                 hue_order = order_legend[::-1],
                                 edgecolor = 'k',
                                 linewidth = 0.1,
                                 palette = color_palette,
                                 legend = legend)

        domains_in_graph = list(graph_res_data.domain.unique())
        new_labels = []
        for dom in order_domains:
            if dom == "C2DIS":
                new_labels.append("DIS-C2")
            elif dom == "PLD":
                new_labels.append("PLC/PLD")
            else:
                new_labels.append(dom)
        graph_res.set_yticklabels(new_labels) # NB! legacy warning here

        if PERTYPE:
            for rec, label in zip(graph_res.patches,graph_res_data['Percentage_Type'].round(1).astype(str)):
                height = rec.get_height()
                width = rec.get_width()
                val = f"{rec.get_width():.1f} "
                size=fontsize
                #if PERTYPE:
                #    size = 8
                #else:
                #    size = 4
                ax.text((rec.xy[0]+rec.get_width()/2), (rec.xy[1]+rec.get_height()/2), val, size=size, color="#383838",
                        ha = 'center', va='center',)

        if savedata != None:
            graph_res_data.to_excel(f"{self.settings.FIGURESFOLDER}article/{savedata}.xlsx")

        if legend:
            self._move_seaborn_legend(graph_res, 
                                      legend_loc, 
                                      title = "",
                                      order = order_legend,
                                      ncol = math.ceil(len(Labels)/nrow), 
                                      bbox_to_anchor = (0.5,-0.23))

        _ = graph_res.set(title=title, 
                          xlabel="Percentage", 
                          ylabel="",
                          xlim=(-1,101),)

        """
        #Add Groups
        rect1 = patches.Rectangle((-0.5, -0.5), 101, 2.95, linewidth=0.5, edgecolor='k', facecolor='none', linestyle="-")
        rect2 = patches.Rectangle((-0.5, 2.55), 101, 3.9, linewidth=0.5, edgecolor='k', facecolor='none', linestyle="-")
        rect3 = patches.Rectangle((-0.5, 6.55), 101, 2.9, linewidth=0.5, edgecolor='k', facecolor='none', linestyle="-")
        _ = ax.add_patch(rect1)
        _ = ax.add_patch(rect2)
        _ = ax.add_patch(rect3)
        """

        if envir == True:
            self.pepr2ds.domainDf = local_backup.copy()

        if showstat:
            if type(showstat) != type([]):
                showstat = [showstat]
            for typequery in showstat:
                print(typequery)
                print(graph_res_data.query("type == @typequery").describe())
                print(graph_res_data.query("type == @typequery").describe())

        if return_data:
            plt.close()
            return(graph_res_data)

        if ax == None:
            self._save_fig(outputname)
        else:
            return graph_res

    ######################################################################################################
    # Extra figure: exposed environment of hydrophobic protrusions with AlphaFold structures
    # NB! On Notebook #4, this code is deactivated, labeled "Figure 8" but it does not appear on the paper
    def make_figure_exposed_HP_env_with_AF(self, fig_filename = "Fig 8 - OLD"):
        if 'alphafold' not in self._data_type:
            print("\nmake_figure_exposed_HP_env_with_AF() depends on data from")
            print("AlphaFold entries, which are not in the tagged dataset provided to the")
            print("current object. Check the 'db' argument given to load_IBS_data() or")
            print("add_IBS_data() methods of DatasetManager\n")
            return

        # this is exactly Figure 5, with AF entries taken into account
        self.make_figure_neighbourhood_composition(fig_filename = fig_filename, drop_AF_below_b_factor = 70)

    # Table 4: p-values for wilcoxon signed rank test
    def make_table_p_values_for_wilcoxon_test(self):
        pass

    # Supplement Figure 1: composition of the exposed IBS (DATASET augmented)
    def make_figure_composition_of_exposed_IBS_on_augmented_dataset(self):
        pass

    # Supplement Figure 2: amino acid composition per secondary structures
    def make_figure_aminoacid_composition_per_secondary_structures(self):
        pass

    # Supplement Figure 3: odds ratio for secondary structures
    def make_figure_odds_ratio_for_secondary_structures(self):
        pass

    # Supplement Figure 4: increasing our dataset size by integrating alphafold2 models
    def make_figure_dataset_size_increase_with_alphafold(self):
        pass

    # Supplement Figure 5: number of structures per domains with alphafold2
    def make_figure_structures_per_domains_with_alphafold(self):
        pass

    # Supplement Table 1: origin of the data
    def make_table_origin_of_data(self):
        pass

    # Supplement Figure 6: hydrophobic protrusions exposed environment at the IBS?
    def make_figure_hydro_protrusions_per_domain_at_IBS(self): 
        pass

    # Export dataset for journal
    def export_dataset(self):
        pass

    # Export format file for journal
    def export_format_file(self):
        pass

