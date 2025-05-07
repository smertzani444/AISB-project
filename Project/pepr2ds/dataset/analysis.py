from .tagibs import *

from scipy.stats import fisher_exact
import pandas as pd
import numpy as np
import math
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams["figure.dpi"] = 200
# %matplotlib inline
sns.set_style("darkgrid")

from IPython.display import display
# from tqdm.auto import tqdm
from tqdm.notebook import tqdm

from termcolor import colored
from IPython.display import display, HTML
import weasyprint

tqdm.pandas()  # activate tqdm progressbar for pandas apply
pd.options.mode.chained_assignment = (
    None  # default='warn', remove pandas warning when adding a new column
)
pd.set_option("display.max_columns", None)
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# %config InlineBackend.figure_format ='svg' #better quality figure figure
np.seterr(divide='ignore', invalid='ignore')


class Analysis():
        def __init__(self, parent):
            self.parent = parent

        def report(self, domain=None, displayHTML=False, customFolder=None, mainReportFolderName="report"):

            #First thing to do: Calc summary table. This is needed for next analysis.
            self.calc_summary_table()


            from matplotlib.backends.backend_pdf import PdfPages #Multipage PDF



            domain = self.parent.domainLabel

            #DEBUG
            #self.oddsratio_graph(among="convhull_vertex", "neighbour", category='type', )


            print(f"making report for {domain}")

            if not customFolder == None:
                folder = self.parent.FIGURESFOLDER + mainReportFolderName+ "/" + customFolder + '/'
                print(f"custom folder = {folder}")

            else:
                folder = self.parent.FIGURESFOLDER + mainReportFolderName + "/" + domain + '/'
            figsize = (12,16)
            if not os.path.exists(folder):
                os.makedirs(folder)

            page=0
            with PdfPages(folder + f'/results_{domain}.pdf') as pdf:
                Npage = 15
                ###################################
                ## PAGE 1
                page+=1
                fig = plt.figure(figsize=figsize, constrained_layout=False)
                gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.4) #Grid configuration
                self.plot_residue_composition(gs[0, :]) #Make analysis 1
                self.protrusion_is_coinsertable(gs[1, :]) #Analysis 2
                try:
                    self.shannon_entropy(gs[2, :]) #Analysis 3
                except:
                    print("Error while generating shannon entropy graphs")

                #fig.suptitle(f"REPPORT FOR {domain}")
                # plt.tight_layout()

                plt.text(0.5, 0.95, "PePrMInt analysis report", horizontalalignment='center', transform=fig.transFigure,
                         size=24, fontweight='bold')
                plt.text(0.5, 0.93, f"for {domain} domain", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                if self.parent.noAlignment == False:
                    plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                             size=10)


                pdf.savefig(fig) #Save 1st page
                plt.close()

                ###################################
                ## PAGE 2
                page+=1
                fig = plt.figure(figsize=figsize, constrained_layout=False)
                gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 2],hspace=0.3)  # Grid configuration
                self.plot_residue_composition(type="co-insertable", axs=gs[0, 0])
                self.plot_residue_composition(type="hydrophobic protrusion", axs=gs[0, 1])
                self.protein_fraction(axs = gs[1, :])  # Analysis 2
                if domain in ["PH","C2", 'PX'] and self.parent.noAlignment == False:
                    self.binding_loop(axs=gs[2,:])
                #self.oddsratio_graph(among="is_hydrophobic_protrusion","sec_struc_full",gs[2, :])  # Analysis 3

                #fig.suptitle(f"REPPORT FOR {domain}")
                plt.tight_layout()
                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)

                pdf.savefig(fig)  # Save page
                plt.close()

                ###################################
                ## PAGE 3
                page+=1
                fig = plt.figure(figsize=(12, 16), constrained_layout=False)
                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1],hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="convhull_vertex",feature="sec_struc_full",axs=gs[0])  # Analysis 1
                self.oddsratio_graph(among="convhull_vertex",feature="type",axs=gs[1])  # Analysis 1
                self.oddsratio_graph(among="convhull_vertex",feature="residue_name", axs=gs[2])  # Analysis 1
                #self.oddsratio_graph(among="convhull_vertex","prot_block",axs=gs[2])  # Analysis 1

                plt.text(0.5, 0.95, "Oddsratio analysis", horizontalalignment='center', transform=fig.transFigure,
                         size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among convexhull vertices", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)
                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')


                pdf.savefig(fig)  # Save page
                plt.close()
                ###################################
                ## PAGE 3
                page += 1
                fig = plt.figure(figsize=(12, 16), constrained_layout=False)
                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="protrusions", feature="sec_struc_full", axs=gs[0])  # Analysis 1
                self.oddsratio_graph(among="protrusions", feature="type", axs=gs[1])  # Analysis 1
                self.oddsratio_graph(among="protrusions", feature="residue_name", axs=gs[2])  # Analysis 1
                # self.oddsratio_graph(among="convhull_vertex","prot_block",axs=gs[2])  # Analysis 1

                plt.text(0.5, 0.95, "Oddsratio analysis", horizontalalignment='center', transform=fig.transFigure,
                         size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among Protrusions", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)
                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')

                pdf.savefig(fig)  # Save page
                plt.close()

                ###################################
                ## PAGE 4
                page+=1
                fig = plt.figure(figsize=figsize, constrained_layout=False)
                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="is_hydrophobic_protrusion", feature="sec_struc_full", axs=gs[0])  # Analysis 1
                self.oddsratio_graph(among="is_hydrophobic_protrusion",feature= "type", axs=gs[1])  # Analysis 1
                self.oddsratio_graph(among="is_hydrophobic_protrusion",feature="residue_name",axs=gs[2])  # Analysis 1

                plt.text(0.5, 0.95, "Oddsratio analysis", horizontalalignment='center', transform=fig.transFigure,
                         size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among hydrophobic protrusions", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')
                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)
                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')

                pdf.savefig(fig)  # Save page
                plt.close()



                ###################################
                ## PAGE 5
                page+=1
                fig = plt.figure(figsize=figsize, constrained_layout=False)
                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1,1], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="is_co_insertable",feature= "sec_struc_full", axs=gs[0])  # Analysis 1
                self.oddsratio_graph(among="is_co_insertable", feature="type", axs=gs[1])  # Analysis 1
                self.oddsratio_graph(among="is_co_insertable",feature="residue_name",axs=gs[2])  # Analysis 1

                plt.text(0.5, 0.95, "Oddsratio analysis", horizontalalignment='center', transform=fig.transFigure,
                         size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among co-insertables", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')
                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)
                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')

                pdf.savefig(fig)  # Save page
                plt.close()
                # fig.suptitle(f"REPPORT FOR {domain}")

                # ##################################
                # ## PAGE 11
                # page += 1
                # fig = plt.figure(figsize=figsize, constrained_layout=False)
                #
                # gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                # self.oddsratio_graph(among="protrusions", "type", axs=gs[0], )
                # self.oddsratio_graph(among="protrusions", "residue_name", axs=gs[1], )
                # plt.text(0.5, 0.95, "Protrusion (all type) analysis", horizontalalignment='center',
                #          transform=fig.transFigure,
                #          size=24, fontweight='bold')
                # plt.text(0.5, 0.93, "Among all Protrusions", horizontalalignment='center',
                #          transform=fig.transFigure,
                #          size=18, style='italic')
                #
                # plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                #          size=10)
                # plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                #          transform=fig.transFigure,
                #          size=10, style='italic')
                # pdf.savefig(fig)  # Save page
                # plt.close()

                ##################################
                ## PAGE 9
                page += 1
                fig = plt.figure(figsize=figsize, constrained_layout=False)

                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="convhull_vertex", feature='type', envir=True,  axs=gs[0], category='type', )
                self.oddsratio_graph(among="convhull_vertex", feature='residue_name', envir=True, axs=gs[1], category='aa', )
                plt.text(0.5, 0.95, "Environment Analysis ", horizontalalignment='center', transform=fig.transFigure,
                         size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among convexhull vertices", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)

                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                pdf.savefig(fig)  # Save page
                plt.close()


                ##################################
                ## PAGE 10
                page += 1
                fig = plt.figure(figsize=figsize, constrained_layout=False)

                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="protrusions", feature="type", envir=True, axs=gs[0], category='type', )
                self.oddsratio_graph(among="protrusions", feature="residue_name", envir=True, axs=gs[1], category='aa', )
                plt.text(0.5, 0.95, "Environment Analysis ", horizontalalignment='center', transform=fig.transFigure,
                         size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among PROTRUSIONS", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page 10/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)
                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                pdf.savefig(fig)  # Save page
                plt.close()


                ##################################
                ## PAGE 10
                page += 1
                fig = plt.figure(figsize=figsize, constrained_layout=False)

                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="is_hydrophobic_protrusion",feature='type',envir=True, axs=gs[0] )
                self.oddsratio_graph(among="is_hydrophobic_protrusion",feature='residue_name',envir=True, axs=gs[1])
                plt.text(0.5, 0.95, "Environment Analysis ", horizontalalignment='center', transform=fig.transFigure,
                         size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among hydrophobic protrusions", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page 10/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)
                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                pdf.savefig(fig)  # Save page
                plt.close()

                ##################################
                ## PAGE 11
                fig = plt.figure(figsize=figsize, constrained_layout=False)

                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="is_co_insertable", feature='type',envir=True, axs=gs[0] )
                self.oddsratio_graph(among="is_co_insertable", feature='residue_name',envir=True, axs=gs[1], )
                plt.text(0.5, 0.95, "Environment Analysis", horizontalalignment='center', transform=fig.transFigure,
                         size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among convexhull co-insertables", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)
                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                pdf.savefig(fig)  # Save page
                plt.close()


                ##################################
                ## PAGE 6
                fig = plt.figure(figsize=figsize, constrained_layout=False)
                page+=1
                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="convhull_vertex", feature = 'type', envir=True, axs = gs[0],condition='exposed')
                self.oddsratio_graph(among="convhull_vertex", feature = 'residue_name',envir=True, axs=gs[1], condition='exposed')
                plt.text(0.5, 0.95, "Environment Analysis (EXPOSED)",horizontalalignment = 'center',  transform=fig.transFigure, size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among convexhull vertices", horizontalalignment='center', transform=fig.transFigure,
                         size=18,style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right',transform=fig.transFigure,
                         size=10)

                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                pdf.savefig(fig)  # Save page
                plt.close()

                ##################################
                ## PAGE 6+1
                fig = plt.figure(figsize=figsize, constrained_layout=False)
                page+=1
                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="protrusions", feature = 'type', envir=True, axs = gs[0], category='type',condition='exposed')
                self.oddsratio_graph(among="protrusions", feature='residue_name',envir=True, axs=gs[1], condition='exposed')
                plt.text(0.5, 0.95, "Environment Analysis (EXPOSED)",horizontalalignment = 'center',  transform=fig.transFigure, size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among all Protrusions", horizontalalignment='center', transform=fig.transFigure,
                         size=18,style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right',transform=fig.transFigure,
                         size=10)

                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                pdf.savefig(fig)  # Save page
                plt.close()

                ##################################
                ## PAGE 7
                page+=1
                fig = plt.figure(figsize=figsize, constrained_layout=False)

                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="is_hydrophobic_protrusion", feature='type',envir=True, axs=gs[0], condition='exposed')
                self.oddsratio_graph(among="is_hydrophobic_protrusion", feature='residue_name',envir=True, axs=gs[1],condition='exposed')
                plt.text(0.5, 0.95, "Environment Analysis (EXPOSED)", horizontalalignment='center', transform=fig.transFigure,
                         size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among hydrophobic protrusions", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)
                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                pdf.savefig(fig)  # Save page
                plt.close()

                ##################################
                ## PAGE 8
                page+=1
                fig = plt.figure(figsize=figsize, constrained_layout=False)

                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="is_co_insertable", feature='type',envir=True, axs=gs[0], condition='exposed')
                self.oddsratio_graph(among="is_co_insertable", feature='residue_name',envir=True, axs=gs[1],condition='exposed')
                plt.text(0.5, 0.95, "Environment Analysis (EXPOSED)", horizontalalignment='center', transform=fig.transFigure,
                         size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among convexhull co-insertables", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)
                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                pdf.savefig(fig)  # Save page
                plt.close()







                ##################################
                ## PAGE 12
                page+=1
                fig = plt.figure(figsize=figsize, constrained_layout=False)

                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="convhull_vertex", feature='type',envir=True, axs=gs[0] , condition='buried')
                self.oddsratio_graph(among="convhull_vertex", feature='residue_name',envir=True, axs=gs[1], condition='buried')
                plt.text(0.5, 0.95, "Environment Analysis (BURIED)", horizontalalignment='center',
                         transform=fig.transFigure, size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among convexhull vertices", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)

                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                pdf.savefig(fig)  # Save page
                plt.close()

                ##################################
                ## PAGE 13
                page+=1
                fig = plt.figure(figsize=figsize, constrained_layout=False)

                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="protrusions", feature='type',envir=True, axs=gs[0], condition='buried')
                self.oddsratio_graph(among="protrusions", feature='residue_name',envir=True, axs=gs[1], condition='buried')
                plt.text(0.5, 0.95, "Environment Analysis (BURIED)", horizontalalignment='center',
                         transform=fig.transFigure, size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among all Protrusions", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)

                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                pdf.savefig(fig)  # Save page
                plt.close()

                ##################################
                ## PAGE 14
                page+=1
                fig = plt.figure(figsize=figsize, constrained_layout=False)

                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="is_hydrophobic_protrusion", feature='type',envir=True, axs=gs[0],
                                     condition='buried')
                self.oddsratio_graph(among="is_hydrophobic_protrusion", feature='residue_name',envir=True, axs=gs[1],
                                     condition='buried')
                plt.text(0.5, 0.95, "Environment Analysis (BURIED)", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among hydrophobic protrusions", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)
                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                pdf.savefig(fig)  # Save page
                plt.close()

                ##################################
                ## PAGE 15
                page+=1
                fig = plt.figure(figsize=figsize, constrained_layout=False)

                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="is_co_insertable", feature='type',envir=True, axs=gs[0], condition='buried')
                self.oddsratio_graph(among="is_co_insertable", feature='residue_name',envir=True, axs=gs[1], condition='buried')
                plt.text(0.5, 0.95, "Environment Analysis (BURIED)", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=24, fontweight='bold')
                plt.text(0.5, 0.93, "Among convexhull co-insertables", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)
                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                pdf.savefig(fig)  # Save page
                plt.close()

                ##################################
                ## non hydrophobic protrusion COMPOSITION
                ##################################
                page += 1
                fig = plt.figure(figsize=(12, 16), constrained_layout=False)
                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="protrusions", feature="sec_struc_full", nohydrprot=True,
                                     axs=gs[0])  # Analysis 1
                self.oddsratio_graph(among="protrusions", feature="type", nohydrprot=True, axs=gs[1])  # Analysis 1
                self.oddsratio_graph(among="protrusions", feature="residue_name", nohydrprot=True,
                                     axs=gs[2])  # Analysis 1
                # self.oddsratio_graph(among="convhull_vertex","prot_block",axs=gs[2])  # Analysis 1

                plt.text(0.5, 0.95, "Oddsratio analysis", horizontalalignment='center', transform=fig.transFigure,
                         size=24, fontweight='bold')
                plt.text(0.5, 0.93, "non-hydrophobic protrusions", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)
                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')

                pdf.savefig(fig)  # Save page
                plt.close()

                ##################################
                ## non hydrophobic protrusion ENVIR
                ##################################
                fig = plt.figure(figsize=figsize, constrained_layout=False)

                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="protrusions", feature='type', nohydrprot=True, envir=True, axs=gs[0])
                self.oddsratio_graph(among="protrusions", feature='residue_name', nohydrprot=True, envir=True,
                                     axs=gs[1], )
                plt.text(0.5, 0.95, "Environment Analysis", horizontalalignment='center', transform=fig.transFigure,
                         size=24, fontweight='bold')
                plt.text(0.5, 0.93, "non-hydrophobic protrusions", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)
                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                pdf.savefig(fig)  # Save page
                plt.close()

                #########################################
                # non hydrophobic protrusion EXPOSED ENVIR
                #########################################
                fig = plt.figure(figsize=figsize, constrained_layout=False)
                page += 1
                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="protrusions", feature='type', envir=True, nohydrprot=True, axs=gs[0],
                                     category='type', condition='exposed')
                self.oddsratio_graph(among="protrusions", feature='residue_name', envir=True, nohydrprot=True,
                                     axs=gs[1], condition='exposed')
                plt.text(0.5, 0.95, "Environment Analysis (EXPOSED)", horizontalalignment='center',
                         transform=fig.transFigure, size=24, fontweight='bold')
                plt.text(0.5, 0.93, "non-hydrophobic protrusions", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)

                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                pdf.savefig(fig)  # Save page
                plt.close()

                #########################################
                # non hydrophobic protrusion BURIED ENVIR
                #########################################
                fig = plt.figure(figsize=figsize, constrained_layout=False)
                page += 1
                gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 0], hspace=0.3)  # Grid configuration
                self.oddsratio_graph(among="protrusions", feature='type', envir=True, nohydrprot=True, axs=gs[0],
                                     category='type', condition='buried')
                self.oddsratio_graph(among="protrusions", feature='residue_name', envir=True, nohydrprot=True,
                                     axs=gs[1], condition='buried')
                plt.text(0.5, 0.95, "Environment Analysis (BURIED)", horizontalalignment='center',
                         transform=fig.transFigure, size=24, fontweight='bold')
                plt.text(0.5, 0.93, "non-hydrophobic protrusions", horizontalalignment='center',
                         transform=fig.transFigure,
                         size=18, style='italic')

                plt.text(0.95, 0.05, f"page {page}/{Npage}", horizontalalignment='right', transform=fig.transFigure,
                         size=10)

                plt.text(0.95, 0.98, f"analysis for {domain} domain(s)", horizontalalignment='right',
                         transform=fig.transFigure,
                         size=10, style='italic')
                pdf.savefig(fig)  # Save page
                plt.close()




                #plt.savefig(folder + "graphs.pdf")

            self.pdf_report(domain, displayHTML)
            # plt.show()

        def pdf_report(self, domain="", displayHTML=False):
            from jinja2 import Environment, FileSystemLoader
            string = """
            <!DOCTYPE html>
            <html>
            <head lang="en">
                <meta charset="UTF-8">
                <title>{{ title }}</title>
                <style>table, td { border: 1px solid black }</style>

            </head>
            <body>
                <h1> {{title}} </h1>
                <div>
                    <p>
                        Count table of protrusion, co-insertable and LDCI (Lowest Density Co-Insertable)
                        {{count_table}}
                    </p>
                </div>                
                {{oddratios}}
                
            </body>
            </html>
            
            """
            if domain == "":
                domain = self.parent.domainDf.domain.unique()[0]
            template = Environment(loader=FileSystemLoader('.')).from_string(string)

            oddratios_html = ''
            oddratios_html += self.oddsratio_dataset(output="html")[3]
            oddratios_html += self.oddsratio_dataset("is_hydrophobic_protrusion", "LDCI", output="html")[3]
            oddratios_html += self.oddsratio_dataset("is_co_insertable", "LDCI", output="html")[3]
            oddratios_html += self.oddsratio_dataset(var1="hydrophobic", var2="buried", among="all", output="html")[3]

            template_vars = {"title": f"ODD RATIO REPORTS for {domain} domain(s)",
                             "count_table": self.tableCount.to_html(border=10),
                             "oddratios": oddratios_html}
            html_out = template.render(template_vars)

            folder = self.parent.FIGURESFOLDER + "report/" + domain + '/'
            if not os.path.exists(folder):
                os.makedirs(folder)

            # Generate the pdf report
            weasyprint.HTML(string=html_out).write_pdf(folder + "report.pdf")
            with open(folder + "report.html", 'w') as output:
                output.write(html_out)
            if displayHTML == True:
                display(HTML(html_out))
            # display(HTML(html_out))

        def plot_residue_composition(self, axs=None, type="all"):
            ibs = self.parent.domainDf.query("IBS == True")
            nonibs = self.parent.domainDf.query("IBS == False")
            if type == "all":
                ibsStat = ibs.query("atom_name == 'CA'").residue_name.value_counts(normalize=True).to_frame()
                ibsStat.columns = ["ibs"]
                nonibsStat = nonibs.query("atom_name == 'CA'").residue_name.value_counts(normalize=True).to_frame()
                nonibsStat.columns = ["nonibs"]
            elif type == "co-insertable":
                ibsStat = ibs.query("atom_name == 'CB' and is_co_insertable == True").residue_name.value_counts(normalize=True).to_frame()
                ibsStat.columns = ["ibs"]
                nonibsStat = nonibs.query("atom_name == 'CB' and is_co_insertable == True").residue_name.value_counts(normalize=True).to_frame()
                nonibsStat.columns = ["nonibs"]
            elif type == "hydrophobic protrusion":
                ibsStat = ibs.query("atom_name == 'CB' and is_hydrophobic_protrusion == True").residue_name.value_counts(
                    normalize=True).to_frame()
                ibsStat.columns = ["ibs"]
                nonibsStat = nonibs.query("atom_name == 'CB' and is_hydrophobic_protrusion == True").residue_name.value_counts(
                    normalize=True).to_frame()
                nonibsStat.columns = ["nonibs"]
            elif type == "protrusion":
                ibsStat = ibs.query("atom_name == 'CB' and protrusion == True").residue_name.value_counts(
                    normalize=True).to_frame()
                ibsStat.columns = ["ibs"]
                nonibsStat = nonibs.query("atom_name == 'CB' and protrusion == True").residue_name.value_counts(
                    normalize=True).to_frame()
                nonibsStat.columns = ["nonibs"]
            stat = pd.concat([ibsStat, nonibsStat], axis=1)
            stat = stat.reset_index().rename(columns={"index": 'Residue'})

            dfGraph = pd.melt(stat, id_vars="Residue", var_name="Localisation", value_name="Composition")
            stat["Difference"] = stat.ibs - stat.nonibs

            #TODO FIX.

            #Remove empty rows
            dfGraph = dfGraph[dfGraph["Composition"] != 0]
            stat = stat[stat["Difference"] != 0]


            dfGraph.Residue = dfGraph.Residue.astype(str)
            stat.Residue = stat.Residue.astype(str)

            if axs == None:
                gs = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[3, 1])
                ax0 = plt.subplot(gs[0])
                ax1 = plt.subplot(gs[1])
            else:
                gs = gridspec.GridSpecFromSubplotSpec(ncols=1, nrows=2, height_ratios=[3, 1], subplot_spec=axs,
                                                      hspace=0.05)
                ax0 = plt.subplot(gs[0])
                ax1 = plt.subplot(gs[1])

            g1 = sns.barplot(x="Residue", y="Composition", hue="Localisation", data=dfGraph, ax=ax0, palette='pastel')
            ax0.text(0.5, 1.1, "Amino acid Composition",  horizontalalignment='center', transform=ax0.transAxes, fontsize=16)
            ax0.text(0.5, 1.02, f"({type}) depending (IBS vs non-IBS)", horizontalalignment='center', transform=ax0.transAxes)
            ax0.set(xlabel="",)
                    #suptitle=f"Amino acid composition:",
                    #title=f"({type}) depending (IBS vs non-IBS)")
            sns.despine(left=True, bottom=True)
            ax0.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=True,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off

            # Second graph
            # Check this for color palette
            colors = [0 if c >= 0 else 1 for c in stat.Difference]
            stat['positive'] = stat['Difference'] > 0
            g2 = sns.barplot(x="Residue", y="Difference", data=stat, palette=stat.positive.map({True: '#ACC9E9', False: '#EEB895'}),
                             ax=ax1)
            #g2.legend_.remove()

            ax1.tick_params(rotation=45)

            if axs == None:
                plt.tight_layout()
                plt.show()
                plt.close()

        def calc_summary_table(self, showTable = False):

            ibs = self.parent.ibs
            nonibs = self.parent.nonibs
            ibsCathPdb = ibs.query("atom_name == 'CB' and data_type != 'prosite' and convhull_vertex == True")
            nonIbsCathPdb = nonibs.query("atom_name == 'CB' and data_type != 'prosite' and convhull_vertex == True")

            # Hydrophobic protrusion
            tc_HP_IBS = _safe_value_counts_with_bools(ibsCathPdb.is_hydrophobic_protrusion.value_counts())
            tc_HP_nonIBS = _safe_value_counts_with_bools(nonIbsCathPdb.is_hydrophobic_protrusion.value_counts())
            percentageHPibs = tc_HP_IBS[1] / tc_HP_IBS[0] * 100
            pertentageHPnonibs = tc_HP_nonIBS[1] / tc_HP_nonIBS[0] * 100

            # CO-insertable
            tc_CO_IBS = _safe_value_counts_with_bools(ibsCathPdb.is_co_insertable.value_counts())
            tc_CO_nonIBS = _safe_value_counts_with_bools(nonIbsCathPdb.is_co_insertable.value_counts())
            percentageCOibs = tc_CO_IBS[1] / tc_CO_IBS[0] * 100
            pertentageCOnonibs = tc_CO_nonIBS[1] / tc_CO_nonIBS[0] * 100

            # MLIP
            tc_LDCI_IBS = _safe_value_counts_with_bools(ibsCathPdb.LDCI.value_counts())
            tc_LDCI_nonIBS = _safe_value_counts_with_bools(nonIbsCathPdb.LDCI.value_counts())
            percentageLDCIibs = tc_LDCI_IBS[1] / tc_LDCI_IBS[0] * 100
            try:
                pertentageLDCInonibs = tc_LDCI_nonIBS[1] / tc_LDCI_nonIBS[0] * 100
            except:
                pertentageLDCInonibs = 0

            #print("###### Hydrophobic protrusion composition #########")
            columns = (("hydrophobic_prostrusion", "yes"),
                       ("hydrophobic_prostrusion", "no"),
                       ("co_insertable", "yes"),
                       ("co_insertable", "no"),
                       ('LDCI', "yes"),
                       ("LDCI", "no")
                       )

            indexes = (("IBS", "Count"),
                       ("IBS", "Percentage"),
                       ("nonIBS", "Count"),
                       ("nonIBS", "Percentage"))

            columns = pd.MultiIndex.from_tuples(columns)
            indexes = pd.MultiIndex.from_tuples(indexes)

            try:
                values = [
                    [tc_HP_IBS[1], tc_HP_IBS[0], tc_CO_IBS[1], tc_CO_IBS[0], tc_LDCI_IBS[1], tc_LDCI_IBS[0]],
                    [percentageHPibs, 100 - percentageHPibs, percentageCOibs, 100 - percentageCOibs, percentageLDCIibs,
                     100 - percentageLDCIibs],
                    [tc_HP_nonIBS[1], tc_HP_nonIBS[0], tc_CO_nonIBS[1], tc_CO_nonIBS[0], tc_LDCI_nonIBS[1],
                     tc_LDCI_nonIBS[0]],
                    [pertentageHPnonibs, 100 - pertentageHPnonibs, pertentageCOnonibs, 100 - pertentageCOnonibs,
                     pertentageLDCInonibs, 100 - pertentageLDCInonibs],

                ]
            except:
                print(tc_HP_IBS)
                print(tc_CO_IBS)
                print(tc_LDCI_IBS)
                print(percentageHPibs)
                print(percentageCOibs)
                print(percentageLDCIibs)

            protrusions = pd.DataFrame(values, columns=columns, index=indexes)
            protrusions.loc[(slice(None), "Count"), :] = protrusions.loc[(slice(None), "Count"), :].astype(
                int).to_numpy()

            self.tableCount = protrusions
            if showTable:
                display(HTML(self.tableCount.to_html()))

        def protrusion_is_coinsertable(self, axs=None):
            sns.set_style(style='whitegrid')

            if axs == None:
                fig = plt.figure(figsize=(10, 16), constrained_layout=True)

                gs = gridspec.GridSpec(ncols=3, nrows=1)
                ax0 = plt.subplot(gs[0])
                ax1 = plt.subplot(gs[1])
                ax2 = plt.subplot(gs[2])
            else:
                gs = gridspec.GridSpecFromSubplotSpec(ncols=3, nrows=1, subplot_spec=axs)
                ax0 = plt.subplot(gs[0])
                ax1 = plt.subplot(gs[1])
                ax2 = plt.subplot(gs[2])

            # ------- IBS
            table_IBS = pd.Series([self.tableCount.loc[("IBS", "Count"), ("co_insertable", "no")] - self.tableCount.loc[
                ("IBS", "Count"), ("hydrophobic_prostrusion", "no")],
                                   self.tableCount.loc[("IBS", "Count"), ("co_insertable", "yes")]])

            table_IBS.index = ["Non Co-insertable", "Co-Insertable"]  #
            colors = sns.color_palette("Set2", len(table_IBS))
            colors = colors[::-1]
            pie1 = table_IBS.plot.pie(
                colors=colors,
                explode=[0.02] * len(table_IBS),
                autopct=lambda p: '{:.2f}% ({:,.0f})'.format(p, p * sum(table_IBS) / 100),
                title="IBS",
                ax=ax0
            )
            # ax0.text(0.5, 0.9, 'IBS', horizontalalignment='center', transform=ax1.transAxes)
            _ = pie1.set_ylabel('')
            # plt.show()

            # --------- NON IBS
            table_nonIBS = pd.Series([self.tableCount.loc[("nonIBS", "Count"), ("co_insertable", "no")] -
                                      self.tableCount.loc[("nonIBS", "Count"), ("hydrophobic_prostrusion", "no")],
                                      self.tableCount.loc[("nonIBS", "Count"), ("co_insertable", "yes")]])

            table_nonIBS.index = ["Non Co-insertable", "Co-Insertable"]  #
            colors = sns.color_palette("Set2", len(table_nonIBS))
            colors = colors[::-1]
            pie2 = table_nonIBS.plot.pie(
                colors=colors,
                explode=[0.02] * len(table_nonIBS),
                autopct=lambda p: '{:.2f}% ({:,.0f})'.format(p, p * sum(table_nonIBS) / 100),
                title="non-IBS",
                ax=ax1
            )
            # ax1.text(0.5, 0.9, 'non-IBS', horizontalalignment='center', transform=ax1.transAxes)
            _ = pie2.set_ylabel('')
            # plt.show()

            # Co-insertable
            tcCI = pd.Series([self.tableCount.loc[("IBS", "Count"), ("co_insertable", "yes")],
                              self.tableCount.loc[("nonIBS", "Count"), ("co_insertable", "yes")]])

            tcCI.index = ["IBS", "Non-IBS"]  #
            colors = sns.color_palette("pastel", len(tcCI))
            #colors = colors[::-1]
            pie3 = tcCI.plot.pie(
                colors=colors,
                explode=[0.02] * len(tcCI),
                autopct=lambda p: '{:.2f}% ({:,.0f})'.format(p, p * sum(tcCI) / 100),
                title="IBS vs non-IBS",
                ax=ax2
            )
            # ax2.text(0.5, 0.9, 'IBS vs non-IBS', horizontalalignment='center', transform=ax1.transAxes)
            _ = pie3.set_ylabel('')

            ax1.text(0.5, 1.2, 'Number of co-insertable amoung hydrophobic protrusions',
                     horizontalalignment='center', transform=ax1.transAxes, fontsize=14, fontweight='bold')

            if axs == None:
                # fig.suptitle("Number of co-insertable amoung hydrophobic protrusions")
                plt.tight_layout()
                plt.show()

        def shannon_entropy(self, axs=None, entropytype='shannonH10', remove_gaps_colums=True):
            from itertools import chain
            graphDataDict = {}

            if axs == None:
                gs = gridspec.GridSpec(ncols=2, nrows=1)
                ax0 = plt.subplot(gs[0])
                ax1 = plt.subplot(gs[1])
            else:
                gs = gridspec.GridSpecFromSubplotSpec(ncols=2, nrows=1, subplot_spec=axs)
                ax0 = plt.subplot(gs[0])
                ax1 = plt.subplot(gs[1])

            coInsertableEntropy = [
                ["IBS" if x else "nonIBS" for x in
                 self.parent.domainDf.query(f'is_co_insertable == is_hydrophobic_protrusion == 1').IBS],
                ["Co-insertable" for x in
                 self.parent.domainDf.query(f'is_co_insertable == is_hydrophobic_protrusion == 1').IBS],
                self.parent.domainDf.query(
                    'is_co_insertable == is_hydrophobic_protrusion ==1')[entropytype].to_numpy().tolist()
            ]

            HydrophobicProtrusionEntropy = [
                ["IBS" if x else "nonIBS" for x in self.parent.domainDf.query('is_hydrophobic_protrusion == 1').IBS],
                ["Hydrophobic protrusion" for x in self.parent.domainDf.query('is_hydrophobic_protrusion == 1').IBS],
                self.parent.domainDf.query('is_hydrophobic_protrusion == 1')[entropytype].to_numpy().tolist()
            ]

            LDCIEntropy = [
                ["IBS" if x else "nonIBS" for x in self.parent.domainDf.query('LDCI == True').IBS],
                ["LDCI" for x in self.parent.domainDf.query('LDCI == True').IBS],
                self.parent.domainDf.query('LDCI == 1')[entropytype].to_numpy().tolist()
            ]

            graphDataDict["Localisation"] = list(chain.from_iterable([HydrophobicProtrusionEntropy[0],
                                                                      coInsertableEntropy[0],
                                                                      LDCIEntropy[0]]))
            graphDataDict["Property"] = list(chain.from_iterable([HydrophobicProtrusionEntropy[1],
                                                                  coInsertableEntropy[1],
                                                                  LDCIEntropy[1]]))
            graphDataDict["Entropy"] = list(chain.from_iterable([HydrophobicProtrusionEntropy[2],
                                                                 coInsertableEntropy[2],
                                                                 LDCIEntropy[2]]))

            graphData = pd.DataFrame.from_dict(graphDataDict)
            #Since we normalize the entropy between 0 and 1 and columns with too much gaps are set to -1, we remove them :-)
            if remove_gaps_colums:
                graphData = graphData.query("Entropy >= 0")
            graphData = graphData.sort_values(by="Localisation")

            g1 = sns.boxplot(x="Property", y="Entropy", hue="Localisation",
                             data=graphData, palette="pastel", ax=ax0).set_title(
                "Shannon Entropy (conservation) per localisation and property")

            # plt.show()

            g2 = sns.violinplot(x="Property", y="Entropy", hue="Localisation",
                                data=graphData, palette="pastel", split=True,
                                scale="count", inner="quartile", ax=ax1).set_title(
                "Shannon Entropy (conservation) per localisation and property")

            if axs == None:
                plt.show()
                plt.tight_layout()
                plt.close()

        def oddsratio_dataset(self, var1="is_hydrophobic_protrusion", var2="is_co_insertable", output="standard", among="convhull_vertex", filter=None):
            """
            give the oddratio and statistical significance of a contingency table (2 variables)
            """
            if among == "all": #All will be considered as CB + GLY's CA (since the properties are tagged on CB only and GLY don't have CB)
                ibsCathPdb = self.parent.domainDf.query(
                    "IBS == True and atom_name == 'CB' or (atom_name == 'CA' and residue_name == 'GLY') and data_type != 'prosite'")
                nonIbsCathPdb = self.parent.domainDf.query(
                    "IBS == False and atom_name == 'CB' or (atom_name == 'CA' and residue_name == 'GLY') and data_type != 'prosite'")
            else:
                ibsCathPdb = self.parent.domainDf.query(
                    "IBS == True and atom_name == 'CB' and data_type != 'prosite' and convhull_vertex == True")
                nonIbsCathPdb = self.parent.domainDf.query(
                    "IBS == False and atom_name == 'CB' and data_type != 'prosite' and convhull_vertex == True")

            if filter == "buried":
                ibsCathPdb = self.parent.domainDf.query(
                    "IBS == True and atom_name == 'CB' and data_type != 'prosite' and convhull_vertex == True")
                nonIbsCathPdb = self.parent.domainDf.query(
                    "IBS == False and atom_name == 'CB' and data_type != 'prosite' and convhull_vertex == True")

            if var1 == "hydrophobic":
                hydrophobic_resname = ["LEU","ILE","CYS","MET","TYR","TRP","PHE"]
                ibsCathPdb["hydrophobic"] = ibsCathPdb.residue_name.apply(lambda x: True if x in hydrophobic_resname else False)
                nonIbsCathPdb["hydrophobic"] = nonIbsCathPdb.residue_name.apply(lambda x: True if x in hydrophobic_resname else False)
            elif var1 == "non_hydrophobic_protrusion_but_hydrophobic":
                hydrophobic_resname = ["LEU", "ILE", "CYS", "MET", "TYR", "TRP", "PHE"]
                ibsCathPdb["non_hydrophobic_protrusion_but_hydrophobic"] = ibsCathPdb.apply(
                    lambda x: True if (x.residue_name in hydrophobic_resname and x.is_hydrophobic_protrusion == False) else False, axis=1)
                nonIbsCathPdb["non_hydrophobic_protrusion_but_hydrophobic"] = nonIbsCathPdb.apply(
                    lambda x: True if (x.residue_name in hydrophobic_resname and x.is_hydrophobic_protrusion == False) else False, axis=1)

            if var2 == "buried":
                ibsCathPdb["buried"] = ibsCathPdb.RSA_freesasa_florian.apply(lambda x: True if x < 20 else False)
                nonIbsCathPdb["buried"] = nonIbsCathPdb.RSA_freesasa_florian.apply(lambda x: True if x < 20 else False)
            elif var2 == "exposed":
                ibsCathPdb["exposed"] = ibsCathPdb.RSA_freesasa_florian.apply(lambda x: True if x >= 20 else False)
                nonIbsCathPdb["exposed"] = nonIbsCathPdb.RSA_freesasa_florian.apply(lambda x: True if x >= 20 else False)

            NumProtIBS = len(ibsCathPdb.query(f"{var1} == True"))
            NumProtNONIBS = len(nonIbsCathPdb.query(f"{var1} == True"))
            sns.set_style(style='whitegrid')
            # TO DO: discuss proper handling of the border case where all entries are True/False
            #tc_CI_ibs = ibsCathPdb.query(f"{var1} == True")[var2].value_counts()  # Create table count
            #tc_CI_ibs.rename(index={0: "No", 1: "Yes"}, inplace=True)
            tc_CI_ibs = _safe_value_counts_with_bools(ibsCathPdb.query(f"{var1} == True")[var2].value_counts())  # Create table count
            #print(tc_CI_ibs)
            fracCI_IBS = tc_CI_ibs[1] / sum(tc_CI_ibs)

            # TO DO: discuss proper handling of the border case where all entries are True/False
            #tc_CI_nonIBS = nonIbsCathPdb.query(f"{var1} == True")[var2].value_counts()  # Create table count
            #tc_CI_nonIBS.rename(index={0: "No", 1: "Yes"}, inplace=True)
            tc_CI_nonIBS = _safe_value_counts_with_bools(nonIbsCathPdb.query(f"{var1} == True")[var2].value_counts())  # Create table count
            fracCI_NONIBS = tc_CI_nonIBS[1] / sum(tc_CI_nonIBS)

            table_oddsratio = [[tc_CI_ibs[1], tc_CI_ibs[0]],
                               [tc_CI_nonIBS[1], tc_CI_nonIBS[0]]
                               ]

            oddsratioDf = pd.DataFrame(table_oddsratio)
            oddsratioDf.columns = pd.MultiIndex.from_product([[var2], ['yes', 'no']])
            oddsratioDf.index = pd.MultiIndex.from_product([["IBS"], ['yes', 'no']])

            oddsratio, pvalue = fisher_exact(table_oddsratio)
            flag_null = (oddsratio == 0)

            # Edvin's formula
            Rab = (fracCI_IBS * (1 - fracCI_NONIBS)) / (fracCI_NONIBS * (1 - fracCI_IBS))

            # oddsratio = math.log(oddsratio)
            # Standard deviation
            # TO DO: discuss proper handling of null odds ratio
            ###
            stddev_summands = [table_oddsratio[0][0], 
                               table_oddsratio[0][1],
                               table_oddsratio[1][0],
                               table_oddsratio[1][1]]
            se = sum([1/x for x in stddev_summands if x != 0])**0.5
            #se = (1 / table_oddsratio[0][0] + 1 / table_oddsratio[0][1] + 1 / table_oddsratio[1][0] + 1 /
            #      table_oddsratio[1][1]) ** 0.5

            # Confidance Interval at 95% Wald, z=1.96
            if not flag_null:
                lower_CI = math.exp(math.log(oddsratio) - 1.96 * se)
                upper_CI = math.exp(math.log(oddsratio) + 1.96 * se)
            else:
                lower_CI = np.NaN
                upper_CI = np.NaN

            if output == 'standard':
                print("")
                print(colored("###### - ODDSRATIO ESTIMATOR - IBS/NONIBS ######", 'blue', attrs=['bold']))
                print("...................")
                print(f"Variable 1: {var1}")
                print(f"Variable 2: {var2}")
                print("...................")
                display(HTML(oddsratioDf.to_html()))
                print("-------")
                if not flag_null:
                    print(f"Oddratio = {oddsratio} - LOG -> {np.log(oddsratio):.2f}")
                    print(f"Pvalue = {pvalue} (fisher)")
                    print(f"Uncertainty = {abs(oddsratio - lower_CI)} (fisher) - LOG -> {abs(np.log(oddsratio) - np.log(lower_CI)):.2f}")
                    print(f"Standard deviation = {se:.2f} - LOG -> {np.log(se):.2f}")

                    print(f"Confidance Interval at 95% (Wald) {lower_CI:.2f}, {upper_CI:.2f}")
                    print("-------")
                    if 1 < lower_CI:
                        print(colored(">> 1 lower than lower confidence interval, result is statistically relevant.",
                                      "green"))
                        print(f">> {var1} are {oddsratio:.2f} times more likely to be {var2} on IBS than non-IBS")
                    else:
                        print(colored(">> 1 in confidence interval, result IS NOT statistically relevant.", "red"))
                        print(f"{var1} are {oddsratio:.2f} times more likely to be {var2} on IBS than non-IBS")
                    return (oddsratio, lower_CI, upper_CI)
                else:
                    print("* Warning: the Fisher exact test returned an odds ratio of zero")
                    print("*          ignore confidence intervals and statistical relevance")
                    print(f"Oddratio = {oddsratio} - LOG -> {np.NaN}")
                    print(f"Pvalue = {pvalue} (fisher)")
                    print("-------")
                    return (oddsratio, np.NaN, np.NaN)


            elif output == "html":
                HTMLstring = f"""
                <div style="border:3px; border-style:dashed; border-color:#808080; padding: 1em; margin:10px;>
                <p">
                {self.span("###### - ODDSRATIO ESTIMATOR - IBS/NONIBS ######", 'blue', 'bold')}
                <br />
                {self.span('Variable 1', 'black', 'bold')}: {var1}<br />
                {self.span('Variable 2', 'black', 'bold')}: {var2}<br />
                {oddsratioDf.to_html()}
                """
                if not flag_null:
                    HTMLstring += f"""
                    <ul>
                    <li>OddRatio = {oddsratio:.2}</li>
                    <li>Pvalue = {pvalue:.2} (fisher)</li>
                    <li>Standard deviation = {se:.2f}</li>
                    <li>Confidance Interval at 95% (Wald) {lower_CI:.2f}, {upper_CI:.2f}</li>
                    <br />
                    """
                    if 1 < lower_CI:
                        HTMLstring += self.span(
                            ">> 1 lower than lower confidence interval, result is statistically relevant",
                            "green") + ".<br />"
                        HTMLstring += f">> {var1} are " + self.span(f"{oddsratio:.2f}", "black",
                                                                    "bold") + f" times more likely to be {var2} on IBS than non-IBS<br />"
                    else:
                        HTMLstring += self.span(">> 1 is in the confidence interval, result is NOT statistically relevant",
                                                "red", "bold") + ".<br />"
                        HTMLstring += f">> {var1} are " + self.span(f"{oddsratio:.2f}", "black",
                                                                    "bold") + f" times more likely to be {var2} on IBS than non-IBS<br />"
                    HTMLstring += "</p></div>"
                    # display(HTML(HTMLstring))
                    return (oddsratio, lower_CI, upper_CI, HTMLstring)
                else:
                    HTMLstring += f"""
                    <ul>
                    <li>OddRatio = {oddsratio:.2}</li>
                    <li>Pvalue = {pvalue:.2} (fisher)</li>
                    <br />
                    """
                    HTMLstring += self.span("The Fisher exact test returned an odds ratio of zero, cannot determine confidence intervals and statistical relevance",
                                            "red", "bold") + ".<br />"
                    return (oddsratio, np.NaN, np.NaN, HTMLstring)

        def span(self, string, color, style="standard"):
            cdict = {"red": "#FF0000",
                     "blue": "#0000FF",
                     "green": "#32CD32",
                     "black": "#000000"}
            sdict = {"standard": "",
                     "bold": "; font-weight: bold;"}

            span = f'<span style = "color: {cdict[color]}{sdict[style]}" > {string} </span>'
            return span

        def protein_fraction(self, axs=None):


            if axs == None:
                fig = plt.figure(figsize=(15,3), constrained_layout=True)

                gs = gridspec.GridSpec(ncols=3, nrows=1)
                ax0 = plt.subplot(gs[0])
                ax1 = plt.subplot(gs[1])
                ax2 = plt.subplot(gs[2])
            else:
                gs = gridspec.GridSpecFromSubplotSpec(ncols=3, nrows=1, subplot_spec=axs)
                ax0 = plt.subplot(gs[0])
                ax1 = plt.subplot(gs[1])
                ax2 = plt.subplot(gs[2])
                plt.tight_layout()

            ####################################################################################################
            ibsStat = self.parent.ibs.query("atom_name == 'CB' and convhull_vertex == True").groupby(
                "cathpdb").is_co_insertable.sum().to_frame()
            ibsStat.columns = ["ibs"]
            nonibsStat = self.parent.nonibs.query("atom_name == 'CB' and convhull_vertex == True").groupby(
                "cathpdb").is_co_insertable.sum().to_frame()
            nonibsStat.columns = ["nonibs"]
            stat = pd.concat([ibsStat, nonibsStat], axis=1)
            stat = stat.reset_index().rename(columns={"index": 'cathpdb'})

            dfGraph = pd.melt(stat, id_vars="cathpdb", var_name="Localisation", value_name="hydrophobic_protrusions")
            test = dfGraph.groupby("Localisation").apply(lambda x: (x.hydrophobic_protrusions.value_counts()))
            try:
                test = test.reindex(range(0,
                                   int(np.nanmax(test.index.get_level_values(1))) +1),
                          fill_value=0,
                          level=1)




            except: #I don't know why but sometimes I have a dataframe. I have to handle it differently....

                test = test.T.unstack(level=1,fill_value=0)
                test.index.set_names( ['Localisation', None], inplace=True)
                test.rename("hydrophobic_protrusions", inplace=True)


            # NB! FutureWarning: Using the level keyword in DataFrame and Series aggregations is deprecated and will be removed in a future version. Use groupby instead. df.sum(level=1) should use df.groupby(level=1).sum().
            #test = test/test.sum(level=0)
            test = test/(test.groupby(level=0).sum())
            dataGraph = test.to_frame()
            # dataGraph = dfGraph.groupby("Localisation").apply(
            #     lambda x: (x.hydrophobic_protrusions.value_counts() \
            #                .reindex(range(0, int(np.nanmax(x.hydrophobic_protrusions + 1))), fill_value=0) \
            #                / sum(x.hydrophobic_protrusions.value_counts()))
            # ).to_frame()
            dataGraph.reset_index(inplace=True)
            dataGraph.rename(
                columns={"hydrophobic_protrusions": "Protein fraction", "level_1": "hydrophobic_protrusions"},
                inplace=True)

            colors = sns.color_palette("pastel", 2)

            sns.barplot(x="hydrophobic_protrusions", y="Protein fraction", hue="Localisation", data=dataGraph,
                            palette=colors, ax=ax0,linewidth=0)
            ax0.set(ylabel='Frac. Protein (%)')
            ax0.text(0.5,1.1, "Protein Fraction", horizontalalignment='center', transform=ax0.transAxes, fontsize=15)
            ax0.text(0.5, 1.02, "Number of hydrophobic Protrusions", horizontalalignment='center', transform=ax0.transAxes, fontsize=12)


            ####################################################################################################
            # Percentage of protrusions that are hydrophobic.
            ibsStat = self.parent.ibs.query("atom_name == 'CB' and convhull_vertex == True").groupby("cathpdb").apply(lambda
                                                                                                                      x: x.is_hydrophobic_protrusion.sum() / x.protrusion.sum() if x.protrusion.sum() > 0 else 0).to_frame()
            ibsStat.columns = ["ibs"]
            nonibsStat = self.parent.nonibs.query("atom_name == 'CB' and convhull_vertex == True").groupby("cathpdb").apply(
                lambda
                    x: x.is_hydrophobic_protrusion.sum() / x.protrusion.sum() if x.protrusion.sum() > 0 else 0).to_frame()
            nonibsStat.columns = ["nonibs"]
            stat = pd.concat([ibsStat, nonibsStat], axis=1)
            stat = stat.reset_index().rename(columns={"index": 'cathpdb'})

            dfGraph = pd.melt(stat, id_vars="cathpdb", var_name="Localisation", value_name="hydrophobic_protrusions")

            dataGraph = pd.concat([ibsStat, nonibsStat], axis=1).to_numpy()

            # Quick trick to get data into percentage
            weights = [np.ones(len(dataGraph)) / len(dataGraph),
                       np.ones(len(dataGraph)) / len(dataGraph)]

            ax1.hist(dataGraph, label=['IBS', 'nonIBS'], weights=weights, align='left', edgecolor=None, linewidth=0,color=colors)
            ax1.legend(loc='upper right')


            ax1.set(ylabel='Frac. Protein (%)',
                    xlabel = 'protrusion | hydrophobe (%)',
                    ylim = (0,1))
            ax1.text(0.5, 1.1, "Protein Fraction", horizontalalignment='center', transform=ax1.transAxes, fontsize=15)
            ax1.text(0.5, 1.02, "Protrustion that are also hydrophobic", horizontalalignment='center',
                     transform=ax1.transAxes, fontsize=12)


            ####################################################################################################

            ibsStat = self.parent.ibs.query("atom_name == 'CB' and convhull_vertex == True").groupby("cathpdb").apply(lambda
                                                                                                                      x: x.is_co_insertable.sum() / x.is_hydrophobic_protrusion.sum() if x.is_hydrophobic_protrusion.sum() > 0 else 0).to_frame()
            ibsStat.columns = ["ibs"]
            nonibsStat = self.parent.nonibs.query("atom_name == 'CB' and convhull_vertex == True").groupby("cathpdb").apply(
                lambda
                    x: x.is_co_insertable.sum() / x.is_hydrophobic_protrusion.sum() if x.is_hydrophobic_protrusion.sum() > 0 else 0).to_frame()
            nonibsStat.columns = ["nonibs"]
            stat = pd.concat([ibsStat, nonibsStat], axis=1)
            stat = stat.reset_index().rename(columns={"index": 'cathpdb'})

            dataGraph = pd.concat([ibsStat, nonibsStat], axis=1).to_numpy()

            # Quick trick to get data into percentage
            weights = [np.ones(len(dataGraph)) / len(dataGraph),
                       np.ones(len(dataGraph)) / len(dataGraph)]

            ax2.hist(dataGraph, label=['IBS', 'nonIBS'], weights=weights, align='left',linewidth=0,color=colors)
            ax2.legend(loc='upper right')

            ax2.set(ylabel='Frac. Protein (%)',
                    xlabel='hydrophobic protrusion | co-insertable (%)',
                    ylim=(0, 1))
            ax2.text(0.5, 1.1, "Protein Fraction", horizontalalignment='center', transform=ax2.transAxes, fontsize=15)
            ax2.text(0.5, 1.02, "Hydrophobic protrusions that are co-insertable", horizontalalignment='center',
                     transform=ax2.transAxes, fontsize=12)

            if axs == None:
                plt.tight_layout()
                plt.show()
                plt.close()

        def oddsratio_calculation(self, group):
            """
            TODO -> Definition
            :param group:
            :return:
            """
            table_oddsratio = [[group.loc[(True,), ("Count")].values[0], group.loc[(True,), ("nonSS")].values[0]],
                               [group.loc[(False,), ("Count")].values[0], group.loc[(False,), ("nonSS")].values[0]]
                               ]


            ss = group.index[0][1]  # get SS Type
            oddsratioDf = pd.DataFrame(table_oddsratio)
            oddsratioDf.columns = pd.MultiIndex.from_product([[f"SS:{ss}"], ['yes', 'no']])
            oddsratioDf.index = pd.MultiIndex.from_product([["IBS"], ['yes', 'no']])

            oddsratio, pvalue = fisher_exact(table_oddsratio)


            se = (1 / table_oddsratio[0][0] + 1 / table_oddsratio[0][1] + 1 / table_oddsratio[1][0] + 1 /
                  table_oddsratio[1][1]) ** 0.5

            if not oddsratio == 0:
                lower_CI = math.exp(math.log(oddsratio) - 1.96 * se)
                upper_CI = math.exp(math.log(oddsratio) + 1.96 * se)
            else:
                lower_CI = np.NaN
                upper_CI = np.NaN

            return pd.DataFrame([[oddsratio, pvalue, lower_CI, upper_CI]],
                                columns=["oddsratio", "pvalue", "lower_CI", "upper_CI"]
                                )

        def generate_OR_graph(self,oddsGraph, **kwargs):
            among = kwargs.get('among')
            feature = kwargs.get('feature')
            axs = kwargs.get('axs')
            category = kwargs.get('category')
            condition = kwargs.get('condition')
            xlim = kwargs.get('xlim')
            exclude = kwargs.get('exclude')
            output = kwargs.get('output')
            transparent = kwargs.get('transparent')
            removeXAxis = kwargs.get('removeXAxis')
            colorZone = kwargs.get('colorZone')
            return_dataset = kwargs.get('return_dataset')
            palette = kwargs.get('palette')
            title = kwargs.get('title')
            hue = kwargs.get('hue')
            figsize = kwargs.get('figsize')

            if feature == "residue_name":
                # Very specific to new pandas version (1.1.x?)
                # I convert the residue name as category to have a smaller dataset in memory or disk
                # But right know Pandas seaborn think that I have 21 amino acids (because some amino acids can be "UNK")
                # So I have to convert to string again to be sure that I have only the right amont of data...
                oddsGraph.residue_name = oddsGraph.residue_name.astype(str)

            # Reorder dataframe to be consistant with Seaborn order
            if hue != None:
                # oddsGraph = oddsGraph.sort_values([feature], axis=0, ascending=True, ignore_index=True).sort_values(
                #    [hue], axis=0, ascending=True, ignore_index=True)
                oddsGraph = oddsGraph.sort_values([hue, feature], ignore_index=True)
                order = oddsGraph[feature].unique()
                hueorder = oddsGraph[hue].unique()
            else:
                oddsGraph = oddsGraph.sort_values([feature], axis=0, ignore_index=True)
                order = oddsGraph[feature].unique()
                hueorder = None

            # Create an axis if nothing is given
            if axs == None:
                if figsize == None:
                    figsize = (10, 5)
                fig, ax = plt.subplots(figsize=figsize)
            else:
                gs = gridspec.GridSpecFromSubplotSpec(ncols=1, nrows=1, subplot_spec=axs)
                ax = plt.subplot(gs[0])


            # EditGraph for black and white output
            if palette == 'bw':
                cpalette = sns.color_palette("Greys_r", as_cmap=False)
                g = sns.barplot(data=oddsGraph, y=feature, x='oddsratio', orient='h', hue=hue, ax=ax,
                                palette=cpalette, order=order, hue_order=hueorder)
            else:
                g = sns.barplot(data=oddsGraph, y=feature, x='oddsratio', orient='h', hue=hue, ax=ax, order=order,
                                hue_order=hueorder, palette=sns.color_palette("pastel")
                                )
            # Get the ylim, needed for some features
            ylim = g.get_ylim()
            # get the position of each bar (usefull to add error bar and pvalue)
            positions = [x.get_y() + (x.get_height() / 2) for x in g.patches]

            # Defin Xlim if xlim is not given
            if xlim == None:
                left = oddsGraph.replace([np.inf, -np.inf], np.nan).lower_CI.min() - 0.25
                right = oddsGraph.replace([np.inf, -np.inf], np.nan).upper_CI.max() + 0.25
                # Include 0 in the intervale, if there are only negative or positive values
                if not left < 0 < right:
                    if left < 0:
                        right = 0.25
                    else:
                        left = -0.25
                # Last check, if we only have inf or nan values.
                if np.isnan(left):
                    left = -0.25
                if np.isnan(right):
                    right = 0.25

                xlim = (left, right)






            ################
            # DRAW MARKERS #
            ################
            markersize = g.patches[0].get_window_extent().height / 0.7
            #Sometimes, when the first bar is np.inf or -np.iinf, the height is nan. which
            # means that is will not be possible to estimate the marker size
            # this patch is to get a markersize (realnumber) no matter what
            if np.isnan(markersize):
                for patchnum in range(1,len(g.patches)):
                    markersize = g.patches[patchnum].get_window_extent().height / 0.7
                    if not np.isnan(markersize):
                        break
                if np.isnan(markersize):
                    markersize = 1


            if markersize < 20:  # Arbitrary value
                linewidth = 0.45
            else:
                linewidth = 1
            for i in range(len(oddsGraph)):
                y = positions[i]
                row = oddsGraph.loc[i]
                # Draw error bar
                ax.plot((row["lower_CI"], row["upper_CI"]),
                        (y, y),
                        '-',
                        c='black',
                        linewidth=linewidth)
                #adjust linewidth
                if g.patches[0].get_window_extent().height < 30:
                    g.patches[i].set_linewidth(0.05)

                if row["oddsratio"] == -np.inf:
                    ax.plot([0, xlim[0]], (y, y), color='gray')
                elif row["oddsratio"] == np.inf:
                    ax.plot([0, xlim[1]], (y, y), color='gray')


                #Draw pvalue markers
                if row["pvalue"] < 0.05:
                    plt.scatter(0, y, marker='o', c='g', zorder=100, s=markersize)
                else:
                    plt.scatter(0, y, marker='X', c='r', zorder=101, s=markersize)

            # DRAW MAIN AXIS
            ax.plot([0, 0], ylim, 'k--', )

            if title == None:
                title = f"(Log) Odds ratio for '{feature}' among '{among}' group"

            ax.set(title=title, xlim=xlim, ylim=ylim, xlabel='LOG(oddsratio)')

            # Add rectangle (IBS/NonIBS)
            # Create a Rectangle patch
            if (not palette == 'bw') or (palette == None):
                from matplotlib import patches
                rectNONIBS = patches.Rectangle((xlim[0], -1), np.abs(0 - xlim[0]), len(oddsGraph) + 1, linewidth=0,
                                               edgecolor='none', facecolor='#ffd7c9', zorder=-10)
                rectIBS = patches.Rectangle((0, -1), xlim[1], len(oddsGraph) + 1, linewidth=0,
                                            edgecolor='none', facecolor='#e9e9ff', zorder=-10)

                # Add the patch to the Axes
                if colorZone != False:
                    ax.add_patch(rectNONIBS)
                    ax.add_patch(rectIBS)

            if removeXAxis == True:
                plt.tick_params(
                    axis='y',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    # bottom=False,  # ticks along the bottom edge are off
                    # top=False,  # ticks along the top edge are off
                    labelleft=False)  # labels along the bottom edge are off

            if axs == None:
                if not output is None:
                    plt.savefig(output, transparent=transparent)
                else:
                    plt.show()


            return g

        def oddsratio_graph(self,
                            among='CB',
                            feature='sec_struc_full',
                            category = 'Type',
                            nohydrprot = None,
                            axs = None,
                            condition = None,
                            xlim = None,
                            envir=None,
                            exclude = None,
                            output = None,
                            transparent = None,
                            removeXAxis = None,
                            colorZone = None,
                            return_dataset = None,
                            palette = None,
                            title = None,
                            hue = None,
                            include = None,
                            subsel=None,
                            return_tablecount=False,
                            exclude_protrusion=False,
                            envirPerDomain=False,
                            **kwargs,
                            ):
            """

            Args:
                among: Among which type of AA ? (Protrusions/co-insertable/convex_hull/CB...)
                feature: "Which feature to look ? (Secondary structures, type, residue_name...)
                **kwargs: Possible arguments :
                - categogy : Group by category ? ('type','aa')
                - axs: Insert output graph into given axis
                - condition: Specific condition for analysis ('exposed'/'buried')
                - xlim: xlim for axis ([left,right])
                - exclude: Exclude a specific group of feature (like ['polar','apolar'] for aa type)
                - output: Output path for graph is wanted
                - transperent: With (True) transparency background on the figure ?
                - removeXAxis: Remove Xaxis
                -

            Returns:

            """


            AATYPE = {
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

            #Reset AATYpe if needed... #TODO -> Remove this for final prod, it's only a trick to avoid recalculating the full dataset if we want to change a definition
            self.parent.domainDf["type"] = self.parent.domainDf.residue_name.apply(lambda x: AATYPE[x])


            # ## Add "exposition columns" #TODO --> Add it in the generation of the dataset
            # self.parent.domainDf["exposition"] = np.where(self.parent.domainDf['RSA_freesasa_florian'] >= 20, "exposed", "buried")
            # self.parent.ibs["exposition"] = np.where(self.parent.ibs['RSA_freesasa_florian'] >= 20, "exposed",
            #                                               "buried")
            # self.parent.nonibs["exposition"] = np.where(self.parent.nonibs['RSA_freesasa_florian'] >= 20, "exposed",
            #                                          "buried")
            # self.parent.dataset["exposition"] = np.where(self.parent.dataset['RSA_freesasa_florian'] >= 20, "exposed",
            #                                          "buried")



            if among == 'CB':
                df = self.parent.domainDf.query("atom_name == 'CB'")
            elif among in ['protrusions','protrusion']:
                df = self.parent.domainDf.query("protrusion == True")
            elif among in ['HP']:
                df = self.parent.domainDf.query("is_hydrophobic_protrusion == True")
            elif among in ['positive_protrusions']:
                df = self.parent.domainDf.query("protrusion == True and type == 'Positive'")
            else:
                if subsel:
                    df = self.parent.domainDf.query(f"{among} == True and {subsel}")
                else:
                    df = self.parent.domainDf.query(f"{among} == True")

            # Reasign type, to have the same AATYPE everywhere
            df["type"] = df.residue_name.apply(lambda x: AATYPE[x])

            if nohydrprot == True:
                def count_hydr_protr_perpdb(group):
                    g = group.query("IBS == True and is_hydrophobic_protrusion == True and atom_name == 'CB'")[
                        ["residue_name", "residue_number"]].drop_duplicates()
                    return (len(g))

                hydr_per_pdbs = df.groupby("cathpdb").apply(lambda group: count_hydr_protr_perpdb(group))
                pdbs_no_hydr = list(hydr_per_pdbs[hydr_per_pdbs == 0].index)
                df = df.query("cathpdb in @pdbs_no_hydr")



            if feature == "sec_struc_full":
                transformSSFull = {"H":'Helix',
                                   "G":'Helix',
                                   "I":'Helix',
                                   "B":'Beta',
                                   "E":'Beta',
                                   "T":'Bend',
                                   "S":'Turn',
                                   "-":'Coil',}
                df.sec_struc_full = df.sec_struc_full.apply(lambda x: transformSSFull[x])

            if envir == True:
                # note: same as Florian's definition (which is same as Edvin's definition )


                def calc_envir_counttable(group,feature,datasetLight, condition=None, exclude_protrusion=True):
                    cathpdb = group.cathpdb.unique()[0]
                    cathpdbDF = datasetLight.query("cathpdb == @cathpdb")

                    number_as_index = cathpdbDF.set_index("residue_number", drop=False).drop_duplicates('residue_number')


                    if condition == 'exposed':
                        rel_cutoff = 20
                        sign = '>='
                    elif condition == 'buried':
                        rel_cutoff = 20
                        sign = '<'
                    else:
                        rel_cutoff = 0
                        sign = '>='


                    neighboursID = group[group.neighboursID.notnull()].neighboursID
                    envir = []
                    #here we will go through every neighbors
                    for idstr in neighboursID:
                        if isinstance(idstr, str) and len(idstr) > 0: #if we actually have neighbors
                            for id in map(int,idstr.split(";")): #The neighbors list looks like "12;23;2" with the respective residue_number, so we have to split it and convert it into a integer
                                try:
                                    sasa = number_as_index.loc[id,"RSA_freesasa_florian"] # we get the SASA value
                                    is_protrusion = number_as_index.loc[id,"protrusion"] # we take its protrusion tag
                                    if exclude_protrusion == True: # If we choose to ignore protrusion from the analysis
                                        if is_protrusion == False or pd.isna(is_protrusion): # BUT if the neighbor is not a protrusion
                                            if eval(f"sasa {sign} {rel_cutoff}") : # and its sasa value is respected
                                                envir.append(number_as_index.loc[id,feature]) # Then we add it to our neighbours list

                                    else: #Otherwise, if we want to include all amino acid (protrusion/non protrusion)
                                        if eval(f"sasa {sign} {rel_cutoff}"): # we check the SASA values
                                            envir.append(number_as_index.loc[id, feature]) # we add it to our neighbours list
                                except: #This exception is for Alphafold filter on pLDDT
                                    # print("ID", id)
                                    # print("NUMBER_AS_INDEX", number_as_index)
                                    # print("CATHPDBDF", cathpdbDF)
                                    #print(f"error with pdb {cathpdb} - ignoring it.")
                                    #return None #Use return None to ignore FULL Structure
                                    pass


                    envir=pd.Series(envir)


                    #Each row is a list of neighbour type/aa. Explode will flatten this to have a propoer table count
                    envir = envir.explode().reset_index(drop=True)
                    if feature == 'type':
                        #if not exclude is None:
                        #    envir = envir[~envir.isin(exclude)]
                        #BUT, since amino acids can have several types (aromatic/hydrophobic) we have to transform this type
                        #with a 'split' (since types are separated by a coma
                        envir = envir.dropna().apply(lambda x: x.split(','))
                        #And flattend the list again
                        envir = envir.explode().reset_index(drop=True)


                    tc = envir.value_counts().to_frame().reset_index().rename(
                        columns={"index": feature, 'neighboursID': 'Count', 0: 'Count', 'residue_name':'Count'}
                    )
                    tc["IBS"] = group.IBS.unique()[0]
                    tc.set_index(['IBS', feature], inplace=True)
                    return (tc)

                domainList = self.parent.domainLabel.split('+')

                datasetLight = self.parent.domainDf.query("atom_name in 'CA' and data_type != 'prosite' and domain in @domainList") #get a 'light' version of the original dataset (only CA for pdbs) to have faster queries.
                #Apply "PROTRUSION" labels on CA as well.

                #Get the tags
                protrusion_tags = self.parent.domainDf.query("atom_name == 'CB'").set_index(["cathpdb", "residue_number"]).protrusion
                #set the same index for our "light dataset"
                datasetLight = datasetLight.set_index(["cathpdb", "residue_number"])
                #set the dag
                datasetLight["protrusion"] = protrusion_tags
                #reset the index (the way it was before)
                datasetLight = datasetLight.reset_index()



                if envirPerDomain:#specific patch for "Per domain + environment + composition analysis":
                    df2 = df.copy()
                    temp_list = []
                    for dom in df.domain.unique():
                        dfgroup = df2.query('domain == @dom')
                        r = dfgroup.query("IBS == True").groupby(["cathpdb"], as_index=False).apply(lambda x: calc_envir_counttable(x,
                                                                                                                 feature,
                                                                                                                 datasetLight,
                                                                                                                 condition,
                                                                                                                 exclude_protrusion,
                                                                                                                 )
                                                                                               )

                        r = r.dropna()
                        r = r.sum(axis=0, level=[1, 2])


                        r["domain"] = dom
                        temp_list.append(r)

                    df2 = pd.concat(temp_list).droplevel('IBS')
                    values = df2.groupby("domain").apply(lambda x: x["Count"] / x["Count"].sum() * 100).to_frame()
                    df2 = df2.reset_index().set_index(["domain", "residue_name"])
                    df2["Percentage"] = values
                    df2 = df2[['Percentage']]
                    return(df2)

                else:
                    r = df.groupby(["IBS", "cathpdb"], as_index=False).apply(lambda x: calc_envir_counttable(x,
                                                                                                             feature,
                                                                                                             datasetLight,
                                                                                                             condition,
                                                                                                             exclude_protrusion,
                                                                                                             ))

                    # NB! FutureWarning: Using the level keyword in DataFrame and Series aggregations is deprecated and will be removed in a future version. Use groupby instead. df.sum(level=1) should use df.groupby(level=1).sum().
                    #tableCount = r.sum(axis=0, level=[1, 2])
                    tableCount = r.groupby(axis=0, level=[1, 2]).sum()

                    index_l1 = tableCount.index.get_level_values(1).unique().tolist()

            else:
                if feature == 'type':
                    df[feature] = df[feature].str.split(',')
                    df = df.explode(feature)

                #if not exclude is None:
                #    df = df.query("type not in @exclude")
                tableCount = df.groupby(["IBS"])[feature].value_counts() \
                    .rename('Count') \
                    .to_frame()

                index_l1 = df[feature].unique()


            # Automatic filling missing values
            index_l0 = [False, True]
            index = []
            for l0 in index_l0:
                for l1 in index_l1:
                    index.append((l0, l1))

            tableCount = tableCount.reindex(index, fill_value=0)
            tableCount["total"] = tableCount.groupby('IBS')['Count'].transform(sum)
            tableCount["nonSS"] = tableCount["total"] - tableCount["Count"]


            odds = tableCount.groupby(feature).apply(lambda x: self.oddsratio_calculation(x)).reset_index(level=1, drop=True)

            if (feature == "type" or category == "type") and not exclude is None:
                odds = odds.query("type not in @exclude")

            #if (feature in "type" or category == "type") and not include is None:
            #    odds = odds.query("type in @include")

            if not include is None:
                odds = odds.query(f"{feature} in @include")


            # convert to log
            odds.oddsratio = np.log(odds.oddsratio)
            odds.upper_CI = np.log(odds.upper_CI)
            odds.lower_CI = np.log(odds.lower_CI)



            odds = odds.rename_axis(feature).reset_index()
            odds['err'] = odds.oddsratio.values - odds.lower_CI.values

            if return_dataset or return_tablecount:
                if return_dataset and not return_tablecount:
                    return odds
                if return_tablecount and not return_dataset:
                    return tableCount
                else:
                    return(tableCount, odds)
            else:
                self.generate_OR_graph(odds,
                                            among=among,
                                            feature=feature,
                                            axs=axs,
                                            category=category,
                                            condition=condition,
                                            xlim=xlim,
                                            exclude=exclude,
                                            output=output,
                                            transparent=transparent,
                                            removeXAxis=removeXAxis,
                                            colorZone=colorZone,
                                            return_dataset=return_dataset,
                                            palette=palette,
                                            title=title,
                                            hue=hue,
                                       )


        #-------------------

        def binding_loop(self, axs=None, labels=["1", "2"], normalize=False):
            if axs == None:
                fig = plt.figure(figsize=(10, 5))
                gs = gridspec.GridSpec(ncols=2, nrows=1)
                ax0 = plt.subplot(gs[0])
                ax1 = plt.subplot(gs[1])
            else:
                gs = gridspec.GridSpecFromSubplotSpec(ncols=2, nrows=1, subplot_spec=axs)
                ax0 = plt.subplot(gs[0])
                ax1 = plt.subplot(gs[1])

            domain = self.parent.domainLabel
            if domain == 'PH':
                limit = 240
                labels = ["β1/β2","β3/β4"]
            elif domain == 'C2':
                limit = 275
                labels = ["L1 (173-177)", "L2 (232,238)"]
            elif domain == 'PX':
                limit = 224
                labels = ["L1 (34-69)", "L2 (77,85)"]


            self.parent.ibs["bindingLoop"] = [1 if x < limit else 2 for x in self.parent.ibs.alignment_position]
            tableCI = self.parent.ibs.query("atom_name == 'CB' and is_co_insertable == True").bindingLoop.value_counts(
                normalize=normalize)
            tableHP = self.parent.ibs.query("atom_name == 'CB' and is_hydrophobic_protrusion == True").bindingLoop.value_counts(
                normalize=normalize)


            tableCI = tableCI.rename({1:labels[0],2:labels[1]})
            tableHP = tableHP.rename({1: labels[0], 2: labels[1]})


            maxValue = max([*tableCI.values, *tableHP.values])
            if not normalize:  # If it is not a frequency
                maxValue += 5
                ylabel = "Count"
            else:
                ylabel = "Frequency"

            colors = sns.color_palette("hls", len(tableHP))
            # TO DO: discuss proper handling of the border case where some of these is null
            if (len(tableCI) > 0):
                sns.barplot(x=tableCI.index, y=tableCI, ax=ax0, palette=colors)
            if (len(tableHP) > 0):
                sns.barplot(x=tableHP.index, y=tableHP, ax=ax1, palette=colors)

            ax0.set(xlabel="loop", ylabel=ylabel, title="Where are the co-insertables ?", ylim=(0, maxValue))
            ax1.set(xlabel="loop", ylabel=ylabel, title="Where are the hydrophobic protrusions ?", ylim=(0, maxValue))

            if axs == None:
                fig.tight_layout()
                plt.show()
                plt.close()

# TO DO: discuss proper handling of the border case where all entries are True/False
def _safe_value_counts_with_bools(d):
    """ 
        Useful to proxy the pandas series returned by value_counts()
        with "False" and "True" keys (mapping to 0) when they are missing
    """
    safe_dict = {}
    for k in d.keys():
        safe_dict[k] = d[k]
    if False not in d.keys():
        safe_dict[False] = 0
    if True not in d.keys():
        safe_dict[True] = 0
    return safe_dict
