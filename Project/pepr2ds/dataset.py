## Imports and settings
import pandas as pd
import numpy as np
import math
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import ConvexHull
from biopandas.pdb import PandasPdb


plt.rcParams["figure.dpi"] = 200

# %matplotlib inline
sns.set_style("darkgrid")
import ipywidgets as widgets
from IPython.display import display, Markdown, clear_output
# from tqdm.auto import tqdm
from tqdm.notebook import tnrange, tqdm

from termcolor import colored
from IPython.display import display, HTML
import weasyprint
from scipy.stats import fisher_exact

tqdm.pandas()  # activate tqdm progressbar for pandas apply
pd.options.mode.chained_assignment = (
    None  # default='warn', remove pandas warning when adding a new column
)
pd.set_option("display.max_columns", None)
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# %config InlineBackend.figure_format ='svg' #better quality figure figure
np.seterr(divide='ignore', invalid='ignore')

import MDAnalysis as mda
import nglview as nv

from Bio import PDB
from Bio.PDB import PDBParser
import numpy as np


from pandarallel import pandarallel

class Dataset(object):
    class Widgets():
        def __init__(self, parent):
            self.parent = parent
            self.create_widgets()
            self.watch_widgets()
            self.set_layout()

            #instantiate features
            self.alignment_group = []

        def create_widgets(self):
            self.dropdownDomainsList = self.parent.dataset.domain.unique()  # drodown
            self.domainsWidget = widgets.Dropdown(description="Domain",
                                                  options=self.dropdownDomainsList,
                                                  value='PH')
            self.pdbWidget = widgets.Dropdown(description='PDB Ref',
                                              disabled=True)
            self.extendSSWidget = widgets.Checkbox(value=True,
                                                   description='Extend SS ?')
            self.alignmentWidget = widgets.Checkbox(value=True,
                                                    description='enrich with 2D prosite sequence ?')
            self.onlyCoilSSWidget = widgets.Checkbox(value=False,
                                                     description='Only Coil ?')
            self.cathClusterWidget = widgets.Dropdown(description="Cath Cluster",
                                                      options=[None, "S35", "S60", "S95", "S100"],
                                                      value="S100")
            self.showCoinsertableWidget = widgets.Checkbox(value=False,
                                                           description='Show co-insertables ?')

            self.showProtrusionWidget = widgets.Checkbox(value=False,
                                                         description='Show Protrusion ?')
            self.excludeStrand = widgets.Checkbox(value=False,
                                                  description='ExcludeStrand ?')

            self.unirefWidget = widgets.Dropdown(description="Uniref",
                                                 options=[None, "uniref50", "uniref90", "uniref100"],
                                                 value="uniref100")

            self.IncludeFromWidget = widgets.IntText(value=20, description='From:', disabled=True, )
            self.IncludeToWidget = widgets.IntText(value=26, description='To:', disabled=True, )
            self.IncludeSegments = [widgets.HBox(children=[self.IncludeFromWidget, self.IncludeToWidget])]

            self.ExcludeFromWidget = widgets.IntText(value=0, description='From:', disabled=True, )
            self.ExcludeToWidget = widgets.IntText(value=0, description='To:', disabled=True, )
            self.ExcludeSegments = [widgets.HBox(children=[self.ExcludeFromWidget, self.ExcludeToWidget])]

            self.entropyWidget = widgets.FloatSlider(value=4.22, min=0, max=4.22,
                                                     step=0.01, description='', )
            self.expositionWidget = widgets.IntSlider(value=100, min=0, max=100,
                                                      step=1, description='', )

            self.entropySignWidget = widgets.Dropdown(description="Entropy",
                                                      options=['<', '<=', '=', '>', '>='],
                                                      value='<=',
                                                      layout=widgets.Layout(width="20%"))

            self.expositionSignWidget = widgets.Dropdown(description="Exposition (%)",
                                                         options=['<', '<=', '=', '>', '>='],
                                                         value='<=',
                                                         layout=widgets.Layout(width="20%"))

            # Buttons
            self.createDatasetButton = widgets.Button(description="Create subdataset")
            self.residueAnalysisButton = widgets.Button(description="Amino acid Analysis", disabled=True)
            self.addIncludeSegmentButton = widgets.Button(description="Add new Segment", disabled=True)
            self.addExcludeSegmentButton = widgets.Button(description="Add new Segment", disabled=True)
            self.checkSelectionButton = widgets.Button(description="Create ConvexHull", disabled=True)
            self.quickRunPHButton = widgets.Button(description="Quick setup PH")
            self.quickRunC2Button = widgets.Button(description="Quick setup C2")

            # Output
            self.out = widgets.Output()
            self.displayMol = nv.NGLWidget()  # widgets.Text("select a PDB")

        def watch_widgets(self):
            # Action
            self.domainsWidget.observe(self.update_domainsWidget, names=['value'])
            self.pdbWidget.observe(self.update_pdbWidget, names=['value'])
            self.createDatasetButton.on_click(self.on_run_button_clicked)
            self.residueAnalysisButton.on_click(self.on_residueAnalysis_button_clicked)
            self.addIncludeSegmentButton.on_click(self.on_addIncludeSegment_button_clicked)
            self.addExcludeSegmentButton.on_click(self.on_addExcludeSegment_button_clicked)
            self.checkSelectionButton.on_click(self.on_createConvexhull_button_clicked)
            self.checkSelectionButton.on_click(self.update_selection)
            self.expositionWidget.observe(self.update_selection, names=["value"])
            self.entropyWidget.observe(self.update_selection, names=["value"])
            self.IncludeFromWidget.observe(self.update_selection, names=["value"])
            self.IncludeToWidget.observe(self.update_selection, names=["value"])
            self.ExcludeFromWidget.observe(self.update_selection, names=["value"])
            self.ExcludeToWidget.observe(self.update_selection, names=["value"])
            self.showCoinsertableWidget.observe(self.on_coinsertable_ticked, names=["value"])
            self.showProtrusionWidget.observe(self.on_protrusion_ticked, names=["value"])
            self.quickRunPHButton.on_click(self.quickRunPH)
            self.quickRunC2Button.on_click(self.quickRunC2)

        def set_layout(self):
            self.tab1HBox_layout = widgets.Layout(display='flex',
                                                  # flex_flow='column',
                                                  align_items='stretch',
                                                  border='solid',
                                                  width='90%')

            self.IncludeVBoxSegments = widgets.VBox(children=self.IncludeSegments)
            self.MetaIncludeVboxSegment = widgets.VBox(
                children=[self.IncludeVBoxSegments, self.addIncludeSegmentButton])

            self.ExcludeVBoxSegments = widgets.VBox(children=self.ExcludeSegments)
            self.ExcludeMetaVboxSegment = widgets.VBox(
                children=[self.ExcludeVBoxSegments, self.addExcludeSegmentButton])

            self.entropyBox = widgets.HBox([self.entropySignWidget, self.entropyWidget])
            self.expositionBox = widgets.HBox([self.expositionSignWidget, self.expositionWidget])

            self.selectionTab = widgets.Tab()
            self.selectionTab.set_title(0, 'Include in IBS')
            self.selectionTab.set_title(1, 'Exclude from IBS')
            self.selectionTab.children = [self.MetaIncludeVboxSegment, self.ExcludeMetaVboxSegment]
            # TAB1 BOX. Configuration
            self.tab1VBox = widgets.VBox(children=[widgets.HBox(children=[self.quickRunPHButton,
                                                                          self.quickRunC2Button]),
                                                   self.domainsWidget,
                                                   self.pdbWidget,
                                                   self.selectionTab,
                                                   self.entropyBox,
                                                   self.expositionBox,
                                                   self.showCoinsertableWidget,
                                                   self.showProtrusionWidget,
                                                   self.checkSelectionButton,
                                                   self.extendSSWidget,
                                                   self.alignmentWidget,
                                                   self.onlyCoilSSWidget,
                                                   self.excludeStrand,
                                                   self.cathClusterWidget,
                                                   self.unirefWidget,
                                                   self.createDatasetButton])
            self.tab1 = widgets.HBox(children=[self.tab1VBox,
                                               self.displayMol],
                                     layout=self.tab1HBox_layout)
            self.tab2 = widgets.VBox([self.residueAnalysisButton])

        def display(self):

            self.masterTab = widgets.Tab()
            self.masterTab.set_title(0, 'Create IBS dataset')
            self.masterTab.set_title(1, 'Analysis')

            self.masterTab.children = [self.tab1,
                                       self.tab2]
            display(self.masterTab)
            display(self.out)

            self.update_domainsWidget(self.domainsWidget.value)

        def quickRunPH(self, change):
            with self.out:
                self.domainsWidget.value = "PH"
                self.pdbWidget.value = "2da0A00"
                # Add sew segment
                newSegment1 = widgets.HBox([widgets.IntText(value=20, description='From:', disabled=False),
                                            widgets.IntText(value=26, description='to:', disabled=False)])
                newSegment1.children[0].observe(self.update_selection, names=["value"])
                newSegment1.children[1].observe(self.update_selection, names=["value"])

                newSegment2 = widgets.HBox([widgets.IntText(value=42, description='From:', disabled=False),
                                            widgets.IntText(value=50, description='to:', disabled=False)])
                newSegment2.children[0].observe(self.update_selection, names=["value"])
                newSegment2.children[1].observe(self.update_selection, names=["value"])

                self.IncludeSegments = [newSegment1, newSegment2]
                self.IncludeVBoxSegments.children = tuple(self.IncludeSegments)

                self.update_selection()
                self.unirefWidget.value = "uniref90"
                self.cathClusterWidget.value = "S95"

        def quickRunC2(self, change):
            with self.out:
                self.domainsWidget.value = "C2"
                self.pdbWidget.value = "1rsyA00"
                # Add sew segment
                newSegment1 = widgets.HBox([widgets.IntText(value=173, description='From:', disabled=False),
                                            widgets.IntText(value=177, description='to:', disabled=False)])
                newSegment1.children[0].observe(self.update_selection, names=["value"])
                newSegment1.children[1].observe(self.update_selection, names=["value"])

                newSegment2 = widgets.HBox([widgets.IntText(value=232, description='From:', disabled=False),
                                            widgets.IntText(value=238, description='to:', disabled=False)])
                newSegment2.children[0].observe(self.update_selection, names=["value"])
                newSegment2.children[1].observe(self.update_selection, names=["value"])

                self.IncludeSegments = [newSegment1, newSegment2]
                self.IncludeVBoxSegments.children = tuple(self.IncludeSegments)

                self.update_selection()
                self.unirefWidget.value = "uniref90"
                self.cathClusterWidget.value = "S95"

        def on_addIncludeSegment_button_clicked(self, change):
            """
            Add a new residue range.
            """
            with self.out:
                newSegment = widgets.HBox([widgets.IntText(value=0, description='From:', disabled=False),
                                           widgets.IntText(value=0, description='to:', disabled=False)])
                newSegment.children[0].observe(self.update_selection, names=["value"])

                newSegment.children[1].observe(self.update_selection, names=["value"])
                self.IncludeSegments += [newSegment]
                self.IncludeVBoxSegments.children = tuple(self.IncludeSegments)

        def on_addExcludeSegment_button_clicked(self, change):
            """
            Add a new residue range.
            """
            with self.out:
                newSegment = widgets.HBox([widgets.IntText(value=0, description='From:', disabled=False),
                                           widgets.IntText(value=0, description='to:', disabled=False)])
                newSegment.children[0].observe(self.update_selection, names=["value"])

                newSegment.children[1].observe(self.update_selection, names=["value"])
                self.ExcludeSegments += [newSegment]
                self.ExcludeVBoxSegments.children = tuple(self.ExcludeSegments)

        def update_domainsWidget(self, *args):
            domain = self.domainsWidget.value
            cathpdblist = self.parent.dataset.query("domain == @domain and data_type != 'prosite'").cathpdb.unique()
            cathpdblist = sorted(cathpdblist)
            self.pdbWidget.disabled = False
            self.pdbWidget.options = cathpdblist

        def update_pdbWidget(self, *args):
            with self.out:
                self.IncludeFromWidget.disabled = False
                self.IncludeToWidget.disabled = False
                self.ExcludeFromWidget.disabled = False
                self.ExcludeToWidget.disabled = False
                self.checkSelectionButton.disabled = False
                self.addIncludeSegmentButton.disabled = False
                self.addExcludeSegmentButton.disabled = False

                self.showCoinsertableWidget.value = False
                self.showProtrusionWidget.value = False
                domain = self.domainsWidget.value
                pdbname = self.pdbWidget.value

                if domain and pdbname:
                    structure = self.parent.CATHFOLDER + "domains/" + domain + "/raw/" + pdbname + ".pdb"
                    self.mdaMol = mda.Universe(structure)
                    # cog = self.mdaMol.atoms.center_of_geometry()
                    # self.mdaMol.atoms.translate(-cog)

                    #self.set_entropy_as_bfactor(pdbname)

                    self.displayMol = nv.show_mdanalysis(self.mdaMol)
                    self.displayMol.clear_representations()
                    self.displayMol.add_representation('cartoon', selection='protein', color='sstruc')
                    # self.displayMol.add_representation('cartoon', selection='protein',
                    #                                    color='bfactor')  # Representation 0 = Cartoon
                    self.displayMol.add_representation('hyperball', selection='@1',
                                                       color='red')  # Representation 1 = selected residues
                    self.displayMol.add_representation('ball+stick', selection='@1',
                                                       color='blue', opacity=0)  # Representation 2 = co_insertable
                    self.displayMol.add_representation('ball+stick', selection='@1',
                                                       color='green', opacity=0)  # Representation 3 = protrusion

                    self.displayMol.layout.width = '600px'
                    self.displayMol.layout.height = '600px'
                    # self.displayMol._set_selection('@', repr_index=1)
                    self.tab1.children = (list(self.tab1.children)[0], self.displayMol)

                    # Create a copy of the pdb so request are faster
                    self.parent.pdb = self.parent.dataset.query('cathpdb == @self.pdbWidget.value')
                    self.parent.haspdb = True
                    # Update exposition and entropy values

                    rangeExposition = (self.parent.pdb.sasa_rel_dssp.min(), self.parent.pdb.sasa_rel_dssp.max())
                    self.expositionWidget.min = rangeExposition[0]
                    self.expositionWidget.max = rangeExposition[1]
                    self.expositionWidget.value = rangeExposition[1]

                    rangeEntropy = (self.parent.pdb.shannonH10.min(), self.parent.pdb.shannonH10.max())
                    if np.isnan(rangeEntropy[0]) and np.isnan(rangeEntropy[1]):
                        self.entropyWidget.disable = True
                    else:
                        self.entropyWidget.disable = False
                        self.entropyWidget.min = rangeEntropy[0]
                        self.entropyWidget.max = rangeEntropy[1]
                        self.entropyWidget.value = rangeEntropy[1]

                    # self.out.clear_output()

        def set_entropy_as_bfactor(self, pdbname):
            entropy = self.parent.dataset.query("cathpdb == @pdbname").shannonH10.to_numpy()
            entropy = np.nan_to_num(entropy, nan=-1)
            ptp = np.ptp(
                entropy[~np.isnan(entropy)])  # copy of entropy without nan. ptp is the range that takes values.
            entropy = ((entropy - np.nanmin(entropy)) / ptp)
            self.mdaMol.atoms.tempfactors = entropy

        def update_selection(self, change=None):
            with self.out:
                # exposition
                exposed = self.parent.pdb.query(
                    f"sasa_rel_dssp {self.expositionSignWidget.value} @self.expositionWidget.value").residue_number.unique()
                # exposition
                conserved = self.parent.pdb.query(
                    f"shannonH10 {self.entropySignWidget.value} @self.entropyWidget.value").residue_number.unique()

                # Add segment
                IncludeResid_segments = []
                for segment in self.IncludeSegments:
                    low = segment.children[0].value
                    up = segment.children[1].value
                    if low != 0 and up != 0:
                        IncludeResid_segments.extend(list(range(low,
                                                                up)))
                # Exclude segment
                ExcludeResid_segments = []
                for segment in self.ExcludeSegments:
                    low = segment.children[0].value
                    up = segment.children[1].value
                    if low != 0 and up != 0:
                        ExcludeResid_segments.extend(list(range(low,
                                                                up)))
                # take common amino acids between 3 lists
                s1 = set(exposed)
                s2 = set(conserved)
                s3 = set(IncludeResid_segments)
                set1 = s1.intersection(s2)
                selected_resid = list(set1.intersection(s3))

                selection_string = ' or '.join(f"resid {x}" for x in selected_resid)
                if len(ExcludeResid_segments) > 0:
                    selection_string = '(' + selection_string + ')' + "and not (" + ' or '.join(
                        f"resid {x}" for x in ExcludeResid_segments) + ')'

                # Remove N,C,O
                # selection_string = "not type H and (" + selection_string + ")" #+" and not (name C or name O or name N)"
                ids = self.mdaMol.select_atoms(selection_string).ix
                self.displayMol._set_selection("not hydrogen and sidechainAttached and @" + ','.join(map(str, ids)),
                                               repr_index=1)
                self.displayMol.update_representation(component=0, repr_index=1, colorScheme='element')

        def on_createConvexhull_button_clicked(self, change):
            def normal(self, points, face):
                u = self.vect(points, face[0], face[1])
                v = self.vect(points, face[0], face[-1])
                return self.cross(u, v)

            with self.out:

                positions = self.mdaMol.select_atoms("name CB").positions
                hull = ConvexHull(positions)

                mesh = []
                for triangles in hull.simplices[hull.good]:
                    triangles.sort()
                    vertices = []
                    for vertex in triangles:
                        vertices.extend(positions[vertex].tolist())
                    mesh.append(vertices)
                mesh = np.asarray(mesh).flatten().tolist()
                color = [[0, 0, 1]] * len(mesh)  # RGB, for now let's fix it to blue
                color = np.asarray(color).flatten().tolist()
                shape = self.displayMol.shape
                # shape.add_mesh(mesh,color)

                shape.add_mesh(
                    mesh,
                    color,
                )

                self.displayMol.update_representation(component=1, repr_index=0, opacity=0.5, side="double", )

        def on_coinsertable_ticked(self, change):
            with self.out:
                if self.parent.haspdb:
                    if self.showCoinsertableWidget.value == True:
                        selected_resid = self.parent.pdb.query("is_co_insertable == 1").residue_number.unique()
                        selection_string = ' or '.join(f"resid {x}" for x in selected_resid)
                        ids = self.mdaMol.select_atoms(selection_string).ids
                        self.displayMol._set_selection(".CB and @" + ','.join(map(str, ids)), repr_index=2)
                        self.displayMol.update_representation(component=0, repr_index=2, opacity=1, radiusScale=5)
                    else:
                        self.displayMol._set_selection('@2', repr_index=2)
                        self.displayMol.update_representation(component=0, repr_index=2, opacity=0)

        def on_protrusion_ticked(self, change):
            with self.out:
                if self.parent.haspdb:
                    if self.showProtrusionWidget.value == True:
                        selected_resid = self.parent.pdb.query("is_hydrophobic_protrusion == 1").residue_number.unique()
                        selection_string = ' or '.join(f"resid {x}" for x in selected_resid)
                        ids = self.mdaMol.select_atoms(selection_string).ids
                        self.displayMol._set_selection(".CB and @" + ','.join(map(str, ids)), repr_index=3)
                        self.displayMol.update_representation(component=0, repr_index=3, opacity=1, radiusScale=5)

                    else:
                        self.displayMol._set_selection('@3', repr_index=3)
                        self.displayMol.update_representation(component=0, repr_index=3, opacity=0)

        def on_residueAnalysis_button_clicked(self, change):
            with self.out:
                self.parent.plot_residue_composition(self.parent.ibs, self.parent.nonibs)

        def on_run_button_clicked(self, change):
            with self.out:
                self.out.clear_output()

                domain = self.domainsWidget.value

                df = self.parent.dataset.query("domain == @domain")
                pdb = self.pdbWidget.value

                IncludeResidueSegments = []
                for segment in self.IncludeSegments:
                    start = segment.children[0].value
                    end = segment.children[1].value
                    if start != 0 and end != 0:
                        IncludeResidueSegments.append([start, end])

                # Exclude segment
                ExcludeResidueSegments = []
                for segment in self.ExcludeSegments:
                    start = segment.children[0].value
                    end = segment.children[1].value
                    if start != 0 and end != 0:
                        ExcludeResidueSegments.append([start, end])

                extendSS = self.extendSSWidget.value
                alignment = self.alignmentWidget.value
                onlyCoilSS = self.onlyCoilSSWidget.value
                cathCluster = self.cathClusterWidget.value
                uniref = self.unirefWidget.value
                excludeStrand = self.excludeStrand.value


                self.parent.tag_ibs(dataset=df,
                                    domain=domain,
                                    pdbreference=pdb,
                                    includeResidueRange=IncludeResidueSegments,
                                    excludeResidueRange=ExcludeResidueSegments,
                                    extendSS=extendSS,
                                    withAlignment=alignment,
                                    onlyC=onlyCoilSS,
                                    cathCluster=cathCluster,
                                    Uniref=uniref,
                                    addSequence=True,  # Always true for now,
                                    extendAlign=True,  # Always true for now,
                                    excludeStrand=excludeStrand,
                                    )

                # # Put dataset in the parent class
                # self.parent.ibs = tempout[0]
                # self.parent.nonibs = tempout[1]

                print("> IBS/NON-IBS datasets generated, you can run analysis now")
                self.residueAnalysisButton.disabled = False
            # with output:
            #    print(domain,pdb,residueRange,extendSS, alignment, onlyCoilSS, cathCluster, uniref)



    # PARRENT CLASS, DATASET.
    def __init__(self, dataset, PEPRMINT_FOLDER):
        self.dataset = dataset
        self.haspdb = False
        self.CATHFOLDER = f"{PEPRMINT_FOLDER}/databases/cath/"
        self.FIGURESFOLDER = f"{PEPRMINT_FOLDER}/figures/"
        self.PEPRMINT_FOLDER = PEPRMINT_FOLDER
        self.ui = self.Widgets(self)
        self.analysis = self.Analysis(self)
        self.ibs = None
        self.nonibs = None
        self.domainDf = None
        self.noAlignment = False
        self.domainLabel = ''

    def get_df_objects(self):
        return self.df_objects




    def load_dataset(self, name, path=None):
        if path == None:
            path = f"{self.PEPRMINT_FOLDER}/dataset"

        self.domainDf = pd.read_pickle(f"{path}/{name}.pkl")
        self.domainLabel = "+".join(self.domainDf.domain.unique())


    def save_dataset(self, name, path=None):
        if path == None:
            path = f"{self.PEPRMINT_FOLDER}/dataset"
        self.domainDf.to_pickle(f"{path}/{name}.pkl")



    #############
    ## DATASET FUNCTIONS
    #############
    def tag_ibs(self, dataset,
                domain,
                pdbreference,
                includeResidueRange=[],
                excludeResidueRange=[],
                extendSS=True,
                withAlignment=True,
                onlyC=False,
                cathCluster=None,
                Uniref=None,
                addSequence=True,
                extendAlign=True,
                excludeStrand=False,
                overide_axis_mode=False,
                zaxis=0,
                extendCoilOnly=True,
                coordinates_folder_name = None, #If coordinate folder is given, all the X,Y,W coordinates will be updated from the PDB inside the folder.
                filter_uniprot_acc = None,
                data_type = None,
                base_folder = 'cath',
                ):
        """
        TODO
        """

        AATYPE = {
            "LEU": "Hydrophobic,H-non-aromatic",
            "ILE": "Hydrophobic,H-non-aromatic",
            "CYS": "Hydrophobic,H-non-aromatic",
            "MET": "Hydrophobic,H-non-aromatic",
            "TYR": "Hydrophobic,H-aromatic",
            "TRP": "Hydrophobic,H-aromatic",
            "PHE": "Hydrophobic,H-aromatic",
            "HIS": "Positive",
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



        # Reset AATYpe if needed... #TODO -> Remove this for final prod, it's only a trick to avoid recalculating the full dataset if we want to change a definition
        print("Domain=",domain)
        dataset["type"] = dataset.residue_name.apply(lambda x: AATYPE[x])
        dataset.uniprot_acc = dataset.uniprot_acc.astype(str)


        dataset["exposition"] = np.where(dataset['RSA_freesasa_florian'] >= 20,
                                         "exposed",
                                         "buried")
        #Same to "exposed" condition
        dataset["exposed"] = dataset["RSA_freesasa_florian"].apply(lambda x: True if x >= 20 else False)

        #with self.ui.out:
        print("selecting amino acids")
        #######################
        # Defining borders and alignment position
        #######################
        # Checking
        if not domain in dataset.domain.unique():
            raise ValueError("domain not recognized")

        # Get domain
        df = dataset.query("domain == @domain")
        self.domainLabel = domain


        if data_type == 'cath':
            df = df.query("data_type == 'cathpdb'")
        elif data_type == 'alphafold':
            df = df.query("data_type == 'alphafold'")
        elif data_type == 'cath+af':
            df = df.query("data_type in ['cathpdb','alphafold']")


        df["matchIndex"] = list(range(len(df)))

        #If SH2, clean with CHO data

        from sys import platform
        if platform == "linux" or platform == "linux2":
            sh2_cho = "/mnt/g/clouds/OneDrive - University of Bergen/projects/peprmint/data/Cho_SH2_transformed.xlsx"
        else:
            sh2_cho = "/Users/thibault/OneDrive - University of Bergen/projects/peprmint/data/Cho_SH2_transformed.xlsx"

        if domain == 'SH2':
            cho = pd.read_excel(
                sh2_cho,
            engine='openpyxl').dropna(subset=["Range"])
            uniprot_cho = list(cho.uniprot_acc)
            uniprot_dataset = list(
                df.query("domain == 'SH2' and atom_name == 'CA' and alignment_position == 0").uniprot_acc.unique())
            common = list(set(uniprot_cho).intersection(uniprot_dataset))
            df = df.query("uniprot_acc in @common")


        # CLUSTER REDUNDANCY.
        if cathCluster and Uniref:
            df = self.selectUniquePerCluster(df, cathCluster, Uniref, withAlignment, pdbreference)

        # KEEP ONLY MATCH WITH UNIPROT_ACC
        if filter_uniprot_acc:
            def select_only_one(group):
                keep = group.cathpdb.unique()[0]
                return group.query('cathpdb == @keep')

            df = df.query("uniprot_acc in @filter_uniprot_acc")
            df = df.groupby('uniprot_acc', as_index=False).apply(lambda group: select_only_one(group))


        # check if several borders given, the format should be a list
        if not any(isinstance(i, list) for i in includeResidueRange):
            includeResidueRange = [includeResidueRange]

        if not overide_axis_mode:
            if extendAlign:
                includeAliRange = []
                for s, e in includeResidueRange:
                    start_ali = df.query(
                        "cathpdb == @pdbreference and atom_name == 'CA' and residue_number == @s").alignment_position.values[
                        0]
                    end_ali = df.query(
                        "cathpdb == @pdbreference and atom_name == 'CA' and residue_number == @e").alignment_position.values[
                        0]
                    includeAliRange.append([start_ali, end_ali])
                SelectionString = ' or '.join(
                    ["{} <= alignment_position <= {}".format(s, e) for s, e in includeAliRange])
            else:
                SelectionString = ' or '.join(
                    ["{} <= residue_number <= {}".format(s, e) for s, e in includeResidueRange])

            # Exclusion
            if len(excludeResidueRange) > 0:
                if not any(isinstance(i, list) for i in excludeResidueRange):
                    excludeResidueRange = [excludeResidueRange]
                SelectionString = '(' + SelectionString + ') and not ( ' + ' or '.join(
                    ["{} <= residue_number <= {}".format(s, e) for s, e in excludeResidueRange]) + ')'

            # if extendSS:
            #    ssSegments = df.query("cathpdb == @pdbreference and atom_name == 'CA' and ({0})".format(SelectionString)).sec_struc_segment.unique()
            #    alignmentPos = df.query("cathpdb == @pdbreference and atom_name == 'CA' and sec_struc_segment in @ssSegments").alignment_position
            # else:
            #    alignmentPos = df.query("cathpdb ==  @pdbreference and atom_name == 'CA' and ({0})".format(SelectionString)).alignment_position

            # Change IBS status to True
            df.loc[df.eval(f"{SelectionString}", engine='python'), "IBS"] = True
            df.loc[~df.eval(f"{SelectionString}", engine='python'), "IBS"] = False
        elif overide_axis_mode == True:
            self.noAlignment = True
            if not coordinates_folder_name is None:
                if base_folder == 'cath':
                    base_folder = f"{self.CATHFOLDER}/domains"
                else:
                    base_folder = f"{self.PEPRMINT_FOLDER}/databases/{base_folder}"
                coordinates_folder = f"{base_folder}/{domain}/{coordinates_folder_name}"

                print(coordinates_folder)
                if os.path.exists(coordinates_folder):
                    print("UPDATING COORDINATES")

                    def update_coords(group, coordinates_folder):
                        pdb = group.cathpdb.unique()[0]

                        if not os.path.isfile(f"{coordinates_folder}/{pdb}.pdb"):
                            return None
                        pl1 = PandasPdb().read_pdb(f"{coordinates_folder}/{pdb}.pdb").df["ATOM"]
                        # When sometimes we remove duplicated residues we have to be sure that we update
                        # the residues list we take from the new coordinates files
                        residues_number_list = group.residue_name.unique()
                        pl1 = pl1.query("residue_name in @residues_number_list")

                        pdbdf = pl1[["atom_name", "residue_number", "x_coord", "y_coord", "z_coord"]]


                        _merged = group.merge(pdbdf, on=["atom_name", "residue_number"], how="left")

                        # Sometimes there is duplicated atoms, we just keep the first one by removing duplicates.
                        if len(_merged["x_coord_y"].values) != len(group["x_coord"]):
                            _merged = _merged.drop_duplicates(subset=["atom_name", "residue_number"])

                        try:
                            group["x_coord"] = _merged["x_coord_y"].values
                            group["y_coord"] = _merged["y_coord_y"].values
                            group["z_coord"] = _merged["z_coord_y"].values
                        except:
                            print(group)
                            1/0

                        return (group)
                df = df.groupby("cathpdb", as_index=False).progress_apply(
                    lambda x: update_coords(x, coordinates_folder))

            df.loc[df.eval("z_coord <= @zaxis ", engine='python'), "IBS"] = True
            df.loc[df.eval("z_coord > @zaxis", engine='python'), "IBS"] = False






        def tag_MLIP(df):
            # TODO: Improve this and put it directly in the dataset generation and calculate the real MLIP
            df['LDCI'] = False
            try:
                minIndex = df.query('is_co_insertable == 1').density.idxmin()
                df.at[minIndex, "LDCI"] = True
            except:
                return df

            return df

        if "LDCI" not in df.columns:
            print("taggin MLIP")
            df = df.groupby("cathpdb").progress_apply(tag_MLIP)

        self.domainDf = df


        print("taggin IBS")
        ibs_nonibs = df.groupby('cathpdb').progress_apply(
            lambda x: self.get_ibs_and_non_ibs(x, extendSS, onlyC, excludeStrand,extendCoilOnly))

        ibs = pd.concat([x[0] for x in ibs_nonibs]).reset_index(drop=True)
        nonibs = pd.concat([x[1] for x in ibs_nonibs]).reset_index(drop=True)

        print(f"len IBS {len(ibs.cathpdb.unique())}")
        print(f"len nonIBS {len(nonibs.cathpdb.unique())}")

        if addSequence:
            print("adding sequences")
            ibsSeq = df.query(f"data_type == 'prosite' and ({SelectionString})", engine='python')
            nonIbsSeq = df.query(f"data_type == 'prosite' and not ({SelectionString})", engine='python')
            ibs = pd.concat([ibs, ibsSeq])
            nonibs = pd.concat([nonibs, nonIbsSeq])

        self.ibs = ibs
        self.nonibs = nonibs

        #Update "domainDf" with new tag

        matchIndexIBS = ibs.matchIndex
        matchIndexnonIBS = nonibs.matchIndex

        self.domainDf.loc[self.domainDf.matchIndex.isin(matchIndexIBS), "IBS"] = True
        self.domainDf.loc[self.domainDf.matchIndex.isin(matchIndexnonIBS), "IBS"] = False


        #remove NaN Values
        self.domainDf = self.domainDf.dropna(subset=["residue_name"])
        #return (ibs, nonibs)

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
        seqdf = df.query("data_type == 'prosite' or data_type == 'alphafold'")

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

        dfReprCathNames = list(cathdf.groupby(["domain", cathCluster]).apply(selectUniqueCath).to_numpy())

        if len(dfReprCathNames) > 0:
            excludeUniref = df.query(
                "cathpdb in @dfReprCathNames").uniprot_acc.unique()  # Structures are prior to sequences.
            dfReprUnirefNames = list(seqdf.groupby(["domain", Uniref]).apply(selectUniqueUniref,
                                                                        exclusion=excludeUniref).to_numpy())


        else:
            dfReprUnirefNames = list(seqdf.groupby(["domain", Uniref]).apply(selectUniqueUniref,
                                                                        exclusion = []).to_numpy())



        dfReprCath = cathdf.query("cathpdb in @dfReprCathNames")
        uniproc_acc_cath = dfReprCath.uniprot_acc.unique()
        dfReprUniref = seqdf.query("uniprot_acc in @dfReprUnirefNames")

        return (pd.concat([dfReprCath, dfReprUniref]))

    def get_ibs_and_non_ibs(self, cathpdb, extendSS=True, onlyC=False, excludeStrand=False, extendCoilOnly=False):
        # get secondary structure loops
        # _ssSegments = cathpdb.query("@start <= alignment_position <= @end").sec_struc_segment.unique()
        _ssSegments = list(map(str,cathpdb.query("IBS == True").sec_struc_segment.unique()))
        # check if segment is a loop

        if onlyC:
            ssSegments = []
            for segment in _ssSegments:
                if segment.startswith('C'):
                    ssSegments.append(segment)

            ibs = cathpdb.query("sec_struc_segment in @ssSegments")
            nonibs = cathpdb.query("sec_struc_segment not in @ssSegments")

        else:
            # ibs = cathpdb.loc[cathpdb.eval("@start <= alignment_position <= @end")]
            # nonibs = cathpdb.loc[~cathpdb.eval("@start <= alignment_position <= @end")]
            if extendSS:
                ssSegs = cathpdb.query("IBS == True").sec_struc_segment.unique()

                #TODO: EXTEND COIL ONLY

                if extendCoilOnly:
                    ssSegs = [x for x in ssSegs if x.startswith('C')]
                    ibs = cathpdb.query("IBS == True or sec_struc_segment in @ssSegs")
                    nonibs = cathpdb.query("IBS == False and sec_struc_segment not in @ssSegs")
                    ibs.IBS = True
                    nonibs.IBS = False
                    return(ibs,nonibs)



                if not excludeStrand:
                    ibs = cathpdb.query("sec_struc_segment in @ssSegs")
                else:
                    ibs = cathpdb.query("sec_struc_segment in @ssSegs and not sec_struc == 'E'")



                nonibs = cathpdb.query("sec_struc_segment not in @ssSegs")
                # Update IBS
                ibs.IBS == True

            else:
                ibs = cathpdb.query("IBS == True")
                nonibs = cathpdb.query("IBS == False")




        return (ibs, nonibs)

    def generate_picutre_of_IBS(self, subfolder='raw'):
        import MDAnalysis as mda

        cathfolder = f"{self.PEPRMINT_FOLDER}/databases/cath"
        domain = self.ibs.domain.unique()[0]
        pdbfolder = f"{cathfolder}/domains/{domain}/{subfolder}"
        outputPDB = f"{cathfolder}/domains/{domain}/IBS/pdb"
        outputPNG = f"{cathfolder}/domains/{domain}/IBS/png"

        view = {
            "PH": "set_view (0.579240322,    0.505582690,   -0.639426887,    -0.760016978,    0.618558824,   -0.199398085,     0.294711024,    0.601475298,    0.742546558,    -0.000000262,    0.000000060, -192.327224731,     1.087764502,   -0.489078194,    0.073249102,   155.025390625,  229.629058838,  -20.000000000 )",
            'C2': "set_view (0.732501030,    0.379622459,   -0.565089643, -0.677367389,    0.323625565,   -0.660633683, -0.067914240,    0.866690934,    0.494200438, -0.000011362,   -0.000007764, -146.056518555,  4.814969063,   -0.570566535,   10.014616013,115.152069092,  176.960968018,  -20.000000000 )",
            'START':"set_view (     0.997152150,   -0.059061568,    0.046864305,     0.044335004,   -0.043418523,   -0.998070359,     0.060982693,    0.997307599,   -0.040675215,     0.000018209,   -0.000058129, -155.363037109,   -10.252207756,    6.747013569,   12.991518974,   115.271072388,  195.459533691,  -20.000000000 )",
            'C1':"set_view (     0.313648969,    0.817849040,    0.482437909,    -0.183394849,    0.550686538,   -0.814315557,    -0.931659758,    0.166932240,    0.322710901,     0.000000011,    0.000000708, -117.470191956,    -2.323507547,    1.068632245,    1.322908878,    99.383796692,  135.557312012,  -20.000000000 )",
            'SH2':"set_view (     0.552586615,   -0.076772571,    0.829912364,     0.832381070,    0.101391248,   -0.544850767,    -0.042316202,    0.991879404,    0.119931221,     0.000000000,    0.000000000, -132.395614624,    -2.113309860,    3.920848846,    8.741382599,   109.715965271,  155.075241089,  -20.000000000 )",
            "C2DIS":"set_view (     0.143325791,    0.803093255,   -0.578357637,     0.467322886,    0.460218340,    0.754856706,     0.872391641,   -0.378470331,   -0.309342563,     0.000000000,    0.000000000, -154.193801880,     0.543474197,   -0.320308685,    1.579742432,   121.567565918,  186.820037842,  -20.000000000 )",
            "FYVE":"set_view (     0.886591494,    0.351747036,   -0.300380319,    -0.339166492,    0.052801486,   -0.939242423,    -0.314515173,    0.934602559,    0.166114911,    -0.000005476,    0.000001019, -188.998291016,    -1.669404864,   -6.018759727,    0.102616847,   112.505050659,  265.490844727,  -20.000000000 )",
            "PX":"set_view (     0.807477295,    0.112026922,    0.579163909,     0.589899063,   -0.153209910,   -0.792809069,    -0.000082285,    0.981823087,   -0.189797714,     0.000000000,    0.000000000, -159.604339600,     4.754449844,    6.864978790,    9.598649979,   125.833282471,  193.375396729,  -20.000000000 )",
            "ENTH":"set_view (     0.679595053,    0.116319597,    0.724306464,     0.711486697,   -0.345033497,   -0.612157106,     0.178704709,    0.931353927,   -0.317244768,     0.000000000,    0.000000000, -156.046157837,     0.794780731,   -7.394954681,   11.364852905,   102.870040894,  209.222259521,  -20.000000000 )",
            "PLD":"set_view (    -0.942126930,    0.016740968,    0.334838033,     0.324299693,    0.298773795,    0.897533059,    -0.085015431,    0.954176843,   -0.286911786,     0.000000000,    0.000000000, -155.298629761,    -3.902690887,   -0.973564148,   13.115238190,   124.827445984,  185.769866943,  -20.000000000 )",
            "ANNEXIN":"set_view (    -0.082639754,    0.010267707,    0.996525764,     0.996567369,    0.005868928,    0.082582556,    -0.005001415,    0.999930799,   -0.010716723,     0.000000000,    0.000000000, -107.264595032,    -3.597750664,   -2.025382519,    5.951478481,    86.362419128,  128.166778564,  -20.000000000 )",
            "PLA":"set_view (    -0.986954212,   -0.063351221,   -0.148008540,    -0.151886493,    0.061519083,    0.986482680,    -0.053388733,    0.996094048,   -0.070337638,    -0.000003427,   -0.000011377, -137.629989624,     4.810346603,   -4.336011410,    8.157093048,   108.358093262,  166.902557373,  -20.000000000 )",
        }
        if not os.path.isdir(outputPDB):
            os.makedirs(outputPDB)
        if not os.path.isdir(outputPNG):
            os.makedirs(outputPNG)

        pdblist = self.ibs.cathpdb.unique()
        print(len(pdblist))
        for pdb in tqdm(pdblist):
            # read pdb and tag IBS in the beta-factor column
            file = f"{pdbfolder}/{pdb}.pdb"
            ibsResidue = list(map(int, self.ibs.query("cathpdb == @pdb").residue_number.unique()))
            selectionString = " or ".join([f"resnum {x}" for x in ibsResidue])
            U = mda.Universe(file)
            U.atoms.tempfactors = 0
            U.select_atoms(selectionString).tempfactors = 1
            U.atoms.write(f"{outputPDB}/{pdb}.pdb")

            # generate picture.

            pymolCmd = f"load {outputPDB}/{pdb}.pdb; " + \
                       f"{view[domain]};" + \
                       "as cartoon; select ibs, b > 0.5; color red, ibs;" + \
                       f"bg white; png {outputPNG}/{pdb}.png, ray=0"
            # "select ax, z < 0.0001; show spheres, ax and name CA; set sphere_color, blue; set sphere_scale, 0.4;" + \

            _ = os.system(f'pymol -Q -c -d "{pymolCmd}"')



    # START
    def show_structure_and_plane(self, idpdb, folder='raw'):
        parser = PDB.PDBParser()
        from Bio.PDB.PDBExceptions import PDBConstructionWarning
        import warnings
        import nglview as nv

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            structure = parser.get_structure(id=idpdb,
                                             file=f"/Users/thibault/Documents/WORK/peprmint/databases/cath/domains/{self.domainLabel}/{folder}/{idpdb}.pdb")


        view = nv.show_biopython(structure)
        shape = view.shape
        chain = structure[0].child_list[0].id


        shape.add_sphere([0, 0, 0], [1, 0, 0], 1)
        shape.add_sphere([10, 10, 0], [0, 1, 0], 0.5)
        shape.add_sphere([-10, -10, 0], [0, 1, 0], 0.5)
        shape.add_sphere([10, -10, 0], [0, 1, 0], 0.5)
        shape.add_sphere([-10, 10, 0], [0, 1, 0], 0.5)

        shape.add_arrow([0, 0, 0], [0, 0, -10], [1, 0, 0], 1.0)

        mesh = [20, 20, 0,
                20, -20, 0,
                -20, -20, 0,
                -20, -20, 0,
                20, 20, 0,
                -20, 20, 0]

        color = [[0, 1, 0]] * len(mesh)  # RGB, for now let's fix it to blue
        color = np.asarray(color).flatten().tolist()
        shape.add_mesh(mesh, color)

        view.update_representation(component=7, repr_index=0, opacity=0.5, side="double")
        return view




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
            tc_HP_IBS = ibsCathPdb.is_hydrophobic_protrusion.value_counts()
            tc_HP_nonIBS = nonIbsCathPdb.is_hydrophobic_protrusion.value_counts()
            percentageHPibs = tc_HP_IBS[1] / tc_HP_IBS[0] * 100
            pertentageHPnonibs = tc_HP_nonIBS[1] / tc_HP_nonIBS[0] * 100

            # CO-insertable
            tc_CO_IBS = ibsCathPdb.is_co_insertable.value_counts()
            tc_CO_nonIBS = nonIbsCathPdb.is_co_insertable.value_counts()
            percentageCOibs = tc_CO_IBS[1] / tc_CO_IBS[0] * 100

            pertentageCOnonibs = tc_CO_nonIBS[1] / tc_CO_nonIBS[0] * 100

            # MLIP
            tc_LDCI_IBS = ibsCathPdb.LDCI.value_counts()
            tc_LDCI_nonIBS = nonIbsCathPdb.LDCI.value_counts()
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
            tc_CI_ibs = ibsCathPdb.query(f"{var1} == True")[var2].value_counts()  # Create table count
            tc_CI_ibs.rename(index={0: "No", 1: "Yes"}, inplace=True)
            #print(tc_CI_ibs)
            fracCI_IBS = tc_CI_ibs[1] / sum(tc_CI_ibs)

            tc_CI_nonIBS = nonIbsCathPdb.query(f"{var1} == True")[var2].value_counts()  # Create table count
            tc_CI_nonIBS.rename(index={0: "No", 1: "Yes"}, inplace=True)
            fracCI_NONIBS = tc_CI_nonIBS[1] / sum(tc_CI_nonIBS)

            table_oddsratio = [[tc_CI_ibs[1], tc_CI_ibs[0]],
                               [tc_CI_nonIBS[1], tc_CI_nonIBS[0]]
                               ]

            oddsratioDf = pd.DataFrame(table_oddsratio)
            oddsratioDf.columns = pd.MultiIndex.from_product([[var2], ['yes', 'no']])
            oddsratioDf.index = pd.MultiIndex.from_product([["IBS"], ['yes', 'no']])

            oddsratio, pvalue = fisher_exact(table_oddsratio)
            # Edvin's formula
            Rab = (fracCI_IBS * (1 - fracCI_NONIBS)) / (fracCI_NONIBS * (1 - fracCI_IBS))

            # oddsratio = math.log(oddsratio)
            # Standart deviation
            se = (1 / table_oddsratio[0][0] + 1 / table_oddsratio[0][1] + 1 / table_oddsratio[1][0] + 1 /
                  table_oddsratio[1][1]) ** 0.5
            # Confidance Interval at 95% Wald, z=1.96
            lower_CI = math.exp(math.log(oddsratio) - 1.96 * se)
            upper_CI = math.exp(math.log(oddsratio) + 1.96 * se)

            if output == 'standard':
                print("")
                print(colored("###### - ODDSRATIO ESTIMATOR - IBS/NONIBS ######", 'blue', attrs=['bold']))
                print("...................")
                print(f"Variable 1: {var1}")
                print(f"Variable 2: {var2}")
                print("...................")
                display(HTML(oddsratioDf.to_html()))

                print("-------")
                print(f"Oddratio = {oddsratio} - LOG -> {np.log(oddsratio):.2f}")
                print(f"Pvalue = {pvalue} (fisher)")
                print(f"Incertitude = {abs(oddsratio - lower_CI)} (fisher) - LOG -> {abs(np.log(oddsratio) - np.log(lower_CI)):.2f}")
                print(f"Standart deviation = {se:.2f} - LOG -> {np.log(se):.2f}")

                print(f"Confidance Interval at 95% (Wald) {lower_CI:.2f}, {upper_CI:.2f}")
                print("-------")
                if 1 < lower_CI:
                    print(colored(">> 1 lower than lower confidance intervale, result is statistically relevant.",
                                  "green"))
                    print(f">> {var1} are {oddsratio:.2f} times more likely to be {var2} on IBS than non-IBS")
                else:
                    print(colored(">> 1 in confidance intervale, result IS NOT statistically relevant.", "red"))
                    print(f"{var1} are {oddsratio:.2f} times more likely to be {var2} on IBS than non-IBS")
                return (oddsratio, lower_CI, upper_CI)

            elif output == "html":
                HTMLstring = f"""
                <div style="border:3px; border-style:dashed; border-color:#808080; padding: 1em; margin:10px;>
                <p">
                {self.span("###### - ODDSRATIO ESTIMATOR - IBS/NONIBS ######", 'blue', 'bold')}
                <br />
                {self.span('Variable 1', 'black', 'bold')}: {var1}<br />
                {self.span('Variable 2', 'black', 'bold')}: {var2}<br />
                {oddsratioDf.to_html()}
                <ul>
                <li>OddRatio = {oddsratio:.2}</li>
                <li>Pvalue = {pvalue:.2} (fisher)</li>
                <li>Standart deviation = {se:.2f}</li>
                <li>Confidance Interval at 95% (Wald) {lower_CI:.2f}, {upper_CI:.2f}</li>
                <br />"""
                if 1 < lower_CI:
                    HTMLstring += self.span(
                        ">> 1 lower than lower confidance intervale, result is statistically relevant",
                        "green") + ".<br />"
                    HTMLstring += f">> {var1} are " + self.span(f"{oddsratio:.2f}", "black",
                                                                "bold") + f" times more likely to be {var2} on IBS than non-IBS<br />"
                else:
                    HTMLstring += self.span(">> 1 is in the confidance intervale, result is NOT statistically relevant",
                                            "red", "bold") + ".<br />"
                    HTMLstring += f">> {var1} are " + self.span(f"{oddsratio:.2f}", "black",
                                                                "bold") + f" times more likely to be {var2} on IBS than non-IBS<br />"

                HTMLstring += "</p></div>"

                # display(HTML(HTMLstring))

                return (oddsratio, lower_CI, upper_CI, HTMLstring)

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


            test = test/test.sum(level=0)
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
                "HIS": "Positive",
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
                                except:
                                    print(f"error with pdb {cathpdb} - ignoring it.")
                                    return None
                                is_protrusion = number_as_index.loc[id,"protrusion"] # we take its protrusion tag
                                if exclude_protrusion == True: # If we choose to ignore protrusion from the analysis
                                    if is_protrusion == False or pd.isna(is_protrusion): # BUT if the neighbor is not a protrusion
                                        if eval(f"sasa {sign} {rel_cutoff}") : # and its sasa value is respected
                                            envir.append(number_as_index.loc[id,feature]) # Then we add it to our neighbours list

                                else: #Otherwise, if we want to include all amino acid (protrusion/non protrusion)
                                    if eval(f"sasa {sign} {rel_cutoff}"): # we check the SASA values
                                        envir.append(number_as_index.loc[id, feature]) # we add it to our neighbours list

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

                    tableCount = r.sum(axis=0, level=[1, 2])

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
            sns.barplot(x=tableCI.index, y=tableCI, ax=ax0, palette=colors)
            sns.barplot(x=tableHP.index, y=tableHP, ax=ax1, palette=colors)

            ax0.set(xlabel="loop", ylabel=ylabel, title="Where are the co-insertables ?", ylim=(0, maxValue))
            ax1.set(xlabel="loop", ylabel=ylabel, title="Where are the hydrophobic protrusions ?", ylim=(0, maxValue))

            if axs == None:
                fig.tight_layout()
                plt.show()
                plt.close()



if __name__ == "__main__":
    #PEPRMINT_FOLDER = "/Users/thibault/Documents/WORK/peprmint"
    PEPRMINT_FOLDER = "/mnt/g/WORK/projets/peprmint"
    WORKDIR = f"{PEPRMINT_FOLDER}/dataset/"
    DATASET = pd.read_pickle(f"{WORKDIR}/DATASET_peprmint_d25.pkl")
    # PH = Dataset(DATASET, PEPRMINT_FOLDER)
    # PH.tag_ibs(DATASET,
    #            domain='PH',  # Domain
    #            pdbreference="2da0A00",  # PDB Template
    #            includeResidueRange=[[20, 26], [42, 50]],  # Include those residues in IBS
    #            excludeResidueRange=[],  # Exclude thoses residues from IBS
    #            extendSS=False,  # Extend the secondary structures
    #            withAlignment=False,  # restrict the results with pdb that have a sequences.
    #            onlyC=False,  # get only COIL in the IBS.
    #            cathCluster="S95",  # Structure redundancy filter
    #            Uniref="uniref90",  # Sequence redundancy filter
    #            addSequence=False,  # add the non structural data in the IBS/NONIBS dataset.
    #            extendAlign=False,
    #            # Extend the secondary structure instead of a raw "cut" based on the alignment position
    #            excludeStrand=False,  # Exclude "strand" From secondary structure
    #            overide_axis_mode=True,  # use the Zaxis instead of the alignment to tag the IBS
    #            zaxis=0,  # Z axis plane to define "IBS" or not IBS
    #            extendCoilOnly=False,  # Extend coil only.
    #            coordinates_folder_name='zaligned',  # Where are the PDBs
    #            )
    #PH.analysis.report(displayHTML=False)

    C1 = Dataset(DATASET,PEPRMINT_FOLDER)
    C1.tag_ibs(DATASET,
                domain = 'C1', #Domain
                pdbreference = "1ptrA00",
                includeResidueRange = [], #CHANGE 173 to 171 and run again!
                excludeResidueRange=[], #Exclude thoses residues from IBS
                extendSS=False, #Extend the secondary structures
                withAlignment=False, #restrict the results with pdb that have a sequences.
                onlyC=False, #get only COIL in the IBS.
                cathCluster='S100', #Structure redundancy filter
                Uniref="uniref90", #Sequence redundancy filter
                addSequence=False, #add the non structural data in the IBS/NONIBS dataset.
                extendAlign=False, #Extend the secondary structure instead of a raw "cut" based on the alignment position
                excludeStrand=False, #Exclude "strand" From secondary structure
                overide_axis_mode = True, #use the Zaxis instead of the alignment to tag the IBS
                zaxis=0, #Z axis plane to define "IBS" or not IBS
                extendCoilOnly = False, #Extend coil only.
                coordinates_folder_name = "zaligned"
              )

    odds = C1.analysis.oddsratio_graph(among="is_hydrophobic_protrusion", feature="type",envir=True, return_dataset=False, exclude_protrusion=False)  # Analysis 1

    print(odds)


    #C1.analysis.report(displayHTML=False)