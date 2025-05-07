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


""" Handle to use the software from Jupyter notebooks with the adequate settings

__author__ = ["Thibault Tubiana", "Phillippe Samer"]
__organization__ = "Computational Biology Unit, Universitetet i Bergen"
__copyright__ = "Copyright (c) 2022 Reuter Group"
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Phillippe Samer"
__email__ = "samer@uib.no"
__status__ = "Prototype"
"""

import ipywidgets as widgets
from IPython.display import display

class NotebookHandle:

    def __init__(self):
        print("Notebook settings loaded (general)")

    def dataset_manager_options(self):
        # Notebook #2 settings
        from IPython.core.interactiveshell import InteractiveShell
        InteractiveShell.ast_node_interactivity = "all"
        get_ipython().run_line_magic('config', "InlineBackend.figure_format ='svg' #better quality figure figure")
        print("Notebook settings loaded (for dataset creation)")

    def alphafold_utils_options(self):
        # Notebook #3 settings
        from IPython.display import display, Markdown, clear_output
        
        from IPython.core.interactiveshell import InteractiveShell
        InteractiveShell.ast_node_interactivity = "all"
        #%config InlineBackend.figure_format ='svg' #better quality figure
        
        print("Notebook settings loaded (for alphafold data download)")

    def figure_generator_options(self):
        # Notebook #4 settings
        from IPython.display import display, Markdown, clear_output

        from IPython.core.interactiveshell import InteractiveShell
        InteractiveShell.ast_node_interactivity = "all"
        #%config InlineBackend.figure_format ='svg' #better quality figure figure

        get_ipython().run_line_magic('matplotlib', 'inline')
        
        print("Notebook settings loaded (for figure generation)")

    def ibs_options(self):
        # Notebook "tools/GENERATE_IBS_DATASETS" settings
        from IPython.display import display, Markdown, clear_output

        from IPython.core.interactiveshell import InteractiveShell
        InteractiveShell.ast_node_interactivity = "all"
        #%config InlineBackend.figure_format ='svg' #better quality figure figure

        print("Notebook settings loaded (for IBS tagging)")
