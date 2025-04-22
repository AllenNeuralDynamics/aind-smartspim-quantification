# aind-smartspim-quantification

Code for quantifying cell counts for whole brain lighsheet imaging. This repository assumes that we have the cell locations in a XML or CSV and the image transformations from the CCF alignment.
Please, refer to these repositories to be able to generate these results:

- [aind-smartspim-segmentation](https://github.com/AllenNeuralDynamics/aind-SmartSPIM-segmentation)
- [aind-ccf-registration](https://github.com/AllenNeuralDynamics/aind-ccf-registration)

## Outputs

The <Cell_count_by_region.csv> file contains the number of cells in each region of the 25um Allen Brain atlas:

'ID': Unique numerical identifier for a given region
'Acronym': Acronym associated with region
'Name': Full name of region
'Parent Region': 
'Layer': Cortical layer of region if applicable
'Ancestors': Full list of upstream parent regions
'Graph Order': Location within the Allen Brain Atlas structure graph https://api.brain-map.org/api/v2/structure_graph_download/1.json
'Struct_Info': Identifies regions that exist across the ML midline (mid) or are seperated in each hemisphere (hemi)
'Struct_area_um3': The total volume of a region in voxels 
'Left': The total number of cells identified in the left hemisphere of a region
'Right': The total number of cells identified in the right hemisphere of a region
'Total': The total number of cells identified within a region
'Left_Density': The density of cells (cells/voxel) identified in the left hemisphere of a region
'Right_Density': The density of cells (cells/voxel) identified in the right hemisphere of a region
'Total_Density': The total density of cells (cells/voxel) identified within a region

The transformed_cells.xml contains the coordinates of cells in CCF space:

'x': location in the Anterior-Posterior axis
'y': location in the Dorsal-Ventral axis
'x': location in the Medial-Lateral axis

