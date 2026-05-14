# aind-smartspim-quantification

Code for quantifying cell counts for whole brain lighsheet imaging. This repository assumes that we have the cell locations in a XML or CSV and the image transformations from the CCF alignment.
Please, refer to these repositories to be able to generate these results:

- [aind-smartspim-segmentation](https://github.com/AllenNeuralDynamics/aind-SmartSPIM-segmentation)
- [aind-ccf-registration](https://github.com/AllenNeuralDynamics/aind-ccf-registration)

## Outputs

The `cell_count_by_region.csv` file contains the number of cells in each region of the 25um Allen Brain atlas:

`Id`: Unique numerical identifier for a given region  
`Acronym`: Acronym associated with region  
`Struct_Info`: Identifies regions that span the midline or are separated in each hemisphere  
`Struct_area_um3`: The total volume of a region in cubic microns  
`Left`: Total number of cells identified in the left hemisphere of a region  
`Right`: Total number of cells identified in the right hemisphere of a region  
`Total`: Total number of cells identified within a region  
`Left_Density`: Density of cells (cells/voxel) identified in the left hemisphere of a region  
`Right_Density`: Density of cells (cells/voxel) identified in the right hemisphere of a region  
`Total_Density`: Total density of cells (cells/voxel) identified within a region  
`Left_Median_Foreground`: Median foreground likelihood of cells in the left hemisphere of a region  
`Right_Median_Foreground`: Median foreground likelihood of cells in the right hemisphere of a region  
`Total_Median_Foreground`: Median foreground likelihood of all cells within a region  
`Left_Median_Background`: Median background likelihood of cells in the left hemisphere of a region  
`Right_Median_Background`: Median background likelihood of cells in the right hemisphere of a region  
`Total_Median_Background`: Median background likelihood of all cells within a region  

The `transformed_cells.csv` contains cell coordinates in CCF space and classifier metrics:

`x`: location in the Anterior-Posterior axis  
`y`: location in the Dorsal-Ventral axis  
`z`: location in the Medial-Lateral axis  
`Foreground`: cell classifier foreground likelihood  
`Background`: cell classifier background likelihood  
`Cell ID`: unique identifier for the cell  

