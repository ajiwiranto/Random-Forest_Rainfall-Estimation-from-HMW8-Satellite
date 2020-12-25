# Random-Forest_Rainfall-Estimation-from-HMW8-Satellite
rainfall estimation from satellite himawari-8 Multiband data using random forest

Model Random Forest Machine Learning

The data didn't uploaded because too large and a bit confindental because it has trough data processing.
you could mail me if you interest and want to see the data : ajiwiranto96@gmail.com

Data used
IR band Himawari 8
Spatial Resolution : 2kmx2km 
Temporal Resolution : 10 menit
Combination from all band.
9 band + 36 Split Window.
in a year Aug 2018- Jul 2019

Downloaded Free at ftp://hmwr829gr.cr.chiba-u.ac.jp/gridded/FD/V20151105/

2. GPM DPR KuPR
Spatial Resolution : 5.2km x 5.2km

Downloaded Free at https://worldview.earthdata.nasa.gov/

Data sampling was carried out by extending data collection to one island of Java with
the assumption of similar atmospheric conditions to overcome data limitations in
the Bandung Basin. The algorithm used the random forest model with several stage which is 
1.  classify the rain area
2.  classify the type of rain
3.  regress to get the rain value.

There are 3 part script which is make the model, implement the model, and evaluate the rainfall estimate by the model with other rainfall estimation by other satellite in Bandung Basin. 

The library use in this project are

numpy, pandas, os, matplotlib, scikit-learn, basemap, scipy, glob, netCDF4, H5py (HDF5), global_land_mask. 

