# Implementation of Pan-Tomkins Algorithm 
This project is implementation of ECG QRS Detection based on the work of Pan-Tomkins algorithm.


## Dependencies

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all these libraries. Original work was done using Python3, but Python2 also support all these libraries.

```bash
Numpy
Matplotlib
Scipy
Pyserial
```
If you face installation issues: Using Conda Environment is highly recommended for Windows Users. The following command after installing Anaconda resolves all dependencies.
```
$ python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
```

## Repository Structure
```

├── README.md          				 <- The top-level README for developers using this project.

├── main.py   			 <- Offline QRS Detector module.

├── report.pdf    			 <- Online QRS Detector module.

```

## MIT-BIH Arrhythmia Database 
MIT-BIH Arrhythmia Database was used to calibrate and verify the algorithm. Our offline code requires .csv file input of the sample data with 3 coloumns (as in the MIT-BIH Database). First coloumn is the timestamp, second and third are ECG signlas taken from two different leads. As discussed in Pan - Tompin's work, only one channel input is processed at a time (for efficient accuracy). This .csv file can be directly found from this website:
 [ MIT-BIH Database ](https://www.physionet.org/cgi-bin/atm/ATM).
Here 1 Hour signal record is used. From TOOLBOX on the website you can choose to export file as .csv and then click on samples.csv to save it.
File samples.csv should be in the same folder as QRSdetector.py.

## main.py
This file assume you have saved samples.csv file from above instructions. Now you can directly run this file by the command through terminal once in the same directory in which your files reside:
```
$ python main.py
```
