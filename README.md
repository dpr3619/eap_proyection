# Projecting the Future of Employment in Colombia: 2025–2040
The main objective of this repository is to document the process and the code developing a model that proyect 15 years in the future the economically active population (EAP) for Colombia taking into account the possible impact of two emerging technologies **Artificial Intelligence (AI)** and **Industrial Internet of Thins (IIoT).** The repository is divided in 3 main folders, one containing all the ETL (Extract, Transform and load) process, the other folder is the one containing all the models and the selected models.

## ETL

We use the GEIH (Gran encuesta integrada de hogares) data that contains all the information about the EAP in Colombia, the data has monthly data about the EAP population and information about employment in each one of the major economic sectors. We also use the GEIHISS data that contains data about the formal and informal employment, we used this to get the proportion of the occupied population that is formal and informal, nevertheless, this information is only available for 2021 and above for the Total National information, before that there is information about 23 cities and metopolitan areas. Given this issue of missing values for all the data, we will use an imputation approach for the less years.

Also when we check the data about the forman and informal employment in the economic sector we noticed that the data is not reported by months but in moving average of three months. Given this we will not try to get more data about this, we will work with the proportion and the values from this data.

## Preporseccing

## Models

* ARIMAX
* Check Prophet.
