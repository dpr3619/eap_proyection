# Projecting the Future of Employment in Colombia: 2025–2040
The main objective of this repository is to document the process and the code developing a model that proyect 15 years in the future the economically active population (EAP) for Colombia taking into account the possible impact of two emerging technologies **Artificial Intelligence (AI)** and **Industrial Internet of Thins (IIoT).** The repository is divided in 3 main folders, one containing all the ETL (Extract, Transform and load) process, the other folder is the one containing all the models and their pipeline, third is the generate_analysis_table that joins the two mentioned folder creating a table for the future analysis.

## ETL

We use the GEIH (Gran encuesta integrada de hogares) data that contains all the information about the EAP in Colombia, the data has monthly data about the EAP population and information about employment in each one of the major economic sectors. We also use the GEIHISS data that contains data about the formal and informal employment, we used this to get the proportion of the occupied population that is formal and informal, nevertheless, this information is only available for 2021 and above for the Total National information, before that there is information about 23 cities and metopolitan areas.

Also when we check the data about the forman and informal employment in the economic sector we noticed that the data is not reported by months but in moving average of three months. Given this we will not try to get more data about this, we will work with the proportion and the values from this data.

## Preporseccing

Adding the supply shocks: Desde 2002 en Colombia se han presentado 3 choques de oferta de importante magnitud.
El primero está vinculado al incremento de los precios del petróleo entre enero de 2007 y
julio de 2008, justo antes de la crisis financiera de 2008. El segundo, asociado al fenómeno
de El Niño que se presentó entre 2014 y 2015, y que se agudizó con el paro camionero
que atravesó el país a mediados de 2016.

## Models

Two models were used in this analysis, first an ARIMAX approach and then a Machine Learning approach using Catboost for prediction.


While the sources mention the total percentage of the workforce employed in agriculture and provide a general percentage point difference in automation risk between formal and informal workers, they do not indicate a specific percentage for the rate of formal employment within the agricultural sector itself.