# Training a Transformer model for Predicting if statements

## Recent updates
- 🔥 **News**: ``2024/10/31``: The [Dataset](https://drive.google.com/drive/folders/100X2rtYo3oV4Rt9cPjkDi3z2hU9_csr7?usp=sharing) creation has been completed.
- 🔥 **News**: ``2024/11/2``: The model pre-training done.


# 📗 Dataset Creation Procedure
For every research data is very important. In this assignment, we need almost ```200k``` of data where we'll use ```150k``` for training the model and ```50k``` for testing the model. To collect the data we use [SEART](https://seart-ghs.si.usi.ch/) website. We use only 2 main parameters to collect the github repositories, first of all the number of contributors should be at least ```100``` and the code lines should be more than ```1000``` lines. We use this because we want to ensure that we can collect larger projects and our procedure will be easy.  
## 🌟 Summary of datacollection procedure
```Step 1:``` Collect the data from ```SEART``` website.\
```Step 2: ``` In the dataset what we've collected from the SEART website. There are several columns such as id, name, branch, commits etc. From all of those column we only need the name column which contains the repo owener name and the repo name.  
```Step 3:``` Save all the name into ```repo_links.csv```\
```Step 4:``` To clone the repository, we add ```https://github.com/``` at the beginning of the name and add ```.git``` at the end of the name. Now we the url is ready to clone the repository. If our name column contains ```tensorflow/tensorflow```, we will get our url as ```https://github.com/tensorflow/tensorflow.git```\
```Step 5:``` Now, we have the repository links. First we will clone the repository and save the folder into local machine.
```Step 6:``` We'll crawling the folder to find a file that endswith ```.py```\
```Step 7:``` Check if there any methods present in that file that contains ```if``` conditions\
```Step 8:``` If any methods contains the ```if``` conditions save that file into ```csv``` file\
```Step 9:``` Repeat this procedure untill the length of the ```csv``` file more than ```200k```\
```Step 10:``` Saved the train ```150k``` data as ```train_data.csv``` and test data as ```generated-testset.csv```
