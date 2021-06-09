# Penguin Classification

![SS1](https://github.com/Hongyanlee0614/Streamlit/blob/main/Penguin%20Classification/ss.PNG)

## Install

#### Open your conda's terminal and run the following:

```
$ pip install streamlit
```

#### Clone this repo using the following command:

```
$ git clone https://github.com/Hongyanlee0614/Streamlit.git
```

#### Or you can go to [this link](https://github.com/Hongyanlee0614/Streamlit) and download the zip file for entire repo

#### After that, open the clonned or unzipped Streamlit folder in your preferred editor (e.g. VS Code)


```
$ cd '.\Penguin Classification\'
```

#### Run the following command to generate the pickle file so we no need to train the model each time we change the streamlit parameter input. Note that there is a pickle file once you have clonned this repo. You can delete it at first to build from your own.

```
$ streamlit run penguins-model-building.py
```

#### Now a pickle file named penguins_clf.pkl is generated. Run the following command to start working on/customizing the main file - penguins-app.py!

```
$ streamlit run penguins-app.py
```

#### Open [http://localhost:8501](http://localhost:8501) with your browser to see the result.


## References
- Data Professor ([Link to YouTube channel](https://www.youtube.com/channel/UCV8e2g4IWQqK71bbzGDEI4Q))
- Pratik Mukherjee ([Link to Kaggle Notebook](https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering))
- Allison Horst ([Link to Data Source)](https://github.com/allisonhorst/palmerpenguins)

