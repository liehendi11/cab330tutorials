# Week 1: Installing and Understanding Python's Data Mining Libraries

### What's on this week
1. [Why Python?](#whypython)
2. [Installing Python and its data mining libraries](#install)
3. [Process flow of predictive mining in Python](#processflow)
4. [Interactive prototyping in ipython](#ipython)
5. [Defining problem and purpose of data mining](#purpose)
---

The practical note for this week introduces you to Python and its common machine learning libraries. Python is a high-level, interpreted programming language. It is used for wide range of purposes, from web servers to scientific computing. Its syntax emphasizes on readibility, which allow anyone to learn and use it quickly.

The practical sessions in this unit will be covering the usage of Python for data mining and machine learning purposes. We **WILL NOT ** cover the basics of Python. Fortunately, there is a lot of resources for learning Python from scratch, and you can reasonably learn the basics in a week.

We will use Python 3 in this unit. All examples are written using Python 3.5.2, but any version of Python 3 above 3.4 should work just fine. 

## 1. Why Python? <a name="whypython"></a>
In the field of machine learning, Python is arguably the fastest growing and most widely used programming language alongside R. There are a number of reasons for this:

### 1.1. Interpreted language.
Python is designed as an interpreted language, which allow users to test and prototype models really quickly.

### 1.2. Open-source
Python is free and has no ties to any propertiary/corporate technologies, which makes Python the top choice for students, academics and startups.

### 1.3. Fast and wide support for almost anything that you want to do

Vast range of libraries for almost every data mining task.

* **pandas** for data wrangling and preprocessing ([link](http://pandas.pydata.org/))
* **scikit-learn** for supervised and unsupervised learning ([link](http://scikit-learn.org/stable/))
* **numpy** for matrix manipulation ([link](http://www.numpy.org/))
* **seaborn** and **matplotlib** for visualization ([link](https://seaborn.pydata.org/)) ([link2](https://matplotlib.org/))
* **ipython** for interactive prototyping ([link](https://ipython.org/))


### 1.4. Production ready
Models and pipelines built with Python are very suitable to deployment in production systems.

## 2. Installing Python and its data mining libraries <a name="install"></a>

### 2.1. For Windows Users

((to do for windows users))

Google "Python 3" and download it from the official website.

### 2.2. For Linux Users

Ubuntu/Linux Mint users are covered in this section. Typically, Python 3 comes pre-packaged with your distro installation. To check, write:

``` bash
whereis python3
```

Which should return the location of Python 3 binaries in your system
``` bash
python3: /usr/bin/python3 /usr/bin/python3.5m /usr/bin/python3.5 /usr/lib/python3 /usr/lib/python3.5 /etc/python3 /etc/python3.5 /usr/local/lib/python3.5 /usr/include/python3.5m /usr/share/python3 /usr/share/man/man1/python3.1.gz
```

If this is not the case with your system, please read section 2.2.1. Otherwise, you can skip to section 2.2.2.

#### 2.2.1. Download and install Python
Using your distro's package manager, download and install Python3. In Ubuntu/Linux Mint/Debian-based distros, type the following lines in your terminal

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3
```

We also need to install additional libraries such as pip (Python's package manager).
```bash
sudo apt-get install python3-pip
sudo apt-get install build-essential libssl-dev libffi-dev python-dev
```

#### 2.2.2. Setting up virtual environment
Virtual environments enable you to have an isolated space for your Python projects, ensuring that each of your projects won't disrupt each other. This allows greater control over our Python projects and over how different versions of packages are handled. You can have as many virtual environments as you want.

We need to first install the venv module. Let's do it by typing:
```bash
sudo apt-get install python3-venv
```

Once installed, we need to create environments. Choose which directory you would like to put the projects of this units in, or you could make a new one
```bash
mkdir /dir/to/your/project
cd /dir/to/your/project
```

After you are in the directory, you can create an environment by running:
```bash
python3 -m venv my_env
```

This command creates a new directory that specifies the virtual environment. To activate it, type the following command:
```bash
source my_env/bin/activate
```

Your prompt will now be prefixed with the name of your environment, which in this case is called **my_env**. It should looks like this:
```bash
(my_env) hendi@hendi-HP-Pavilion-15-Notebook-PC ~/Documents/Tutoring/dataminingtutorials/week1 $
```

Within the virtual environment, you can use the command ```python``` instead of ```python3``` and pip instead of ```pip3```. If you use Python 3 outside of the environment, you would need to use ```python3``` and ```pip3``` commands as ```python``` and ```pip``` refers to the Python 2 packages.


**NOTE:** You need to activate the virtual environment everytime you open a new terminal session to work on this directory. Otherwise, all libraries and setting you set up in the virtual environment would not be applied.

#### 2.2.2. Install machine learning packages
To install the libraries that we will use in this unit, we can use ```pip``` package manager by typing:
```bash
pip install ipython pandas sklearn matplotlib numpy seaborn nltk
```

## 3. Process flow for predictive mining using Python<a name="processflow"></a>
![Predictive mining process flow in Python](/home/hendi/Documents/Tutoring/dataminingtutorialsresources/process_flow_python.png  "Predictive mining process flow in Python")

The diagram above presents the steps we will take in this unit to perform predictive mining on the dataset. The first and most important step is to define problem and purpose of the data mining. You need to ask questions such as:

* What kind of data do we have?
* Why are we performing predictive mining on this data?
* What information are we trying to predict?
* How could the stakeholders (including yourself) use the insights we gained from the data mining?

After we understand the problem and purpose of the data mining process, next step is to explore the data. In this step, we try to understand patterns and distributions in the data. We should also identifies problems in the dataset, such as noise and missing values, to be cleaned and processed out in the next step. Both steps will be performed mainly using ```pandas``` with some help from ```sklearn```'s preprocessing modules.

Once the data is clean, it can be used to built predictive models. There are many algorithms available in ```sklearn```, each with its own characteristics. We will explore one algorithm at a time in the upcoming weeks.

In all stages, we also need to visualize the patterns and trends found in the data. Visualization allows us to understand the data better. In this unit, all visualizations will be done using ```seaborn``` and ```matplotlib``` with data presented by ```pandas``` dataframes.


## 4. Interactive prototyping with ipython<a name="ipython"></a>

```ipython``` is an interactive Python shell designed for fast prototyping. In data mining/machine learning, many engineers use ipython to quickly review the data and process they are working on. We can call ipython the same way as we call the python interpreter itself:

```bash
ipython
```

```bash
# Output
Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
Type 'copyright', 'credits' or 'license' for more information
IPython 6.1.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: 
```

All examples in this unit are shown using ipython console.

## 5. Defining problem and purpose of data mining process<a name="purpose"></a>

Let's start the data mining process by defining why we are performing data mining on this data. To do this, we need to take a look on the supplied **pva97nk** data.

Start by importing the dataset into our ipython console. We will use pandas for this purpose.
```python
# input
import pandas as pd
df = pd.read_csv('path/to/your/pva97nk.csv')
```

Once the dataset is imported, let's see what information it has.
```python
# input
df.info()
```

```python
#output
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9686 entries, 0 to 9685
Data columns (total 28 columns):
TargetB             9686 non-null int64
ID                  9686 non-null int64
TargetD             4843 non-null float64
GiftCnt36           9686 non-null int64
GiftCntAll          9686 non-null int64
GiftCntCard36       9686 non-null int64
GiftCntCardAll      9686 non-null int64
GiftAvgLast         9686 non-null float64
GiftAvg36           9686 non-null float64
GiftAvgAll          9686 non-null float64
GiftAvgCard36       7906 non-null float64
GiftTimeLast        9686 non-null int64
GiftTimeFirst       9686 non-null int64
PromCnt12           9686 non-null int64
PromCnt36           9686 non-null int64
PromCntAll          9686 non-null int64
PromCntCard12       9686 non-null int64
PromCntCard36       9686 non-null int64
PromCntCardAll      9686 non-null int64
StatusCat96NK       9686 non-null object
StatusCatStarAll    9686 non-null int64
DemCluster          9686 non-null int64
DemAge              7279 non-null float64
DemGender           9686 non-null object
DemHomeOwner        9686 non-null object
DemMedHomeValue     9686 non-null int64
DemPctVeterans      9686 non-null int64
DemMedIncome        9686 non-null int64
dtypes: float64(6), int64(19), object(3)
memory usage: 2.1+ MB
```

PVA97NK dataset is about a national veterans’ organization that seeks to better target its solicitations for donation. By only soliciting the most likely donors, less money will be spent on solicitation efforts and more money will be available for charitable concerns. Of particular interest is the class of individuals identified as lapsing donors. The organization seeks to classify its lapsing donors based on their responses to a greeting card mailing and calls it as the 97NK Campaign. With this classification, a decision can be made to either solicit or ignore a lapsing individual in next year campaign.

The PVA97NK dataset contains 29 variables including identifiers, demographics of members, donation history of members, etc. In the upcoming weeks, we aim to predict TARGETB, a binary variable corresponding to whether or not someone responded to the greeting card mailing sent in June, 1997.

## End notes and next week
This week, we learned how to install Python and its libraries in a virtual enviroment. We also learned about the typical data mining process flow in Python and explored a bit of the dataset to understand why we are performing data mining on it.

Next week, we will focus on exploring trends and performing data cleaning/preprocessing on the PVA97NK dataset.

