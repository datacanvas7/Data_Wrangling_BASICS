# %% [markdown]
# # Data Wrangling
# 
# `Data wrangling`  is the process of `cleaning`, `transforming` and `organizing` data source can be used for `analysis and visualization`. It is an important step in `data analysis` process because `raw data` is often `incomplete`, `inconsistent` and is in an `unstructured format`, which can make it `difficult to work with`. 
# - Data wrangling helps us to make data `more consistent`, `accurate` and `useful for analysis and decision making`.

# %% [markdown]
# ## Steps: 
# 1. Gathering data
# 2. Tools to clean data (libraries)
# 3. How to clean data (steps)
#    1. Dealing with missing values
#    2. Correcting errors in data
#       1. Outliers removal
#          1. Visualization
#          2. IQR Method
#          3. Z-score
#    3. Dropping duplicates   
# 4. Transforming the data 
#    1. Normalize the data
#       1. Min-Max Normalization/Scaling
#       2. Standard Scaling
#       3. Winsorization
#       4. Z-score Normalization
#       5. Decimal Scaling
#       6. Log Transformation
# 5. Feature Engineering
# 6. Organizing the data 
#    1. Columns creation 
#    2. Renaming 
# 7. Saving the wrangled data

# %% [markdown]
# ## 01- Import libraries

# %%
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

# %% [markdown]
# ## 02- Load dataset

# %%
df = sns.load_dataset('titanic')

# %% [markdown]
# ## 03- Apply EDA steps

# %%
df.shape


# %%
df.info()

# %%
df.describe()

# %%
df.info()    

# %%
df.isnull().sum()

# %%
df.isnull().sum()/len(df) * 100

# %%
df.drop(columns='deck', inplace=True)

# %%
df["age"] = df.fillna(value=df["age"].mean())

# %%
## Convert Column to Numeric First
# Temporarily treat as non-categorical
df["age"] = df["age"].astype("float")

# Fill NaN and convert back if needed
df["age"] = df["age"].fillna(df["age"].mean())

# %%
df.isnull().sum()/len(df) * 100

# %% [markdown]
# ## 04- Outliers Removal

# %% [markdown]
# ## 4.1- Visualization

# %%
import seaborn as sns
sns.boxplot(x="sex", y='age', data=df)

# %% [markdown]
# ## 4.2- IQR Method

# %%
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
IQR
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['age'] >= lower_bound) & (df['age'] <= upper_bound)]

# %%
df.shape

# %%
sns.histplot(df['age'])

# %% [markdown]
# ## 4.3- Z-Score Method

# %%
pip install scipy

# %%
from scipy import stats
import numpy as np

zscore = np.abs(stats.zscore(df['age']))
threshold = 3
df = df[(zscore < threshold)]

# %% [markdown]
# ## Explanation
# - `Z-Score Calculation`: Computes how many standard deviations each age value is from the mean (stats.zscore), then takes absolute values (np.abs)
# 
# - `Outlier Removal`: Keeps only rows where age is within ±3 standard deviations (threshold = 3)
# 
# - ` Result`: Filters out extreme age values, creating a cleaner dataset (df = df[(zscore < threshold)])
# 
# - (`Bonus`: Visualize with df['age'].plot(kind='box') before/after to see the effect!) 🚀
# 

# %% [markdown]
# ## 05- Finding and dropping duplicates

# %%
df.shape

# %%
df.duplicated().sum()

# %%
df_duplicates = df[df.duplicated()]
df_duplicates.head()

# %%
df_duplicates.shape

# %%
df_duplicates = df.duplicated()
df_duplicates_count = df_duplicates.value_counts()

plt.bar(df_duplicates_count.index, df_duplicates_count.values)
plt.xlabel('Duplicates')
plt.ylabel('Count')
plt.title('Duplicated Rows Count')
plt.show()

# %%
df.drop_duplicates(inplace=True)

# %%
df.shape

# %% [markdown]
# ## 06- Normalizing the data 

# %%
pip install scikit-learn

# %% [markdown]
# ## 6.1- Min-Max Normalization

# %%
#1 - Import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#2 - Load the dataset
df
#3 - Select the columns to normalize
cols_to_normalize = ['age', 'fare']
#4 - Create a scaler function/tool
scaler = MinMaxScaler()
#5 - Fit the scaler to the data
df[cols_to_normalize] = scaler.fit(df[cols_to_normalize])
#6 - Check the data
df


# %%
df.describe()

# %% [markdown]
# ## 6.2- Standard Scaling 

# %%
#1 - Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
#2 - Load the dataset
df
#3 - Select the columns to normalize
cols_to_normalize = ['age', 'fare']
#4 - Create a scaler function/tool
scaler = StandardScaler()
#5 - Fit the scaler to the data
df[cols_to_normalize] = scaler.fit(df[cols_to_normalize])
#6 - Check the data
df


# %%
df.describe()

# %% [markdown]
# ### Standardization
# 
# x_stand = (x-mean(X))/ std(x)
# 

# %% [markdown]
# ## 6.3- Log Transformation

# %%
ship = sns.load_dataset('titanic')
ship.head()

# %%
ship.info()

# %%
ship.head(5)

# %%
import numpy as np

ship ["age"] = ship ["age"].fillna(ship ["age"].median())
ship["fare"] = ship["fare"].fillna(ship["fare"].median())

#log transformation
ship["fare"] = np.log(ship["fare"])
ship["age"] = np.log(ship["age"])

# %%
ship.head(5)

# %%
sns.boxplot(x="sex", y="fare", data=ship)

# %%
sns.histplot(ship['age'], kde=True)

# %% [markdown]
# ## 07- Organizing the data 

# %%
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["family_size"].head(5)

# %%
sns.swarmplot(data=df, x="sex", y="family_size")    

# %%
sns.swarmplot(data=df, x="sex", y="family_size")

# %%
df = df.rename(columns={"survived": "survival"})
df.columns

# %%
table = pd.pivot_table(ship, values="fare", index="pclass", 
    columns= "survived", aggfunc=np.sum)
table

# %%
sns.scatterplot(data=df, x="fare", y="age")

# %% [markdown]
# ## 08- Save the Data

# %%
df.to_csv("titanic_cleaned.csv", index=False)

# %% [markdown]
# ---
# 


