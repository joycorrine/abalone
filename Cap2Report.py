import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt 
import seaborn as sns
from PIL import Image

image = Image.open('src/AbaloneShellsimage.jpg')

st.image(image)
st.caption('Photo by Content Pixie on Unsplash')

st.markdown('''
# Predicting the Age of Abalones

Abalone is a common name for any of a group of small to very large marine snails. 
    The age of an abalone can be determined by cutting the shell through the cone, staining it, and counting the number of 
    rings through a microscope. The age can then be calculated in years by adding 1.5 to the number of rings on the shell.

Counting the rings of an abalone shell can be an incredibly tedious and time-consuming task, so 
    I am going to attempt to build a regression model that can accurately predict the rings (age) of 
    an abalone based on knowing the shells' physical features.

### Source

To build my model, I’ll be using a dataset on abalone shells from the University of California, Irvine: 
''')
st.write(" - [UCI Machine Learning](https://archive.ics.uci.edu/ml/datasets/abalone)")
st.write(" - [Kaggle Source](https://www.kaggle.com/rodolfomendes/abalone-dataset)")

st.markdown('''
### Overview of the Abalone Shell Data

Here is an overview of the first few lines of data.  We have nine features and 4175 unique entries. 
''')

with st.expander("Abalone Shell Features ", expanded=False):
    st.write(
        """    
- **Sex**: Male (M), Female (F), Infant (I)
- **Length**: Longest shell measurement
- **Diameter**: Perpendicular to length
- **Height**: With meat in shell
- **Whole weight**: Whole abalone weight
- **Shucked weight**: Weight of meat
- **Viscera weight**: Gut weight (after bleeding)
- **Shell Weight**: after being dried
- **Rings**: Number of rings on shell
   
        """
    )


abalone_data = pd.read_csv('src/abaloneEDA_cleaned.csv')
st.dataframe(abalone_data.head())

st.markdown('''
And here are the summary statistics.
''')

st.dataframe(abalone_data.describe())

st.markdown('''
Looking at a heatmap of the data, we can see that age is most correlated with _Height_ 
    and _Shell weight_ and least correlated with _Shucked weight_.
''')

fig, ax = plt.subplots()
sns.heatmap(abalone_data.corr(), ax=ax)
st.write(fig)

st.markdown('''
Let’s look more closely at how age is correlated with the shell's height and weight.
''')

scatter = alt.Chart(abalone_data).mark_circle(size=60).encode(x=alt.X('Age', title='Age'), y=alt.Y('Height', title='Shell Height'))
st.altair_chart(scatter, use_container_width=True)

scatter = alt.Chart(abalone_data).mark_circle(size=60).encode(x=alt.X('Age', title='Age'), y=alt.Y('Shell weight', title='Shell Weight'))
st.altair_chart(scatter, use_container_width=True)

st.markdown('''
Reviewing the above graphs, it appears that abalones typically reach their peak 
    height and weight around 12-15 years old, and then their weight and diameter measurements start to decline.
''')

st.markdown('''
### Building a Predictive Model

For this dataset, I built multiple models in order to identify the best performing one at predicting the age of an 
    abalone shell. I tested with multiple linear regression, ridge regression, random forest, gradient boost, 
    support vector regression, and K nearest neighbors.

I used RMSE (root mean squared error) to evaluate and compare the effectiveness of my models, 
    and ultimately, my multiple linear regression model performed the best with an RMSE of 2.06.
''')

data = {'Model': ['Multiple Linear Regression', 'Ridge Regression', 'Random Forest', 'Gradient Boost', 'SVR', 'KNN'], 
        'RMSE': [2.061, 2.088, 2.100, 2.166, 2.162, 2.216]}
rmsedf = pd.DataFrame(data)

st.dataframe(rmsedf)

st.markdown('''
**While in the process of evaluating my multiple regression model, I made 
an important discovery.**

I compared my model’s worst and best predictions (in terms of absolute difference 
    between the predicted value and real value) and found that my model performed 
    well at predicting ages around 12.5 and below, but did not perform as well 
    predicting older abalones (20+ years old).

Below you can see the top three worst predicted values - abalone 628, 678, and 233. They are all older than 22.5 years old,
    and my model prediction was 11-14 years off for each one of them. You can also see the results for the younger abalones - abalone 462,
    2983, and 3043. My model perfectly predicted their ages.

''')

diffdata = {'Abalone': [628, 678, 2333, 462, 2983, 3043], 
        'Real Values': [22.5, 24.5, 24.5, 7.5, 9.5, 11.5],
        'Predicted Values': [11.87, 14.15, 14.48, 7.50, 9.50, 11.50],
        'Difference': [10.62, 10.34, 10.01, 0, 0, 0]
        
        }
diff = pd.DataFrame(diffdata)
diff.set_index('Abalone') 

st.dataframe(diff)

st.markdown('''
I decided to take a deeper dive into my multiple regression model to better understand where 
    the cutoff was for my model’s performance. Below you can see how the RMSE score to 
    evaluate my model increases as the age threshold for my abalones increases.

''')

st.image('src/abalonegraphimage.png')
st.caption('Increasing the age threshold of the abalone data vs. RMSE')

st.markdown('''
### Conclusion

In conclusion, I would be confident using the multiple regression model for abalones 
aged 12.5+ or younger, but for older abalones, I would either: 
    1) Build a new model which takes into account the skewed performance on older abalones, or 
    2) Stick to counting the shell rings under a microscope. :)  

''')
