import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from scipy.stats import gmean, variation


def read_df(fname, countries, years):
    """
    Function to read the csv file and return 2 dataframes, one with years
    as columns and the other with countries as columns. Takes the filename as
    the argument.
    """
    # read file into a dataframe
    df0 = pd.read_csv(fname, skiprows=4, index_col=0)
    # some cleaning
    df0.drop(columns=["Country Code"], axis=1, inplace=True)
    df1 = df0.loc[countries, years]
    # some dataframe methods
    df1 = df1.sort_index().rename_axis("Years", axis=1).fillna(0)
    # transpose
    df2 = df1.T

    return df1, df2


def skew(dist):
    """Calculates the centralised and normalised skewness of dist."""
    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)
    # now calculate the skewness
    value = np.sum(((dist - aver) / std) ** 3) / len(dist - 2)
    return value


def kurtosis(dist):
    """Calculates the centralised and normalised excess kurtosis of dist."""
    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)
    # now calculate the kurtosis
    value = np.sum(((dist - aver) / std) ** 4) / len(dist - 3) - 3.0
    return value


def bootstrap(dist, function, confidence_level=0.90, nboot=10000):
    """Carries out a bootstrap of dist to get the uncertainty of statistical
    function applied to it. Dist can be a numpy array or a pandas dataseries.
    confidence_level specifies the quantile (defaulted to 0.90). E.g 0.90
    means the quantile from 0.05 to 0.95 is evaluated.
    confidence_level=0.682 gives the range corresponding to 1 sigma,
    but evaluated using the corresponding
    quantiles.
    nboot (default 10000) is the number of bootstraps to be evaluated.
    Returns the lower and upper quantiles.
    A call of the form low, high = bootstrap(dist, np.mean, confidence_level=0.682)
    will return the lower and upper limits of the 1 sigma range"""
    fvalues = np.array([])  # creates an empty array to store function values
    dlen = len(dist)
    for i in range(nboot):
        rand = np.random.choice(dist, dlen, replace=True)
        fvalues = np.append(fvalues, function(rand))

    quantiles = [1 - confidence_level, confidence_level]
    lower_limit, upper_limit = np.percentile(fvalues, [100 * q for q in quantiles])

    return lower_limit, upper_limit


def stats_df(df):
    """
    Function to do some basic statistics on the dataframes.
    Takes the dataframe with countries as columns as the argument.
    """
    # exploring the dataset
    print("The summary statistics of the dataframe are: \n", df.describe())
    # some basic stats with custom functions and dataframe methods
    print(
        "Weighted geometric means of each country are ", gmean(df, axis=0),
        "\n Coefficient of variation of each country are ",
        variation(df, axis=0),
        "\n Maximum values for each country in these years are ",
        df.max(axis=0),
        "\n Minimum values for each country in these years are ",
        df.min(axis=0),
        "\n Skewness is ", skew(df))

    return

def plot_df(df, knd, title, color, y_label=None):
    """
    Function to plot the dataframes using the dataframe.plot method.

    Arguments:
    The dataframe to be plotted.
    The kind of plot required.
    The title of the figure.
    The color scheme for the plot.
    The label for the y-axis (optional).

    """
    # using conditional statements for different kinds for better customization
    if knd == "line":
        ax = df.plot(kind=knd, figsize=(7, 5), rot=20, color=color)
        ax.legend(loc='best', fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize='x-large',
                     fontname="Times New Roman")
        ax.set_xlabel("Years", fontweight='bold')
        if y_label:
            ax.set_ylabel(y_label, fontweight='bold')
        ax.grid(axis='x', alpha=.85, linewidth=.75)
        plt.savefig(title + ".png", dpi=600)
    else:
        ax = df.plot(kind=knd, figsize=(6.5, 5), rot=20, color=color)
        ax.legend(loc='best', fontsize=10)
        ax.set_title(title, fontweight='bold', fontsize='x-large',
                     fontname="Times New Roman")
        if y_label:
            ax.set_ylabel(y_label, fontweight='bold')
        plt.savefig(title + ".png", dpi=600, bbox_inches='tight')

    return


def makeheatmap(filename, country, indicators, c):
    """
    Function to plot the heatmap of a country's indicators. Parameters:
    The name of the csv file containing data of all indicators of
    all countries as a string(should end in .csv).
    The country of which we're plotting the heatmap of.
    The list of indicators we're considering for the heat map.
    The color scheme.
    """
    # making the dataframe with which the
    # correlation matrix is to be calculated
    df0 = pd.read_csv(filename, skiprows=4)
    df0.drop(columns=["Country Code", "Indicator Code"], inplace=True)
    # setting multi-index to easily select the country
    df0.set_index(["Country Name", "Indicator Name"], inplace=True)
    df1 = df0.loc[country].fillna(0).T

    # Print the columns to check available indicators
    print(f"Available Indicators for {country}: {df1.columns.tolist()}")

    # Slicing the dataframe to have only the years with nonzero data
    df = df1.loc["1970":"2020", indicators]
    df.rename(columns={
        "CO2 emissions (kt)":
            "CO2 emissions \n(kt)",
        "Energy use (kg of oil equivalent) per $1,000 GDP (constant 2017 PPP)":
            "Energy use (kg of oil equivalent) \nper $1,000 GDP (constant 2017 PPP)",
        "Agricultural land (% of land area)":
            "Agricultural land \n(% of land area)",
        "Electric power consumption (kWh per capita)":
            "Electric power consumption \n(kWh per capita)",
        "Methane emissions (kt of CO2 equivalent)":
            "Methane emissions \n(kt of CO2 equivalent)",
        "Total greenhouse gas emissions (kt of CO2 equivalent)":
            "Total greenhouse emissions \n(kt of CO2 equivalent)",
        "Arable land (% of land area)":
            "Arable land \n(% of land area)",
        "Forest area (% of land area)":
            "Forest area \n(% of land area)",
        "Population, total":
            "Population, total",
        "Mortality rate, under-5 (per 1,000 live births)":
             "Mortality rate, under-5 \n(per 1,000 live births))",
        "Urban population growth \n(annual %)":
            "Urban population growth (annual %)",
        "Urban population (% of total population)":
            "Urban population (% of total population)",
        "Access to electricity \n(% of population)":
            "Access to electricity (% of population)"         
    }, inplace=True)
    # Plotting the heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(df.corr(), cmap=c, annot=True)
    plt.xticks(rotation=90)
    # Setting a title and saving the figure
    plt.title(country, fontweight='bold', fontsize='x-large',
              fontname="Times New Roman")
    plt.savefig(country + "'s Heatmap" + ".png", dpi=450,
                bbox_inches='tight')

    return


# Choosing the countries and years for the dataframes
cntrs = ["Indonesia", "Brazil", "China", "Russian Federation", "India", "United States", "Nigeria", "Germany", "Japan"]
yrs = ["1990", "1995", "2000", "2005", "2010", "2015"]

# Creating the dataframes using the function
a, b = read_df("API_SP.POP.TOTL_DS2_en_csv_v2_6224560.csv", cntrs, yrs)
total_pop_1, total_pop_2 = a, b
c, d = read_df("API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_6225987.csv", cntrs, yrs)
pop_accs_1, pop_accs_2 = c, d
e, f = read_df("API_EN.ATM.CO2E.KT_DS2_en_csv_v2_6224818.csv", cntrs, yrs)
co2_e_1, co2_e_2 = e, f
g, h = read_df("API_EN.ATM.METH.KT.CE_DS2_en_csv_v2_6232520.csv", cntrs, yrs)
methane_e_1, methane_e_2 = g, h

# Creating some cool colormaps for the bar plots
c1 = cm.viridis(np.linspace(.1, .9, 6)[::-1])
c2 = cm.inferno(np.linspace(.2, .9, 6)[::-1])
# Some distinctive colors for the line plots
c3 = ['black', 'maroon', 'goldenrod', 'green',
      'teal', 'navy', 'hotpink', 'red', 'yellow']
c4 = ['b', 'g', 'r', 'k', 'm', 'y', 'c', 'brown', 'olive']

# Plotting dataframes with the function
plot_df(total_pop_1, 'bar', 'Population, total (annual %)', c2, y_label='Total population of each')
plt.show()
plot_df(pop_accs_1, 'bar','Access to electricity (% of population)', c1, y_label='Access to electricity rate (% of population)')
plt.show()
plot_df(co2_e_2, 'line', 'Carbon Dioxide emissions(kt)', c3, y_label='Carbon Dioxide emissions rate (kt)')
plt.show()
plot_df(methane_e_2, 'line', 'Methane emissions (kt of CO2 equivalent)', c4, y_label='Methane emissions rate (kt of CO2 equivalent)')
plt.show()

# Suppressing scientific notation to use the statistical tools
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# Doing the basic statistics with the function
stats_df(total_pop_2)
stats_df(pop_accs_2)

# Choosing the indicators to make heatmaps
indicators_to_plot = [
    'Population, total',
    'CO2 emissions (kt)',
    'Energy use (kg of oil equivalent) per $1,000 GDP (constant 2017 PPP)',
    'Electric power consumption (kWh per capita)',
    'Methane emissions (kt of CO2 equivalent)',
    'Total greenhouse gas emissions (kt of CO2 equivalent)',
    'Agricultural land (% of land area)',
    'Urban population (% of total population)',
    'Access to electricity (% of population)'
]

# Creating some heatmaps to compare indicators of countries, explore its correlations(or lack of)
makeheatmap("API_19_DS2_en_csv_v2_6224512.csv", "Russian Federation", indicators_to_plot, cm.winter)
plt.show()
makeheatmap("API_19_DS2_en_csv_v2_6224512.csv", "China", indicators_to_plot, cm.cool)
plt.show()
makeheatmap("API_19_DS2_en_csv_v2_6224512.csv", "Germany", indicators_to_plot, cm.jet)
plt.show()

