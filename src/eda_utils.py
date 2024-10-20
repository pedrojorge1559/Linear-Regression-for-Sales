import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def eda_utils(data):
    data.head()
    data.tail()
    data.info()
    data.describe().T
    data.shape()
    data.columns()
    data.dtypes()
    return


def plot_missing_values(data):
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    missing_values.plot(kind='bar')
    plt.show()

#utils for analysis plot from https://github.com/allmeidaapedro
def analysis_plots(data, features, histplot=True, barplot=False, mean=None, text_y=0.5,    
                   outliers=False, boxplot=False, boxplot_x=None, kde=False, hue=None, 
                   nominal=False, color='#023047', figsize=(24, 12)):
    '''
    Gera gráficos para análise univariada e bivariada.

    Esta função gera histogramas, gráficos de barras horizontais 
    e boxplots com base nos dados e características fornecidos. 

    Args:
        data (DataFrame): O DataFrame contendo os dados a serem visualizados.
        features (list): Uma lista de nomes de características a serem visualizadas.
        histplot (bool, optional): Gera histogramas. Padrão é True.
        barplot (bool, optional): Gera gráficos de barras horizontais. Padrão é False.
        mean (str, optional): Gera gráficos de barras de média da característica especificada. Padrão é None.
        text_y (float, optional): Coordenada Y para texto em gráficos de barras. Padrão é 0.5.
        outliers (bool, optional): Gera boxplots para visualização de outliers. Padrão é False.
        boxplot (bool, optional): Gera boxplots para comparação de distribuições de categorias. Padrão é False.
        boxplot_x (str, optional): A característica à qual as categorias terão suas distribuições comparadas. Padrão é None.
        kde (bool, optional): Plota a Estimativa de Densidade de Kernel nos histogramas. Padrão é False.
        hue (str, optional): Hue para histogramas e gráficos de barras. Padrão é None.
        nominal (bool, optional): Indica se as características são nominais. Padrão é False.
        color (str, optional): A cor do gráfico. Padrão é '#023047'.
        figsize (tuple, optional): O tamanho da figura do gráfico. Padrão é (24, 12).

    Returns:
        None

    '''
    
    try:
        # Obter num_features e num_rows e iterar sobre as dimensões do subplot.
        num_features = len(features)
        num_rows = num_features // 3 + (num_features % 3 > 0) 
        
        fig, axes = plt.subplots(num_rows, 3, figsize=figsize)  

        for i, feature in enumerate(features):
            row = i // 3  
            col = i % 3  

            ax = axes[row, col] if num_rows > 1 else axes[col] 
            
            if barplot:
                if mean:
                    data_grouped = data.groupby([feature])[mean].mean().reset_index()
                    data_grouped[mean] = round(data_grouped[mean], 2)
                    ax.barh(y=data_grouped[feature], width=data_grouped[mean], color=color)
                    for index, value in enumerate(data_grouped[mean]):
                        ax.text(value + text_y, index, f'{value:.1f}', va='center', fontsize=15)
                else:
                    if hue:
                        data_grouped = data.groupby([feature])[hue].mean().reset_index().rename(columns={hue: 'pct'})
                        data_grouped['pct'] *= 100
                    else:
                        data_grouped = data.groupby([feature])[feature].count().rename(columns={feature: 'count'}).reset_index()
                        data_grouped['pct'] = data_grouped['count'] / data_grouped['count'].sum() * 100
        
                    ax.barh(y=data_grouped[feature], width=data_grouped['pct'], color=color)
                    
                    if pd.api.types.is_numeric_dtype(data_grouped[feature]):
                        ax.invert_yaxis()
                        
                    for index, value in enumerate(data_grouped['pct']):
                        ax.text(value + text_y, index, f'{value:.1f}%', va='center', fontsize=15)
                
                ax.set_yticks(ticks=range(data_grouped[feature].nunique()), labels=data_grouped[feature].tolist(), fontsize=15)
                ax.get_xaxis().set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.grid(False)
        
            elif outliers:
                # Plot boxplot univariado.
                sns.boxplot(data=data, x=feature, ax=ax, color=color)
            
            elif boxplot:
                # Plot boxplot multivariado.
                sns.boxplot(data=data, x=boxplot_x, y=feature, showfliers=outliers, ax=ax, palette='Set2')

            else:
                # Plot histplot.
                sns.histplot(data=data, x=feature, kde=kde, ax=ax, color=color, stat='proportion', hue=hue)

            ax.set_title(feature)  
            ax.set_xlabel('')  
        
        # Remover eixos não utilizados.
        if num_features < len(axes.flat):
            for j in range(num_features, len(axes.flat)):
                fig.delaxes(axes.flat[j])

        plt.tight_layout()
    
    except Exception as e:
        print(f"An error occurred: {e}")

#utils check outliers from https://github.com/allmeidaapedro
def check_outliers(data, features, visualize=False):
    '''
    Check for outliers in the given dataset features.

    This function calculates and identifies outliers in the specified features
    using the Interquartile Range (IQR) method.

    Args:
        data (DataFrame): The DataFrame containing the data to check for outliers.
        features (list): A list of feature names to check for outliers.
        visualize (bool, optional): If True, plots the features with outliers highlighted. Default is False.

    Returns:
        tuple: A tuple containing three elements:
            - outlier_indexes (dict): A dictionary mapping feature names to lists of outlier indexes.
            - outlier_counts (dict): A dictionary mapping feature names to the count of outliers.
            - total_outliers (int): The total count of outliers in the dataset.

    '''
    
    try:
        outlier_counts = {}
        outlier_indexes = {}
        total_outliers = 0
        
        for feature in features:
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            feature_outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
            outlier_indexes[feature] = feature_outliers.index.tolist()
            outlier_count = len(feature_outliers)
            outlier_counts[feature] = outlier_count
            total_outliers += outlier_count
            
            if visualize:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=data[feature], color='#023047')
                plt.scatter(feature_outliers[feature], [0]*outlier_count, color='red', label='Outliers', zorder=5)
                plt.title(f'Outliers in {feature}')
                plt.legend()
                plt.show()
        
        print(f'There are {total_outliers} outliers in the dataset.')
        print()
        print(f'Number (percentage) of outliers per feature: ')
        print()
        for feature, count in outlier_counts.items():
            print(f'{feature}: {count} ({round(count/len(data)*100, 2)})%')

        return outlier_indexes, outlier_counts, total_outliers
    
    except Exception as e:
        print(f"An error occurred: {e}")