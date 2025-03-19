import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def load_data(file_path):
    """
    Carrega o CSV, imprime as primeiras linhas, informações gerais e os tipos de dados.
    """
    df = pd.read_csv(file_path)
    print("Primeiras linhas do DataFrame:")
    print(df.head())
    print("\nInformações do DataFrame:")
    print(df.info())
    print("\nTipos de dados:")
    print(df.dtypes)
    return df

def plot_scatter_matrix(df, columns, title):
    """
    Plota uma matriz de dispersão para as colunas especificadas.
    """
    scatter_matrix(df[columns], figsize=(10, 10), diagonal='kde')
    plt.suptitle(title)
    plt.show()
    plt.clf()

def plot_scatter(x, y, xlabel, ylabel, title, color='b'):
    """
    Plota um gráfico de dispersão com os rótulos e título informados.
    """
    plt.scatter(x, y, color=color, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.clf()

def linear_regression_single_feature(feature, target, feature_name):
    """
    Realiza regressão linear com uma única variável.
    Separa os dados em treino/teste, treina o modelo, imprime o score
    e plota as previsões.
    """
    X_train, X_test, y_train, y_test = train_test_split(feature, target, train_size=0.8, test_size=0.2)
    # Converter para array 2D
    X_train = X_train.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    print(f"Score do modelo com {feature_name}: {score:.2f}")
    
    y_pred = model.predict(X_test)
    plot_scatter(y_test, y_pred, "Vitórias reais", "Vitórias previstas",
                 f'Previsão de vitória usando "{feature_name}"', color='b')
    return model

def linear_regression_two_features(features, target, feature_names):
    """
    Realiza regressão linear com duas variáveis.
    Separa os dados, treina o modelo, imprime o score e plota as previsões.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    print(f"Score usando as variáveis de previsão {', '.join(feature_names)}: {score:.2f}")
    
    y_pred = model.predict(X_test)
    plot_scatter(y_test, y_pred, "Vitórias reais", "Vitórias previstas",
                 f'Previsão de vitória usando {", ".join(feature_names)}', color='m')
    return model

def linear_regression_multiple_features(features, target):
    """
    Realiza regressão linear com múltiplas variáveis.
    Separa os dados, treina o modelo, imprime o score e plota as previsões.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    print(f"Score usando múltiplos parâmetros: {score:.2f}")
    
    y_pred = model.predict(X_test)
    plot_scatter(y_test, y_pred, "Vitórias reais", "Vitórias previstas",
                 "Previsão de vitória usando múltiplas variáveis", color='c')
    return model

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    # Carrega os dados
    df = load_data('tennis_stats.csv')
    
    # =============================================================================
    # Análise Exploratória (EDA)
    # =============================================================================
    print("\n--- Scatter Matrix: Estatísticas de Return Match vs Winnings ---")
    plot_scatter_matrix(df, 
                        ['FirstServeReturnPointsWon', 'SecondServeReturnPointsWon', 
                         'ReturnGamesWon', 'ReturnPointsWon', 'Winnings'],
                        "Relação entre estatísticas de Return e Winnings")
    
    print("\n--- Scatter Matrix: Estatísticas de Break Points vs Winnings ---")
    plot_scatter_matrix(df, 
                        ['BreakPointsConverted', 'BreakPointsFaced', 
                         'BreakPointsOpportunities', 'BreakPointsSaved', 'Winnings'],
                        "Relação entre estatísticas de Break Points e Winnings")
    
    # Gráficos individuais que chamaram a atenção
    plot_scatter(df['BreakPointsOpportunities'], df['Winnings'],
                 'Break Points Opportunities', 'Total Winnings',
                 'Relação: Break Points Opportunities vs Winnings', color='g')
    
    plot_scatter(df['BreakPointsFaced'], df['Winnings'],
                 'Break Points Faced', 'Total Winnings',
                 'Relação: Break Points Faced vs Winnings', color='r')
    
    # =============================================================================
    # Modelos de Regressão Linear
    # =============================================================================
    print("\n--- Regressão Linear: Previsão de Winnings ---")
    target = df['Winnings']
    
    # Modelo com uma única variável: Break Points Opportunities
    model_bpo = linear_regression_single_feature(df['BreakPointsOpportunities'], target, "Break Points Opportunities")
    
    # Modelo com uma única variável: Break Points Faced
    model_bpf = linear_regression_single_feature(df['BreakPointsFaced'], target, "Break Points Faced")
    
    print("\nObservação: O modelo baseado em 'Break Points Opportunities' apresentou melhor performance.")
    
    # Modelos com duas variáveis (usando Break Points Opportunities em conjunto com outra variável)
    features_fsrpw = df[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]
    model_two_fsrpw = linear_regression_two_features(features_fsrpw, target, 
                                                     ["BreakPointsOpportunities", "FirstServeReturnPointsWon"])
    
    features_ssrpw = df[['BreakPointsOpportunities', 'SecondServeReturnPointsWon']]
    model_two_ssrpw = linear_regression_two_features(features_ssrpw, target, 
                                                     ["BreakPointsOpportunities", "SecondServeReturnPointsWon"])
    
    features_rpw = df[['BreakPointsOpportunities', 'ReturnPointsWon']]
    model_two_rpw = linear_regression_two_features(features_rpw, target, 
                                                   ["BreakPointsOpportunities", "ReturnPointsWon"])
    
    print("Nota: A combinação com 'FirstServeReturnPointsWon' resultou em um aumento no score do modelo.")
    
    # Modelo com múltiplas variáveis
    features_multi = df[['FirstServe', 'FirstServePointsWon', 'FirstServeReturnPointsWon',
                         'SecondServePointsWon', 'SecondServeReturnPointsWon', 'Aces',
                         'BreakPointsConverted', 'BreakPointsFaced', 'BreakPointsOpportunities',
                         'BreakPointsSaved', 'DoubleFaults', 'ReturnGamesPlayed', 'ReturnGamesWon',
                         'ReturnPointsWon', 'ServiceGamesPlayed', 'ServiceGamesWon', 'TotalPointsWon',
                         'TotalServicePointsWon']]
    model_multi = linear_regression_multiple_features(features_multi, target)
    
    print("\nConclusão: O uso de múltiplas variáveis melhorou a capacidade preditiva do modelo.")

if __name__ == "__main__":
    main()
