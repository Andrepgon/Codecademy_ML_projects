# 📊 Wrapper Methods for Feature Selection  

Análise de seleção de atributos utilizando **wrapper methods** em um conjunto de dados sobre hábitos alimentares e obesidade. O objetivo é identificar as melhores variáveis para prever se um indivíduo é obeso com base em suas respostas a um questionário.  

## 🔍 Sobre o Projeto  

O conjunto de dados utilizado foi obtido do [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+). Todas as variáveis categóricas foram convertidas para valores numéricos para facilitar a análise.  

Inicialmente, um modelo de **Regressão Logística** é ajustado com todas as features disponíveis. Em seguida, três diferentes **wrapper methods** são aplicados para selecionar um subconjunto otimizado de variáveis:  

- **Sequential Forward Selection (SFS)**  
- **Sequential Backward Floating Selection (SBFS)**  
- **Recursive Feature Elimination (RFE)**  

A performance do modelo é avaliada em cada cenário, comparando a acurácia ao utilizar todas as features e os subconjuntos reduzidos.  

## 📂 Estrutura do Repositório  

- `wrapper_method_solution.ipynb` → Notebook principal com código e análise detalhada.  
- `data/` → Diretório onde os dados podem ser armazenados (caso aplicável).  
- `README.md` → Este arquivo com informações sobre o projeto.  

## 🛠 Tecnologias Utilizadas  

- Python  
- Pandas  
- Scikit-Learn  
- MLxtend  
- Matplotlib  

## 🚀 Como Usar  

1. Clone este repositório:  
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   ```
2. Instale as dependências:  
   ```bash
   pip install -r requirements.txt
   ```
3. Abra o notebook e execute os passos da análise:  
   ```bash
   jupyter notebook wrapper_method_solution.ipynb
   ```

## 📈 Resultados  

Os resultados obtidos mostram como diferentes métodos de seleção de features impactam a performance do modelo. A comparação entre os métodos ajuda a entender quais variáveis são mais relevantes para prever obesidade e quais podem ser descartadas sem comprometer a acurácia.  

## 📚 Referências  

- [UCI Machine Learning Repository - Obesity Dataset](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+)  
- [Documentação do Scikit-Learn](https://scikit-learn.org/stable/)  
- [MLxtend Feature Selection](http://rasbt.github.io/mlxtend/)  
