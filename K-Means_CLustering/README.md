# 🌟 K-Means Clustering do Zero!  

Este projeto implementa o algoritmo **K-Means Clustering** sem o uso da função pronta do Scikit-Learn. O objetivo é consolidar o aprendizado sobre aprendizado não supervisionado, aplicando a técnica ao conjunto de dados **Iris**.  

---

## 📌 Sobre o Projeto  

O K-Means Clustering é um dos algoritmos mais populares para agrupamento de dados. Neste projeto, seguimos os seguintes passos:  

1. **Inicialização**: Escolhemos aleatoriamente **K** centroides iniciais.  
2. **Atribuição de Grupos**: Cada ponto de dados é atribuído ao centróide mais próximo.  
3. **Atualização dos Centroides**: Os centroides são recalculados com base na média dos pontos atribuídos a eles.  
4. **Convergência**: O processo se repete até que os centroides parem de se mover significativamente.  

---

## 🛠 Tecnologias Utilizadas  

- **Python** 🐍  
- **NumPy** para manipulação de arrays  
- **Matplotlib** para visualização dos clusters  
- **Scikit-Learn** (somente para carregar o dataset)  

---

## 📊 Visualização  

O resultado final mostra os dados do conjunto **Iris**, com suas amostras agrupadas em três clusters distintos, representados por cores diferentes:  

![Gráfico de Clustering](C:\Users\andre\Downloads\FireShot\gráfico_kmean)   


