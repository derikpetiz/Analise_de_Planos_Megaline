# 📱 Análise Preditiva e Classificação de Planos Telecom — Megaline

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## 👨‍💻 Apresentação & Prólogo

* **Autor:** Derik Petiz
* **Contexto:** Projeto Prático do Módulo de Introdução ao Aprendizado de Máquina — MBA em Data Science & Artificial Intelligence (TripleTen)
* **Objetivo:** Construir e comparar modelos de aprendizado de máquina supervisionado para classificar o comportamento de consumo de clientes de telecomunicações e recomendar a migração para planos de telefonia mais adequados.
* **Repositório Principal:** [GitHub](https://github.com/derikpetiz/Analise_de_Planos_Megaline)

---

## 📌 Descrição do Problema de Negócio

A operadora de telefonia celular **Megaline** identificou um desafio comercial relevante: um volume expressivo de sua base de clientes ainda utiliza planos legados (antigos). Para otimizar a receita e melhorar a retenção, a empresa deseja migrar estrategicamente esses usuários para seus dois novos planos principais: **Smart** e **Ultra**.

### 🎯 Objetivo do Modelo
Desenvolver um modelo preditivo de classificação supervisionada capaz de analisar o comportamento de uso mensal dos clientes (quantidade de chamadas, minutos falados, mensagens de texto enviadas e consumo de internet em MB) e recomendar o plano ideal (**Smart** = `0` ou **Ultra** = `1`).

### 📐 Critério de Aceite (Métrica de Sucesso)
* **Métrica Principal:** Acurácia (*Accuracy*).
* **Meta do Projeto:** Alcançar uma acurácia mínima de **0.75 (75%)** no conjunto de teste não observado.

---

## 📊 Estrutura e Distribuição dos Dados

O conjunto de dados (`users_behavior.csv`) contém **3.214 registros** e **5 colunas**, sem valores ausentes ou nulos. As variáveis comportamentais observadas por usuário/mês são:

| Variável | Tipo | Descrição |
| :--- | :---: | :--- |
| `calls` | `float64` | Número de chamadas telefônicas realizadas no mês |
| `minutes` | `float64` | Duração total das chamadas (em minutos) |
| `messages` | `float64` | Quantidade de mensagens de texto enviadas |
| `mb_used` | `float64` | Tráfego de dados de internet consumido (em MB) |
| `is_ultra` | `int64` | **Variável Alvo:** `1` se o cliente usa o plano Ultra, `0` se usa o plano Smart |

![Distribuição da Variável Alvo](./images/class_distribution.png)

> **Distribuição da Base:** Cerca de **69.35%** dos usuários pertencem ao plano Smart (classe 0) e **30.65%** ao plano Ultra (classe 1). Essa proporção serviu de referência para a amostragem estratificada e para a Prova Real (*Sanity Check*).

---

## ⚙️ Metodologia e Arquitetura da Solução

1. **Análise Exploratória & Balanceamento:** Verificação da integridade dos dados e identificação do desbalanceamento moderado das classes.
2. **Divisão Estruturada dos Dados (60/20/20):** Separação em **Treino (60%)**, **Validação (20%)** e **Teste (20%)** utilizando `stratify=y` e `SEED=12345` para garantir a reprodutibilidade dos experimentos.
3. **Seleção de Modelos & Tuning de Hiperparâmetros:** Avaliação de três algoritmos no conjunto de validação:
   * *Decision Tree Classifier* (variação de `max_depth` de 1 a 10)
   * *Random Forest Classifier* (variação de `n_estimators` de 10 a 100 e `max_depth` de 1 a 10)
   * *Logistic Regression* (modelo baseline linear com `max_iter=1000`)
4. **Avaliação Final:** Avaliação do modelo campeão no conjunto de teste inédito.
5. **Prova Real (*Sanity Check*):** Comparação do desempenho com o classificador ingênuo `DummyClassifier(strategy='most_frequent')`.

---

## 📈 Resultados e Comparativo dos Modelos

### Performance no Conjunto de Validação

| Algoritmo | Hiperparâmetros Otimizados | Acurácia na Validação | Status |
| :--- | :---: | :---: | :---: |
| **Árvore de Decisão** | `max_depth = 8` | 80.25% | Aprovado |
| **Floresta Aleatória** | `n_estimators = 40`, `max_depth = 9` | **82.12%** | **Modelo Campeão** |
| **Regressão Logística** | `max_iter = 1000`, `solver = 'lbfgs'` | 73.87% | Reprovado (< 75%) |

![Acurácia dos Modelos no Conjunto de Validação](./images/model_comparison.png)

### Avaliação Final no Conjunto de Teste & Prova Real

* **Acurácia no Conjunto de Teste (Random Forest):** **80.87%** (Superando a meta de 75.00%).
* **Acurácia do Baseline (Dummy Classifier):** **69.36%** (Previsão constante da classe majoritária Smart).
* **Ganho Real do Modelo:** **+11.51 pontos percentuais** superior ao chute aleatório/majoritário.

---

## 🖼️ Importância das Variáveis (Feature Importances)

Analisando a estrutura do modelo **Floresta Aleatória** através do atributo `feature_importances_`, identificamos a relevância de cada variável comportamental na tomada de decisão:

![Importância das Variáveis na Predição do Plano](./images/feature_importance.png)

> **Insight Preditivo:** O tráfego de dados (`mb_used`) representa sozinho **37.60%** da relevância relativa do modelo, consolidando-se como o principal fator diferenciador de perfil entre os clientes dos planos Smart e Ultra.

---

## 💡 Insights e Conclusões de Negócio

1. **O Consumo de Internet é o Principal Driver de Upgrade:**
   * O volume de megabytes consumidos (`mb_used`) lidera isoladamente a relevância preditiva (**37.60%**). Usuários com consumo intenso de internet apresentam alta probabilidade de aderir ao plano **Ultra**.
2. **Minutos, Chamadas e Mensagens em Equilíbrio:**
   * O número de chamadas (`calls` com **21.58%**), a quantidade de mensagens (`messages` com **20.80%**) e os minutos falados (`minutes` com **20.02%**) possuem pesos equilibrados, demonstrando que a intensidade geral de uso do smartphone complementa a decisão do upgrade.
3. **Viabilidade Técnica e Financeira:**
   * O modelo atingiu **80.87% de acurácia** no conjunto de teste não observado, superando o baseline em **11.51 pontos percentuais**. Isso permite à Megaline automatizar ofertas personalizadas via CRM, reduzindo custos com campanhas ineficientes e aumentando a taxa de conversão na migração de planos.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.9+
* **Manipulação e Análise de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`train_test_split`, `DecisionTreeClassifier`, `RandomForestClassifier`, `LogisticRegression`, `DummyClassifier`, `accuracy_score`)
* **Visualização de Dados:** Matplotlib, Seaborn
* **Ambiente de Desenvolvimento:** Jupyter Notebook / VS Code

---

## 📂 Como Executar o Projeto

1. Clone o repositório:
   ```bash
   git clone [https://github.com/derikpetiz/Analise_de_Planos_Megaline.git](https://github.com/derikpetiz/Analise_de_Planos_Megaline.git)
