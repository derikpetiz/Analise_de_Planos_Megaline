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

---

## 📌 Descrição do Problema de Negócio

A operadora de telefonia celular **Megaline** identificou um desafio comercial relevante: um volume expressivo de sua base de clientes ainda utiliza planos legados (antigos). Para otimizar a receita e melhorar a retenção, a empresa deseja migrar estrategicamente esses usuários para seus dois novos planos principais: **Smart** e **Ultra**.

### 🎯 Objetivo do Modelo
Desenvolver um modelo preditivo de classificação supervisionada capaz de analisar o comportamento de uso mensal dos clientes (quantidade de chamadas, minutos falados, mensagens de texto enviadas e consumo de internet em MB) e recomendar o plano ideal (**Smart** = `0` ou **Ultra** = `1`).

### 📐 Critério de Aceite (Métrica de Sucesso)
* **Métrica Principal:** Acurácia (*Accuracy*).
* **Meta do Projeto:** Alcançar uma acurácia mínima de **0.75 (75%)** no conjunto de teste não observado.

---

## 📊 Estrutura dos Dados

O conjunto de dados (`users_behavior.csv`) contém **3.214 registros** e **5 colunas**, sem valores ausentes ou nulos. As variáveis comportamentais observadas por usuário/mês são:

| Variável | Tipo | Descrição |
| :--- | :---: | :--- |
| `calls` | `float64` | Número de chamadas telefônicas realizadas no mês |
| `minutes` | `float64` | Duração total das chamadas (em minutos) |
| `messages` | `float64` | Quantidade de mensagens de texto enviadas |
| `mb_used` | `float64` | Tráfego de dados de internet consumido (em MB) |
| `is_ultra` | `int64` | **Variável Alvo:** `1` se o cliente usa o plano Ultra, `0` se usa o plano Smart |

---

## ⚙️ Metodologia e Arquitetura da Solução

1. **Análise Exploratória & Balanceamento:** Verificação da integridade dos dados e identificação da proporção das classes (69.35% Smart / 30.65% Ultra).
2. **Divisão de Dados (60/20/20):** Separação em **Treino (60%)**, **Validação (20%)** e **Teste (20%)** utilizando amostragem estratificada (`stratify=y`) para manter a proporção da classe alvo constante entre todas as bases.
3. **Seleção de Modelos & Tuning de Hiperparâmetros:** Avaliação de três algoritmos no conjunto de validação:
   * *Decision Tree Classifier* (variação de `max_depth` de 1 a 10)
   * *Random Forest Classifier* (combinação de `n_estimators` de 10 a 100 e `max_depth` de 1 a 10)
   * *Logistic Regression* (modelo baseline linear com `max_iter=1000`)
4. **Avaliação Final:** Teste do modelo campeão no conjunto de teste mantido isolado.
5. **Prova Real (*Sanity Check*):** Comparação do desempenho com o algoritmo ingênuo `DummyClassifier(strategy='most_frequent')`.

---

## 📈 Resultados e Comparativo dos Modelos

### Performance no Conjunto de Validação

| Algoritmo | Hiperparâmetros Otimizados | Acurácia na Validação |
| :--- | :---: | :---: |
| **Árvore de Decisão** | `max_depth = 8` | 80.25% |
| **Floresta Aleatória** | `n_estimators = 40`, `max_depth = 9` | **82.12% (Campeão)** |
| **Regressão Logística** | `max_iter = 1000`, `solver = 'lbfgs'` | 73.87% |

### Avaliação Final no Conjunto de Teste & Prova Real

* **Acurácia no Conjunto de Teste (Random Forest):** **80.87%** (Superando a meta de 75.00%).
* **Acurácia do Baseline (Dummy Classifier):** **69.36%** (Previsão constante da classe majoritária Smart).
* **Ganho do Modelo Proposto:** **+11.51 pontos percentuais** sobre o chute aleatório/majoritário.

---

## 🖼️ Importância das Variáveis (Feature Importances)

Analisando a estrutura da **Floresta Aleatória**, identificamos quais fatores comportamentais têm maior peso preditivo para prever a adesão ao plano **Ultra**:
