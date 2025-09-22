# UFRJ Storm - Sistema de Previsão de Raios

Sistema de Machine Learning para previsão da quantidade diária de raios acumulada usando dados meteorológicos termodinâmicos.

## 🎯 Objetivos

O projeto implementa dois modelos de ML funcionando em cascata:

1. **Modelo Principal**: Prevê a quantidade de raios no dia (`contagem_raios`)
2. **Modelo de Incerteza**: Prevê o intervalo de confiança (95%) da predição

## 📊 Dataset

- **Arquivo**: `data/tma_sp.csv`
- **Período**: 2000-2019 (20 anos)
- **Registros**: 5.857 observações diárias
- **Features**: 30 variáveis meteorológicas termodinâmicas
- **Target**: `contagem_raios` (quantidade diária de raios)

### Variáveis Principais

- Índices termodinâmicos (Showalter, Lifted, K-index, etc.)
- Energia potencial convectiva (CAPE)
- Inibição convectiva (CIN)
- Parâmetros de cisalhamento do vento
- Água precipitável
- Temperaturas e pressões atmosféricas

### Divisão Temporal

- **Treinamento**: 2000-2014 (4.751 registros)
- **Teste**: 2015-2019 (1.106 registros)

## 🏗️ Estrutura do Projeto

```
ufrj-storm/
├── data/
│   └── tma_sp.csv                 # Dataset principal
├── src/
│   └── models.py                  # Classes dos modelos ML
├── notebooks/
│   └── data_quality_tests.ipynb   # Notebook principal TDD
├── config/
│   └── settings.py                # Configurações do projeto
├── models/                        # Modelos treinados (ignorado no git)
├── docs/                          # Documentação técnica
├── tests/                         # Testes automatizados
├── requirements.txt               # Dependências Python
├── .gitignore                     # Arquivos ignorados
└── README.md                      # Este arquivo
```

## 🚀 Como Executar

### 1. Pré-requisitos

```bash
# Python 3.8+
pip install -r requirements.txt
```

### 2. Executar Análise Completa

```bash
# Abrir Jupyter Notebook
jupyter notebook

# Executar notebook principal
# notebooks/data_quality_tests.ipynb
```

### 3. Executar Apenas Testes de Qualidade

```python
# No notebook, executar seções 1-4 para validação dos dados
```

## 🧪 Abordagem TDD

O projeto segue Test-Driven Development com testes automatizados para:

- ✅ Validação da estrutura dos dados
- ✅ Verificação de valores ausentes
- ✅ Consistência temporal
- ✅ Integridade das variáveis target
- ✅ Qualidade das features

### Testes Implementados

1. **Teste de Dimensões**: Verificar shape esperado do dataset
2. **Teste de Colunas**: Validar presença de colunas obrigatórias
3. **Teste de Data**: Verificar formato e validade das datas
4. **Teste de Target**: Validar variável de resposta
5. **Teste de Consistência**: Verificar coerência entre variáveis
6. **Teste Temporal**: Verificar continuidade da série temporal
7. **Teste de Features**: Validar tipos de dados das features
8. **Teste de Completude**: Verificar integridade dos dados

## 🤖 Modelos Implementados

### Modelo 1: Previsão de Quantidade de Raios

**Algoritmos Testados:**
- Linear Regression
- Ridge Regression  
- Lasso Regression
- Random Forest ⭐ (Melhor: RMSE = 2.792)
- Gradient Boosting
- XGBoost
- LightGBM

**Métricas de Performance:**
- RMSE: 2.792
- MAE: 1.376
- R²: 0.344

### Modelo 2: Intervalo de Confiança (95%)

**Algoritmo Usado:**
- Random Forest para predição da incerteza

**Métricas:**
- Cobertura: 62.1% (subótima, requer ajustes)
- Largura média do intervalo: 1.948

## 📈 Resultados Principais

### Análise dos Dados

- **Dias sem raios**: 37.1% (2.171 dias)
- **Dias com raios**: 62.9% (3.686 dias)
- **Valor máximo**: 57.034 raios/dia
- **Outliers**: 16.5% dos dados (968 registros)

### Features Mais Importantes

1. **Equivalentpotentialtemp[K]oftheLCL**: 19.6%
2. **Meanmixedlayerpotentialtemperature**: 8.0%
3. **Precipitablewater[mm]forentiresounding**: 5.9%
4. **Totalstotalsindex**: 4.7%
5. **Verticaltotalsindex**: 4.4%

### Qualidade dos Dados

- ✅ **Sem valores ausentes**
- ⚠️ **321 gaps na série temporal**
- ✅ **Consistência entre variáveis**
- ✅ **100% dos testes de qualidade aprovados**

## 🔧 Pipeline de Processamento

### 1. Carregamento e Validação
- Leitura do CSV
- Conversão de tipos
- Testes de qualidade automatizados

### 2. Pré-processamento
- Limpeza de nomes de colunas
- Criação de features temporais
- Features cíclicas (sazonalidade)
- Escalamento robusto

### 3. Divisão dos Dados
- Separação temporal (2000-2014 vs 2015-2019)
- Preservação da ordem temporal

### 4. Treinamento
- Múltiplos algoritmos em paralelo
- Validação cruzada (5-fold)
- Seleção automática do melhor modelo

### 5. Avaliação
- Métricas de regressão
- Análise de resíduos
- Importância das features
- Intervalos de confiança

## 📋 Melhorias Futuras

### Modelo Principal
- [ ] Otimização de hiperparâmetros (Optuna)
- [ ] Ensemble de modelos
- [ ] Feature selection automática
- [ ] Tratamento de outliers

### Modelo de Incerteza
- [ ] Calibração do intervalo de confiança
- [ ] Métodos quantile regression
- [ ] Bootstrap para incerteza
- [ ] Validação cruzada temporal

### Infraestrutura
- [ ] Pipeline automatizado (MLflow)
- [ ] Monitoramento de drift
- [ ] API para predições
- [ ] Dashboard interativo

## 📚 Documentação Técnica

Para documentação detalhada:

- **Roteiro de Testes**: `docs/test_protocol.md`
- **Roteiro de Produção**: `docs/production_guide.md`
- **API Reference**: Docstrings nos módulos Python

## 🐛 Troubleshooting

### Problemas Comuns

1. **Erro de importação do XGBoost/LightGBM**
   ```bash
   pip install xgboost lightgbm
   ```

2. **Caracteres especiais em features**
   - O pré-processamento automaticamente limpa os nomes

3. **Memória insuficiente**
   - Reduzir `n_estimators` nos modelos
   - Usar amostragem dos dados

## 👥 Contribuições

Para contribuir com o projeto:

1. Fork do repositório
2. Criar branch para feature
3. Executar todos os testes
4. Submit pull request

## 📄 Licença

Este projeto está sob licença MIT. Veja o arquivo LICENSE para detalhes.

---

**Desenvolvido para UFRJ** | **Previsão de Descargas Atmosféricas**