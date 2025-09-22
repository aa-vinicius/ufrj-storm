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
│   └── models.py                  # Classes dos modelos ML (biblioteca principal)
├── notebooks/
│   └── data_quality_tests.ipynb   # Notebook TDD para desenvolvimento
├── config/
│   └── settings.py                # Configurações do projeto  
├── models/                        # Modelos treinados (ignorado no git)
├── results/                       # Plots e relatórios gerados
├── docs/                          # Documentação técnica
├── tests/                         # Testes automatizados
├── run_quick_pipeline.py          # Pipeline rápido (demonstração)
├── run_production_pipeline.py     # Pipeline completo (produção)
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

### 2. Pipeline Rápido (Demonstração)

```bash
# Executar pipeline completo com 5 visualizações
python run_quick_pipeline.py
```

### 3. Pipeline de Produção (Completo)

```bash
# Executar pipeline completo com todos os algoritmos
python run_production_pipeline.py
```

### 4. Desenvolvimento Interativo (Jupyter)

```bash
# Abrir Jupyter Notebook para desenvolvimento
jupyter notebook

# Executar notebook principal para testes TDD
# notebooks/data_quality_tests.ipynb
```

### 5. Executar Apenas Testes de Qualidade

```python
# No notebook, executar células 1-6 para validação dos dados
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

### Arquitetura: Dois Modelos em Cascata

O sistema utiliza as classes `LightningPredictor` e `UncertaintyPredictor` da biblioteca `src/models.py`:

### Modelo 1: Previsão de Quantidade de Raios (`LightningPredictor`)

**Algoritmos Testados:**
- Linear Regression
- Ridge Regression  
- Lasso Regression
- Random Forest ⭐ (Melhor: RMSE = 2.888)
- Gradient Boosting
- XGBoost
- LightGBM

**Métricas de Performance (Melhor Modelo):**
- RMSE: 2.888
- MAE: 1.447
- R²: 0.297

### Modelo 2: Intervalo de Confiança 95% (`UncertaintyPredictor`)

**Algoritmo Usado:**
- Random Forest para predição da incerteza

**Métricas:**
- Cobertura: ~59.5% (em desenvolvimento)
- Largura média do intervalo: ~2.071

## � Visualizações Geradas

O pipeline gera **5 plots** automaticamente em `/results/`:

1. **Plot 1: Série Temporal Completa (2000-2019)**
   - Treino vs teste com intervalos de confiança
   - Círculos coloridos para observado/predito

2. **Plot 2: Análise Detalhada do Período de Teste**
   - Zoom no período 2015-2019
   - Métricas de performance detalhadas

3. **Plot 3: Acumulados Mensais**
   - Comparação mensal observado vs predito
   - Intervalos de confiança 95% mensais

4. **Plot 4: Período Contínuo de 30 Dias**
   - Zoom em 30 dias consecutivos do teste
   - Visualização detalhada dia a dia

5. **Plot 5: Período Específico (Nov/2018 - Mar/2019)**
   - Análise de período histórico específico
   - 150 dias com marcadores mensais

## �📈 Resultados Principais

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

### Arquitetura do Sistema

O sistema possui **2 pipelines principais** que utilizam a biblioteca `src/models.py`:

#### Pipeline Rápido (`run_quick_pipeline.py`)
- Demonstração com Random Forest
- 5 visualizações automáticas
- Relatório JSON completo
- Execução ~2-3 minutos

#### Pipeline de Produção (`run_production_pipeline.py`)
- Todos os 7 algoritmos de ML
- Validação cruzada completa
- Seleção automática do melhor modelo
- Relatórios detalhados
- Execução ~10-15 minutos

### Etapas do Processamento

### 1. Carregamento e Validação
- Leitura do CSV via `src/models.py`
- Conversão de tipos automática
- Testes de qualidade TDD (8 testes)

### 2. Pré-processamento
- Limpeza de nomes de colunas
- Criação de features temporais
- Escalamento robusto automático

### 3. Divisão dos Dados
- Separação temporal (2000-2014 vs 2015-2019)
- Preservação da ordem temporal

### 4. Treinamento
- Classe `LightningPredictor`: múltiplos algoritmos
- Classe `UncertaintyPredictor`: intervalos de confiança
- Validação cruzada (5-fold)
- Seleção automática do melhor modelo

### 5. Avaliação e Visualização
- Métricas de regressão automáticas
- Geração de 5 plots
- Análise de incerteza
- Relatórios JSON/texto

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

## � Arquivos Gerados

Após executar os pipelines, os seguintes arquivos são criados em `/results/`:

### Visualizações (5 plots)
- `temporal_series_complete_*.png` - Série temporal completa
- `detailed_analysis_*.png` - Análise detalhada do teste
- `monthly_accumulations_*.png` - Acumulados mensais
- `detailed_30days_*.png` - Período de 30 dias
- `specific_period_2018-2019_*.png` - Nov/2018 a Mar/2019

### Relatórios e Modelos
- `quick_report_*.json` - Relatório completo (pipeline rápido)
- `production_report_*.txt` - Relatório detalhado (pipeline produção)
- `lightning_model_*.joblib` - Modelo treinado salvo

## �🐛 Troubleshooting

### Problemas Comuns

1. **Erro de importação do XGBoost/LightGBM**
   ```bash
   pip install xgboost lightgbm
   ```

2. **Erro "ModuleNotFoundError: No module named 'src'"**
   ```bash
   # Execute sempre a partir do diretório raiz do projeto
   cd ufrj-storm/
   python run_quick_pipeline.py
   ```

3. **Memória insuficiente**
   - Use o pipeline rápido: `python run_quick_pipeline.py`
   - Edite `n_estimators` em `config/settings.py`

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