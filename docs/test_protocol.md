# Roteiro Técnico - Testes de Qualidade

Este documento descreve o protocolo completo para execução dos testes de qualidade dos dados do projeto UFRJ Storm.

## 🎯 Objetivo

Garantir a integridade, consistência e qualidade dos dados meteorológicos antes do treinamento dos modelos de ML.

## 📋 Pré-requisitos

- Python 3.8+
- Jupyter Notebook
- Dependências instaladas (`pip install -r requirements.txt`)
- Dataset `data/tma_sp.csv` disponível

## 🧪 Protocolo de Testes

### 1. Preparação do Ambiente

```bash
# Clonar repositório
git clone <repo-url>
cd ufrj-storm

# Instalar dependências
pip install -r requirements.txt

# Iniciar Jupyter
jupyter notebook
```

### 2. Execução dos Testes

Abrir o notebook `notebooks/data_quality_tests.ipynb` e executar as células na ordem:

#### Seção 1: Configuração (Células 1-2)
- **Objetivo**: Importar bibliotecas e configurar ambiente
- **Tempo esperado**: ~10 segundos
- **Saída esperada**: Mensagens de sucesso na importação

#### Seção 2: Carregamento dos Dados (Célula 3)
- **Objetivo**: Carregar e explorar dataset
- **Tempo esperado**: ~5 segundos
- **Validações**:
  - Dataset com 5.857 registros e 33 colunas
  - Período: 2000-12-01 a 2019-09-30
  - Memória: ~1.7 MB

#### Seção 3: Análise de Qualidade (Células 4-5)
- **Objetivo**: Analisar missing values, gaps temporais, outliers
- **Tempo esperado**: ~30 segundos
- **Validações esperadas**:
  - ✅ Nenhum valor ausente
  - ⚠️ ~321 gaps na série temporal (esperado)
  - 16.5% de outliers (968 registros)
  - 37.1% dias sem raios, 62.9% com raios

#### Seção 4: Testes Automatizados (Célula 6)
- **Objetivo**: Executar bateria de testes TDD
- **Tempo esperado**: ~2 segundos
- **Resultado esperado**: 100% dos testes aprovados (8/8)

### 3. Interpretação dos Resultados

#### ✅ Testes que DEVEM passar:
1. **Teste dimensões**: Dataset com shape correto
2. **Teste colunas obrigatórias**: Presença de `data`, `contagem_raios`, `sim_nao`
3. **Teste formato de data**: Coluna `data` em datetime
4. **Teste variável target**: `contagem_raios` não-negativa e numérica
5. **Teste consistência**: Coerência entre `contagem_raios` e `sim_nao`
6. **Teste continuidade temporal**: >90% intervalos de 1 dia
7. **Teste features numéricas**: Todas features são numéricas
8. **Teste completude**: Sem registros completamente vazios

#### ⚠️ Alertas esperados:
- **Gaps temporais**: 321 gaps são esperados (dados históricos incompletos)
- **Outliers**: 16.5% é aceitável para dados meteorológicos extremos

#### ❌ Condições de falha:
- Taxa de aprovação < 100% nos testes
- Valores ausentes na target
- Inconsistências entre variáveis
- Tipos de dados incorretos

### 4. Critérios de Aprovação

Para prosseguir com o treinamento dos modelos:

- [ ] **100% dos testes automatizados aprovados**
- [ ] **Nenhum valor ausente nas variáveis críticas**
- [ ] **Dataset com pelo menos 5.000 registros**
- [ ] **Período temporal cobrindo treino e teste**
- [ ] **Features numéricas com distribuições válidas**

### 5. Troubleshooting

#### Problema: Erro na importação de bibliotecas
```bash
# Solução: Reinstalar dependências
pip install --upgrade -r requirements.txt
```

#### Problema: Dataset não encontrado
```bash
# Verificar localização do arquivo
ls -la data/tma_sp.csv

# Verificar path no notebook
print(DATA_DIR / config.input_file)
```

#### Problema: Falha nos testes de qualidade
1. Examinar mensagem de erro específica
2. Verificar integridade do dataset original
3. Revisar critérios de validação nos testes
4. Documentar desvios encontrados

### 6. Documentação dos Resultados

Após execução dos testes, documentar:

#### Métricas de Qualidade
- Taxa de aprovação dos testes: ____%
- Número de registros válidos: _____
- Período efetivo dos dados: ______
- Gaps temporais identificados: _____
- Outliers detectados: ____%

#### Observações
- Problemas encontrados: _____________
- Ações corretivas aplicadas: ________
- Aprovação para treinamento: [ ] Sim [ ] Não

### 7. Próximos Passos

Se todos os testes passaram:
- ✅ Prosseguir para Seção 5 (Pré-processamento)
- ✅ Executar pipeline completo de treinamento

Se houver falhas:
- ❌ Investigar causas raiz
- ❌ Aplicar correções necessárias
- ❌ Re-executar testes

## 📊 Exemplo de Relatório

```
=== RELATÓRIO DE TESTES DE QUALIDADE ===
Data: 2024-XX-XX
Executor: [Nome]

RESULTADOS:
✅ Todos os 8 testes aprovados (100%)
✅ 5.857 registros carregados
✅ 30 features numéricas validadas
⚠️ 321 gaps temporais identificados (esperado)
⚠️ 968 outliers detectados (16.5% - aceitável)

APROVAÇÃO: SIM
Dados prontos para treinamento dos modelos.
```

---

**Nota**: Este protocolo deve ser executado sempre que houver mudanças no dataset ou nas regras de validação.