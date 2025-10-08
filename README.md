# Processamento de Dados de Descargas Atmosféricas RINDAT

Este projeto processa dados brutos de descargas atmosféricas do formato RINDAT e gera análises, visualizações e relatórios detalhados.

## Estrutura do Projeto

```
ufrj-storm/
├── data/
│   └── RINDAT/           # Arquivos de dados (.dat)
├── src/
│   ├── data_processor.py # Módulo de carregamento de dados
│   └── data_analyzer.py  # Módulo de análise e visualização
├── tests/
│   └── test_data_processing.py # Testes unitários
├── output/               # Arquivos gerados (criado automaticamente)
│   ├── reports/         # Relatórios de processamento
│   ├── plots/           # Gráficos e visualizações
│   ├── maps/            # Mapas de cobertura
│   └── filtered_data/   # Dados filtrados
├── main.py              # Script principal
├── requirements.txt     # Dependências Python
└── README.md           # Este arquivo
```

## Instalação e Configuração

### 1. Criar Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate  # No Linux/Mac
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

## Formato dos Dados

O projeto suporta dois formatos de dados RINDAT:

### Formato Antigo (2000-2017)
```
MM/DD/YY HH:MM:SS.mmm Latitude Longitude Classificação Intensidade Unidade Col8 Col9
```

Exemplo:
```
04/18/00 00:00:04.338 -23.6257 -52.7229 G -15.2 kA 1.3 6
```

### Formato Novo (2018-2021)
```
YYYY-MM-DD HH:MM:SS.nnnnnnnnn Latitude Longitude Intensidade Unidade Classificação Col8 Col9
```

Exemplo:
```
2021-01-01 00:00:05.859154688 -26.6665 -44.2377 -36.2 kA G 1 4
```

## Estrutura das Colunas

1. **Data/Datetime**: Data e hora da descarga
2. **Latitude**: Coordenada geográfica (-90 a +90)
3. **Longitude**: Coordenada geográfica (-180 a +180)
4. **Classificação**: Tipo da descarga (G, C, etc.)
5. **Intensidade**: Valor da intensidade (pode ser positiva ou negativa)
6. **Unidade**: Unidade de medida (kA)
7. **Col8, Col9**: Metadados adicionais

## Processamento Executado

O pipeline de processamento executa as seguintes etapas:

### 1. Carregamento dos Dados
- Identifica automaticamente o formato dos arquivos
- Carrega e combina todos os arquivos RINDAT
- Gera relatório de carregamento

### 2. Remoção de Duplicatas
- Identifica registros duplicados baseado em datetime, coordenadas, classificação e intensidade
- Remove duplicatas mantendo o primeiro registro
- Gera relatório de redução

### 3. Filtro por Horário
- Filtra descargas ocorridas entre 12:00:00 e 23:59:59
- Gera relatório de filtros aplicados

### 4. Análises e Visualizações

#### Arquivo por Dia do Ano
- Agrupa dados por dia (DD/MM)
- Calcula estatísticas por dia

#### Mapa de Cobertura Espacial
- Cria mapa interativo com bounding box
- Mostra área geográfica dos dados

#### Gráficos por Ano
- Total de descargas por ano
- Análise por classificação e sinal de intensidade
- Heatmap combinado

#### Gráficos por Mês
- Total de descargas por mês
- Análise sazonal
- Distribuição por classificação

#### Arquivos Filtrados
- Dados de classificação G com intensidade positiva
- Dados de classificação G com intensidade negativa

## Execução

### Processamento Completo
```bash
python main.py
```

### Executar Testes
```bash
python -m pytest tests/ -v
```

### Executar Teste Específico
```bash
python -m pytest tests/test_data_processing.py::TestClass::test_method -v
```

## Arquivos Gerados

Após a execução, os seguintes arquivos são gerados na pasta `output/`:

### Relatórios (`output/reports/`)
- `relatorio_carregamento_dados.txt` - Estatísticas de carregamento
- `relatorio_remocao_duplicatas.txt` - Resultado da limpeza
- `relatorio_filtro_horario.txt` - Aplicação de filtros
- `dados_por_dia_do_ano.csv` - Dados agrupados por dia
- `relatorio_dados_diarios.txt` - Estatísticas diárias
- `relatorio_arquivos_filtrados.txt` - Resumo dos filtros aplicados

### Visualizações (`output/plots/`)
- `graficos_analise_anual.png` - Gráficos por ano
- `graficos_analise_mensal.png` - Gráficos por mês

### Mapas (`output/maps/`)
- `mapa_cobertura_espacial.html` - Mapa interativo de cobertura
- `relatorio_cobertura_espacial.txt` - Dados da área de cobertura

### Dados Filtrados (`output/filtered_data/`)
- `descargas_G_intensidade_positiva.csv` - Classificação G positiva
- `descargas_G_intensidade_negativa.csv` - Classificação G negativa

## Bibliotecas Utilizadas

- **pandas**: Manipulação de dados
- **numpy**: Operações numéricas  
- **matplotlib**: Gráficos estáticos
- **seaborn**: Visualizações estatísticas
- **folium**: Mapas interativos
- **pytest**: Testes unitários
- **geopandas**: Dados geoespaciais

## Estrutura de Classes

### `RindatDataProcessor`
Responsável pelo carregamento e processamento inicial dos dados:
- `identify_file_format()`: Identifica formato do arquivo
- `parse_old_format()`: Processa formato antigo
- `parse_new_format()`: Processa formato novo
- `load_single_file()`: Carrega arquivo individual
- `load_all_files()`: Carrega todos os arquivos
- `generate_loading_report()`: Gera relatório de carregamento

### `DataAnalyzer`
Responsável pelas análises e visualizações:
- `remove_duplicates()`: Remove duplicatas
- `filter_by_time_period()`: Filtra por horário
- `create_daily_summary()`: Resumo por dia
- `create_coverage_map()`: Mapa de cobertura
- `create_yearly_plots()`: Gráficos anuais
- `create_monthly_plots()`: Gráficos mensais
- `create_filtered_files()`: Arquivos filtrados

## Testes

O projeto inclui testes unitários abrangentes que verificam:
- Identificação correta dos formatos de arquivo
- Processamento adequado dos dados
- Remoção eficiente de duplicatas
- Filtros por período de tempo
- Geração de relatórios
- Criação de arquivos de saída

## Logs

O sistema gera logs detalhados durante o processamento, salvos em:
- `processing.log` - Log completo da execução
- Console - Output em tempo real

## Exemplo de Uso

```python
from src.data_processor import RindatDataProcessor
from src.data_analyzer import DataAnalyzer

# Carregar dados
processor = RindatDataProcessor('data')
df = processor.load_all_files()

# Analisar dados
analyzer = DataAnalyzer(df)
df_clean = analyzer.remove_duplicates('output')
df_filtered = analyzer.filter_by_time_period(df_clean, "12:00:00", "23:59:59", "output")

# Gerar análises
analyzer.create_yearly_plots(df_filtered, 'output')
analyzer.create_monthly_plots(df_filtered, 'output')
```

## Contribuição

Para contribuir com o projeto:

1. Execute os testes: `pytest tests/ -v`
2. Mantenha a cobertura de testes
3. Documente novas funcionalidades
4. Siga as convenções de código Python (PEP 8)

## Licença

Este projeto foi desenvolvido para análise acadêmica de dados meteorológicos da UFRJ.