# Tech Challenge - Fine-tuning e RAG com DistilBERT

Este projeto tem como objetivo realizar o fine-tuning de um modelo pré-treinado `distilbert-base-uncased`, gerar embeddings e aplicar técnicas de Recuperação Aumentada por Geração (RAG) para responder perguntas com base em dados fornecidos.

## Estrutura do Projeto

- **requirements.txt**: Contém as bibliotecas necessárias para o projeto, juntamente com comentários explicativos sobre cada uma.
  
- **notebooks/**:
  - `1_preparacao_dados.ipynb`: Realiza o download, a preparação e a conversão do arquivo `trn.json` em formato parquet para otimizar o processamento de dados.
  - `2_fine_tunning.ipynb`: Executa o fine-tuning do modelo `distilbert-base-uncased` nos dados do arquivo `trn.parquet`.
  - `3_Geracao_Embeddings_Em_Batches.ipynb`: Gera embeddings em lotes usando o modelo treinado.
  - `4_RAG.ipynb`: Implementa a técnica RAG para responder perguntas com base nos dados processados.

- **arquivos/**:
  - `trn.json`: Arquivo JSON com os dados brutos (baixado [aqui](https://drive.google.com/uc?id=12zH4mL2RX8iSvH0VCNnd3QxO4DzuHWnK)).
  - `trn.parquet`: Arquivo parquet gerado no notebook de preparação de dados, otimizado para processamento.
  - **finetunning/**: Contém os arquivos e resultados gerados durante o processo de fine-tuning.
  - **embeedings/**: Contém os arquivos e resultados dos embeddings gerados em lotes.

- **include/**:
  - `utils.py`: Arquivo que contém funções de utilidade usadas nos notebooks para facilitar operações como leitura de dados, preparação de batches, entre outras.

## Objetivo

O pipeline principal do projeto é o seguinte:
1. Baixar e preparar o arquivo `trn.json` contendo os dados.
2. Realizar o fine-tuning do modelo `distilbert-base-uncased`.
3. Gerar embeddings a partir do modelo ajustado em batches.
4. Aplicar a técnica RAG para responder perguntas com base nos dados ajustados.

Os motivos para a escolha do modelo e tokenizer (`DistilBertTokenizerFast`) serão explicados detalhadamente na apresentação do projeto.

## Como Rodar o Projeto

Execute os notebooks na ordem indicada para seguir o pipeline completo:

  1. Preparacao_dados.ipynb
  2. Fine_tunning.ipynb
  3. Geracao_Embeddings_Em_Batches.ipynb
  4. RAG.ipynb
