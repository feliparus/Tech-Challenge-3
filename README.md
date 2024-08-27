# Preparação do Dataset AmazonTitles-1.3MM para Fine-Tuning

Este guia descreve como preparar o dataset AmazonTitles-1.3MM para o fine-tuning de modelos de aprendizado de máquina. O dataset inclui vários arquivos que precisam ser processados e organizados adequadamente.

## Arquivos Disponíveis

O dataset contém os seguintes arquivos:

- `filter_labels_train.txt` e `filter_labels_test.txt`: Contêm rótulos filtrados para os conjuntos de treinamento e teste, respectivamente. Esses arquivos são úteis para associar os exemplos de texto com suas categorias.

- `tst.json` e `trn.json`: Contêm os dados de texto para os conjuntos de teste e treinamento, respectivamente. Normalmente, incluem o texto dos exemplos e possivelmente alguns metadados.

- `lbl.json`: Contém a definição dos rótulos (ou classes) usadas para classificar os exemplos. Inclui informações como o nome das classes e identificadores associados a cada rótulo.

## Preparação dos Dados

Siga os passos abaixo para preparar os dados para o fine-tuning:

1. **Carregar e Explorar os Dados**
   - Carregue os arquivos JSON (`trn.json` e `tst.json`) para explorar o formato dos dados e garantir que contenham as informações necessárias. Verifique se cada entrada inclui o texto e, no caso de tarefas de classificação, o rótulo correspondente.

2. **Mapeamento de Rótulos**
   - Carregue o arquivo `lbl.json` para entender a estrutura dos rótulos. Você precisará mapear esses rótulos para os índices ou IDs usados nos arquivos `filter_labels_train.txt` e `filter_labels_test.txt`.

3. **Preparar os Dados**
   - Combine os dados de texto dos arquivos `trn.json` e `tst.json` com os rótulos apropriados dos arquivos `filter_labels_train.txt` e `filter_labels_test.txt`. Isso pode envolver a leitura dos arquivos JSON e TXT, o mapeamento dos rótulos e a organização dos dados em um formato adequado para o treinamento.

4. **Formatar para o Fine-Tuning**
   - Dependendo do modelo que você está usando para fine-tuning (por exemplo, um modelo BERT, GPT, etc.), você pode precisar formatar os dados em um formato específico, como JSON Lines, CSV ou TFRecord. Certifique-se de seguir o formato esperado pelo seu modelo.
