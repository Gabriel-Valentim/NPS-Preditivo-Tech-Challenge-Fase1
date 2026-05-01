# NPS Preditivo — Tech Challenge Fase 1

## Objetivo

Identificar os fatores operacionais que influenciam a satisfação do cliente em um e-commerce brasileiro, medida pelo Net Promoter Score (NPS), e gerar insights acionáveis para que a empresa possa atuar de forma proativa na melhoria da experiência antes mesmo do disparo da pesquisa.

Pergunta central: quais fatores operacionais realmente influenciam a satisfação do cliente e como a empresa pode agir de forma proativa para melhorar a experiência antes mesmo da aplicação da pesquisa de NPS?

Perguntas complementares:
- Quais fatores parecem mais críticos para a satisfação?
- O que mais gera detratores?
- Existe algum "ponto de ruptura" na experiência do cliente?
- Que tipo de cliente tende a ter NPS mais alto ou mais baixo?

## Base de Dados

A base contém 2.500 registros de pedidos com 19 variáveis cobrindo:
- Dados do cliente: idade, região, tempo de relacionamento
- Dados do pedido: valor, quantidade de itens, desconto, parcelas
- Dados logísticos: tempo de entrega, atraso, tentativas, frete
- Dados de atendimento: contatos com SAC, tempo de resolução, reclamações
- Indicadores: nota NPS (0 a 10), recompra em 30 dias, CSAT interno

Nenhum valor nulo foi encontrado na base.

As colunas foram renomeadas para português para facilitar a leitura e interpretação.

## Metodologia

1. Tratamento das notas NPS (arredondamento para classificação)
2. Análise Exploratória dos Dados (EDA) com foco em negócio
3. Análise de outliers (clientes nota 0)
4. Análise de churn (proxy: recompra em 30 dias)
5. Modelo preditivo de classificação (Árvore de Decisão)

## Tratamento das Notas

As notas NPS originais possuem valores decimais (ex: 6.3, 7.8). Para classificar corretamente em Detrator/Neutro/Promotor, aplicamos arredondamento padrão:
- >= X.5 → arredonda para cima
- < X.5 → arredonda para baixo

Exemplo: 6.5 → 7 (Neutro), 6.4 → 6 (Detrator). Isso evita ambiguidade na classificação.

Variável target: classificação do NPS (não a nota numérica)
- Detrator: notas 0 a 6
- Neutro: notas 7 e 8
- Promotor: notas 9 e 10

## Principais Resultados da EDA

### Distribuição do NPS

| Categoria | Quantidade | Percentual |
|-----------|-----------|------------|
| Detrator (0-6) | 2.001 | 80,0% |
| Neutro (7-8) | 351 | 14,0% |
| Promotor (9-10) | 148 | 5,9% |

Nota NPS média: 4,38. O cenário é crítico — 8 em cada 10 clientes são detratores.

### Correlações com o NPS

Correlações negativas (geram detratores):
1. Dias de atraso na entrega (−0,60)
2. Quantidade de reclamações (−0,50)
3. Contatos com atendimento (−0,35)

Correlações positivas (associadas a promotores):
1. Recompra nos próximos 30 dias (+0,57)
2. CSAT interno (+0,56)

As demais variáveis (idade, região, valor do pedido, parcelas, frete) apresentaram correlação fraca, indicando que o problema está na operação pós-venda, não no perfil do cliente.

### Pontos de Ruptura

A variável de atraso na entrega divide muito bem os públicos. Clientes detratores com notas menores que 4 possuem pelo menos 2 dias de atraso, enquanto as maiores notas estão concentradas nas entregas em dia ou até 1 dia de atraso.

Clientes com 2 ou mais reclamações são detratores. É de suma importância que as áreas de ouvidoria se atentem para solucionar o problema do cliente antes da reclamação e no máximo na primeira reclamação.

Uma boa experiência fideliza o cliente — clientes que recompram nos próximos 30 dias são significativamente mais satisfeitos com sua experiência.

### Análise de Outliers: Clientes com Nota 0

213 clientes (8,5% da base) atribuíram nota 0. Esse grupo apresenta perfil extremo e diferente dos demais detratores — atrasos maiores, mais reclamações e mais contatos com SAC. A correlação interna desse grupo é diferente dos demais detratores, indicando que são casos críticos que merecem tratamento diferenciado pela empresa.

### Análise de Churn (Proxy: Recompra em 30 dias)

| Categoria NPS | Taxa de Churn |
|--------------|---------------|
| Detrator | 100,0% |
| Neutro | 80,1% |
| Promotor | 0,0% |

Nenhum detrator recomprou em 30 dias. Todos os promotores recompraram. A relação entre NPS e retenção é direta e absoluta nesta base.

A partir de 3 dias de atraso, o churn é praticamente total (99,1%). Com 5+ dias de atraso, chega a 100%.

### Análises Complementares (Região e Faixa Etária)

A variação de NPS entre regiões é pequena (de 4,21 a 4,49), sugerindo que o problema de satisfação é generalizado. O mesmo padrão se repete por faixa etária — a insatisfação é transversal a todos os perfis demográficos.

## Modelo Preditivo

Foi aplicado um modelo de Árvore de Decisão para classificar clientes em Detrator, Neutro ou Promotor com base nos dados operacionais.

Principais resultados:
- O modelo identificou muito bem os detratores
- Para promotores, tem bom acerto mas confunde um pouco com neutros
- Nenhum promotor foi predito como detrator
- Para neutros, a maioria dos casos foi alocada como detrator

O modelo foi desenvolvido para que os times de CX usufruam dos dados preditos para promover ações de melhoria da experiência em clientes preditos como não-promotores, antes do disparo da pesquisa.

O modelo NÃO deve ser utilizado para selecionar o público elegível à pesquisa — isso controlaria o resultado sem efetuar melhorias nos processos internos.

## Plano de Ação para o Negócio

1. Contato proativo com clientes preditos como Detratores dentro de 24h após a entrega (cupom de desconto, pedido de desculpas)
2. Reduzir atrasos de entrega para no máximo 3 dias — principal fator de insatisfação e churn
3. Resolver problemas no primeiro contato com SAC — cada contato adicional reduz o NPS
4. Monitorar clientes com mais de 3 reclamações como "alto risco de churn" para ação preventiva
5. Retroalimentar TODOS os setores com os insights de NPS — a centralidade no cliente é responsabilidade de todos

## Estrutura do Projeto

```
├── data/
│   └── desafio_nps_fase_1.csv
├── notebooks/
│   ├── Trabalho_fiap.ipynb        # Notebook principal (entrega)
│   └── analise_nps.py             # Script auxiliar de análise
├── reports/
│   ├── figures/                   # Gráficos gerados
│   └── 01_entendimento_negocio_e_target.txt
├── relatorio.md                   # Relatório detalhado da EDA
└── README.md
```

## Como Reproduzir

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Abrir e executar o notebook `notebooks/Trabalho_fiap.ipynb` (Google Colab ou Jupyter).

Os gráficos são gerados inline no notebook.
