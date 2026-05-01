# NPS Preditivo — Tech Challenge Fase 1

## Objetivo

Analisar dados operacionais de um e-commerce brasileiro para compreender quais fatores influenciam a satisfação do cliente, medida pelo Net Promoter Score (NPS), e gerar insights acionáveis para que a empresa possa atuar de forma proativa na melhoria da experiência antes mesmo do disparo da pesquisa.

## Base de Dados

A base contém **2.500 registros** de pedidos com 19 variáveis, incluindo dados do cliente (idade, região, tempo de relacionamento), do pedido (valor, itens, desconto, parcelas), logísticos (tempo de entrega, atraso, tentativas, frete) e de atendimento (contatos com SAC, tempo de resolução, reclamações). A variável de interesse é a nota NPS (0 a 10), coletada após a experiência de compra.

Não foram encontrados valores nulos na base.

## Tratamento das Notas

As notas NPS originais possuem valores decimais (ex: 6.3, 7.8). Para classificar corretamente em Detrator, Neutro ou Promotor, aplicamos arredondamento padrão (≥ 0.5 arredonda para cima, < 0.5 para baixo). Exemplo: 6.5 → 7 (Neutro), 6.4 → 6 (Detrator).

## Panorama Geral do NPS

Após o arredondamento e classificação:

| Categoria | Quantidade | Percentual |
|-----------|-----------|------------|
| Detrator (0-6) | 2.001 | 80,0% |
| Neutro (7-8) | 351 | 14,0% |
| Promotor (9-10) | 148 | 5,9% |

- Nota NPS média: **4,38**
- Nota NPS mediana: **4,40**
- NPS oficial (%Promotores − %Detratores): **−74,1**

O cenário é crítico: 8 em cada 10 clientes são detratores. A empresa tem um NPS fortemente negativo, indicando que a maioria dos clientes não recomendaria a marca.

## Fatores que Mais Impactam a Satisfação

A análise de correlação revelou três fatores operacionais com impacto negativo forte e direto no NPS:

| Variável | Correlação com NPS | Interpretação |
|----------|-------------------|---------------|
| Dias de atraso na entrega | −0,60 | Quanto maior o atraso, menor o NPS |
| Quantidade de reclamações | −0,50 | Mais reclamações = mais insatisfação |
| Contatos com atendimento | −0,35 | Múltiplos contatos indicam problema não resolvido |

As demais variáveis (idade, região, valor do pedido, quantidade de itens, parcelas, frete, tentativas de entrega) apresentaram correlação fraca ou insignificante com o NPS, indicando que o problema está concentrado na **operação pós-venda** (entrega e atendimento), não no perfil do cliente ou nas características do pedido.

## Pontos de Ruptura na Experiência

A análise identificou limiares claros onde a experiência do cliente se deteriora de forma abrupta:

**Atraso na entrega — o fator mais destrutivo:**

| Dias de atraso | NPS médio |
|---------------|-----------|
| 0 (sem atraso) | 6,86 |
| 1 dia | 5,55 |
| 2 dias | 4,58 |
| 3 dias | 3,44 |
| 4 dias | 2,44 |
| 5 dias | 1,48 |
| 6+ dias | ≤ 1,10 |

A cada dia de atraso, o NPS cai em média 0,86 pontos. A partir de 5 dias de atraso, o NPS médio cai abaixo de 1,5 — praticamente todos os clientes se tornam detratores. Esse é o principal ponto de ruptura.

**Contatos com atendimento:**

| Contatos com SAC | NPS médio |
|-----------------|-----------|
| 0 | 5,54 |
| 1 | 4,66 |
| 2 | 4,12 |
| 3 | 3,20 |
| 4+ | ≤ 2,65 |

Cada contato adicional com o SAC reduz o NPS. Clientes que precisam entrar em contato 4 ou mais vezes têm NPS abaixo de 3. Isso indica falha na resolução no primeiro contato.

**Reclamações:**

| Reclamações | NPS médio |
|------------|-----------|
| 0 | 8,52 |
| 1 | 7,77 |
| 2 | 6,05 |
| 3 | 4,91 |
| 5+ | ≤ 3,65 |

Clientes sem reclamações têm NPS médio de 8,52 (promotores). A partir de 2 reclamações, o NPS já cai para a zona de detratores. Acima de 5 reclamações, o NPS fica consistentemente abaixo de 4.

## Análise de Outliers: Clientes com Nota 0

213 clientes (8,5% da base) atribuíram nota 0. Esse grupo apresenta um perfil extremo e diferente dos demais detratores:

| Indicador | Nota 0 | Demais clientes |
|-----------|--------|-----------------|
| Atraso médio (dias) | 3,97 | 2,02 |
| Reclamações médias | 5,56 | 4,02 |
| Contatos SAC médios | 2,52 | 1,43 |
| CSAT interno médio | 0,81 | 3,14 |

Esses clientes tiveram quase o dobro de atraso, 38% mais reclamações e 76% mais contatos com o SAC. A correlação interna das variáveis com o NPS nesse grupo é diferente da observada nos demais detratores (notas 1-6), indicando que são casos críticos que merecem tratamento diferenciado — como ação imediata de recuperação (contato proativo, compensação).

## Análise por Região e Faixa Etária

**Por região**, as diferenças de NPS são pequenas:

| Região | NPS médio |
|--------|-----------|
| Sul | 4,49 |
| Nordeste | 4,42 |
| Norte | 4,38 |
| Sudeste | 4,37 |
| Centro-Oeste | 4,21 |

A variação entre regiões é de apenas 0,28 pontos, sugerindo que o problema de satisfação é generalizado e não concentrado em uma região específica.

**Por faixa etária**, o padrão é semelhante — sem diferenças significativas entre grupos. A insatisfação é transversal a todos os perfis demográficos.

## Análise de Churn (Proxy: Recompra em 30 dias)

Utilizamos a variável de recompra em 30 dias como proxy inversa de churn (sem recompra = possível churn). Os resultados são contundentes:

| Categoria NPS | Taxa de churn |
|--------------|---------------|
| Detrator | 100,0% |
| Neutro | 80,1% |
| Promotor | 0,0% |

Nenhum detrator recomprou em 30 dias. Todos os promotores recompraram. A relação entre NPS e retenção é direta e absoluta nesta base.

**Churn por faixa de atraso:**

| Faixa de atraso | Taxa de churn |
|----------------|---------------|
| Sem atraso | 66,1% |
| 1-2 dias | 90,7% |
| 3-4 dias | 99,1% |
| 5+ dias | 100,0% |

A partir de 3 dias de atraso, o churn é praticamente total. Isso reforça que a logística é o principal alavanca de retenção.

## Simulação de Cenários de Melhoria

Com base nos ganhos observados na EDA (cada dia de atraso reduzido melhora o NPS em ~0,86 pontos; cada reclamação reduzida melhora em ~0,70 pontos), simulamos três cenários:

| Cenário | NPS projetado | Melhoria |
|---------|--------------|----------|
| Atual | −74,1 | — |
| Reduzir atrasos para máx 2 dias | −70,4 | +3,7 pp |
| Reduzir reclamações para máx 3 | −60,2 | +13,9 pp |
| Combinado (ambas as ações) | −53,8 | +20,3 pp |

O cenário combinado projeta uma melhoria de 20,3 pontos percentuais no NPS. Embora o indicador ainda fique negativo, representa uma redução significativa na insatisfação e um primeiro passo concreto para a recuperação.

## Recomendações para o Negócio

1. **Logística como prioridade #1**: reduzir atrasos de entrega para no máximo 2 dias. É o fator de maior impacto no NPS e no churn.
2. **Resolução no primeiro contato**: cada contato adicional com o SAC reduz o NPS. Investir em capacitação e autonomia do atendimento para resolver no primeiro contato.
3. **Monitoramento de reclamações**: clientes com mais de 3 reclamações devem ser sinalizados como "alto risco de churn" para ação preventiva imediata.
4. **Tratamento diferenciado para casos críticos**: os 8,5% de clientes com nota 0 possuem perfil extremo e devem receber ação de recuperação proativa (contato, compensação).
5. **Retroalimentação entre setores**: os insights de NPS devem ser compartilhados com todos os setores (logística, atendimento, produto, marketing). A centralidade no cliente é responsabilidade de todos.

## Limitações

- A análise captura correlações, não necessariamente causalidade
- A base tem 2.500 registros — mais dados melhorariam a robustez das conclusões
- Variáveis externas (concorrência, sazonalidade, eventos) não estão disponíveis
- O NPS é autorreportado e sujeito a vieses de resposta e seleção
- As simulações de cenário assumem que as relações observadas se mantêm após as intervenções

## Estrutura do Projeto

```
├── data/                  # Base de dados
├── notebooks/
│   └── analise_nps.py     # Script principal (EDA + modelo opcional)
├── reports/
│   ├── figures/           # 12 gráficos gerados
│   └── 01_entendimento_negocio_e_target.txt
└── README.md
```

## Como Reproduzir

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python notebooks/analise_nps.py
```

Os gráficos são salvos automaticamente em `reports/figures/`.
