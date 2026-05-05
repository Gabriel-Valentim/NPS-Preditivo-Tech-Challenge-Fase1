# =============================================================================
# NPS PREDITIVO - TECH CHALLENGE FASE 1
# =============================================================================
# Case: Análise e predição do Net Promoter Score (NPS) em e-commerce
# =============================================================================

# Este documento tem por objetivo demonstrar as análises realizadas no projeto de NPS voltado para o e-commerce.
#
# Pergunta central: quais fatores operacionais realmente influenciam a satisfação do cliente
# e como a empresa pode agir de forma proativa para melhorar a experiência antes mesmo da aplicação da pesquisa de NPS?
#
# Perguntas complementares:
# - Quais fatores parecem mais críticos para a satisfação?
# - O que mais gera detratores?
# - Existe algum "ponto de ruptura" na experiência do cliente?
# - Que tipo de cliente tende a ter NPS mais alto ou mais baixo?


# =============================================================================
# 1) Bibliotecas
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             classification_report, confusion_matrix, accuracy_score)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

import os
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# 2) Análise exploratória dos dados (EDA)
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1) Importando dados
# -----------------------------------------------------------------------------

df = pd.read_csv('desafio_nps_fase_1.csv')

colunas_pt = {
    'customer_id': 'id_cliente',
    'customer_age': 'idade_cliente',
    'customer_region': 'regiao_cliente',
    'customer_tenure_months': 'tempo_relacionamento_meses',
    'order_id': 'id_pedido',
    'order_value': 'valor_pedido',
    'items_quantity': 'qtd_itens',
    'discount_value': 'valor_desconto',
    'payment_installments': 'parcelas_pagamento',
    'delivery_time_days': 'tempo_entrega_dias',
    'delivery_delay_days': 'dias_atraso_entrega',
    'freight_value': 'valor_frete',
    'delivery_attempts': 'tentativas_entrega',
    'customer_service_contacts': 'contatos_atendimento',
    'resolution_time_days': 'tempo_resolucao_dias',
    'nps_score': 'nota_nps',
    'repeat_purchase_30d': 'recompra_30d',
    'complaints_count': 'qtd_reclamacoes',
    'csat_internal_score': 'csat_interno'
}
df.rename(columns=colunas_pt, inplace=True)


# -----------------------------------------------------------------------------
# 2.2) Tratamento de notas NPS
# -----------------------------------------------------------------------------
# As notas NPS possuem valores decimais (ex: 6.3, 7.8). Para categorizar
# corretamente em Detrator/Neutro/Promotor, aplicamos arredondamento:
# - >= X.5 → arredonda para cima
# - < X.5  → arredonda para baixo
#
# Isso evita ambiguidade na classificação de notas como 6.5 ou 8.5.

print("=" * 80)
print("TRATAMENTO: ARREDONDAMENTO DAS NOTAS NPS")
print("=" * 80)

print(f"\nExemplos de notas originais (primeiros 10):")
print(df['nota_nps'].head(10).tolist())

df['nota_nps_original'] = df['nota_nps'].copy()
df['nota_nps_arredondada'] = df['nota_nps'].round(0).astype(int)

print(f"\nNotas após arredondamento:")
print(df['nota_nps_arredondada'].head(10).tolist())

# Verificar impacto do arredondamento
mudaram = (df['nota_nps_arredondada'] != df['nota_nps']).sum()
print(f"\nRegistros com nota alterada pelo arredondamento: {mudaram} ({mudaram/len(df)*100:.1f}%)")

# Distribuição antes e depois
print(f"\nDistribuição das notas arredondadas:")
print(df['nota_nps_arredondada'].value_counts().sort_index())


# -----------------------------------------------------------------------------
# 2.3) Criação da variável Target: Classificação do NPS
# -----------------------------------------------------------------------------
# Alinhado com a definição do grupo: a variável target é a CLASSIFICAÇÃO
# do cliente (Detrator, Neutro ou Promotor), não a nota numérica.
# - Detrator: notas 0 a 6
# - Neutro: notas 7 e 8
# - Promotor: notas 9 e 10

def classificar_nps(nota):
    if nota <= 6:
        return 'Detrator'
    elif nota <= 8:
        return 'Neutro'
    else:
        return 'Promotor'

df['categoria_nps'] = df['nota_nps_arredondada'].apply(classificar_nps)


# -----------------------------------------------------------------------------
# 2.4) Análises iniciais
# -----------------------------------------------------------------------------

# Volumetria da base
print(f"\A base possui: {df.shape[0]} registros e {df.shape[1]} colunas")

# Nulos
print(f"\nValores nulos por coluna:\n{df.isnull().sum()}")

# Estatística descritiva
print(f"\nEstatísticas descritivas:\n{df.describe().round(2)}")


# -----------------------------------------------------------------------------
# 2.5) Distribuição do NPS
# -----------------------------------------------------------------------------

# Histograma de notas
fig, ax = plt.subplots(figsize=(7, 5))

ax.hist(df['nota_nps_arredondada'], bins=20, edgecolor='black', alpha=0.7, color='#3498db')
ax.set_xlabel('NPS Score')
ax.set_ylabel('Frequência')
ax.set_title('Distribuição do NPS Score')
ax.axvline(df['nota_nps_arredondada'].mean(), color='red', linestyle='--',
           label=f"Média: {df['nota_nps_arredondada'].mean():.1f}")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports/figures/01_Distribuicao__NPS_score.png'))
plt.show()
plt.close()

# Distribuição por categorias do NPS
fig, ax = plt.subplots(figsize=(6, 6))

cat_counts = df['categoria_nps'].value_counts() \
    .reindex(['Detrator', 'Neutro', 'Promotor'], fill_value=0)

cores_nps = ['#e74c3c', '#f39c12', '#2ecc71']

ax.pie(
    cat_counts,
    labels=cat_counts.index,
    autopct='%1.1f%%',
    colors=cores_nps,
    startangle=90
)

ax.set_title('Distribuição por Categoria NPS')
ax.axis('equal')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports/figures/02_distribuicao_categoria_NPS.png'))
plt.show()
plt.close()

nps_data = df['nota_nps_arredondada'].dropna()

media_nps = nps_data.mean()

print(f"\nDistribuição por categoria NPS:")
for cat, qtd in cat_counts.items():
    print(f"  {cat}: {qtd} ({qtd/len(df)*100:.1f}%)")
print(f"\nNota NPS média: {media_nps:.2f}")


# -----------------------------------------------------------------------------
# 2.6) Análises de correlação
# -----------------------------------------------------------------------------

# Correlação entre variáveis numéricas e NPS
colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
for col in ['id_cliente', 'id_pedido']:
    colunas_numericas.remove(col)

correlacoes = df[colunas_numericas].corr()['nota_nps_arredondada'].drop(['nota_nps_arredondada','nota_nps_original','nota_nps']).sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
correlacoes.plot(kind='barh', ax=ax,
                 color=['#e74c3c' if x < 0 else '#2ecc71' for x in correlacoes])
ax.set_title('Correlação das Variáveis com a Nota NPS')
ax.set_xlabel('Coeficiente de Correlação (Pearson)')
ax.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports/figures/03_correlacao_nps.png'))
plt.show()
plt.close()

print(f"\nCorrelação com a Nota NPS:")
print(correlacoes.round(3))

fig, ax = plt.subplots(figsize=(14, 10))
corr_full = df[colunas_numericas].corr()
mask = np.triu(np.ones_like(corr_full, dtype=bool))
sns.heatmap(corr_full, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title('Matriz de Correlação - Todas as Variáveis')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports/figures/04_heatmap_correlacao.png'))
plt.show()
plt.close()

# Principais variáveis correlacionadas c/ NPS:
# CORRELAÇÕES NEGATIVAS (DETRATORES):
# 1. Dias de atraso na entrega
# 2. Quantidade de reclamações
# 3. Quantidade de atendimentos do cliente
#
# CORRELAÇÕES POSITIVAS (PROMOTORES):
# 1. Recompra nos próximos 30 dias
# 2. CSAT

# Impacto do atraso da entrega
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].grid(True)
axes[1].grid(True)

sns.lineplot(data=df[df['categoria_nps']=='Detrator'], x='nota_nps_arredondada', y='dias_atraso_entrega', ax=axes[0], color='red')
axes[0].set_title('1. dias atraso vs NPS (detrator)')
axes[0].set_xlabel('NPS')
axes[0].set_ylabel('atraso')

sns.lineplot(data=df[df['categoria_nps']!='Detrator'], x='nota_nps_arredondada', y='dias_atraso_entrega', ax=axes[1], color='green')
axes[1].set_title('2. dias atraso vs NPS (neutro/promotor)')
axes[1].set_xlabel('NPS')
axes[1].set_ylabel('atraso')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports/figures/05_dias_atraso_vs_nps.png'))
plt.show()
plt.close()

# Gráfico 1 x Gráfico 2:
# A variável de atraso na entrega divide muito bem os públicos.
# Clientes detratores com notas menores que 4, possuem pelo menos 2 dias de atraso.
# Enquanto as maiores notas estão concentradas nas entregas em dia ou até 1 dia de atraso.

# Impacto do atendimento
fig, ax = plt.subplots(figsize=(8, 5))

nps_contatos = (
    df.groupby('contatos_atendimento')['nota_nps_arredondada']
    .mean()
    .sort_index()
)

ax.plot(
    nps_contatos.index,
    nps_contatos.values,
    's-',
    color='#e67e22',
    markersize=6
)

ax.set_title('NPS Médio vs Contatos com Atendimento')
ax.set_xlabel('Nº de Contatos com SAC')
ax.set_ylabel('NPS Médio')

ax.axhline(
    y=6,
    color='gray',
    linestyle='--',
    alpha=0.7,
    label='Limite Detrator (≤6)'
)

ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports/figures/06_contatos_atendimento_vs_nps.png'))
plt.show()
plt.close()

# NPS médio por reclamação
fig, ax = plt.subplots(figsize=(8, 5))

nps_reclamacoes = (
    df.groupby('qtd_reclamacoes')['nota_nps_arredondada']
    .mean()
    .sort_index()
)

ax.bar(
    nps_reclamacoes.index,
    nps_reclamacoes.values,
    color='#c0392b',
    edgecolor='black',
    alpha=0.8
)

ax.set_title('NPS Médio vs Número de Reclamações')
ax.set_xlabel('Número de Reclamações')
ax.set_ylabel('NPS Médio')

# Linha de referência
ax.axhline(y=6, color='gray', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports/figures/07_numero_reclamacao_vs_nps.png'))
plt.show()
plt.close()

# Clientes com 2 ou mais reclamações são detratores.
# É de suma importância que as áreas de ouvidoria se atentem para solucionar o problema
# do cliente antes da reclamação e no máximo na primeira reclamação.

# NPS por recompra em 30 dias
fig, ax = plt.subplots(figsize=(8, 5))

nps_recompra = df.groupby('recompra_30d')['nota_nps_arredondada'].mean()

labels = ['Sem Recompra', 'Com Recompra']

ax.bar(
    labels,
    nps_recompra.values,
    color=['#e74c3c', '#2ecc71'],
    edgecolor='black',
    alpha=0.8
)

ax.set_title('NPS Médio: Com vs Sem Recompra em 30 dias')
ax.set_ylabel('NPS Médio')

# Valores nas barras
for i, v in enumerate(nps_recompra.values):
    ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports/figures/08_nps_com_sem_recompra.png'))
plt.show()
plt.close()

# Uma boa experiência fideliza o cliente. O gráfico acima demonstra de forma clara,
# como clientes que recompram nos próximos 30 dias, são mais satisfeitos com sua experiência.


# Comparação Detratores vs Promotores
print("\n" + "-" * 60)
print("COMPARAÇÃO: DETRATORES vs PROMOTORES")
print("-" * 60)

detratores = df[df['categoria_nps'] == 'Detrator']
promotores = df[df['categoria_nps'] == 'Promotor']

cols_comparacao = [c for c in colunas_numericas if c != ['nota_nps_arredondada','nota_nps','nota_nps_original']]
comparacao = pd.DataFrame({
    'Detratores': detratores[cols_comparacao].mean(),
    'Promotores': promotores[cols_comparacao].mean(),
}).round(2)
comparacao['Diferença'] = (comparacao['Detratores'] - comparacao['Promotores']).round(2)
print(comparacao.sort_values('Diferença', ascending=False))

# Análise por boxplot
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
variaveis_chave = ['dias_atraso_entrega', 'qtd_reclamacoes', 'contatos_atendimento',
                   'valor_desconto', 'tempo_entrega_dias', 'csat_interno']
titulos = ['Dias de Atraso', 'Reclamações', 'Contatos SAC',
           'Valor Desconto', 'Tempo Entrega (dias)', 'CSAT Interno']

ordem_cat = ['Detrator', 'Neutro', 'Promotor']
for ax, var, titulo in zip(axes.flat, variaveis_chave, titulos):
    sns.boxplot(data=df, x='categoria_nps', y=var, order=ordem_cat, ax=ax,
                palette=['#e74c3c', '#f39c12', '#2ecc71'])
    ax.set_title(titulo)
    ax.set_xlabel('')

plt.suptitle('Variáveis-Chave por Categoria NPS', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports/figures/09_boxplots_variaveis_chave.png'))
plt.show()
plt.close()


# -----------------------------------------------------------------------------
# 2.7) Análise de detratores com notas 0
# -----------------------------------------------------------------------------

nota_zero = df[df['nota_nps_arredondada'] == 0]
demais = df[df['nota_nps_arredondada'] > 0]

print(f"\nClientes com nota 0: {len(nota_zero)} ({len(nota_zero)/len(df)*100:.1f}%)")
print(f"Clientes com nota > 0: {len(demais)} ({len(demais)/len(df)*100:.1f}%)")

# Comparação de perfil: nota 0 vs demais
cols_analise = ['dias_atraso_entrega', 'qtd_reclamacoes', 'contatos_atendimento',
                'tempo_resolucao_dias', 'valor_desconto', 'valor_pedido',
                'tempo_entrega_dias', 'valor_frete', 'recompra_30d', 'csat_interno']

comparacao_outlier = pd.DataFrame({
    'Nota 0': nota_zero[cols_analise].mean(),
    'Nota > 0': demais[cols_analise].mean(),
}).round(2)
comparacao_outlier['Diferença'] = (comparacao_outlier['Nota 0'] - comparacao_outlier['Nota > 0']).round(2)
print(f"\nPerfil comparativo - Nota 0 vs Demais:")
print(comparacao_outlier.sort_values('Diferença', ascending=False))

# Correlação clientes nota 0 vs demais detratores (1-6)
detratores_nao_zero = df[(df['nota_nps_arredondada'] >= 1) & (df['nota_nps_arredondada'] <= 6)]

print(f"\nCorrelação com NPS - Nota 0 vs Detratores (1-6):")
corr_zero = nota_zero[cols_analise + ['nota_nps_arredondada']].corr()['nota_nps_arredondada'].drop('nota_nps_arredondada')
corr_det = detratores_nao_zero[cols_analise + ['nota_nps_arredondada']].corr()['nota_nps_arredondada'].drop('nota_nps_arredondada')
comp_corr = pd.DataFrame({
    'Corr Nota=0': corr_zero,
    'Corr Detratores(1-6)': corr_det,
}).round(3)
print(comp_corr)

# Perfil dos clientes nota 0
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

vars_destaque = ['dias_atraso_entrega', 'qtd_reclamacoes', 'contatos_atendimento']
titulos_destaque = ['Dias de Atraso', 'Qtd Reclamações', 'Contatos SAC']
for ax, var, titulo in zip(axes, vars_destaque, titulos_destaque):
    dados_plot = pd.DataFrame({
        'Nota 0': nota_zero[var].describe()[['mean', '50%', 'max']],
        'Demais': demais[var].describe()[['mean', '50%', 'max']],
    })
    dados_plot.index = ['Média', 'Mediana', 'Máximo']
    dados_plot.plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db'],
                    edgecolor='black', alpha=0.8)
    ax.set_title(titulo)
    ax.tick_params(axis='x', rotation=0)
    ax.legend(fontsize=8)

plt.suptitle('Perfil dos Clientes com Nota 0 vs Demais', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports/figures/10_perfil_clientes_nota_zero_vs_demais.png'))
plt.show()
plt.close()

# Clientes com nota 0 apresentam perfil extremo — atrasos maiores, mais reclamações
# e mais contatos com SAC. A correlação interna desse grupo é diferente dos demais
# detratores, indicando que são casos críticos que merecem tratamento diferenciado pela empresa.


# -----------------------------------------------------------------------------
# 2.8) Análise de Churn
# -----------------------------------------------------------------------------
# A variável recompra_30d funciona como proxy inversa de churn:
# - recompra_30d = 0 → cliente NÃO recomprou → possível churn
# - recompra_30d = 1 → cliente recomprou → retido

print("\n" + "=" * 80)
print("ANÁLISE DE CHURN (PROXY: RECOMPRA EM 30 DIAS)")
print("=" * 80)

df['risco_churn'] = (df['recompra_30d'] == 0).astype(int)

print(f"\nDistribuição de risco de churn:")
print(f"- Risco de churn (sem recompra): {df['risco_churn'].sum()} ({df['risco_churn'].mean()*100:.1f}%)")
print(f"- Retido (com recompra): {(df['risco_churn'] == 0).sum()} ({(1-df['risco_churn'].mean())*100:.1f}%)")

# Definições necessárias
ordem_cat = ['Detrator', 'Neutro', 'Promotor']
cores_nps = ['#e74c3c', '#f39c12', '#2ecc71']

# Proxy churn
df['risco_churn'] = (df['recompra_30d'] == 0).astype(int)

# Churn por categoria
churn_por_cat = (
    df.groupby('categoria_nps')['risco_churn']
    .mean()
    .reindex(ordem_cat)
    * 100
)

# Faixas
df['faixa_atraso'] = pd.cut(
    df['dias_atraso_entrega'],
    bins=[-1, 0, 2, 4, float('inf')],
    labels=['Sem atraso', '1-2 dias', '3-4 dias', '5+ dias']
)

df['faixa_reclamacao'] = pd.cut(
    df['qtd_reclamacoes'],
    bins=[-1, 0, 2, 4, float('inf')],
    labels=['Nenhuma', '1-2', '3-4', '5+']
)

# Agrupamentos
churn_por_atraso = df.groupby('faixa_atraso', observed=True)['risco_churn'].mean() * 100
churn_por_reclamacao = df.groupby('faixa_reclamacao', observed=True)['risco_churn'].mean() * 100

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# NPS
churn_por_cat.plot(kind='bar', ax=axes[0], color=cores_nps, edgecolor='black')
axes[0].set_title('Churn por NPS')

# Atraso
churn_por_atraso.plot(kind='bar', ax=axes[1], color='#e74c3c')
axes[1].set_title('Churn por Atraso')

# Reclamação
churn_por_reclamacao.plot(kind='bar', ax=axes[2], color='#c0392b')
axes[2].set_title('Churn por Reclamações')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports/figures/11_analise_churn.png'))
plt.show()
plt.close()

tabela_atraso = churn_por_atraso.reset_index()
tabela_atraso.columns = ['Faixa de Atraso', 'Churn (%)']

tabela_reclamacao = churn_por_reclamacao.reset_index()
tabela_reclamacao.columns = ['Faixa de Reclamações', 'Churn (%)']

print("\nChurn por Faixa de Atraso")
print(tabela_atraso.to_string(index=False))

print("\nChurn por Faixa de Reclamações")
print(tabela_reclamacao.to_string(index=False))


# -----------------------------------------------------------------------------
# 2.9) Análises complementares
# -----------------------------------------------------------------------------

# NPS por Região
fig, ax = plt.subplots(figsize=(8, 5))

nps_regiao = (
    df.groupby('regiao_cliente')['nota_nps_arredondada']
    .agg(['mean', 'count'])
    .sort_values('mean', ascending=False)
)

nps_regiao['mean'].plot(
    kind='bar',
    ax=ax,
    color='#3498db',
    edgecolor='black',
    alpha=0.8
)

ax.set_title('NPS Médio por Região')
ax.set_ylabel('NPS Médio')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=45)

for i, v in enumerate(nps_regiao['mean']):
    ax.text(i, v + 0.05, f'{v:.1f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports/figures/12_nps_por_regiao.png'))
plt.show()
plt.close()

# NPS por Faixa Etária
fig, ax = plt.subplots(figsize=(8, 5))

# Criar faixas etárias
df['faixa_etaria'] = pd.cut(
    df['idade_cliente'],
    bins=[17, 25, 35, 45, 55, 70],
    labels=['18-25', '26-35', '36-45', '46-55', '56-70'],
    include_lowest=True
)

nps_idade = (
    df.groupby('faixa_etaria', observed=True)['nota_nps_arredondada']
    .mean()
    .sort_index()
)

nps_idade.plot(
    kind='bar',
    ax=ax,
    color='#9b59b6',
    edgecolor='black',
    alpha=0.8
)

ax.set_title('NPS Médio por Faixa Etária')
ax.set_ylabel('NPS Médio')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=45)

# Valores nas barras
for i, v in enumerate(nps_idade):
    ax.text(i, v + 0.05, f'{v:.1f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports/figures/13_nps_por_faixa_etaria.png'))
plt.show()
plt.close()

print(f"\nNPS por Região:\n{nps_regiao.round(2)}")
print(f"\nNPS por Faixa Etária:\n{nps_idade.round(2)}")


# -----------------------------------------------------------------------------
# 2.10) Conclusões EDA
# -----------------------------------------------------------------------------
# FATORES MAIS CRÍTICOS PARA A SATISFAÇÃO:
# 1. Dias de atraso na entrega: correlação negativa mais forte com o NPS.
#    Acima de 5 dias de atraso, o NPS médio cai drasticamente.
# 2. Quantidade de reclamações: segunda maior correlação negativa.
#    Clientes com mais de 6 reclamações têm NPS próximo de zero.
# 3. Contatos com atendimento: múltiplos contatos indicam que o problema
#    não foi resolvido na primeira interação.
#
# Detratores com nota 0:
# - Clientes com nota 0 possuem perfil extremo e diferente dos demais
#   detratores. Apresentam atrasos maiores, mais reclamações e mais contatos
#   com SAC. Merecem tratamento diferenciado como casos críticos.
#
# CHURN:
# - 100% dos detratores não recompraram em 30 dias (churn total)
# - 100% dos promotores recompraram (retenção total)
# - Atrasos e reclamações são os principais drivers de churn
#
# PLANO DE AÇÃO PARA BUSINESS TEAM:
# ESTRATÉGIA ADOTADA:
# Modelo de classificação multiclasse (Detrator / Neutro / Promotor) utilizando
# árvore de decisão. O objetivo não é acertar a nota exata do NPS, mas sim prever
# a CATEGORIA do cliente antes do disparo da pesquisa.
#
# TRATAMENTO DAS NOTAS:
# As notas decimais foram arredondadas para inteiro antes da classificação,
# evitando ambiguidade (ex: 6.5 → 7 = Neutro, 6.4 → 6 = Detrator).
#
# ANÁLISE DE DETRATORES COM NOTA 0:
# Clientes com nota 0 possuem perfil extremo e diferente dos demais detratores.
# Recomenda-se tratamento diferenciado para esse grupo (ação imediata de recuperação).
#
# PLANO DE AÇÃO PARA REDUÇÃO DE CHURN:
# 1. AÇÃO IMEDIATA: Contato proativo com clientes preditos como Detratores
#    dentro de 24h após a entrega (cupom de desconto, pedido de desculpas).
# 2. LOGÍSTICA: Reduzir atrasos de entrega para no máximo 3 dias.
# 3. ATENDIMENTO: Resolver problemas no primeiro contato.
# 4. MONITORAMENTO: Clientes com mais de 3 reclamações devem ser sinalizados
#    como "alto risco de churn" para ação preventiva.
# 5. RETROALIMENTAÇÃO: Compartilhar insights com TODOS os setores.


# =============================================================================
# 3) Modelos de ML
# =============================================================================

# -----------------------------------------------------------------------------
# 3.1) Criando ABT do modelo
# -----------------------------------------------------------------------------
# Nossa variável target é classe_nps (promotor, neutro ou detrator).
# Portanto, inicialmente, vamos seguir com um modelo de classificação.
#
# Ajustamos o texto de detrator, neutro e promotor e também colunamos as regiões
# para utilizarmos o algoritmo de árvore de decisão.

nps_abt = df.copy()

nps_abt = pd.get_dummies(nps_abt, columns=['regiao_cliente'], drop_first=False)
nps_abt = nps_abt.drop(['faixa_etaria', 'faixa_atraso', 'faixa_reclamacao'], axis=1)

print(nps_abt.head())


# -----------------------------------------------------------------------------
# 3.2) Feature e target
# -----------------------------------------------------------------------------
# Separando as features e nossa target.
# Importante removermos da feature a nota de nps, que resulta na classe do NPS.

X = nps_abt.drop(['categoria_nps', 'nota_nps', 'nota_nps_original', 'nota_nps_arredondada'], axis=1)  # features
y = nps_abt['categoria_nps']  # target


# -----------------------------------------------------------------------------
# 3.3) Treino e teste
# -----------------------------------------------------------------------------
# Adotamos 80% para treinamento e 20% para teste.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------------------------------------------------------
# 3.4) Avaliando treino e teste
# -----------------------------------------------------------------------------

print("\nProporção de classes no teste:")
print(y_test.value_counts(normalize=True))

print("\nProporção de classes no treino:")
print(y_train.value_counts(normalize=True))


# -----------------------------------------------------------------------------
# 3.5) Label encoder
# -----------------------------------------------------------------------------
# Transformando nosso Y em 0,1,2 (detrator, neutro, promotor)
# Para melhor compreensão do modelo.

le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


# -----------------------------------------------------------------------------
# 3.6) Árvore de decisão
# -----------------------------------------------------------------------------
# Aplicando modelo de árvore de decisão, com os seguintes parâmetros:
# - Profundidade máxima de 5;
# - Pelo menos 5 amostras em cada folha.

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# -----------------------------------------------------------------------------
# 3.7) Avaliando resultados do modelo
# -----------------------------------------------------------------------------

from sklearn.tree import plot_tree

plt.figure(figsize=(50, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=True)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports/figures/14_arvore.png'))
plt.show()
plt.close()

model1 = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
model1.fit(X_train, y_train)

from sklearn.tree import export_graphviz

dot_data = export_graphviz(
    model1,
    out_file=None,
    feature_names=X.columns,
    class_names=['Detrator', 'Neutro', 'Promotor'],
    filled=True,
    rounded=True,
    special_characters=True
)

# Nota: para visualizar o graphviz, instale o pacote graphviz
try:
    import graphviz
    graph = graphviz.Source(dot_data)
    graph.render("graph", format="png")
    print("Árvore exportada como graph.png")
except ImportError:
    print("Pacote graphviz não instalado. Pulando renderização da árvore simplificada.")

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\n" + classification_report(y_test, y_pred))

# Matriz de Confusão
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.tight_layout()
plt.show()
plt.close()

# Matriz de confusão com labels nomeados
labels_map = {
    0: 'Detrator',
    1: 'Neutro',
    2: 'Promotor'
}

y_test_named = [labels_map[i] for i in y_test]
y_pred_named = [labels_map[i] for i in y_pred]

labels_order = ['Detrator', 'Neutro', 'Promotor']

cm = confusion_matrix(y_test_named, y_pred_named, labels=labels_order)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels_order,
    yticklabels=labels_order,
    annot_kws={'size': 16}
)

plt.title('Matriz de Confusão', fontsize=14)
plt.xlabel('Previsto')
plt.ylabel('Real')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports/figures/15_matriz_confusao.png'))
plt.show()
plt.close()


# -----------------------------------------------------------------------------
# 3.8) Conclusões - modelo preditivo
# -----------------------------------------------------------------------------
# O modelo de árvore de decisão identificou muito bem os detratores.
# Para promotores, ele tem um bom acerto, mas confunde um pouco com os neutros.
# Não identificamos nenhum promotor predito como detrator, o que é muito bom.
# Porém para identificação de neutros, ele não atingiu as expectativas.
# A maioria dos casos ele acabou alocando como detrator.
