"""
=============================================================================
NPS PREDITIVO - TECH CHALLENGE FASE 1
=============================================================================
Case: Análise e predição do Net Promoter Score (NPS) em e-commerce
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Forçar encoding UTF-8 no console Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150

# =============================================================================
# CARREGAMENTO E RENOMEAÇÃO DAS COLUNAS PARA PORTUGUÊS
# =============================================================================

df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'desafio_nps_fase_1.csv'))

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

# =============================================================================
# PONTO 1 DO PROFESSOR: ARREDONDAMENTO DAS NOTAS QUEBRADAS
# =============================================================================
# As notas NPS possuem valores decimais (ex: 6.3, 7.8). Para categorizar
# corretamente em Detrator/Neutro/Promotor, aplicamos arredondamento:
#   - >= X.5 → arredonda para cima
#   - < X.5  → arredonda para baixo
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

# =============================================================================
# CRIAÇÃO DA VARIÁVEL TARGET: CLASSIFICAÇÃO NPS
# =============================================================================
# Usando a nota arredondada para classificar:
#   - Detrator: notas 0 a 6
#   - Neutro: notas 7 e 8
#   - Promotor: notas 9 e 10

def classificar_nps(nota):
    if nota <= 6:
        return 'Detrator'
    elif nota <= 8:
        return 'Neutro'
    else:
        return 'Promotor'

df['categoria_nps'] = df['nota_nps_arredondada'].apply(classificar_nps)

print(f"\nDistribuição por categoria NPS (após arredondamento):")
cat_counts = df['categoria_nps'].value_counts().reindex(['Detrator', 'Neutro', 'Promotor'])
for cat, qtd in cat_counts.items():
    print(f"  {cat}: {qtd} ({qtd/len(df)*100:.1f}%)")

# =============================================================================
# 3. ANÁLISE EXPLORATÓRIA DOS DADOS (EDA)
# =============================================================================
print("\n" + "=" * 80)
print("3. ANÁLISE EXPLORATÓRIA DOS DADOS (EDA)")
print("=" * 80)

print(f"\nDataset: {df.shape[0]} registros, {df.shape[1]} colunas")
print(f"Valores nulos: {df.isnull().sum().sum()}")
print(f"\nEstatísticas descritivas:\n{df.describe().round(2)}")

# --- 3.1 Distribuição do NPS ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['nota_nps_arredondada'], bins=range(0, 12), edgecolor='black',
             alpha=0.7, color='#3498db', align='left')
axes[0].set_xlabel('Nota NPS (arredondada)')
axes[0].set_ylabel('Frequência')
axes[0].set_title('Distribuição das Notas NPS')
axes[0].set_xticks(range(0, 11))
media_nps = df['nota_nps_arredondada'].mean()
axes[0].axvline(media_nps, color='red', linestyle='--', label=f"Média: {media_nps:.1f}")
axes[0].legend()

cores_nps = ['#e74c3c', '#f39c12', '#2ecc71']
axes[1].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%',
            colors=cores_nps, startangle=90)
axes[1].set_title('Distribuição por Categoria NPS')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '01_distribuicao_nps.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 3.2 Correlação entre variáveis numéricas e nota NPS ---
colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
for col in ['id_cliente', 'id_pedido', 'nota_nps_original', 'nota_nps_arredondada']:
    if col in colunas_numericas:
        colunas_numericas.remove(col)

correlacoes = df[colunas_numericas].corr()['nota_nps'].drop('nota_nps').sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
correlacoes.plot(kind='barh', ax=ax,
                 color=['#e74c3c' if x < 0 else '#2ecc71' for x in correlacoes])
ax.set_title('Correlação das Variáveis com a Nota NPS')
ax.set_xlabel('Coeficiente de Correlação (Pearson)')
ax.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '02_correlacao_nps.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nCorrelação com a Nota NPS:")
print(correlacoes.round(3))

# --- 3.3 Heatmap de correlação ---
fig, ax = plt.subplots(figsize=(14, 10))
corr_full = df[colunas_numericas].corr()
mask = np.triu(np.ones_like(corr_full, dtype=bool))
sns.heatmap(corr_full, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title('Matriz de Correlação - Todas as Variáveis')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '03_heatmap_correlacao.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# PONTO 2 DO PROFESSOR: ANÁLISE DE OUTLIERS (NOTAS 0)
# =============================================================================
print("\n" + "=" * 80)
print("ANÁLISE DE OUTLIERS: CLIENTES COM NOTA 0")
print("=" * 80)

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

# Correlação APENAS para clientes nota 0 vs demais detratores (1-6)
detratores_nao_zero = df[(df['nota_nps_arredondada'] >= 1) & (df['nota_nps_arredondada'] <= 6)]

print(f"\nCorrelação com NPS - Nota 0 vs Detratores (1-6):")
corr_zero = nota_zero[cols_analise + ['nota_nps']].corr()['nota_nps'].drop('nota_nps')
corr_det = detratores_nao_zero[cols_analise + ['nota_nps']].corr()['nota_nps'].drop('nota_nps')
comp_corr = pd.DataFrame({
    'Corr Nota=0': corr_zero,
    'Corr Detratores(1-6)': corr_det,
}).round(3)
print(comp_corr)

# Gráfico: perfil dos clientes nota 0
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
plt.savefig(os.path.join(FIG_DIR, '04_analise_outliers_nota_zero.png'),
            dpi=150, bbox_inches='tight')
plt.close()

print(f"\nINSIGHT: Clientes com nota 0 apresentam perfil extremo — atrasos maiores,")
print(f"mais reclamações e mais contatos com SAC. A correlação interna desse grupo")
print(f"é diferente dos demais detratores, indicando que são casos críticos que")
print(f"merecem tratamento diferenciado pela empresa.")

# --- 3.4 NPS por Região e Faixa Etária ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

nps_regiao = df.groupby('regiao_cliente')['nota_nps'].agg(['mean', 'count']).sort_values('mean', ascending=False)
nps_regiao['mean'].plot(kind='bar', ax=axes[0], color='#3498db', edgecolor='black', alpha=0.8)
axes[0].set_title('NPS Médio por Região')
axes[0].set_ylabel('NPS Médio')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(nps_regiao['mean']):
    axes[0].text(i, v + 0.05, f'{v:.1f}', ha='center', fontsize=10)

df['faixa_etaria'] = pd.cut(df['idade_cliente'], bins=[17, 25, 35, 45, 55, 70],
                             labels=['18-25', '26-35', '36-45', '46-55', '56-70'])
nps_idade = df.groupby('faixa_etaria', observed=True)['nota_nps'].mean()
nps_idade.plot(kind='bar', ax=axes[1], color='#9b59b6', edgecolor='black', alpha=0.8)
axes[1].set_title('NPS Médio por Faixa Etária')
axes[1].set_ylabel('NPS Médio')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(nps_idade):
    axes[1].text(i, v + 0.05, f'{v:.1f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '05_nps_regiao_idade.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 3.5 Impacto do atraso e contatos com atendimento ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

nps_atraso = df.groupby('dias_atraso_entrega')['nota_nps'].mean()
axes[0].plot(nps_atraso.index, nps_atraso.values, 'o-', color='#e74c3c', markersize=6)
axes[0].set_title('NPS Médio vs Dias de Atraso na Entrega')
axes[0].set_xlabel('Dias de Atraso')
axes[0].set_ylabel('NPS Médio')
axes[0].axhline(y=6, color='gray', linestyle='--', alpha=0.5, label='Limite Detrator (≤6)')
axes[0].legend()

nps_contatos = df.groupby('contatos_atendimento')['nota_nps'].mean()
axes[1].plot(nps_contatos.index, nps_contatos.values, 's-', color='#e67e22', markersize=6)
axes[1].set_title('NPS Médio vs Contatos com Atendimento')
axes[1].set_xlabel('Nº de Contatos com SAC')
axes[1].set_ylabel('NPS Médio')
axes[1].axhline(y=6, color='gray', linestyle='--', alpha=0.5, label='Limite Detrator (≤6)')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '06_nps_atraso_contatos.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 3.6 Reclamações e Recompra ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

nps_reclamacoes = df.groupby('qtd_reclamacoes')['nota_nps'].mean()
axes[0].bar(nps_reclamacoes.index, nps_reclamacoes.values,
            color='#c0392b', edgecolor='black', alpha=0.8)
axes[0].set_title('NPS Médio vs Número de Reclamações')
axes[0].set_xlabel('Número de Reclamações')
axes[0].set_ylabel('NPS Médio')
axes[0].axhline(y=6, color='gray', linestyle='--', alpha=0.5)

nps_recompra = df.groupby('recompra_30d')['nota_nps'].mean()
axes[1].bar(['Sem Recompra', 'Com Recompra'], nps_recompra.values,
            color=['#e74c3c', '#2ecc71'], edgecolor='black', alpha=0.8)
axes[1].set_title('NPS Médio: Com vs Sem Recompra em 30 dias')
axes[1].set_ylabel('NPS Médio')
for i, v in enumerate(nps_recompra.values):
    axes[1].text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '07_nps_reclamacoes_recompra.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- 3.7 Comparação Detratores vs Promotores ---
print("\n" + "-" * 60)
print("COMPARAÇÃO: DETRATORES vs PROMOTORES")
print("-" * 60)

detratores = df[df['categoria_nps'] == 'Detrator']
promotores = df[df['categoria_nps'] == 'Promotor']

cols_comparacao = [c for c in colunas_numericas if c != 'nota_nps']
comparacao = pd.DataFrame({
    'Detratores': detratores[cols_comparacao].mean(),
    'Promotores': promotores[cols_comparacao].mean(),
}).round(2)
comparacao['Diferença'] = (comparacao['Detratores'] - comparacao['Promotores']).round(2)
print(comparacao.sort_values('Diferença', ascending=False))

# --- 3.8 Boxplots das variáveis-chave por categoria NPS ---
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
plt.savefig(os.path.join(FIG_DIR, '08_boxplots_variaveis_chave.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# PONTO 4 DO PROFESSOR: PROXY DE CHURN (RECOMPRA)
# =============================================================================
print("\n" + "=" * 80)
print("ANÁLISE DE CHURN (PROXY: RECOMPRA EM 30 DIAS)")
print("=" * 80)

# A variável recompra_30d funciona como proxy inversa de churn:
#   recompra_30d = 0 → cliente NÃO recomprou → possível churn
#   recompra_30d = 1 → cliente recomprou → retido

df['risco_churn'] = (df['recompra_30d'] == 0).astype(int)

print(f"\nDistribuição de risco de churn:")
print(f"  Risco de churn (sem recompra): {df['risco_churn'].sum()} ({df['risco_churn'].mean()*100:.1f}%)")
print(f"  Retido (com recompra): {(df['risco_churn'] == 0).sum()} ({(1-df['risco_churn'].mean())*100:.1f}%)")

# Churn por categoria NPS
churn_por_cat = df.groupby('categoria_nps')['risco_churn'].mean().reindex(ordem_cat) * 100
print(f"\nTaxa de churn por categoria NPS:")
for cat, taxa in churn_por_cat.items():
    print(f"  {cat}: {taxa:.1f}%")

# Churn por faixas de atraso
df['faixa_atraso'] = pd.cut(df['dias_atraso_entrega'], bins=[-1, 0, 2, 4, 8],
                             labels=['Sem atraso', '1-2 dias', '3-4 dias', '5+ dias'])
churn_por_atraso = df.groupby('faixa_atraso', observed=True)['risco_churn'].mean() * 100

# Churn por faixas de reclamação
df['faixa_reclamacao'] = pd.cut(df['qtd_reclamacoes'], bins=[-1, 0, 2, 4, 11],
                                 labels=['Nenhuma', '1-2', '3-4', '5+'])
churn_por_reclamacao = df.groupby('faixa_reclamacao', observed=True)['risco_churn'].mean() * 100

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Churn por categoria NPS
churn_por_cat.plot(kind='bar', ax=axes[0], color=cores_nps, edgecolor='black', alpha=0.8)
axes[0].set_title('Taxa de Churn por Categoria NPS')
axes[0].set_ylabel('Taxa de Churn (%)')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=0)
for i, v in enumerate(churn_por_cat):
    axes[0].text(i, v + 0.5, f'{v:.0f}%', ha='center', fontsize=10, fontweight='bold')

# Churn por atraso
churn_por_atraso.plot(kind='bar', ax=axes[1], color='#e74c3c', edgecolor='black', alpha=0.8)
axes[1].set_title('Taxa de Churn por Faixa de Atraso')
axes[1].set_ylabel('Taxa de Churn (%)')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=0)
for i, v in enumerate(churn_por_atraso):
    axes[1].text(i, v + 0.5, f'{v:.0f}%', ha='center', fontsize=10)

# Churn por reclamações
churn_por_reclamacao.plot(kind='bar', ax=axes[2], color='#c0392b', edgecolor='black', alpha=0.8)
axes[2].set_title('Taxa de Churn por Faixa de Reclamações')
axes[2].set_ylabel('Taxa de Churn (%)')
axes[2].set_xlabel('')
axes[2].tick_params(axis='x', rotation=0)
for i, v in enumerate(churn_por_reclamacao):
    axes[2].text(i, v + 0.5, f'{v:.0f}%', ha='center', fontsize=10)

plt.suptitle('Análise de Churn (Proxy: Ausência de Recompra em 30 dias)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '09_analise_churn.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nTaxa de churn por faixa de atraso:\n{churn_por_atraso.round(1)}")
print(f"\nTaxa de churn por faixa de reclamações:\n{churn_por_reclamacao.round(1)}")

# --- Insights da EDA ---
print("\n" + "=" * 80)
print("INSIGHTS PRINCIPAIS DA EDA")
print("=" * 80)

insights_eda = """
FATORES MAIS CRÍTICOS PARA A SATISFAÇÃO:
1. Dias de atraso na entrega: correlação negativa mais forte com o NPS.
   Acima de 5 dias de atraso, o NPS médio cai drasticamente.
2. Quantidade de reclamações: segunda maior correlação negativa.
   Clientes com mais de 6 reclamações têm NPS próximo de zero.
3. Contatos com atendimento: múltiplos contatos indicam que o problema
   não foi resolvido na primeira interação.

OUTLIERS (NOTA 0):
- Clientes com nota 0 possuem perfil extremo e diferente dos demais
  detratores. Apresentam atrasos maiores, mais reclamações e mais contatos
  com SAC. Merecem tratamento diferenciado como casos críticos.

CHURN:
- 100% dos detratores não recompraram em 30 dias (churn total)
- 100% dos promotores recompraram (retenção total)
- Atrasos e reclamações são os principais drivers de churn
"""
print(insights_eda)

# =============================================================================
# 4. MODELO PREDITIVO DE CLASSIFICAÇÃO
# =============================================================================
# PONTO 3 DO PROFESSOR: a target deve resolver o problema — precisamos
# acertar bem os Promotores e Neutros, que possuem range menor que Detratores.
# Usamos class_weight='balanced' e avaliamos F1 por classe.
# =============================================================================
print("=" * 80)
print("4. MODELO PREDITIVO DE CLASSIFICAÇÃO")
print("=" * 80)

print("\n--- Preparação dos Dados ---")

features = ['idade_cliente', 'tempo_relacionamento_meses', 'valor_pedido',
            'qtd_itens', 'valor_desconto', 'parcelas_pagamento',
            'tempo_entrega_dias', 'dias_atraso_entrega', 'valor_frete',
            'tentativas_entrega', 'contatos_atendimento',
            'tempo_resolucao_dias', 'qtd_reclamacoes']

# Encoding da região
le_regiao = LabelEncoder()
df['regiao_encoded'] = le_regiao.fit_transform(df['regiao_cliente'])
features.append('regiao_encoded')

# Encoding da target
le_target = LabelEncoder()
le_target.fit(['Detrator', 'Neutro', 'Promotor'])
df['target'] = le_target.transform(df['categoria_nps'])

X = df[features]
y = df['target']

print(f"Features utilizadas ({len(features)}):")
for f in features:
    print(f"  - {f}")

print(f"\nVariáveis EXCLUÍDAS (evitar data leakage):")
print("  - recompra_30d (comportamento pós-compra)")
print("  - csat_interno (pode conter informação da própria pesquisa)")

print(f"\nDistribuição da target:")
for classe in ['Detrator', 'Neutro', 'Promotor']:
    qtd = (df['categoria_nps'] == classe).sum()
    print(f"  {classe}: {qtd} ({qtd/len(df)*100:.1f}%)")

# --- Separação treino/teste ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTreino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

# --- Treinamento do modelo ---
print("\n" + "-" * 60)
print("MODELO: Random Forest Classifier (multiclasse)")
print("-" * 60)

modelo = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# --- Métricas ---
nomes_classes = le_target.classes_
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"\nAcurácia: {acc:.3f}")
print(f"F1-Score (macro): {f1_macro:.3f}")
print(f"F1-Score (weighted): {f1_weighted:.3f}")
print(f"\nRelatório de Classificação:")
report = classification_report(y_test, y_pred, target_names=nomes_classes, output_dict=True)
print(classification_report(y_test, y_pred, target_names=nomes_classes))

# PONTO 3: Avaliar acerto de Promotores e Neutros especificamente
print("ATENÇÃO AO ACERTO DE PROMOTORES E NEUTROS:")
print(f"  Promotor - Precision: {report['Promotor']['precision']:.2f} | "
      f"Recall: {report['Promotor']['recall']:.2f} | "
      f"F1: {report['Promotor']['f1-score']:.2f}")
print(f"  Neutro   - Precision: {report['Neutro']['precision']:.2f} | "
      f"Recall: {report['Neutro']['recall']:.2f} | "
      f"F1: {report['Neutro']['f1-score']:.2f}")
print(f"  Detrator - Precision: {report['Detrator']['precision']:.2f} | "
      f"Recall: {report['Detrator']['recall']:.2f} | "
      f"F1: {report['Detrator']['f1-score']:.2f}")

# --- Matriz de confusão ---
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=nomes_classes, yticklabels=nomes_classes)
ax.set_xlabel('Previsto')
ax.set_ylabel('Real')
ax.set_title(f'Matriz de Confusão\nAcurácia: {acc:.1%} | F1 macro: {f1_macro:.2f}')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '10_matriz_confusao.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Importância das variáveis ---
importancia = pd.Series(modelo.feature_importances_, index=features).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
importancia.plot(kind='barh', ax=ax, color='#2ecc71', edgecolor='black', alpha=0.8)
ax.set_title('Importância das Variáveis - Modelo de Classificação NPS')
ax.set_xlabel('Importância')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '11_importancia_variaveis.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nImportância das variáveis:")
print(importancia.sort_values(ascending=False).round(4))

# =============================================================================
# PONTO 5 DO PROFESSOR: PREVISÃO FUTURA - SIMULAÇÃO DE MELHORIA DO NPS
# =============================================================================
# Abordagem: usamos a relação observada nos dados (EDA) para estimar o impacto
# de ações operacionais no NPS. Simulamos cenários alterando variáveis-chave
# e reclassificando os clientes com base na nota ajustada.
# =============================================================================
print("\n" + "=" * 80)
print("5. SIMULAÇÃO DE CENÁRIOS: PREVISÃO DE MELHORIA DO NPS")
print("=" * 80)

# Calcular NPS atual (fórmula oficial: %Promotores - %Detratores)
nps_atual_pct_promotores = (df['categoria_nps'] == 'Promotor').mean() * 100
nps_atual_pct_detratores = (df['categoria_nps'] == 'Detrator').mean() * 100
nps_atual_pct_neutros = (df['categoria_nps'] == 'Neutro').mean() * 100
nps_atual = nps_atual_pct_promotores - nps_atual_pct_detratores

print(f"\nSITUAÇÃO ATUAL:")
print(f"  Promotores: {nps_atual_pct_promotores:.1f}%")
print(f"  Neutros: {nps_atual_pct_neutros:.1f}%")
print(f"  Detratores: {nps_atual_pct_detratores:.1f}%")
print(f"  NPS ATUAL: {nps_atual:.1f}")

# Simulação baseada nos dados observados:
# Calculamos o ganho médio de NPS quando uma variável melhora e aplicamos
# aos clientes afetados.

# Ganho médio por dia de atraso reduzido (observado na EDA)
nps_por_atraso = df.groupby('dias_atraso_entrega')['nota_nps_original'].mean()
ganho_por_dia_atraso = nps_por_atraso.diff().mean() * -1  # positivo = melhoria

# Ganho médio por reclamação reduzida
nps_por_reclamacao = df.groupby('qtd_reclamacoes')['nota_nps_original'].mean()
ganho_por_reclamacao = nps_por_reclamacao.diff().mean() * -1

print(f"\nGanho médio estimado por dia de atraso reduzido: +{ganho_por_dia_atraso:.2f} pontos NPS")
print(f"Ganho médio estimado por reclamação reduzida: +{ganho_por_reclamacao:.2f} pontos NPS")

def simular_cenario(df_sim, nome, ajustes):
    """Simula um cenário ajustando a nota NPS e reclassificando."""
    notas_ajustadas = df_sim['nota_nps_original'].copy()

    for variavel, limite, ganho_unitario in ajustes:
        excesso = (df_sim[variavel] - limite).clip(lower=0)
        notas_ajustadas = notas_ajustadas + excesso * ganho_unitario

    notas_ajustadas = notas_ajustadas.clip(0, 10).round(0).astype(int)
    categorias = notas_ajustadas.apply(classificar_nps)

    pct_prom = (categorias == 'Promotor').mean() * 100
    pct_neu = (categorias == 'Neutro').mean() * 100
    pct_det = (categorias == 'Detrator').mean() * 100
    nps_cenario = pct_prom - pct_det
    return pct_prom, pct_neu, pct_det, nps_cenario

cenarios = [
    ('Reduzir atrasos (máx 2 dias)',
     [('dias_atraso_entrega', 2, ganho_por_dia_atraso)]),
    ('Reduzir reclamações (máx 3)',
     [('qtd_reclamacoes', 3, ganho_por_reclamacao)]),
    ('Combinado (atrasos + reclamações)',
     [('dias_atraso_entrega', 2, ganho_por_dia_atraso),
      ('qtd_reclamacoes', 3, ganho_por_reclamacao)]),
]

print(f"\nSIMULAÇÃO DE CENÁRIOS DE MELHORIA:")
print("-" * 60)

resultados_cenarios = []
for nome, ajustes in cenarios:
    pct_prom, pct_neu, pct_det, nps_cenario = simular_cenario(df, nome, ajustes)
    melhoria = nps_cenario - nps_atual

    resultados_cenarios.append({
        'Cenário': nome,
        'Promotores (%)': round(pct_prom, 1),
        'Neutros (%)': round(pct_neu, 1),
        'Detratores (%)': round(pct_det, 1),
        'NPS': round(nps_cenario, 1),
        'Melhoria (pp)': round(melhoria, 1)
    })

    print(f"\n  {nome}:")
    print(f"    Promotores: {pct_prom:.1f}% | Neutros: {pct_neu:.1f}% | Detratores: {pct_det:.1f}%")
    print(f"    NPS projetado: {nps_cenario:.1f} (melhoria de {melhoria:+.1f} pp)")

# Gráfico de cenários
fig, ax = plt.subplots(figsize=(10, 6))
df_cenarios = pd.DataFrame(resultados_cenarios)
nomes_cenarios = ['Atual'] + df_cenarios['Cenário'].tolist()
valores_nps = [nps_atual] + df_cenarios['NPS'].tolist()
cores_cenario = ['#95a5a6', '#3498db', '#e67e22', '#2ecc71']

bars = ax.bar(range(len(nomes_cenarios)), valores_nps, color=cores_cenario,
              edgecolor='black', alpha=0.85)
ax.set_xticks(range(len(nomes_cenarios)))
ax.set_xticklabels(nomes_cenarios, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('NPS')
ax.set_title('Projeção de NPS por Cenário de Melhoria')
ax.axhline(y=0, color='black', linewidth=0.5)

for bar, val in zip(bars, valores_nps):
    ypos = bar.get_height() + 1 if val >= 0 else bar.get_height() - 3
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, '12_simulacao_cenarios_nps.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# CONCLUSÃO E PLANO DE AÇÃO
# =============================================================================
print("\n" + "=" * 80)
print("CONCLUSÃO, PLANO DE AÇÃO E RECOMENDAÇÕES")
print("=" * 80)

conclusao = """
ESTRATÉGIA ADOTADA:
Modelo de classificação multiclasse (Detrator / Neutro / Promotor) utilizando
Random Forest. O objetivo não é acertar a nota exata do NPS, mas sim prever
a CATEGORIA do cliente antes do disparo da pesquisa.

TRATAMENTO DAS NOTAS:
As notas decimais foram arredondadas para inteiro antes da classificação,
evitando ambiguidade (ex: 6.5 → 7 = Neutro, 6.4 → 6 = Detrator).

ANÁLISE DE OUTLIERS:
Clientes com nota 0 possuem perfil extremo e diferente dos demais detratores.
Recomenda-se tratamento diferenciado para esse grupo (ação imediata de
recuperação).

PLANO DE AÇÃO PARA REDUÇÃO DE CHURN:
1. AÇÃO IMEDIATA: Contato proativo com clientes preditos como Detratores
   dentro de 24h após a entrega (cupom de desconto, pedido de desculpas).
2. LOGÍSTICA: Reduzir atrasos de entrega para no máximo 3 dias. Conforme
   simulação, isso já melhora significativamente o NPS.
3. ATENDIMENTO: Resolver problemas no primeiro contato (FCR). Cada contato
   adicional com SAC reduz o NPS.
4. MONITORAMENTO: Clientes com mais de 3 reclamações devem ser sinalizados
   como "alto risco de churn" para ação preventiva.
5. RETROALIMENTAÇÃO: Compartilhar insights com TODOS os setores — a
   centralidade no cliente é responsabilidade de todos.

COMO O MODELO NÃO DEVE SER USADO:
- NÃO usar para selecionar quem recebe a pesquisa de NPS. Isso controlaria
  o resultado sem efetuar melhorias reais nos processos internos.

LIMITAÇÕES:
- O modelo captura correlações, não necessariamente causalidade
- A base tem 2.500 registros — mais dados melhorariam a generalização
- Variáveis externas (concorrência, sazonalidade) não estão no modelo
- O NPS é autorreportado e sujeito a vieses de resposta e seleção
- As simulações de cenário assumem que o modelo mantém a mesma performance
  com dados modificados (limitação de qualquer análise contrafactual)
"""
print(conclusao)

print("=" * 80)
print("Análise concluída! Gráficos salvos em reports/figures/")
print("=" * 80)
