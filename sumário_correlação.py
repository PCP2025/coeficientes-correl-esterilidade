import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Título
st.title("📊 Análise de Aprovação de Bolsas de Plasma")
st.markdown("""
Este dashboard mostra a **probabilidade estimada de aprovação** de bolsas de plasma em função do tempo de armazenamento até a análise de esterilidade, com base em:
- 🔵 Regressão Logística
- 🟠 Média móvel
""")

# Dados
df = pd.read_csv("Binarios.csv", sep=';')
df.columns = df.columns.str.strip()
df = df[df['Resultado'].isin(['0', '1'])]
df['Resultado'] = df['Resultado'].astype(int)
df['Dias'] = pd.to_numeric(df['Dias'], errors='coerce')
df = df.dropna()

# Modelagem com statsmodels
X = df[['Dias']]
y = df['Resultado']
scaler = StandardScaler()

# Escala e adiciona constante
X_scaled = scaler.fit_transform(X)
X_scaled_const = sm.add_constant(X_scaled)

# Corrige os índices
X_scaled_const = pd.DataFrame(X_scaled_const, columns=["const", "Dias"], index=y.index)

# Ajusta o modelo
model = sm.Logit(y, X_scaled_const).fit(disp=False)

# Predição
dias_range = pd.DataFrame({'Dias': np.linspace(df['Dias'].min(), df['Dias'].max(), 300)})
dias_range_scaled = scaler.transform(dias_range)
dias_range_scaled_df = pd.DataFrame(dias_range_scaled, columns=['Dias'])
dias_range_scaled_df_const = sm.add_constant(dias_range_scaled_df)

proba_pred = model.predict(dias_range_scaled_df_const)

# Sliders
meta = st.slider("Meta mínima de aprovação (%)", 50, 99, 90) / 100
bin_size = st.slider("Tamanho da janela para média móvel (dias):", 10, 60, 20)

# Cálculo do ponto de corte
dias_meta = dias_range['Dias'][proba_pred >= meta]
ponto_meta = float(dias_meta.iloc[-1]) if not dias_meta.empty else None

# Média móvel
df['bin'] = (df['Dias'] // bin_size) * bin_size
media_movel = df.groupby('bin')['Resultado'].mean()

# Classificação de risco
def faixa_risco_dinamica(p, meta):
    if p >= meta:
        return 'Baixo risco'
    elif p >= 0.5:
        return 'Médio risco'
    else:
        return 'Alto risco'

riscos = [faixa_risco_dinamica(p, meta) for p in proba_pred]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dias_range['Dias'], proba_pred, label='Regressão logística (tendência)', color='blue', linewidth=2)
ax.plot(media_movel.index, media_movel, label='Média móvel', color='orange', linestyle='--')

# Faixas coloridas
for i in range(1, len(dias_range)):
    cor = {
        'Baixo risco': '#A8E6A1',
        'Médio risco': '#FFF3B0',
        'Alto risco': '#FFB3B3'
    }[riscos[i]]
    ax.axvspan(dias_range['Dias'].iloc[i-1], dias_range['Dias'].iloc[i], facecolor=cor, alpha=0.2)

# Linha de corte
if ponto_meta:
    ax.axvline(ponto_meta, color='red', linestyle='--',
               label=f'Corte para {meta*100:.0f}%: {int(ponto_meta)} dias')

# Finalização do gráfico
ax.set_title('Resultados da Análise de Esterilidade vs Dias de Armazenamento')
ax.set_xlabel('Dias até a Análise')
ax.set_ylabel('Probabilidade estimada de aprovação')
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Texto com ponto de corte
if ponto_meta:
    st.markdown(f"🔴 O **ponto de corte** para manter a taxa de aprovação acima de {meta*100:.0f}% é cerca de **{int(ponto_meta)} dias**.")
else:
    st.markdown("❌ Nenhum ponto com essa taxa mínima de aprovação foi encontrado.")

# Legenda de risco
st.markdown(f"""
### 🔍 Critérios de risco (baseados na meta de {int(meta * 100)}%):
- 🟢 **Baixo risco**: ≥ {int(meta * 100)}% de aprovação  
- 🟡 **Médio risco**: entre 50% e {int(meta * 100)}%  
- 🔴 **Alto risco**: < 50%
""")

# Coeficientes
coef = model.params['Dias']
intercept = model.params['const']
odds_ratio = np.exp(coef)
z_value = model.tvalues['Dias']
valor_p = model.pvalues['Dias']

st.markdown(f"""
### 📋 Resumo dos Resultados:
- **Coeficiente (inclinação)**: {coef:.4f}  
- **Intercepto**: {intercept:.4f}  
- **Odds Ratio**: {odds_ratio:.2f}
- **Estatística z**: {z_value:.4f}  
- **Valor-p exato**: {valor_p:.50f}  
""")

# Output completo do modelo
st.markdown("### 📈 Coeficientes do Modelo:")
st.text(model.summary())
