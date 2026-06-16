import pandas as pd
import numpy as np
import os 

np.random.seed(42)
n = 1000

id_empleado = [f'E{i:04d}' for i in range(1, n + 1)]
edad = np.clip(np.random.normal(40, 10, n).astype(int), 22, 60)

antiguedad_max = np.maximum(edad - 18, 0)
antiguedad = np.array([min(np.random.poisson(4), m) if m > 0 else 0 for m in antiguedad_max])
antiguedad = np.where(np.random.random(n) < 0.10, np.clip(antiguedad + 10, 0, antiguedad_max), antiguedad)

departamento = np.random.choice(['Sales', 'R&D', 'HR', 'Operations', 'IT'], n, p=[0.28, 0.25, 0.10, 0.22, 0.15])
dept_salary_mult = {'Sales': 1.15, 'R&D': 1.10, 'HR': 0.85, 'Operations': 0.95, 'IT': 1.20}

evaluacion = np.random.choice([1, 2, 3, 4, 5], n, p=[0.04, 0.14, 0.32, 0.38, 0.12])
satisfaccion_base = np.random.choice([1, 2, 3, 4, 5], n, p=[0.10, 0.22, 0.34, 0.26, 0.08])
satisfaccion = np.clip(satisfaccion_base + np.where(evaluacion >= 4, 0.5, 0), 1, 5).astype(int)

mult_array = np.array([dept_salary_mult[d] for d in departamento])
log_salario = 7.15 + 0.018 * edad + 0.035 * antiguedad + 0.10 * evaluacion + np.log(mult_array) + np.random.normal(0, 0.25, n)
nivel_salarial_continuo = np.clip(np.exp(log_salario), 1500, 8000)
nivel_salarial = (np.round(nivel_salarial_continuo / 50) * 50).astype(int)

log_mercado = 7.15 + 0.020 * edad + 0.030 * antiguedad + 0.08 * evaluacion + np.log(mult_array)
salario_mercado = np.exp(log_mercado)
compa_ratio = np.clip((nivel_salarial / salario_mercado) + np.random.normal(0, 0.08, n), 0.65, 1.35).round(2)

distancia = np.random.choice(range(1, 41), n, p=np.random.dirichlet(np.ones(40)*0.9))

horas_extra_probs = np.random.dirichlet(np.concatenate([np.ones(15)*0.6, np.ones(20)*0.3, np.ones(26)*0.15]))
horas_extra = np.clip(np.random.choice(range(0, 61), n, p=horas_extra_probs) + np.where(evaluacion >= 4, 3, 0), 0, 60).astype(int)

meses_max = antiguedad * 12
meses_desde_ascenso = np.array([np.random.randint(0, min(m + 1, 49)) if m > 0 else 0 for m in meses_max])
estancados = np.random.random(n) < 0.20
meses_desde_ascenso = np.where(estancados, np.clip(meses_desde_ascenso + np.random.poisson(15, n), 0, np.minimum(meses_max, 48)), meses_desde_ascenso)

dias_vacaciones_base = np.random.choice(range(0, 31), n, p=np.random.dirichlet(np.concatenate([np.ones(6)*0.15, np.ones(10)*0.50, np.ones(10)*0.30, np.ones(5)*0.05])))
dias_vacaciones = np.clip(dias_vacaciones_base - horas_extra * 0.08 - np.where(satisfaccion <= 2, 4, 0) + antiguedad * 0.5 + np.random.normal(0, 2, n), 0, 30).astype(int)

# ============================================================
# 7. TARGET: FUGA (INTERCEPTO 0.3)
# ============================================================

INTERCEPTO = 0.45  # Subido para ~20% fuga

logit = np.full(n, INTERCEPTO)

logit -= 0.008 * edad
logit += 0.30 * (edad < 28)
logit -= 0.02 * antiguedad
logit += 0.25 * (antiguedad < 1)
logit += 0.15 * (antiguedad > 12)

logit -= 0.000072 * nivel_salarial
logit += 0.002 * distancia

logit -= 1.12 * compa_ratio
logit += 0.68 * (compa_ratio < 0.85)
logit += 0.33 * (compa_ratio < 0.95)
logit -= 0.13 * (compa_ratio > 1.10)

logit += 0.006 * horas_extra
logit += 0.018 * meses_desde_ascenso

logit += 0.80 * (evaluacion == 1).astype(float)
logit += 0.45 * (evaluacion == 2).astype(float)
logit += 0.08 * (evaluacion == 3).astype(float)
logit -= 0.12 * (evaluacion == 4).astype(float)
logit -= 0.25 * (evaluacion == 5).astype(float)

logit -= 0.90 * satisfaccion

logit += 0.45 * ((evaluacion == 5) & (compa_ratio < 0.95)).astype(float)
logit += 0.22 * ((evaluacion == 5) & (meses_desde_ascenso > 24)).astype(float)
logit += 0.30 * ((evaluacion == 5) & (satisfaccion <= 2)).astype(float)

logit += 0.0022 * (30 - dias_vacaciones)
logit += 0.18 * (dias_vacaciones < 5)

dept_fuga = {'Sales': 0.12, 'R&D': 0.05, 'HR': 0.08, 'Operations': 0.06, 'IT': 0.10}
logit += np.array([dept_fuga[dept] for dept in departamento])

logit += 1.30 * ((horas_extra > 20) & (satisfaccion <= 2)).astype(float)
logit += 0.35 * ((distancia > 25) & (nivel_salarial < 3200)).astype(float)
logit += 0.35 * ((meses_desde_ascenso > 24) & (edad < 32)).astype(float)
logit += 0.45 * ((antiguedad < 1) & (evaluacion <= 2)).astype(float)
logit += 0.28 * ((horas_extra > 30) & (distancia > 20)).astype(float)
logit += 1.10 * ((dias_vacaciones < 5) & (horas_extra > 25)).astype(float)
logit += 0.28 * ((compa_ratio < 0.90) & (meses_desde_ascenso > 18)).astype(float)

prob_fuga = 1 / (1 + np.exp(-logit))
fuga = np.random.binomial(1, prob_fuga)

# ============================================================
# 8-9: GUARDAR Y VALIDAR
# ============================================================

df = pd.DataFrame({
    'ID_Empleado': id_empleado, 'Edad': edad, 'Antiguedad_Anios': antiguedad,
    'Departamento': departamento, 'Nivel_Salarial_USD': nivel_salarial,
    'Compa_Ratio': compa_ratio, 'Distancia_Oficina_KM': distancia,
    'Horas_Extra_Mes': horas_extra, 'Meses_Desde_Ascenso': meses_desde_ascenso,
    'Dias_Vacaciones_Tomados_12M': dias_vacaciones,
    'Evaluacion_Desempeno': evaluacion, 'Indice_Satisfaccion': satisfaccion, 'Fuga': fuga
})

assert (df['Antiguedad_Anios'] > df['Edad'] - 18).sum() == 0
assert (df['Meses_Desde_Ascenso'] > df['Antiguedad_Anios'] * 12).sum() == 0
assert ((df['Compa_Ratio'] < 0.65) | (df['Compa_Ratio'] > 1.35)).sum() == 0
assert (df['Dias_Vacaciones_Tomados_12M'] < 0).sum() == 0

os.makedirs('data', exist_ok=True)
df.to_csv('data/Data_Set.csv', index=False)

print("=" * 60)
print("DATASET FINAL v2 - VALIDACION")
print("=" * 60)
print(f"Intercepto: {INTERCEPTO}")
print(f"Tasa de fuga: {df['Fuga'].mean():.1%}")

tasa = df['Fuga'].mean()
if 0.18 <= tasa <= 0.22:
    print("✅ TASA DE FUGA EN RANGO OBJETIVO")
elif tasa < 0.18:
    print(f"⚠️  Subir intercepto a {INTERCEPTO + 0.2:.1f}")
else:
    print(f"⚠️  Bajar intercepto a {INTERCEPTO - 0.2:.1f}")

print()
corr_fuga = df[['Edad', 'Antiguedad_Anios', 'Nivel_Salarial_USD', 'Compa_Ratio',
                'Distancia_Oficina_KM', 'Horas_Extra_Mes', 'Meses_Desde_Ascenso',
                'Dias_Vacaciones_Tomados_12M', 'Evaluacion_Desempeno', 
                'Indice_Satisfaccion']].corrwith(df['Fuga']).sort_values(key=abs, ascending=False)
print("Correlaciones:")
print(corr_fuga.round(3))
print(f"\nRanking: {list(corr_fuga.index)}")

print(f"\n--- COMBOS ---")
print(f"Burnout (HE>20 + Sat<=2):        {df[(df['Horas_Extra_Mes'] > 20) & (df['Indice_Satisfaccion'] <= 2)]['Fuga'].mean():.1%}")
print(f"Vacaciones<5 + HE>25:            {df[(df['Dias_Vacaciones_Tomados_12M'] < 5) & (df['Horas_Extra_Mes'] > 25)]['Fuga'].mean():.1%}")
print(f"Compa<0.90 + Estancado>18:       {df[(df['Compa_Ratio'] < 0.90) & (df['Meses_Desde_Ascenso'] > 18)]['Fuga'].mean():.1%}")

print(f"\n--- SEGMENTOS ---")
print(f"Eval 1 fuga:                     {df[df['Evaluacion_Desempeno'] == 1]['Fuga'].mean():.1%}")
print(f"Compa <0.85 fuga:                {df[df['Compa_Ratio'] < 0.85]['Fuga'].mean():.1%}")
print(f"Sat=1 fuga:                      {df[df['Indice_Satisfaccion'] == 1]['Fuga'].mean():.1%}")
print(f"Estrellas subpagadas (<0.95):     {df[(df['Evaluacion_Desempeno'] == 5) & (df['Compa_Ratio'] < 0.95)]['Fuga'].mean():.1%}")
print(f"Estrellas quemadas (sat<=2):      {df[(df['Evaluacion_Desempeno'] == 5) & (df['Indice_Satisfaccion'] <= 2)]['Fuga'].mean():.1%}")

print("\n" + "=" * 60)
print("Dataset guardado como: data/Data_Set.csv")
print("=" * 60)