import pandas as pd
import numpy as np
from pathlib import Path

# Configurare pentru afișare mai bună
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("=" * 80)
print("PREPROCESAREA DATELOR DESPRE VINURI - VERSIUNE CORECTATĂ")
print("=" * 80)

# ============================================================================
# 1. CITIREA DATELOR CU HEADER CORECT
# ============================================================================
print("\n1. CITIREA DATELOR")
print("-" * 80)

# Înlocuiește cu calea către fișierul tău CSV
file_path = "wine_data_processed.csv"  # Modifică această cale

# Definim explicit headerul conform structurii originale
EXPECTED_COLUMNS = [
    'country', 'description', 'designation', 'points', 'price',
    'province', 'region_1', 'region_2', 'variety', 'winery',
    'title', 'vintage', 'alcohol', 'category'
]

try:
    # Încercăm să citim fișierul
    print("Încercare de citire a fișierului...")

    # Citim primele rânduri pentru a detecta structura
    df_test = pd.read_csv(file_path, nrows=3)

    # Verificăm dacă prima coloană este 'country' sau altceva
    first_col = df_test.columns[0]

    if first_col.lower() == 'country':
        # Headerul este corect
        print("  ✓ Header detectat corect în fișier")
        df = pd.read_csv(file_path)

    elif 'country' in str(df_test.iloc[0, 0]).lower():
        # Prima linie de date este headerul
        print("  ℹ Prima linie de date este headerul, se recitește...")
        df = pd.read_csv(file_path, skiprows=1)

    else:
        # Nu există header, îl setăm manual
        print("  ℹ Nu există header valid, se setează manual...")
        # Verificăm dacă există price_quality_ratio
        num_cols = len(pd.read_csv(file_path, nrows=1).columns)

        if num_cols == 15:
            # Include și price_quality_ratio
            columns = EXPECTED_COLUMNS + ['price_quality_ratio']
        else:
            columns = EXPECTED_COLUMNS

        df = pd.read_csv(file_path, names=columns, skiprows=1)

    initial_rows = len(df)

    print(f"\n✓ Dataset încărcat cu succes!")
    print(f"  • Rânduri inițiale: {initial_rows:,}")
    print(f"  • Coloane: {df.shape[1]}")
    print(f"\nColoane detectate: {list(df.columns)}")

    # Verificăm dacă avem coloanele esențiale
    essential_cols = ['country', 'points', 'price', 'variety', 'category']
    missing_cols = [col for col in essential_cols if col not in df.columns]

    if missing_cols:
        print(f"\n✗ EROARE: Lipsesc coloane esențiale: {missing_cols}")
        print("\nPrimele 3 rânduri pentru diagnostic:")
        print(df.head(3))
        exit()

    print(f"\n✓ Toate coloanele esențiale sunt prezente!")
    print(f"\nPrimele 3 rânduri:")
    print(df.head(3).to_string())

except FileNotFoundError:
    print(f"✗ Eroare: Fișierul '{file_path}' nu a fost găsit!")
    print("Te rog să salvezi fișierul CSV în directorul curent sau să modifici calea.")
    exit()
except Exception as e:
    print(f"✗ Eroare la citirea fișierului: {e}")
    import traceback

    traceback.print_exc()
    exit()

# ============================================================================
# 2. CURĂȚAREA DATELOR
# ============================================================================
print("\n\n2. CURĂȚAREA DATELOR")
print("=" * 80)

# 2.1 Identificarea valorilor lipsă
print("\n2.1 Valori lipsă înainte de curățare:")
print("-" * 80)

missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Coloană': missing_values.index,
    'Valori lipsă': missing_values.values,
    'Procent (%)': missing_percent.values
})
missing_df = missing_df[missing_df['Valori lipsă'] > 0].sort_values('Valori lipsă', ascending=False)

if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
else:
    print("✓ Nu există valori lipsă în dataset!")

# 2.2 Tratarea valorilor lipsă
print("\n\n2.2 Tratarea valorilor lipsă:")
print("-" * 80)

rows_before = len(df)

# Coloane critice (dacă lipsesc, eliminăm rândul)
critical_columns = ['country', 'points', 'price', 'variety', 'category', 'vintage']

print(f"\nPASUL 1: Eliminarea rândurilor cu valori lipsă în coloane CRITICE")
print(f"Coloane critice: {', '.join(critical_columns)}")

# Identificăm rândurile cu valori lipsă în coloane critice
rows_with_missing_critical = df[critical_columns].isnull().any(axis=1)
missing_critical_count = rows_with_missing_critical.sum()

if missing_critical_count > 0:
    print(f"  • Rânduri cu valori lipsă în coloane critice: {missing_critical_count:,}")
    df = df[~rows_with_missing_critical].copy()
    print(f"  • Rânduri eliminate: {missing_critical_count:,}")
    print(f"  • Rânduri rămase: {len(df):,}")
else:
    print(f"  ✓ Nu există valori lipsă în coloanele critice!")

print(f"\nPASUL 2: Completarea valorilor lipsă în coloanele OPȚIONALE")

# Pentru alcohol (numeric): înlocuim cu mediana
if 'alcohol' in df.columns:
    missing_alcohol = df['alcohol'].isnull().sum()
    if missing_alcohol > 0:
        median_value = df['alcohol'].median()
        df['alcohol'].fillna(median_value, inplace=True)
        print(f"  • alcohol: înlocuit {missing_alcohol:,} valori lipsă cu mediana ({median_value:.2f})")
    else:
        print(f"  • alcohol: ✓ nu există valori lipsă")

# Pentru coloane categorice permise: înlocuim cu 'Unknown'
categorical_optional = ['designation', 'province', 'region_1', 'region_2', 'winery', 'title']
for col in categorical_optional:
    if col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col].fillna('Unknown', inplace=True)
            print(f"  • {col}: înlocuit {missing_count:,} valori lipsă cu 'Unknown'")

# Pentru description: înlocuim cu string gol
if 'description' in df.columns:
    missing_desc = df['description'].isnull().sum()
    if missing_desc > 0:
        df['description'].fillna('', inplace=True)
        print(f"  • description: înlocuit {missing_desc:,} valori lipsă cu string gol")

rows_removed = rows_before - len(df)
print(f"\n✓ Procesare completă!")
print(f"  • Total rânduri eliminate: {rows_removed:,}")
print(f"  • Rânduri finale: {len(df):,}")

# 2.3 Verificarea și eliminarea duplicatelor
print("\n\n2.3 Verificarea duplicatelor:")
print("-" * 80)

duplicates = df.duplicated().sum()
print(f"  • Duplicate găsite: {duplicates:,}")

if duplicates > 0:
    before_dup = len(df)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    print(f"  • Duplicate eliminate: {before_dup - len(df):,}")
    print(f"  • Rânduri rămase: {len(df):,}")
else:
    print("  ✓ Nu există duplicate în dataset!")

# ============================================================================
# 3. TRANSFORMĂRI
# ============================================================================
print("\n\n3. TRANSFORMĂRI")
print("=" * 80)

# 3.1 Conversia coloanelor numerice
print("\n3.1 Conversia coloanelor numerice în formate potrivite:")
print("-" * 80)

numeric_cols = ['points', 'price', 'vintage', 'alcohol']
for col in numeric_cols:
    if col in df.columns:
        original_type = df[col].dtype
        # Convertim la numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')

        # Verificăm conversiile
        non_null_count = df[col].notna().sum()
        print(f"  • {col}: {original_type} → {df[col].dtype} ({non_null_count:,} valori valide)")

# 3.2 Standardizarea datelor categorice
print("\n\n3.2 Standardizarea datelor categorice:")
print("-" * 80)

categorical_cols = ['country', 'category', 'variety', 'province', 'designation', 'winery']
for col in categorical_cols:
    if col in df.columns:
        # Eliminăm spațiile goale de la început și sfârșit
        df[col] = df[col].astype(str).str.strip()

        # Convertim la title case pentru consistență (prima literă mare)
        df[col] = df[col].str.title()

        # Înlocuim 'Nan' cu 'Unknown'
        df[col] = df[col].replace('Nan', 'Unknown')

        unique_count = df[col].nunique()
        print(f"  • {col}: standardizat ({unique_count:,} valori unice)")

# 3.3 Crearea variabilei raport preț/calitate
print("\n\n3.3 Crearea variabilei 'price_quality_ratio' (raport preț/calitate):")
print("-" * 80)
print("     Formula: price / points")
print("     (Valoare mai mică = raport calitate-preț mai bun)")

# Verificăm dacă variabila există deja
if 'price_quality_ratio' not in df.columns:
    # Eliminăm rândurile unde price sau points sunt 0 sau NaN
    valid_for_ratio = (df['price'] > 0) & (df['points'] > 0) & df['price'].notna() & df['points'].notna()

    df['price_quality_ratio'] = np.nan
    df.loc[valid_for_ratio, 'price_quality_ratio'] = df.loc[valid_for_ratio, 'price'] / df.loc[
        valid_for_ratio, 'points']

    print(f"\n  ✓ Variabilă creată pentru {valid_for_ratio.sum():,} rânduri")
else:
    # Recalculăm pentru a fi siguri
    valid_for_ratio = (df['price'] > 0) & (df['points'] > 0) & df['price'].notna() & df['points'].notna()
    df.loc[valid_for_ratio, 'price_quality_ratio'] = df.loc[valid_for_ratio, 'price'] / df.loc[
        valid_for_ratio, 'points']
    print(f"\n  ✓ Variabilă recalculată pentru {valid_for_ratio.sum():,} rânduri")

# Statistici despre price_quality_ratio
valid_ratio = df['price_quality_ratio'].dropna()
if len(valid_ratio) > 0:
    print(f"\n  Statistici price_quality_ratio:")
    print(f"    • Medie: {valid_ratio.mean():.4f}")
    print(f"    • Mediană: {valid_ratio.median():.4f}")
    print(f"    • Min: {valid_ratio.min():.4f}")
    print(f"    • Max: {valid_ratio.max():.4f}")
    print(f"    • Std: {valid_ratio.std():.4f}")

# ============================================================================
# 4. VERIFICARE FINALĂ
# ============================================================================
print("\n\n4. VERIFICARE FINALĂ")
print("=" * 80)

print("\nStructura finală a dataset-ului:")
print(f"  • Dimensiuni: {df.shape[0]:,} rânduri × {df.shape[1]} coloane")
print(f"\n  Coloane finale:")
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    non_null = df[col].notna().sum()
    null_count = df[col].isnull().sum()
    print(f"    {i:2d}. {col:25s} | Tip: {str(dtype):10s} | Valide: {non_null:>6,} | Lipsă: {null_count:>5,}")

# ============================================================================
# 5. SALVAREA DATELOR PROCESATE
# ============================================================================
print("\n\n5. SALVAREA DATELOR PROCESATE")
print("=" * 80)

output_file = "wine_data_cleaned.csv"
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"✓ Datele procesate au fost salvate în: {output_file}")
print(f"  • Dimensiune finală: {df.shape[0]:,} rânduri × {df.shape[1]} coloane")

# Salvăm și primele 10 rânduri pentru verificare
print(f"\nPrimele 5 rânduri din fișierul procesat:")
print(df.head(5).to_string())

# ============================================================================
# REZUMAT FINAL
# ============================================================================
print("\n\n" + "=" * 80)
print("REZUMAT FINAL")
print("=" * 80)

print(f"""
Dataset original:        {initial_rows:,} rânduri
Rânduri eliminate:       {initial_rows - len(df):,} (valori lipsă în coloane critice + duplicate)
Dataset final:           {df.shape[0]:,} rânduri × {df.shape[1]} coloane

Coloane în dataset final:
  {', '.join(df.columns)}

Transformări realizate:
  ✓ Rânduri cu valori lipsă în coloane critice eliminate
     Coloane critice: country, points, price, variety, category, vintage

  ✓ Valori lipsă în coloane opționale completate:
     - alcohol: înlocuit cu mediana
     - description, designation, province, region_1, region_2, winery, title: 
       înlocuit cu 'Unknown' sau ''

  ✓ Duplicate eliminate: {duplicates:,}

  ✓ Coloane numerice convertite la tipuri corecte:
     - points, price, vintage, alcohol → float64/int64

  ✓ Date categorice standardizate:
     - Title Case aplicat
     - Spații eliminate
     - 'Nan' înlocuit cu 'Unknown'

  ✓ Variabilă nouă creată: price_quality_ratio
     - Formula: price / points
     - Valori valide: {valid_ratio.sum() if len(valid_ratio) > 0 else 0:,}

Fișier generat:
  📄 {output_file}
""")

print("=" * 80)
print("PROCESARE COMPLETĂ!")
print("=" * 80)

print(f"\nPoți folosi fișierul '{output_file}' pentru analize ulterioare.")
print("Toate coloanele au numele lor originale și corecte!")