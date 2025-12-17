#  Wine Data Analysis & Recommendation Project
##### Student: Iatco Marcel
##  Descriere Generală
Acest proiect are ca scop analiza unui set de date despre vinuri, combinând **analiza numerică**, **analiza textuală a descrierilor**, **corelații statistice** și **vizualizări avansate**, precum și dezvoltarea unei **aplicații interactive** pentru explorarea datelor.  
Proiectul urmărește identificarea relațiilor dintre **preț**, **rating (points)**, **alcool**, **descrierea textuală** și **soiul de struguri**, precum și evaluarea raportului **preț/calitate**.

## 📊 Setul de Date
Setul de date conține informații despre vinuri, având următoarele coloane principale:

- `country` – țara de proveniență  
- `description` – descrierea textuală a vinului  
- `points` – scorul (ratingul) vinului  
- `price` – prețul vinului  
- `alcohol` – procentul de alcool  
- `province`, `region_1`, `region_2` – informații geografice  
- `variety` – soiul de struguri  
- `category` – tipul vinului (Red, White etc.)  
- `price_quality_ratio` – raport preț/calitate (variabilă derivată)

##  1. Curățarea și Preprocesarea Datelor

### ✔ Tratarea valorilor lipsă
- Identificarea valorilor lipsă în coloanele numerice și categorice
- Înlocuirea valorilor lipsă:
  - mediană pentru variabile numerice (`price`, `points`, `alcohol`)
  - modă pentru variabile categorice

### ✔ Eliminarea duplicatelor
- Verificarea și eliminarea înregistrărilor duplicate pentru a evita distorsiuni statistice

### ✔ Transformări
- Conversia coloanelor numerice în formate adecvate (`float`)
- Standardizarea valorilor categorice (`country`, `category`, `variety`)
- Crearea variabilei:

price_quality_ratio = points / price

## 2. Explorarea Inițială a Datelor

### 2.1 Statistici Descriptive pentru Variabilele Numerice

#### 2.1.1 Price (Preț)
- **Medie:** calculată din datele curățate  
- **Mediană:** indicator al prețului tipic al unui vin  
- **Deviație standard (Std):** indică nivelul de variabilitate al prețurilor  
- **Min / Max:** identifică intervalul complet al prețurilor din dataset  

#### 2.1.2 Points (Rating)
- **Medie:** rating-ul mediu al vinurilor analizate  
- **Mediană:** valoarea centrală a distribuției rating-urilor  
- **Distribuție:** majoritatea vinurilor se încadrează între **85 și 95** puncte  

#### 2.1.3 Alcohol (Conținut de Alcool)
- **Medie:** conținutul mediu de alcool al vinurilor  
- **Interval tipic:** între **11% și 15%**  
- **Corelație cu calitatea:** relația dintre alcool și scorul vinului este analizată în secțiunea de corelații  
<img width="496" height="232" alt="image" src="https://github.com/user-attachments/assets/a2e0a27f-cec7-4056-80b1-8b84b7d0d589" />

### 2.2 Distribuția Variabilelor Categorice

#### 2.2.1 Country (Țară)
- Identificarea țărilor cu cel mai mare număr de vinuri în dataset  
- Evidențierea principalilor producători  
- Analiza distribuției geografice a datelor  
<img width="1178" height="580" alt="image" src="https://github.com/user-attachments/assets/95c224f6-f09f-42b5-8464-d685d96502ba" />

#### 2.2.2 Category (Categorie)
- **Red (Roșu):** proporția vinurilor roșii din total  
- **White (Alb):** proporția vinurilor albe  
- **Rosé / Sparkling / Dessert:** categorii secundare, cu pondere mai redusă  
<img width="1184" height="554" alt="image" src="https://github.com/user-attachments/assets/fe357baf-7a94-4181-a310-aa1ec38a24f8" />

#### 2.2.3 Variety (Soi de Struguri)
- Identificarea celor mai populare soiuri de struguri  
- Analiza diversității soiurilor prezente în dataset  
- Evidențierea soiurilor specifice anumitor regiuni  
<img width="1176" height="584" alt="image" src="https://github.com/user-attachments/assets/22e4f34b-d3dd-40ed-ba4a-c9d7906b717f" />

## 3. Analiza Textului (Step 2)

### 3.1 Analiză Preliminară

#### 3.1.1 Cele Mai Frecvente Cuvinte

**Procesul de extracție a cuvintelor frecvente:**

**Preprocesare text:**
- Tokenizare folosind **NLTK**
- Eliminarea stopwords (și, este, de, în etc.)
- Conversie la lowercase
- Eliminarea punctuației

**Top cuvinte identificate (exemple):**
- **Descriptori de gust:** cherry, fruity, oak, tannins, acidity  
- **Descriptori aromatici:** floral, spicy, herbal  
- **Caracteristici structurale:** finish, palate, rich, soft  
<img width="1095" height="580" alt="image" src="https://github.com/user-attachments/assets/723c33f1-f325-4d8a-a0bf-70e6eac6ac67" />

#### 3.1.2 Lungimea Medie a Descrierii
- **Lungime medie:** calculată atât în caractere, cât și în număr de cuvinte  
- **Corelație cu rating:** vinurile mai bine cotate tind să aibă descrieri mai detaliate  
- **Corelație cu preț:** vinurile mai scumpe au, în general, descrieri mai elaborate  

### 3.2 WordCloud

WordCloud-ul este generat pentru:
- Vizualizarea rapidă a cuvintelor dominante
- Identificarea pattern-urilor lingvistice
- Compararea categoriilor (Red vs White) și a regiunilor

**Parametri utilizați:**
- Background color: `white` / `black`
- Max words: `100–200`
- Stopwords: personalizate pentru domeniul vinului

### 3.3 Word Correlation Analysis

#### 3.3.1 Corelația Cuvintelor cu Prețul și Rating-ul

Analiză statistică pentru identificarea cuvintelor care corelează cu:

- **Preț ridicat:** *complex*, *elegant*, *refined*, *estate*  
- **Rating ridicat:** *balanced*, *structured*, *layered*  

**Metodologie:**
- Construirea unei matrici **TF-IDF**
- Calcularea coeficientului de corelație **Pearson**
- Identificarea primelor **20–30** de cuvinte semnificative
<img width="886" height="528" alt="image" src="https://github.com/user-attachments/assets/e2b399c8-7616-45e3-bde1-27f6c4be6389" />

#### 3.3.2 Corelația Cuvintelor cu Soiuri

Identificarea cuvintelor distinctive pentru fiecare soi de struguri:
- **Cabernet Sauvignon:** blackcurrant, cassis, cedar  
- **Pinot Noir:** cherry, earthy, mushroom  
- **Chardonnay:** butter, vanilla, oak  
- **Sauvignon Blanc:** citrus, grass, tropical  
<img width="1096" height="560" alt="image" src="https://github.com/user-attachments/assets/c2db0f8b-dee5-4868-a0cf-9450ddb7a79c" />

## 4. Analiza Corelațiilor (Step 3)

### 4.1 Corelații Numerice

#### 4.1.1 Price vs Points
- **Coeficient de corelație:** calculat folosind Pearson / Spearman  
- **Interpretare:** corelație pozitivă moderată–puternică  
<img width="597" height="575" alt="image" src="https://github.com/user-attachments/assets/6fca4f4e-ef03-4d5e-99fb-37db1195d1a7" />

**Constatări:**
- Vinurile cu rating mai mare tind să fie mai scumpe  
- Există excepții notabile (vinuri scumpe cu rating mediu)  
- **Sweet spot:** rating între **90–93** cu preț moderat  

#### 4.1.2 Alcohol vs Points
- **Coeficient de corelație:** analizat pentru identificarea pattern-urilor  
<img width="574" height="541" alt="image" src="https://github.com/user-attachments/assets/dec5743e-81a6-487b-895f-ca6790681138" />

**Constatări:**
- Corelație slabă până la moderată  
- Conținut optim de alcool: **13–14.5%**  
- Vinurile foarte alcoolice nu sunt automat mai bine cotate  

#### 4.1.3 Regiune vs Preț / Points

**Regiuni premium** (ex: Bordeaux, Napa Valley, Tuscany):
- Prețuri medii mai ridicate  
- Consistență în calitate  

**Regiuni emergente:**
- Raport calitate–preț superior  
- Variabilitate mai mare a rating-urilor  
<img width="1228" height="372" alt="image" src="https://github.com/user-attachments/assets/a20cd846-3557-4ac9-9186-5aee95d93ecd" />

#### 4.1.4 Varietăți de Struguri vs Preț / Points

**Soiuri premium** (Pinot Noir, Cabernet Sauvignon):
- Preț mediu mai ridicat  
- Rating-uri consistente  

**Soiuri accesibile** (Merlot, Chardonnay):
- Gamă largă de prețuri  
- Variabilitate mare în calitate  

### 4.2 Vizualizarea Corelațiilor
<img width="1408" height="391" alt="image" src="https://github.com/user-attachments/assets/2d5f8bb4-8cd0-4b4b-b5b6-9954ebff4d2f" />

#### 4.2.1 Heatmap pentru Corelații Numerice
- Librărie: **seaborn**
- Variabile incluse: `price`, `points`, `alcohol`, `vintage`, `price_quality_ratio`
- Colormap: divergent (`RdBu` / `coolwarm`)
- Adnotări: valori numerice pentru fiecare celulă
<img width="1094" height="569" alt="image" src="https://github.com/user-attachments/assets/ddb76591-f309-48c8-99f4-1671d86f1036" />
<img width="880" height="774" alt="image" src="https://github.com/user-attachments/assets/ddf0b913-3fb6-4777-853c-719b695068dd" />

#### 4.2.2 Scatter Plots pentru Relații Individuale

**Price vs Points:**
- Axă X: Points (80–100)
- Axă Y: Price (log scale – opțional)
- Linie de regresie
- Identificare outliers
<img width="994" height="475" alt="image" src="https://github.com/user-attachments/assets/994f7399-327b-4a5d-b8cf-475663c1ae14" />

**Alcohol vs Points:**
- Trend slab vizibil
- Grupare în jurul valorilor **13–14%**
<img width="977" height="489" alt="image" src="https://github.com/user-attachments/assets/6229b458-0c23-4574-8f4e-fd2bc624e243" />

**Price–Quality Ratio:**
- Identificarea vinurilor cu cel mai bun raport
- Distribuție pe categorii și țări

## 5. Vizualizări Avansate (Step 4)

### 5.1 Distribuția Punctajelor (Points)
- Tip grafic: **Histogramă**
- Axă X: Points (80–100)
- Axă Y: Frecvență
- Bins: 20–30 intervale
<img width="1179" height="563" alt="image" src="https://github.com/user-attachments/assets/bfa50092-b1ac-425a-8131-f4416d704c7c" />

**Observații:**
- Distribuție aproximativ normală  
- Peak în jurul **87–90** puncte  
- Puține vinuri sub 80 sau peste 97  

### 5.2 Prețurile Medii pe Țară
- Tip grafic: **Bar Plot**
- Axă X: Country (Top 15–20)
- Axă Y: Preț mediu (USD)
- Sortare: descrescător după preț
<img width="1475" height="725" alt="image" src="https://github.com/user-attachments/assets/af609ca8-9ab9-4d05-baa9-0ec01e61d57f" />

**Țări cu vinuri scumpe:**
- Franța
- SUA (Napa, Sonoma)
- Italia (regiuni premium)

**Țări cu vinuri accesibile:**
- Spania
- Argentina
- Chile

### 5.3 Distribuția Vinurilor după Categorii și Regiuni
- Tip grafic: **Stacked Bar Chart**
- Axă X: Region / Country
- Axă Y: Număr de vinuri
- Culori: diferențiate pe categorie
<img width="1173" height="681" alt="image" src="https://github.com/user-attachments/assets/0c2ef8d6-3c51-4c1b-80c9-1a596b4cc8b9" />

**Insight-uri:**
- Dominanța vinurilor roșii în anumite regiuni
- Specializarea regiunilor pe categorii specifice
- 
## 6. Aplicație Streamlit

### 6.1 Funcționalități Implementate

#### 6.1.1 Filtre Interactive
- Range preț (slider)
- Range points (slider)
- Country (multiselect)
- Category (multiselect)
- Variety (multiselect)
- Raport preț/calitate (slider – best value)

> Toate filtrele funcționează simultan.

#### 6.1.2 Afișarea Vizualizărilor

**Dashboard principal:**
- **Overview:** statistici generale și KPI-uri
- **Distribuții:** histograme, box plots
- **Corelații:** scatter plots, heatmap-uri, violin plots
- **Analiză geografică:** bar chart-uri și hartă interactivă (opțional)
- **Analiza textului:** WordCloud, top cuvinte, lungime descrieri

#### 6.1.3 Căutare Vinuri după Descriere Textuală
- Introducere text de către utilizator
- Căutare în coloana `description`
- Afișarea rezultatelor relevante:
  - Titlu vin
  - Rating
  - Preț
  - Descriere completă
  - Similarity score (opțional, TF-IDF)
<img width="1897" height="895" alt="image" src="https://github.com/user-attachments/assets/c90b0f1a-38ce-4346-a349-04eae0bc1c15" />

<img width="1911" height="926" alt="image" src="https://github.com/user-attachments/assets/11e3600b-6590-4112-be83-7f941247b4ab" />
