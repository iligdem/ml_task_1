from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from category_encoders import LeaveOneOutEncoder
from feature_engine.outliers import Winsorizer
import re
import warnings
warnings.filterwarnings('ignore')



def extract_torque(s):
    if not isinstance(s, str):
        return None
    s = s.replace(',', '')
    # Ищем число и единицу измерения после него (возможно с @ или пробелами)
    match = re.search(r'([\d\.]+)\s*(?:@|\(|at)?\s*([\d\.]*)\s*([a-z]*)', s,)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(3).strip().lower()

    # Если единица — 'nm', оставляем как есть
    if 'nm' in unit:
        return value
    # Если 'kgm' '@' — считаем, что это kgm и умножаем на 10
    elif 'kgm' in unit or '@' in s or 'kg' in unit:
        return value * 10
    else:
        return value * 10

def extract_max_torque_rpm(s):
    if not isinstance(s, str):
        return None
    s = s.replace(',', '')
     # Ищем диапазон: 1500-2500, 1500~2500 и т.п.
    range_match = re.search(r'(\d+)\s*[-~]\s*(\d+)[^\d]*?(?:rpm|\)|$)', s)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        return round((start + end) / 2)

    # Ищем одиночное значение с rpm или в скобках
    single_match = re.search(r'(\d+)\s*[^\d]*?(?:rpm|\)|$)', s)
    if single_match:
        return int(single_match.group(1))

    # Попытка найти число после @, at, / и т.д., даже если rpm не указан явно
    fallback_match = re.search(r'(?:@|at|\/)\s*(\d+)', s)
    if fallback_match:
        return int(fallback_match.group(1))



# Список марок
BRANDS = [
    'Maruti', 'Hyundai', 'Toyota', 'Ford', 'Mahindra', 'Honda', 'Tata', 'Skoda',
    'Renault', 'Volkswagen', 'Chevrolet', 'Datsun', 'Nissan', 'BMW', 'Mercedes-Benz',
    'Audi', 'MG', 'Jeep', 'Mitsubishi', 'Fiat', 'Isuzu', 'Peugeot', 'Volvo', 'Lexus',
    'Jaguar', 'Land Rover', 'Kia', 'Force', 'Ambassador', 'Daewoo', 'Ssangyong','Opel','Ashok'
]

# Типы топлива
FUEL_TYPES = [
    'Diesel', 'Petrol', 'CNG', 'LPG', 'Electric', 'Hybrid'
]

# Аббревиатуры, указывающие на тип топлива
FUEL_CODES = [
    'TDI', 'CRDi', 'DICOR', 'Revotorq', 'mHawk', 'dCi', 'DDiS', 'Kappa',
    'VTVT', 'VT', 'VVT', 'i-VTEC', 'i DTEC', 'i DTec', 'VX', 'VXi'
]

# Типы трансмиссии
TRANSMISSION = [
    'AMT', 'AT', 'MT', 'DCT', 'CVT', 'Manual', 'Automatic'
]

# Привод
DRIVE_TYPE = [
    '4x4', '4X4', 'AWD', '2WD', 'FWD', '4x2', '4WD'
]

# Экологические стандарты
EMISSION = [
    'BSII', 'BSIII', 'BSIV'
]

# Специальные версии
EDITIONS = [
    'Celebration', 'Limited Edition', 'Special Edition', 'Anniversary Edition',
    'Signature', 'Sports Edition', 'TRD Sportivo', 'Hurricane Limited',
    'Dark Edition', 'Stepway', 'Adventure Edition', 'Windsong Limited',
    'Knightracer', 'Corporate Edition', 'Premium', 'Luxury', 'Prestige',
    'Elegance', 'Style', 'Ambition', 'Option', 'Optional', 'Plus', 'Dual Tone'
]

# Комплектации (часто встречаются)
VARIANTS = [
    'VXI', 'ZXI', 'LDI', 'VDI', 'ZDI', 'LXi', 'LX', 'EX', 'EXi', 'LXI', 'VXi',
    'VX', 'ZXi', 'Alpha', 'Delta', 'Zeta', 'Sigma', 'Era', 'Magna', 'Sportz',
    'Asta', 'SX', 'SX+', 'GLS', 'GLE', 'GLX', 'GVS', 'VTEC', 'CRDi', 'S', 'E',
    'V', 'Z', 'L', 'H', 'X', 'R', 'STD', 'BASE', 'ABS', 'Airbag', 'Sunroof'
]


def extract_features(name):
    if not isinstance(name, str):
        name = ""
    original = name
    features = {
        'brand': None,
        'model': None,
        'variant': None,
        'engine_size': None,
        'fuel_type': None,
        'transmission': None,
        'drive_type': None,
        'seating_capacity': None,
        'edition': None,
        'emission': None
    }

    # 1. Извлечение марки
    for brand in sorted(BRANDS, key=len, reverse=True):
        if brand in name:
            features['brand'] = brand
            name = name.replace(brand, "", 1).strip()
            break

    # 2. Извлечение экологического стандарта
    for em in EMISSION:
        if em in name:
            features['emission'] = em
            name = name.replace(em, "").strip()

    # 3. Извлечение типа трансмиссии
    for trans in TRANSMISSION:
        if trans in name:
            features['transmission'] = trans
            name = name.replace(trans, "").strip()

    # 4. Извлечение привода
    for drive in DRIVE_TYPE:
        if drive in name:
            features['drive_type'] = drive
            name = name.replace(drive, "").strip()

    # 5. Извлечение количества мест
    seating_match = re.search(r'(\d+)\s*Seater|(\d+)\s*Seats|(\d+)\s*STR', original, re.IGNORECASE)
    if seating_match:
        features['seating_capacity'] = int(seating_match.group(1) or seating_match.group(2) or seating_match.group(3))

    # 6. Извлечение объёма двигателя
    engine_match = re.search(r'\b(\d+\.\d+|\d+)\s*(?:Litre|Litre|L|T|Litre|TDI|CRDi|VTVT|Petrol|Diesel|P)\b', original)
    if engine_match:
        features['engine_size'] = float(engine_match.group(1))

    # 7. Извлечение типа топлива
    # Сначала по словам
    for fuel in FUEL_TYPES:
        if fuel in original:
            features['fuel_type'] = fuel
            break
    # Если не найдено — по кодам
    if not features['fuel_type']:
        for code in FUEL_CODES:
            if code in original:
                features['fuel_type'] = 'Diesel' if code in ['TDI', 'CRDi', 'DICOR', 'dCi', 'Revotorq', 'mHawk', 'DDiS'] else 'Petrol'
                break

    # 8. Извлечение специальных версий
    editions_found = []
    for ed in sorted(EDITIONS, key=len, reverse=True):
        if ed in original and ed not in name:  # если было удалено ранее
            editions_found.append(ed)
    features['edition'] = ", ".join(editions_found) if editions_found else None

    # 9. Извлечение комплектации
    variants_found = []
    for v in sorted(VARIANTS, key=len, reverse=True):
        if re.search(rf'\b{re.escape(v)}\b', name):
            variants_found.append(v)
            name = re.sub(rf'\b{re.escape(v)}\b', "", name).strip()
    features['variant'] = ", ".join(variants_found) if variants_found else None

    # 10. Остаток — модель
    model = re.sub(r'[()]', "", name).strip()
    model = re.sub(r'\s+', ' ', model).strip()
    if model and model != features['variant']:
        features['model'] = model

    return pd.Series(features)

#преобразование типов
class TypeConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
       
        X['mileage'] = X['mileage'].astype(str).str.extract(r'(\d+)')[0]

        X['mileage'] = pd.to_numeric(X['mileage'], errors='coerce')
        X['mileage'] = X['mileage'].fillna(X['mileage'].median())

    
        X['engine'] = X['engine'].astype(str).str.extract(r'(\d+)')[0]
        X['engine'] = pd.to_numeric(X['engine'], errors='coerce')
        X['engine'] = X['engine'].fillna(X['engine'].median())

    
        X['max_power'] = X['max_power'].astype(str).str.extract(r'(\d+)')[0]
        X['max_power'] = pd.to_numeric(X['max_power'], errors='coerce')
        X['max_power'] = X['max_power'].fillna(X['max_power'].median())
        return X

#парсинг строки с моментом и оборотами
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['torque_text'] = X['torque']
        X['torque'] = X['torque_text'].apply(lambda x: extract_torque(x) if type(x) != float else x)
        X['max_torque_rpm'] = X['torque_text'].apply(lambda x: extract_max_torque_rpm(x) if type(x) != float else x)
        X = X.drop('torque_text', axis=1)
        return X

#заполнение пропусков медианой
class MedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.medians_ = None

    def fit(self, X, y=None):
        self.medians_ = X[self.columns].median()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].fillna(self.medians_[col])
        return X

#удаление выбросов
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns, fold=1.5):
        self.columns = columns
        self.fold = fold
        self.winsorizer_ = None

    def fit(self, X, y=None):
        self.winsorizer_ = Winsorizer(
            capping_method='iqr',
            tail='both',
            fold=self.fold
        )
        self.winsorizer_.fit(X[self.columns])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns] = self.winsorizer_.transform(X[self.columns])
        return X

#добавление новых признаков
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        temp = X['name'].apply(extract_features)
        temp = temp[['brand', 'model', 'variant']].copy()
        temp['variant'] = temp['variant'].fillna('Unknown')
        X = X.drop('name', axis=1)
        X = pd.concat([X, temp], axis=1)
        X['age'] = 2025 - X['year'] #возраст авто
        X['power_liters'] = X['max_power'] / X['engine'].replace(0, 1)  # мощность/литр
        X['specific_fuel_consumption'] = X['mileage'] / X['max_power'].replace(0, 1) # удельный расход топлива
        X['km_per_year'] = X['km_driven'] / X['age'].replace(0, 1)# коэффициент использования авто
        X['torque_power_ratio'] = X['torque'] / X['max_power'].replace(0, 1)# момент/мощность
        X['age_x_km'] = X['age'] * X['km_driven']# степень износа
        X['is_budget'] = X['brand'].apply(lambda x: 1 if x in ['Maruti', 'Renault', 'Chevrolet', 'Fiat', 'Datsun', 'Tata', 'Daewoo','Force', 'Ambassador'] else 0) #бюджетное авто
        X['is_premium'] = X['brand'].apply(lambda x: 1 if x in ['Mercedes-Benz', 'Audi', 'BMW','Lexus', 'Jaguar', 'Land Rover', 'Volvo'] else 0) #премиум авто
        return X

#добавление полиномиальных признаков
class PolynomialFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, features, degree=2):
        self.features = features
        self.degree = degree
        self.poly = None
        self.feature_names = None

    def fit(self, X, y=None):
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.poly.fit(X[self.features])

        self.feature_names = [f"poly_{i}" for i in range(self.poly.n_output_features_)]
        return self

    def transform(self, X):
        X = X.copy()
        poly_vals = self.poly.transform(X[self.features])
        poly_df = pd.DataFrame(poly_vals, index=X.index, columns=self.feature_names)
        return pd.concat([X, poly_df], axis=1)


#Стандартизация числовых признаков
class StandardScalerCustom(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.scaler_ = StandardScaler()

    def fit(self, X, y=None):
        self.scaler_.fit(X[self.columns])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns] = self.scaler_.transform(X[self.columns])
        return X

#Кодирование высококардинальных признаков: LeaveOneOut
class LOOEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoder_ = LeaveOneOutEncoder(cols=columns)

    def fit(self, X, y):
        self.encoder_.fit(X, y)
        return self

    def transform(self, X):
        return self.encoder_.transform(X)

# OHE для категориальных с низкой кардинальностью
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class OHEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.dummies_ = {}          # Для хранения имён закодированных столбцов
        self.categories_ = {}       # <-- Добавлено: инициализация атрибута

    def fit(self, X, y=None):
        X_copy = X.copy()

        for col in self.columns:
            if col not in X_copy.columns:
                raise ValueError(f"Column '{col}' not found in data")

            # Принудительно приводим к строке
            X_copy[col] = X_copy[col].astype(str)

            # Сохраняем уникальные категории
            self.categories_[col] = X_copy[col].unique()

            # Создаём дамми-столбцы, чтобы запомнить все возможные имена
            dummies = pd.get_dummies(X_copy[col], prefix=col, dummy_na=False)
            self.dummies_[col] = dummies.columns.tolist()
        return self

    def transform(self, X):
        X_copy = X.copy()

        for col in self.columns:
            if col not in X_copy.columns:
                raise ValueError(f"Column '{col}' not found in data during transform")

            # Приводим к строке
            X_copy[col] = X_copy[col].astype(str)

            # Создаём дамми-переменные
            dummies = pd.get_dummies(X_copy[col], prefix=col, dummy_na=False)

            # Восстанавливаем все ожидаемые колонки (включая отсутствующие)
            for dummy_col in self.dummies_[col]:
                if dummy_col not in dummies.columns:
                    dummies[dummy_col] = 0

            # Сортируем колонки в том же порядке, что и при обучении
            dummies = dummies.reindex(columns=self.dummies_[col], fill_value=0)

            # Удаляем исходную колонку и добавляем дамми
            X_copy = X_copy.drop(columns=[col])
            X_copy = pd.concat([X_copy, dummies], axis=1)
        return X_copy
