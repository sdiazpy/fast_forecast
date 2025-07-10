import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

class mco_forecast:
    def __init__(self, tendencia=True, estacionalidad=None, log_y=False, lowess_frac=0.2):
        # tendencia: False, True/1 (lineal), 2...n (polinomial), "lowess"
        if tendencia == "lowess":
            self.tendencia = "lowess"
        elif tendencia is True:
            self.tendencia = 1
        elif tendencia is False:
            self.tendencia = 0
        else:
            self.tendencia = int(tendencia)
        self.lowess_frac = lowess_frac
        self.estacionalidad = estacionalidad or []
        self.log_y = log_y
        self.model = LinearRegression()
        self.feature_names_ = []
        self.last_t_ = None
        self.freq_ = None
        self.last_date_ = None
        self.fittedvalues = None
        self.exog_names_ = []
        self.tendencia_train_ = None  # guarda la tendencia ajustada a los datos

    def _make_features(self, df, start_idx=0, exog_data=None):
        df_ = df.copy()
        n = len(df_)
        feat_cols = []
        # Tendencia polinómica (solo si tendencia no es "lowess")
        if isinstance(self.tendencia, int) and self.tendencia > 0:
            t = np.arange(start_idx, start_idx + n)
            for deg in range(1, self.tendencia + 1):
                col_name = f't{deg}'
                df_[col_name] = t ** deg
                feat_cols.append(col_name)
        # Estacionalidad
        for item in self.estacionalidad:
            if item == 'dayofweek':
                df_['dayofweek'] = df_.index.dayofweek
                dummies = pd.get_dummies(df_['dayofweek'], prefix='dow', drop_first=True)
                for col in dummies.columns:
                    if col not in feat_cols:
                        feat_cols.append(col)
                df_ = pd.concat([df_, dummies], axis=1)
            elif item == 'month':
                df_['month'] = df_.index.month
                dummies = pd.get_dummies(df_['month'], prefix='mon', drop_first=True)
                for col in dummies.columns:
                    if col not in feat_cols:
                        feat_cols.append(col)
                df_ = pd.concat([df_, dummies], axis=1)
        # Exógenas
        if self.exog_names_ and exog_data is not None:
            for col in self.exog_names_:
                df_[col] = exog_data[col].values
                if col not in feat_cols:
                    feat_cols.append(col)
        elif self.exog_names_:  # para el fit
            for col in self.exog_names_:
                if col in df_.columns:
                    if col not in feat_cols:
                        feat_cols.append(col)
        if not self.feature_names_:
            self.feature_names_ = feat_cols
        return df_

    def fit(self, df, y_col, exog=None):
        df_ = df.copy()
        # Si exog es lista de nombres, guárdalos
        if exog is not None:
            if isinstance(exog, list):
                self.exog_names_ = exog
            else:
                raise ValueError("exog debe ser una lista con los nombres de las columnas exógenas.")
        else:
            self.exog_names_ = []
        df_feat = self._make_features(df_)
        X = df_feat[self.feature_names_]
        y_raw = df_feat[y_col]
        if self.log_y:
            if (y_raw <= 0).any():
                raise ValueError("Todos los valores de y deben ser > 0 para aplicar logaritmo.")
            y_raw = np.log(y_raw)
        # LOWESS para tendencia
        if self.tendencia == "lowess":
            tendencia = lowess(y_raw, np.arange(len(y_raw)), frac=self.lowess_frac, return_sorted=False)
            self.tendencia_train_ = tendencia
            y = y_raw - tendencia
        else:
            y = y_raw
        self.model.fit(X, y)
        # Guarda el último valor de t para el forecast (solo si polinómica)
        if isinstance(self.tendencia, int) and self.tendencia > 0:
            self.last_t_ = df_feat[f't1'].iloc[-1]
        else:
            self.last_t_ = None
        self.freq_ = pd.infer_freq(df_.index)
        self.last_date_ = df_.index[-1]
        # fitted values: sumamos la tendencia si LOWESS
        yhat = self.model.predict(X)
        if self.tendencia == "lowess":
            yhat = yhat + self.tendencia_train_
        if self.log_y:
            yhat = np.exp(yhat)
        self.fittedvalues = pd.Series(yhat, index=df_.index)
        return self

    def forecast(self, steps, exog=None):
        if self.freq_ is None or self.last_date_ is None:
            raise ValueError("El modelo debe ser entrenado con fit() antes de pronosticar.")
        fechas = pd.date_range(self.last_date_ + pd.tseries.frequencies.to_offset(self.freq_),
                               periods=steps, freq=self.freq_)
        df_future = pd.DataFrame(index=fechas)
        # Si hay exógenas, exog debe ser un DataFrame con esas columnas
        if self.exog_names_:
            if exog is None:
                raise ValueError(f"Debes pasar un DataFrame exog con columnas: {self.exog_names_}")
            if not all(col in exog.columns for col in self.exog_names_):
                raise ValueError(f"exog debe tener estas columnas: {self.exog_names_}")
            exog_data = exog[self.exog_names_].loc[fechas]
        else:
            exog_data = None
        start_idx = 0
        if isinstance(self.tendencia, int) and self.tendencia > 0 and self.last_t_ is not None:
            start_idx = self.last_t_ + 1
        df_future = self._make_features(df_future, start_idx=start_idx, exog_data=exog_data)
        for col in self.feature_names_:
            if col not in df_future.columns:
                df_future[col] = 0
        Xf = df_future[self.feature_names_]
        yhat = self.model.predict(Xf)
        # Para LOWESS: tendencia futura = último valor observado
        if self.tendencia == "lowess":
            last_trend = self.tendencia_train_[-1]
            yhat = yhat + last_trend
        if self.log_y:
            yhat = np.exp(yhat)
        return pd.Series(yhat, index=fechas)

    def predict(self, df, exog=None):
        if self.exog_names_:
            if exog is None:
                raise ValueError(f"Debes pasar un DataFrame exog con columnas: {self.exog_names_}")
            if not all(col in exog.columns for col in self.exog_names_):
                raise ValueError(f"exog debe tener estas columnas: {self.exog_names_}")
            exog_data = exog[self.exog_names_].loc[df.index]
        else:
            exog_data = None
        df_feat = self._make_features(df, exog_data=exog_data)
        for col in self.feature_names_:
            if col not in df_feat.columns:
                df_feat[col] = 0
        X = df_feat[self.feature_names_]
        yhat = self.model.predict(X)
        if self.tendencia == "lowess":
            # Usar fitted LOWESS sobre las fechas del df dado (solo posible si coincide con el train)
            if len(df) == len(self.tendencia_train_):
                yhat = yhat + self.tendencia_train_
            else:
                yhat = yhat + self.tendencia_train_[-1]
        if self.log_y:
            yhat = np.exp(yhat)
        return pd.Series(yhat, index=df.index)
