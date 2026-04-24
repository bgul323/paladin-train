"""
PALADIN - ML Core
XGBoost + LightGBM + Neural Network ensemble
"""
import os
import json
import warnings
import shutil
warnings.filterwarnings("ignore")

# ══════════════════════════════
# KLASÖR YAPISI KONFİGÜRASYONU
# ══════════════════════════════
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
ML_RESULTS_DIR  = os.path.join(BASE_DIR, "ml_results")
TMP_DIR         = os.path.join(BASE_DIR, "tmp")

# Dosya Yolları
ML_RESULTS_FILE = os.path.join(BASE_DIR, "ml_results.json")
MODEL_STATS_FILE= os.path.join(BASE_DIR, "model_stats.json")
TRAINED_MODEL   = os.path.join(BASE_DIR, "trained_model.joblib")

# Klasörleri otomatik oluştur
for folder in [BASE_DIR, DATA_DIR, ML_RESULTS_DIR, TMP_DIR]:
    os.makedirs(folder, exist_ok=True)

# XGBoost ve ML kütüphaneleri için TMP klasörünü ayarla
os.environ['TMPDIR'] = TMP_DIR
os.environ['TEMP'] = TMP_DIR
os.environ['TMP'] = TMP_DIR
os.environ['XGBOOST_TMPDIR'] = TMP_DIR

def cleanup_ml_results():
    """ml_results klasöründeki eski model kalıntılarını temizle"""
    try:
        deleted = 0
        for filename in os.listdir(ML_RESULTS_DIR):
            if filename.startswith("old_") or filename.endswith(".tmp"):
                file_path = os.path.join(ML_RESULTS_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        deleted += 1
                except:
                    pass
        if deleted > 0:
            print(f"🧹 Temizlik tamamlandı: {deleted} eski dosya silindi")
    except Exception as e:
        print(f"Temizlik hatası: {e}")

# Başlangıçta temizlik çalıştır
cleanup_ml_results()

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ══════════════════════════════
# VERİ HAZIRLAMA
# ══════════════════════════════
def fetch_training_data(ticker: str, years: int = 5) -> pd.DataFrame:
    """5 yıllık veri çek - yahooquery ile"""
    try:
        from yahooquery import Ticker
        import datetime
        
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=years*365)
        
        t = Ticker(ticker)
        df = t.history(start=start_date, end=end_date, interval="1d")
        
        if df is None or df.empty or len(df) < 100:
            return None
        
        # MultiIndex'i düzelt
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Gerekli kolonları seç
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            return None
        
        df = df[required_cols].rename(columns={
            "open": "Open",
            "high": "High", 
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }).dropna()
        
        return df
        
    except Exception as e:
        # Hata durumunda yfinance ile dene
        try:
            period = f"{years * 365}d"
            df = yf.download(ticker, period=period, interval="1d",
                             auto_adjust=False, progress=False, timeout=30)
            if df is None or df.empty or len(df) < 100:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df[["Open","High","Low","Close","Volume"]].dropna()
        except Exception as e2:
            print(f"⚠️ {ticker} için veri çekme hatası: {e} / {e2}")
            return None

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """50+ teknik indikatör feature oluştur"""
    df = df.copy()
    c  = df["Close"]
    h  = df["High"]
    lo = df["Low"]
    v  = df["Volume"]
    
    try:
        import ta
        # RSI
        df["rsi_14"]  = ta.momentum.RSIIndicator(c, 14).rsi()
        df["rsi_7"]   = ta.momentum.RSIIndicator(c, 7).rsi()
        df["rsi_21"]  = ta.momentum.RSIIndicator(c, 21).rsi()
        
        # MACD
        macd         = ta.trend.MACD(c)
        df["macd"]   = macd.macd()
        df["macd_s"] = macd.macd_signal()
        df["macd_h"] = macd.macd_diff()
        
        # Bollinger
        bb           = ta.volatility.BollingerBands(c, 20, 2)
        df["bb_h"]   = bb.bollinger_hband()
        df["bb_l"]   = bb.bollinger_lband()
        df["bb_m"]   = bb.bollinger_mavg()
        df["bb_w"]   = (df["bb_h"] - df["bb_l"]) / df["bb_m"]
        df["bb_pct"] = bb.bollinger_pband()
        
        # EMA
        for span in [5, 10, 20, 50, 100, 200]:
            df[f"ema{span}"] = ta.trend.EMAIndicator(c, span).ema_indicator()
        
        # EMA ratioları
        df["ema_5_20"]   = df["ema5"]   / df["ema20"]
        df["ema_20_50"]  = df["ema20"]  / df["ema50"]
        df["ema_50_200"] = df["ema50"]  / df["ema200"]
        
        # ATR
        df["atr"]     = ta.volatility.AverageTrueRange(h, lo, c, 14).average_true_range()
        df["atr_pct"] = df["atr"] / c * 100
        
        # ADX
        adx          = ta.trend.ADXIndicator(h, lo, c, 14)
        df["adx"]    = adx.adx()
        df["adx_pos"]= adx.adx_pos()
        df["adx_neg"]= adx.adx_neg()
        
        # Stochastic
        stoch        = ta.momentum.StochasticOscillator(h, lo, c, 14, 3)
        df["stoch_k"]= stoch.stoch()
        df["stoch_d"]= stoch.stoch_signal()
        
        # CCI
        df["cci"]    = ta.trend.CCIIndicator(h, lo, c, 20).cci()
        
        # Williams
        df["williams"]= ta.momentum.WilliamsRIndicator(h, lo, c, 14).williams_r()
        
        # OBV
        df["obv"]    = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
        df["obv_pct"]= df["obv"].pct_change(5) * 100
        
        # Hacim
        df["vol_ma5"]  = v.rolling(5).mean()
        df["vol_ma20"] = v.rolling(20).mean()
        df["rel_vol"]  = v / df["vol_ma20"]
        
        # Fiyat değişimleri
        for n in [1, 3, 5, 10, 20]:
            df[f"ret_{n}d"] = c.pct_change(n) * 100
        
        # Momentum
        df["momentum_10"] = c.diff(10)
        df["momentum_20"] = c.diff(20)
        
        # Volatilite
        df["volatility_10"] = c.pct_change().rolling(10).std() * 100
        df["volatility_20"] = c.pct_change().rolling(20).std() * 100
        
        # Mum özellikleri
        df["candle_body"]  = abs(c - df["Open"]) / (h - lo + 1e-9)
        df["upper_shadow"] = (h - np.maximum(c, df["Open"])) / (h - lo + 1e-9)
        df["lower_shadow"] = (np.minimum(c, df["Open"]) - lo) / (h - lo + 1e-9)
        df["is_bullish"]   = (c > df["Open"]).astype(int)
        
        # Fib mesafesi
        rolling_high = h.rolling(80).max()
        rolling_low  = lo.rolling(80).min()
        fib_level    = rolling_high - (rolling_high - rolling_low) * 0.618
        df["fib_dist"]= (c - fib_level) / fib_level * 100
        
    except ImportError:
        # ta yoksa basit özellikler
        df["rsi_14"] = 50
        df["macd"]   = 0
    
    return df

def create_labels(df: pd.DataFrame, days: int = 5, threshold: float = 3.0) -> pd.Series:
    """
    Label oluştur:
    1 = days gün sonra +threshold% üstünde
    0 = değil
    """
    future_return = df["Close"].pct_change(days).shift(-days) * 100
    return (future_return >= threshold).astype(int)

def prepare_dataset(tickers: list, days: int = 5, threshold: float = 3.0):
    """Tüm hisseler için dataset hazırla"""
    all_X = []
    all_y = []
    
    feature_cols = None
    
    for ticker in tickers:
        try:
            df = fetch_training_data(ticker, years=5)
            if df is None or len(df) < 200:
                continue
            
            df_feat = create_features(df)
            labels  = create_labels(df_feat, days=days, threshold=threshold)
            
            df_feat["label"] = labels
            df_feat = df_feat.dropna()
            
            if len(df_feat) < 100:
                continue
            
            exclude = ["Open","High","Low","Close","Volume","label"]
            cols    = [c for c in df_feat.columns if c not in exclude]
            
            if feature_cols is None:
                feature_cols = cols
            
            X = df_feat[feature_cols].values
            y = df_feat["label"].values
            
            all_X.append(X)
            all_y.append(y)
            
        except Exception as e:
            continue
    
    if not all_X:
        return None, None, None
    
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    
    return X_all, y_all, feature_cols

# ══════════════════════════════
# MODEL EĞİTİMİ
# ══════════════════════════════
class PaladinMLModel:
    def __init__(self):
        self.xgb_model   = None
        self.lgbm_model  = None
        self.rf_model    = None
        self.feature_cols = None
        self.is_trained  = False
        self.model_file  = "paladin_model.json"
        self.stats       = {}
    
    def train(self, X, y, feature_cols):
        """3 modeli eğit"""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing   import StandardScaler
        from sklearn.ensemble        import RandomForestClassifier
        from sklearn.metrics         import accuracy_score, classification_report
        
        self.feature_cols = feature_cols
        
        # Train/Test split (zamana göre - son %20 test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📊 Eğitim: {len(X_train)} örnek | Test: {len(X_test)} örnek")
        print(f"📊 Pozitif oran: %{y_train.mean()*100:.1f}")
        
        results = {}
        
        # 1. XGBoost
        try:
            import xgboost as xgb
            print("🔄 XGBoost eğitiliyor...")
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            xgb_pred = self.xgb_model.predict(X_test)
            xgb_acc  = accuracy_score(y_test, xgb_pred)
            results["xgboost"] = xgb_acc
            print(f"✅ XGBoost: %{xgb_acc*100:.1f} doğruluk")
        except ImportError:
            print("⚠️ XGBoost yüklü değil")
        except Exception as e:
            print(f"❌ XGBoost hatası: {e}")
        
        # 2. LightGBM
        try:
            import lightgbm as lgb
            print("🔄 LightGBM eğitiliyor...")
            self.lgbm_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            self.lgbm_model.fit(X_train, y_train)
            lgbm_pred = self.lgbm_model.predict(X_test)
            lgbm_acc  = accuracy_score(y_test, lgbm_pred)
            results["lightgbm"] = lgbm_acc
            print(f"✅ LightGBM: %{lgbm_acc*100:.1f} doğruluk")
        except ImportError:
            print("⚠️ LightGBM yüklü değil")
        except Exception as e:
            print(f"❌ LightGBM hatası: {e}")
        
        # 3. Random Forest
        try:
            print("🔄 Random Forest eğitiliyor...")
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(X_train, y_train)
            rf_pred = self.rf_model.predict(X_test)
            rf_acc  = accuracy_score(y_test, rf_pred)
            results["random_forest"] = rf_acc
            print(f"✅ Random Forest: %{rf_acc*100:.1f} doğruluk")
        except Exception as e:
            print(f"❌ RF hatası: {e}")
        
        self.is_trained = True
        self.stats = {
            "trained_at":    datetime.now().isoformat(),
            "n_samples":     len(X),
            "n_features":    len(feature_cols),
            "model_results": results,
            "avg_accuracy":  float(np.mean(list(results.values()))) if results else 0
        }
        
        print(f"\n🏆 Ortalama Doğruluk: %{self.stats['avg_accuracy']*100:.1f}")
        return self.stats
    
    def predict(self, df_features: pd.DataFrame) -> dict:
        """Tek hisse için tahmin yap"""
        if not self.is_trained:
            return {"error": "Model henüz eğitilmedi"}
        
        try:
            # Feature hazırla
            available = [c for c in self.feature_cols if c in df_features.columns]
            if len(available) < len(self.feature_cols) * 0.8:
                return {"error": "Yetersiz feature"}
            
            X = df_features[available].iloc[-1:].fillna(0).values
            
            probs = []
            
            # XGBoost tahmini
            if self.xgb_model is not None:
                try:
                    p = self.xgb_model.predict_proba(X)[0][1]
                    probs.append(("XGBoost", p))
                except: pass
            
            # LightGBM tahmini
            if self.lgbm_model is not None:
                try:
                    p = self.lgbm_model.predict_proba(X)[0][1]
                    probs.append(("LightGBM", p))
                except: pass
            
            # RF tahmini
            if self.rf_model is not None:
                try:
                    p = self.rf_model.predict_proba(X)[0][1]
                    probs.append(("RandomForest", p))
                except: pass
            
            if not probs:
                return {"error": "Hiç model aktif değil"}
            
            # Ensemble: ortalama
            avg_prob = float(np.mean([p for _, p in probs]))
            
            return {
                "probability": avg_prob,
                "signal":      "AL" if avg_prob >= 0.60 else "NÖTR" if avg_prob >= 0.45 else "SAT",
                "confidence":  "Yüksek" if avg_prob >= 0.70 or avg_prob <= 0.30 else "Orta",
                "models":      {name: round(p*100, 1) for name, p in probs},
                "ensemble":    round(avg_prob * 100, 1)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def save(self):
        """Model istatistiklerini kaydet - ml_results.json ve model_stats.json ikisine birden yaz"""
        try:
            # Ana sonuç dosyası
            with open(ML_RESULTS_FILE, "w") as f:
                json.dump(self.stats, f, indent=2)
            
            # Yedek istatistik dosyası
            with open(MODEL_STATS_FILE, "w") as f:
                json.dump(self.stats, f, indent=2)
            
            print(f"✅ Sonuçlar kaydedildi: {ML_RESULTS_FILE}")
            
            # Gerçek model nesnesini joblib ile kaydet
            try:
                import joblib
                model_data = {
                    "xgb_model": self.xgb_model,
                    "lgbm_model": self.lgbm_model,
                    "rf_model": self.rf_model,
                    "feature_cols": self.feature_cols,
                    "stats": self.stats
                }
                joblib.dump(model_data, TRAINED_MODEL, compress=3)
                print(f"✅ Eğitilmiş model kaydedildi: {TRAINED_MODEL}")
            except Exception as je:
                print(f"⚠️ Model dosyası kaydedilemedi: {je}")
                
        except Exception as e:
            print(f"Model save hatası: {e}")
    
    def load_stats(self) -> dict:
        """Kaydedilmiş istatistikleri yükle"""
        try:
            # Öncelikli ml_results.json'dan yükle
            if os.path.exists(ML_RESULTS_FILE):
                with open(ML_RESULTS_FILE, "r") as f:
                    return json.load(f)
            # Yoksa yedek dosyadan dene
            elif os.path.exists(MODEL_STATS_FILE):
                with open(MODEL_STATS_FILE, "r") as f:
                    return json.load(f)
        except:
            pass
        return {}
    
    def load_model(self):
        """Kaydedilmiş eğitilmiş modeli yükle"""
        try:
            import joblib
            if os.path.exists(TRAINED_MODEL):
                model_data = joblib.load(TRAINED_MODEL)
                self.xgb_model = model_data.get("xgb_model")
                self.lgbm_model = model_data.get("lgbm_model")
                self.rf_model = model_data.get("rf_model")
                self.feature_cols = model_data.get("feature_cols")
                self.stats = model_data.get("stats", {})
                self.is_trained = True
                print(f"✅ Kaydedilmiş model yüklendi: {TRAINED_MODEL}")
                return True
        except Exception as e:
            print(f"⚠️ Model yüklenemedi: {e}")
        return False

# Global model instance
_ml_model = PaladinMLModel()

def get_ml_model() -> PaladinMLModel:
    return _ml_model

def train_ml_model(tickers: list) -> dict:
    """ML modelini eğit - uzun sürer!"""
    print(f"🚀 ML Eğitimi başlıyor: {len(tickers)} hisse")
    print("⏳ Bu işlem 15-30 dakika sürebilir...")
    
    X, y, feature_cols = prepare_dataset(tickers)
    
    if X is None:
        return {"error": "Dataset hazırlanamadı"}
    
    model  = get_ml_model()
    stats  = model.train(X, y, feature_cols)
    model.save()
    
    return stats

def predict_stock(ticker: str) -> dict:
    """Tek hisse tahmin"""
    model = get_ml_model()
    if not model.is_trained:
        return {"error": "Model eğitilmedi. /mlegit komutunu çalıştır."}
    
    df = fetch_training_data(ticker, years=1)
    if df is None:
        return {"error": f"{ticker} için veri alınamadı"}
    
    df_feat = create_features(df)
    df_feat = df_feat.dropna()
    
    if len(df_feat) < 10:
        return {"error": "Yetersiz veri"}
    
    return model.predict(df_feat)
