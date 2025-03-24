import os
import re
import secrets
import logging
import warnings
import datetime as dt
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import feedparser
import urllib.parse
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from transformers import pipeline
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mysql.connector
import bcrypt
import json
import plotly.express as px
import chart_studio.plotly as py
import chart_studio.tools as tls

# 載入環境變數
load_dotenv()

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# 建立需要的資料夾
for path in ['static/charts', 'static/data']:
    os.makedirs(path, exist_ok=True)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# 設定 Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
general_model = genai.GenerativeModel("models/gemini-1.5-pro")
portfolio_model = genai.GenerativeModel("models/gemini-2.0-flash-thinking-exp")

# 資料庫連接設定
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "0912559910"),
            database=os.getenv("DB_NAME", "testdb")
        )
        return conn
    except Exception as e:
        logging.error(f"資料庫連接錯誤: {e}")
        return None

# 初始化資料庫表格
def init_database():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        
        # 創建用戶表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            email VARCHAR(100) UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 創建追蹤清單表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            ticker VARCHAR(20) NOT NULL,
            name VARCHAR(100) NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, ticker)
        )
        ''')
        
        # 創建設定表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            dark_mode BOOLEAN DEFAULT TRUE,
            font_size VARCHAR(10) DEFAULT 'medium',
            price_alert BOOLEAN DEFAULT FALSE,
            market_summary BOOLEAN DEFAULT TRUE,
            data_source VARCHAR(20) DEFAULT 'default',
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id)
        )
        ''')
        
        # 創建股票數據表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20) UNIQUE NOT NULL,
            name VARCHAR(100) NOT NULL,
            market VARCHAR(10) NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ma5 FLOAT DEFAULT NULL,
            ma20 FLOAT DEFAULT NULL,
            ma50 FLOAT DEFAULT NULL,
            ma120 FLOAT DEFAULT NULL,
            ma200 FLOAT DEFAULT NULL,
            bb_upper FLOAT DEFAULT NULL,
            bb_middle FLOAT DEFAULT NULL,
            bb_lower FLOAT DEFAULT NULL,
            rsi FLOAT DEFAULT NULL,
            wmsr FLOAT DEFAULT NULL,
            psy FLOAT DEFAULT NULL,
            bias6 FLOAT DEFAULT NULL,
            macd FLOAT DEFAULT NULL,
            macd_signal FLOAT DEFAULT NULL,
            macd_hist FLOAT DEFAULT NULL,
            k FLOAT DEFAULT NULL,
            d FLOAT DEFAULT NULL,
            j FLOAT DEFAULT NULL,
            pe_ratio FLOAT DEFAULT NULL,
            market_cap BIGINT DEFAULT NULL,
            open_price FLOAT DEFAULT NULL,
            close_price FLOAT DEFAULT NULL,
            high_price FLOAT DEFAULT NULL,
            low_price FLOAT DEFAULT NULL,
            volume BIGINT DEFAULT NULL
        )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        logging.info("資料庫初始化成功")

# ------------------ 股票分析功能 ------------------
class StockAnalyzer:
    def __init__(self, ticker: str, api_key: str, period: str = "10y", market: str = "TW"):
        self.ticker = ticker.strip()
        if market == "TW" and "." not in self.ticker:
            self.ticker = f"{self.ticker}.TW"
        self.period = period
        self.market = market
        self.stock = yf.Ticker(self.ticker)
        self.data = None
        self.company_name = None
        self.currency = None
        self.pe_ratio = None
        self.market_cap = None
        self.forward_pe = None
        self.profit_margins = None
        self.eps = None
        self.roe = None
        self.financials_head = None
        self.balance_sheet_head = None
        self.cashflow_head = None
        self.net_profit_margin_str = None
        self.current_ratio_str = None

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-2.0-flash-thinking-exp")
        self.sentiment_analyzer = pipeline('sentiment-analysis', model='yiyanghkust/finbert-tone')

        self._get_data()
        self._get_financial_data()
        self._calculate_indicators()
        self._update_db_data()

    def _get_data(self):
        try:
            self.data = self.stock.history(period=self.period)
            if self.data.empty:
                raise ValueError(f"無法取得 {self.ticker} 的資料，請確認股票代碼是否正確")
            company_info = self.stock.info
            self.company_name = company_info.get('longName', self.ticker)
            self.currency = company_info.get('currency', 'TWD' if self.market == 'TW' else 'USD')
            logging.info("成功取得 %s 的股票資料", self.ticker)
        except Exception as e:
            logging.error("取得股票資料時發生錯誤: %s", e)
            raise

    def _get_financial_data(self):
        try:
            info = self.stock.info
            self.pe_ratio = info.get('trailingPE', 'N/A')
            self.market_cap = info.get('marketCap', 'N/A')
            self.forward_pe = info.get('forwardPE', 'N/A')
            self.profit_margins = info.get('profitMargins', 'N/A')
            self.eps = info.get('trailingEps', 'N/A')

            annual_financials = self.stock.financials
            annual_balance_sheet = self.stock.balance_sheet
            annual_cashflow = self.stock.cashflow

            self.financials_head = annual_financials.head().to_string()
            self.balance_sheet_head = annual_balance_sheet.head().to_string()
            self.cashflow_head = annual_cashflow.head().to_string()

            try:
                financials = self.stock.financials
                balance_sheet = self.stock.balance_sheet

                net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0
                equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else 0
                self.roe = net_income / equity if equity != 0 else 'N/A'

                if "Total Revenue" in annual_financials.index and "Net Income" in annual_financials.index:
                    revenue = annual_financials.loc["Total Revenue"]
                    net_income = annual_financials.loc["Net Income"]
                    net_profit_margin = (net_income / revenue) * 100
                    net_profit_margin_value = net_profit_margin.iloc[0]
                    self.net_profit_margin_str = f"{net_profit_margin_value:.2f}%"
                else:
                    self.net_profit_margin_str = "無法計算（缺少 Total Revenue 或 Net Income 數據）"

                if ("Total Current Assets" in annual_balance_sheet.index and
                    "Total Current Liabilities" in annual_balance_sheet.index):
                    current_assets = annual_balance_sheet.loc["Total Current Assets"]
                    current_liabilities = annual_balance_sheet.loc["Total Current Liabilities"]
                    current_ratio = current_assets / current_liabilities
                    current_ratio_value = current_ratio.iloc[0]
                    self.current_ratio_str = f"{current_ratio_value:.2f}"
                else:
                    self.current_ratio_str = "無法計算（缺少 Total Current Assets 或 Total Current Liabilities 數據）"

            except Exception as inner_e:
                logging.error("計算財務指標時發生錯誤: %s", inner_e)
                self.roe = 'N/A'
                self.net_profit_margin_str = 'N/A'
                self.current_ratio_str = 'N/A'

            logging.info("成功取得 %s 的財務資料", self.ticker)

        except Exception as e:
            logging.error("取得財務資料時發生錯誤: %s", e)
            raise

    def _calculate_indicators(self):
        try:
            df = self.data.copy()
            df['MA5'] = ta.sma(df['Close'], length=5)
            df['MA20'] = ta.sma(df['Close'], length=20)
            df['MA50'] = ta.sma(df['Close'], length=50)
            df['MA120'] = ta.sma(df['Close'], length=120)
            df['MA200'] = ta.sma(df['Close'], length=200)
            df['RSI'] = ta.rsi(df['Close'], length=12)
            macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            df['MACD'] = macd_df['MACD_12_26_9']
            df['MACD_signal'] = macd_df['MACDs_12_26_9']
            df['MACD_hist'] = macd_df['MACDh_12_26_9']
            stoch_df = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3, smooth_k=3)
            df['K'] = stoch_df['STOCHk_9_3_3']
            df['D'] = stoch_df['STOCHd_9_3_3']
            df['J'] = 3 * df['K'] - 2 * df['D']
            bbands = ta.bbands(df['Close'], length=20, std=2)
            df['BB_lower'] = bbands['BBL_20_2.0']
            df['BB_middle'] = bbands['BBM_20_2.0']
            df['BB_upper'] = bbands['BBU_20_2.0']
            df['WMSR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
            df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
            
            # 計算布林帶寬度
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            
            # 計算成交量變化率
            df['Volume_Change'] = df['Volume'].pct_change() * 100
            
            # 計算價格動量
            df['Momentum'] = df['Close'] - df['Close'].shift(10)
            
            # 計算波動率 (20日標準差)
            df['Volatility'] = df['Close'].rolling(window=20).std()
            
            # 計算心理線指標 (PSY)
            df['PSY'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else 0).rolling(12).sum() / 12 * 100
            
            # 計算乖離率 (BIAS6)
            df['BIAS6'] = (df['Close'] - df['Close'].rolling(window=6).mean()) / df['Close'].rolling(window=6).mean() * 100
            
            self.data = df
            logging.info("成功計算 %s 的技術指標", self.ticker)
        except Exception as e:
            logging.error("計算技術指標時發生錯誤: %s", e)
            raise

    def _update_db_data(self):
        """更新資料庫中的股票數據"""
        try:
            conn = get_db_connection()
            if not conn:
                logging.error("無法連接資料庫，跳過數據更新")
                return
                
            cursor = conn.cursor()
            
            # 檢查股票是否已存在
            cursor.execute("SELECT id FROM stocks WHERE symbol = %s", (self.ticker,))
            result = cursor.fetchone()
            
            latest_data = self.data.iloc[-1]
            
            # 準備數據
            stock_data = {
                'symbol': self.ticker,
                'name': self.company_name,
                'market': self.market,
                'ma5': float(latest_data['MA5']) if not pd.isna(latest_data['MA5']) else None,
                'ma20': float(latest_data['MA20']) if not pd.isna(latest_data['MA20']) else None,
                'ma50': float(latest_data['MA50']) if not pd.isna(latest_data['MA50']) else None,
                'ma120': float(latest_data['MA120']) if not pd.isna(latest_data['MA120']) else None,
                'ma200': float(latest_data['MA200']) if not pd.isna(latest_data['MA200']) else None,
                'bb_upper': float(latest_data['BB_upper']) if not pd.isna(latest_data['BB_upper']) else None,
                'bb_middle': float(latest_data['BB_middle']) if not pd.isna(latest_data['BB_middle']) else None,
                'bb_lower': float(latest_data['BB_lower']) if not pd.isna(latest_data['BB_lower']) else None,
                'rsi': float(latest_data['RSI']) if not pd.isna(latest_data['RSI']) else None,
                'wmsr': float(latest_data['WMSR']) if not pd.isna(latest_data['WMSR']) else None,
                'psy': float(latest_data['PSY']) if not pd.isna(latest_data['PSY']) else None,
                'bias6': float(latest_data['BIAS6']) if not pd.isna(latest_data['BIAS6']) else None,
                'macd': float(latest_data['MACD']) if not pd.isna(latest_data['MACD']) else None,
                'macd_signal': float(latest_data['MACD_signal']) if not pd.isna(latest_data['MACD_signal']) else None,
                'macd_hist': float(latest_data['MACD_hist']) if not pd.isna(latest_data['MACD_hist']) else None,
                'k': float(latest_data['K']) if not pd.isna(latest_data['K']) else None,
                'd': float(latest_data['D']) if not pd.isna(latest_data['D']) else None,
                'j': float(latest_data['J']) if not pd.isna(latest_data['J']) else None,
                'pe_ratio': float(self.pe_ratio) if isinstance(self.pe_ratio, (int, float)) else None,
                'market_cap': int(self.market_cap) if isinstance(self.market_cap, (int, float)) else None,
                'open_price': float(latest_data['Open']) if not pd.isna(latest_data['Open']) else None,
                'close_price': float(latest_data['Close']) if not pd.isna(latest_data['Close']) else None,
                'high_price': float(latest_data['High']) if not pd.isna(latest_data['High']) else None,
                'low_price': float(latest_data['Low']) if not pd.isna(latest_data['Low']) else None,
                'volume': int(latest_data['Volume']) if not pd.isna(latest_data['Volume']) else None
            }
            
            if result:
                # 更新現有記錄
                update_query = """
                UPDATE stocks SET 
                    name = %s, market = %s, last_updated = NOW(),
                    ma5 = %s, ma20 = %s, ma50 = %s, ma120 = %s, ma200 = %s,
                    bb_upper = %s, bb_middle = %s, bb_lower = %s,
                    rsi = %s, wmsr = %s, psy = %s, bias6 = %s,
                    macd = %s, macd_signal = %s, macd_hist = %s,
                    k = %s, d = %s, j = %s,
                    pe_ratio = %s, market_cap = %s,
                    open_price = %s, close_price = %s, high_price = %s, low_price = %s, volume = %s
                WHERE symbol = %s
                """
                cursor.execute(update_query, (
                    stock_data['name'], stock_data['market'],
                    stock_data['ma5'], stock_data['ma20'], stock_data['ma50'], stock_data['ma120'], stock_data['ma200'],
                    stock_data['bb_upper'], stock_data['bb_middle'], stock_data['bb_lower'],
                    stock_data['rsi'], stock_data['wmsr'], stock_data['psy'], stock_data['bias6'],
                    stock_data['macd'], stock_data['macd_signal'], stock_data['macd_hist'],
                    stock_data['k'], stock_data['d'], stock_data['j'],
                    stock_data['pe_ratio'], stock_data['market_cap'],
                    stock_data['open_price'], stock_data['close_price'], stock_data['high_price'], stock_data['low_price'], stock_data['volume'],
                    self.ticker
                ))
            else:
                # 插入新記錄
                insert_query = """
                INSERT INTO stocks (
                    symbol, name, market, last_updated,
                    ma5, ma20, ma50, ma120, ma200,
                    bb_upper, bb_middle, bb_lower,
                    rsi, wmsr, psy, bias6,
                    macd, macd_signal, macd_hist,
                    k, d, j,
                    pe_ratio, market_cap,
                    open_price, close_price, high_price, low_price, volume
                ) VALUES (
                    %s, %s, %s, NOW(),
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s, %s, %s
                )
                """
                cursor.execute(insert_query, (
                    self.ticker, stock_data['name'], stock_data['market'],
                    stock_data['ma5'], stock_data['ma20'], stock_data['ma50'], stock_data['ma120'], stock_data['ma200'],
                    stock_data['bb_upper'], stock_data['bb_middle'], stock_data['bb_lower'],
                    stock_data['rsi'], stock_data['wmsr'], stock_data['psy'], stock_data['bias6'],
                    stock_data['macd'], stock_data['macd_signal'], stock_data['macd_hist'],
                    stock_data['k'], stock_data['d'], stock_data['j'],
                    stock_data['pe_ratio'], stock_data['market_cap'],
                    stock_data['open_price'], stock_data['close_price'], stock_data['high_price'], stock_data['low_price'], stock_data['volume']
                ))
            
            conn.commit()
            cursor.close()
            conn.close()
            logging.info("成功更新 %s 的資料庫數據", self.ticker)
        except Exception as e:
            logging.error("更新資料庫數據時發生錯誤: %s", e)

    def _identify_patterns(self, days=30):
        """識別最近的技術形態"""
        try:
            df = self.data.tail(days).copy()
            patterns = []
            
            # 黃金交叉 (MA5 上穿 MA20)
            if (df['MA5'].iloc[-2] <= df['MA20'].iloc[-2]) and (df['MA5'].iloc[-1] > df['MA20'].iloc[-1]):
                patterns.append("黃金交叉 (短期均線上穿長期均線)")
            
            # 死亡交叉 (MA5 下穿 MA20)
            if (df['MA5'].iloc[-2] >= df['MA20'].iloc[-2]) and (df['MA5'].iloc[-1] < df['MA20'].iloc[-1]):
                patterns.append("死亡交叉 (短期均線下穿長期均線)")
            
            # 突破布林帶上軌
            if df['Close'].iloc[-1] > df['BB_upper'].iloc[-1] and df['Close'].iloc[-2] <= df['BB_upper'].iloc[-2]:
                patterns.append("突破布林帶上軌 (可能超買)")
            
            # 跌破布林帶下軌
            if df['Close'].iloc[-1] < df['BB_lower'].iloc[-1] and df['Close'].iloc[-2] >= df['BB_lower'].iloc[-2]:
                patterns.append("跌破布林帶下軌 (可能超賣)")
            
            # MACD 金叉
            if (df['MACD'].iloc[-2] <= df['MACD_signal'].iloc[-2]) and (df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]):
                patterns.append("MACD 金叉 (看漲信號)")
            
            # MACD 死叉
            if (df['MACD'].iloc[-2] >= df['MACD_signal'].iloc[-2]) and (df['MACD'].iloc[-1] < df['MACD_signal'].iloc[-1]):
                patterns.append("MACD 死叉 (看跌信號)")
            
            # KDJ 金叉
            if (df['K'].iloc[-2] <= df['D'].iloc[-2]) and (df['K'].iloc[-1] > df['D'].iloc[-1]):
                patterns.append("KDJ 金叉 (看漲信號)")
            
            # KDJ 死叉
            if (df['K'].iloc[-2] >= df['D'].iloc[-2]) and (df['K'].iloc[-1] < df['D'].iloc[-1]):
                patterns.append("KDJ 死叉 (看跌信號)")
            
            # RSI 超買
            if df['RSI'].iloc[-1] > 70:
                patterns.append("RSI 超買 (可能即將回檔)")
            
            # RSI 超賣
            if df['RSI'].iloc[-1] < 30:
                patterns.append("RSI 超賣 (可能即將反彈)")
            
            # 頭肩頂形態 (簡化版)
            if len(df) >= 20:
                recent_highs = df['High'].rolling(5).max()
                if (recent_highs.iloc[-20] < recent_highs.iloc[-15] > recent_highs.iloc[-10] < recent_highs.iloc[-5] > recent_highs.iloc[-1]):
                    patterns.append("可能形成頭肩頂形態 (看跌)")
            
            # 頭肩底形態 (簡化版)
            if len(df) >= 20:
                recent_lows = df['Low'].rolling(5).min()
                if (recent_lows.iloc[-20] > recent_lows.iloc[-15] < recent_lows.iloc[-10] > recent_lows.iloc[-5] < recent_lows.iloc[-1]):
                    patterns.append("可能形成頭肩底形態 (看漲)")
            
            # 雙頂形態 (簡化版)
            if len(df) >= 15:
                recent_highs = df['High'].rolling(3).max()
                if abs(recent_highs.iloc[-15] - recent_highs.iloc[-5]) / recent_highs.iloc[-15] < 0.03 and recent_highs.iloc[-10] < recent_highs.iloc[-15]:
                    patterns.append("可能形成雙頂形態 (看跌)")
            
            # 雙底形態 (簡化版)
            if len(df) >= 15:
                recent_lows = df['Low'].rolling(3).min()
                if abs(recent_lows.iloc[-15] - recent_lows.iloc[-5]) / recent_lows.iloc[-15] < 0.03 and recent_lows.iloc[-10] > recent_lows.iloc[-15]:
                    patterns.append("可能形成雙底形態 (看漲)")
            
            return patterns
        except Exception as e:
            logging.error("識別技術形態時發生錯誤: %s", e)
            return ["無法識別技術形態"]

    def _generate_chart(self, days=180):
        """生成股票走勢圖"""
        try:
            df = self.data.tail(days).copy()
            
            # 創建子圖
            fig = make_subplots(rows=4, cols=1, 
                               shared_xaxes=True, 
                               vertical_spacing=0.05, 
                               row_heights=[0.5, 0.15, 0.15, 0.2])
            
            # 添加K線圖
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='K線'
            ), row=1, col=1)
            
            # 添加移動平均線
            fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], name='MA5', line=dict(color='orange', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='blue', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA120'], name='MA120', line=dict(color='green', width=1)), row=1, col=1)
            
            # 添加布林帶
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='布林上軌', line=dict(color='rgba(173, 204, 255, 0.7)', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='布林下軌', line=dict(color='rgba(173, 204, 255, 0.7)', width=1, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], name='布林中軌', line=dict(color='rgba(173, 204, 255, 0.7)', width=1)), row=1, col=1)
            
            # 添加成交量
            colors = ['red' if df['Close'].iloc[i] > df['Open'].iloc[i] else 'green' for i in range(len(df))]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors), row=2, col=1)
            
            # 添加MACD
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='MACD信號', line=dict(color='red', width=1)), row=3, col=1)
            
            colors = ['red' if val >= 0 else 'green' for val in df['MACD_hist']]
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD柱狀', marker_color=colors), row=3, col=1)
            
            # 添加KDJ
            fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K值', line=dict(color='blue', width=1)), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['D'], name='D值', line=dict(color='red', width=1)), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['J'], name='J值', line=dict(color='green', width=1)), row=4, col=1)
            
            # 更新佈局
            fig.update_layout(
                title=f'{self.company_name} ({self.ticker}) 技術分析圖',
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                height=800,
                width=1000,
                margin=dict(l=50, r=50, t=80, b=50),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.2)',
                font=dict(color='white')
            )
            
            # 更新Y軸格式
            fig.update_yaxes(title_text='價格', row=1, col=1)
            fig.update_yaxes(title_text='成交量', row=2, col=1)
            fig.update_yaxes(title_text='MACD', row=3, col=1)
            fig.update_yaxes(title_text='KDJ', row=4, col=1)
            
            # 保存圖表
            chart_path = f"static/charts/{self.ticker.replace('.', '_')}_chart.html"
            fig.write_html(chart_path)
            
            return chart_path
        except Exception as e:
            logging.error("生成圖表時發生錯誤: %s", e)
            raise

    def get_stock_summary(self):
        """獲取股票綜合分析"""
        try:
            # 計算技術指標
            latest = self.data.iloc[-1]
            prev = self.data.iloc[-2]
            
            # 計算漲跌幅
            price_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
            price_change_str = f"{price_change:.2f}%"
            
            # 獲取技術形態
            patterns = self._identify_patterns()
            
            # 生成圖表
            chart_path = self._generate_chart()
            
            # 獲取最新新聞
            news = self._get_stock_news()
            
            # 生成AI分析
            analysis = self._get_ai_analysis()
            
            return {
                "ticker": self.ticker,
                "company_name": self.company_name,
                "currency": self.currency,
                "current_price": latest['Close'],
                "price_change": price_change_str,
                "price_change_value": price_change,
                "volume": latest['Volume'],
                "pe_ratio": self.pe_ratio,
                "market_cap": self.market_cap,
                "eps": self.eps,
                "roe": self.roe if isinstance(self.roe, str) else f"{self.roe:.2%}" if self.roe is not None else "N/A",
                "net_profit_margin": self.net_profit_margin_str,
                "current_ratio": self.current_ratio_str,
                "rsi": latest['RSI'],
                "macd": latest['MACD'],
                "macd_signal": latest['MACD_signal'],
                "k": latest['K'],
                "d": latest['D'],
                "j": latest['J'],
                "patterns": patterns,
                "chart_path": chart_path,
                "news": news,
                "analysis": analysis
            }
        except Exception as e:
            logging.error("獲取股票綜合分析時發生錯誤: %s", e)
            raise

    def _get_stock_news(self, max_news=5):
        """獲取相關股票新聞"""
        try:
            # 使用 Google News RSS
            search_term = f"{self.company_name} stock" if self.market == "US" else f"{self.company_name} 股票"
            rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(search_term)}&hl={'en-US' if self.market == 'US' else 'zh-TW'}&gl={'US' if self.market == 'US' else 'TW'}&ceid={'US:en' if self.market == 'US' else 'TW:zh-Hant'}"
            
            feed = feedparser.parse(rss_url)
            news_list = []
            
            for entry in feed.entries[:max_news]:
                try:
                    published_time = dt.datetime(*entry.published_parsed[:6])
                    
                    # 使用 FinBERT 進行情緒分析
                    result = self.sentiment_analyzer(entry.title)[0]
                    
                    news_entry = {
                        'title': entry.title,
                        'link': entry.link,
                        'date': published_time.strftime('%Y-%m-%d'),
                        'source': entry.source.title if hasattr(entry, 'source') else 'Google News',
                        'sentiment': result['label'],
                        'sentiment_score': result['score']
                    }
                    
                    news_list.append(news_entry)
                except Exception as inner_e:
                    logging.error(f"處理新聞條目時發生錯誤: {inner_e}")
                    continue
                    
            return news_list
        except Exception as e:
            logging.error(f"獲取股票新聞時發生錯誤: {e}")
            return []

    def _get_ai_analysis(self):
        """使用Gemini生成AI分析"""
        try:
            latest = self.data.iloc[-1]
            prev_day = self.data.iloc[-2]
            prev_week = self.data.iloc[-6] if len(self.data) >= 6 else self.data.iloc[0]
            prev_month = self.data.iloc[-23] if len(self.data) >= 23 else self.data.iloc[0]
            
            # 計算各種漲跌幅
            daily_change = ((latest['Close'] - prev_day['Close']) / prev_day['Close']) * 100
            weekly_change = ((latest['Close'] - prev_week['Close']) / prev_week['Close']) * 100
            monthly_change = ((latest['Close'] - prev_month['Close']) / prev_month['Close']) * 100
            
            # 準備提示詞
            prompt = f"""
            請分析以下股票數據並提供專業的投資建議：
            
            股票：{self.company_name} ({self.ticker})
            市場：{'台股' if self.market == 'TW' else '美股'}
            當前價格：{latest['Close']} {self.currency}
            
            技術指標：
            - RSI: {latest['RSI']:.2f}
            - MACD: {latest['MACD']:.4f}
            - KD值: K={latest['K']:.2f}, D={latest['D']:.2f}
            - 布林帶: 上軌={latest['BB_upper']:.2f}, 中軌={latest['BB_middle']:.2f}, 下軌={latest['BB_lower']:.2f}
            
            價格變動：
            - 日漲跌: {daily_change:.2f}%
            - 週漲跌: {weekly_change:.2f}%
            - 月漲跌: {monthly_change:.2f}%
            
            基本面數據：
            - 本益比(P/E): {self.pe_ratio if isinstance(self.pe_ratio, str) else f"{self.pe_ratio:.2f}" if self.pe_ratio is not None else "N/A"}
            - 市值: {self.market_cap if isinstance(self.market_cap, str) else f"{self.market_cap:,}" if self.market_cap is not None else "N/A"}
            - EPS: {self.eps if isinstance(self.eps, str) else f"{self.eps:.2f}" if self.eps is not None else "N/A"}
            - ROE: {self.roe if isinstance(self.roe, str) else f"{self.roe:.2%}" if self.roe is not None else "N/A"}
            - 淨利潤率: {self.net_profit_margin_str}
            - 流動比率: {self.current_ratio_str}
            
            請提供以下分析：
            1. 技術面分析：目前股價位於什麼位置？技術指標顯示什麼信號？
            2. 基本面分析：公司財務狀況如何？估值是否合理？
            3. 短期展望（1-4週）
            4. 中長期展望（1-6個月）
            5. 投資建議（買入/賣出/持有）及理由
            
            請簡潔有力地回答，使用繁體中文，並注意分析的專業性和客觀性。
            """
            
            response = self.model.generate_content(prompt)
            analysis = response.text
            
            return analysis
        except Exception as e:
            logging.error(f"生成AI分析時發生錯誤: {e}")
            return "無法生成AI分析，請稍後再試。"

# ------------------ API 路由 ------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis/<ticker>')
def analysis(ticker):
    market = request.args.get('market', 'TW')
    return render_template('analysis.html', ticker=ticker, market=market)

@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip()
        market = data.get('market', 'TW')
        
        if not ticker:
            return jsonify({'error': '請提供股票代碼'}), 400
            
        analyzer = StockAnalyzer(ticker, GEMINI_API_KEY, period="5y", market=market)
        summary = analyzer.get_stock_summary()
        
        return jsonify({
            'success': True,
            'data': summary
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error("分析股票時發生錯誤: %s", e)
        return jsonify({'error': f"分析股票時發生錯誤: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        market = data.get('market', 'TW')
        
        if not message:
            return jsonify({'error': '請輸入訊息'}), 400
            
        # 檢查是否包含股票代碼查詢
        stock_match = re.search(r'[#＃]([0-9A-Za-z\.]+)', message)
        if stock_match:
            ticker = stock_match.group(1)
            try:
                analyzer = StockAnalyzer(ticker, GEMINI_API_KEY, period="5y", market=market)
                summary = analyzer.get_stock_summary()
                
                # 添加到追蹤清單
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    # 暫時使用固定user_id=1，未來應整合登入系統
                    user_id = 1
                    
                    # 檢查是否已存在
                    cursor.execute('''
                        SELECT id FROM watchlist 
                        WHERE user_id = %s AND ticker = %s
                    ''', (user_id, analyzer.ticker))
                    
                    if not cursor.fetchone():
                        # 添加到追蹤清單
                        cursor.execute('''
                            INSERT INTO watchlist (user_id, ticker, name)
                            VALUES (%s, %s, %s)
                        ''', (user_id, analyzer.ticker, analyzer.company_name))
                        conn.commit()
                    
                    cursor.close()
                    conn.close()
                
                return jsonify({
                    'success': True,
                    'type': 'stock',
                    'data': summary
                })
            except Exception as e:
                logging.error(f"處理股票查詢時發生錯誤: {e}")
                return jsonify({
                    'success': True,
                    'type': 'text',
                    'data': f"無法查詢股票 {ticker}，請確認代碼是否正確。錯誤: {str(e)}"
                })
        
        # 檢查是否為投資組合優化請求
        if re.search(r'(投資|投組|資產|配置|組合).*(優化|建議|推薦|分配)', message):
            try:
                response = portfolio_model.generate_content(f"""
                使用者想要投資組合優化建議。請詢問他們的風險承受度、投資期限和投資目標，
                然後根據當前市場環境提供適合的資產配置建議。
                
                使用者的訊息: {message}
                
                請用繁體中文回覆，並提供具體的資產配置比例建議。
                """)
                
                return jsonify({
                    'success': True,
                    'type': 'text',
                    'data': response.text
                })
            except Exception as e:
                logging.error(f"處理投資組合優化請求時發生錯誤: {e}")
                return jsonify({
                    'success': True,
                    'type': 'text',
                    'data': "無法處理投資組合優化請求，請稍後再試。"
                })
        
        # 一般查詢，使用Gemini模型
        try:
            market_str = "台股" if market == "TW" else "美股"
            response = general_model.generate_content(f"""
            你是一位專業的股票分析師和投資顧問，專精於{market_str}市場分析。
            使用者的訊息: {message}
            
            請用繁體中文回覆，提供專業、有見地且具體的回答。如果使用者詢問特定股票，
            可以建議他們使用 #股票代碼 的格式來查詢詳細資訊。
            """)
            
            return jsonify({
                'success': True,
                'type': 'text',
                'data': response.text
            })
        except Exception as e:
            logging.error(f"使用Gemini處理訊息時發生錯誤: {e}")
            return jsonify({
                'success': True,
                'type': 'text',
                'data': "抱歉，我無法處理您的請求，請稍後再試。"
            })
    except Exception as e:
        logging.error("處理聊天請求時發生錯誤: %s", e)
        return jsonify({'error': f"處理聊天請求時發生錯誤: {str(e)}"}), 500

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    try:
        # 從資料庫獲取資料
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '無法連接資料庫'}), 500
            
        cursor = conn.cursor(dictionary=True)
        
        # 暫時使用固定user_id=1，未來應整合登入系統
        user_id = 1
        cursor.execute('''
            SELECT w.ticker, w.name FROM watchlist w
            WHERE w.user_id = %s
        ''', (user_id,))
        
        watchlist_items = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # 獲取最新價格數據
        watchlist = []
        for item in watchlist_items:
            ticker = item['ticker']
            try:
                # 優先從資料庫獲取數據
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor(dictionary=True)
                    cursor.execute('''
                        SELECT close_price, open_price 
                        FROM stocks 
                        WHERE symbol = %s
                    ''', (ticker,))
                    
                    stock_data = cursor.fetchone()
                    cursor.close()
                    conn.close()
                    
                    if stock_data and stock_data['close_price'] and stock_data['open_price']:
                        price = stock_data['close_price']
                        change = ((stock_data['close_price'] - stock_data['open_price']) / stock_data['open_price']) * 100
                        
                        watchlist.append({
                            'ticker': ticker,
                            'name': item['name'],
                            'price': price,
                            'change': change
                        })
                        continue
                
                # 如果資料庫中沒有數據，則從 yfinance 獲取
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2d")
                if not hist.empty and len(hist) >= 2:
                    current = hist.iloc[-1]
                    prev = hist.iloc[-2]
                    price = current['Close']
                    change = ((current['Close'] - prev['Close']) / prev['Close']) * 100
                else:
                    price = 0
                    change = 0
                    
                watchlist.append({
                    'ticker': ticker,
                    'name': item['name'],
                    'price': price,
                    'change': change
                })
            except Exception as e:
                logging.error(f"獲取股票 {ticker} 數據時出錯: {e}")
                # 添加錯誤項，但不中斷流程
                watchlist.append({
                    'ticker': ticker,
                    'name': item['name'],
                    'price': 0,
                    'change': 0,
                    'error': True
                })
        
        return jsonify({'watchlist': watchlist})
    except Exception as e:
        logging.error(f"獲取追蹤清單時發生錯誤: {e}")
        return jsonify({'error': f"獲取追蹤清單時發生錯誤: {str(e)}"}), 500

@app.route('/api/watchlist/add', methods=['POST'])
def add_to_watchlist():
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip()
        name = data.get('name', '').strip()
        
        if not ticker:
            return jsonify({'error': '請提供股票代碼'}), 400
        
        # 如果沒有提供名稱，嘗試獲取
        if not name:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                name = info.get('longName', ticker)
            except:
                name = ticker
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '無法連接資料庫'}), 500
            
        cursor = conn.cursor()
        
        # 暫時使用固定user_id=1，未來應整合登入系統
        user_id = 1
        
        # 檢查是否已存在
        cursor.execute('''
            SELECT id FROM watchlist 
            WHERE user_id = %s AND ticker = %s
        ''', (user_id, ticker))
        
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': True, 'message': f'{name} 已在追蹤清單中'})
        
        # 添加到追蹤清單
        cursor.execute('''
            INSERT INTO watchlist (user_id, ticker, name)
            VALUES (%s, %s, %s)
        ''', (user_id, ticker, name))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': f'已將 {name} 添加到追蹤清單'})
    except Exception as e:
        logging.error(f"添加到追蹤清單時發生錯誤: {e}")
        return jsonify({'error': f"添加到追蹤清單時發生錯誤: {str(e)}"}), 500

@app.route('/api/watchlist/remove', methods=['POST'])
def remove_from_watchlist():
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip()
        
        if not ticker:
            return jsonify({'error': '請提供股票代碼'}), 400
            
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '無法連接資料庫'}), 500
            
        cursor = conn.cursor()
        
        # 暫時使用固定user_id=1，未來應整合登入系統
        user_id = 1
        
        # 獲取股票名稱用於回應訊息
        cursor.execute('''
            SELECT name FROM watchlist 
            WHERE user_id = %s AND ticker = %s
        ''', (user_id, ticker))
        
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            return jsonify({'error': '該股票不在追蹤清單中'}), 404
            
        name = result[0]
        
        # 從追蹤清單中移除
        cursor.execute('''
            DELETE FROM watchlist 
            WHERE user_id = %s AND ticker = %s
        ''', (user_id, ticker))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': f'已從追蹤清單中移除 {name}'})
    except Exception as e:
        logging.error(f"從追蹤清單移除時發生錯誤: {e}")
        return jsonify({'error': f"從追蹤清單移除時發生錯誤: {str(e)}"}), 500

@app.route('/api/portfolio/optimize', methods=['POST'])
def optimize_portfolio():
    try:
        data = request.get_json()
        
        # 提取表單數據
        risk_level = data.get('risk_level', 'moderate')
        investment_amount = data.get('investment_amount', 10000)
        investment_period = data.get('investment_period', '1-5')
        investment_goal = data.get('investment_goal', 'growth')
        
        # 使用Gemini生成投資組合建議
        prompt = f"""
        請根據以下投資者信息提供詳細的投資組合配置建議：
        
        風險承受度：{risk_level}（保守/適中/積極）
        投資金額：{investment_amount} 元
        投資期限：{investment_period} 年
        投資目標：{investment_goal}（收入/成長/平衡）
        
        請提供：
        1. 資產類別配置比例（股票、債券、現金等）
        2. 不同市場的分配比例（台股、美股、其他國際市場）
        3. 具體的ETF或基金推薦（請提供實際代碼和名稱）
        4. 投資策略建議（定期定額、價值投資等）
        
        回答需要具體、實用，並考慮當前市場環境。請用繁體中文回覆。
        """
        
        response = portfolio_model.generate_content(prompt)
        portfolio_suggestion = response.text
        
        # 生成一個模擬的投資組合分配圖
        try:
            # 解析AI回應中的資產配置比例
            # 這裡使用簡單的正則表達式來提取百分比，實際應用中可能需要更複雜的解析
            stocks_match = re.search(r'股票[：:]\s*(\d+)%', portfolio_suggestion)
            bonds_match = re.search(r'債券[：:]\s*(\d+)%', portfolio_suggestion)
            cash_match = re.search(r'現金[：:]\s*(\d+)%', portfolio_suggestion)
            other_match = re.search(r'(其他|另類資產|黃金|房地產)[：:]\s*(\d+)%', portfolio_suggestion)
            
            stocks = int(stocks_match.group(1)) if stocks_match else 60
            bonds = int(bonds_match.group(1)) if bonds_match else 30
            cash = int(cash_match.group(1)) if cash_match else 10
            other = int(other_match.group(2)) if other_match else 0
            
            # 確保總和為100%
            total = stocks + bonds + cash + other
            if total != 100:
                # 調整比例
                factor = 100 / total
                stocks = int(stocks * factor)
                bonds = int(bonds * factor)
                cash = int(cash * factor)
                other = 100 - stocks - bonds - cash
            
            # 創建餅圖
            labels = ['股票', '債券', '現金']
            values = [stocks, bonds, cash]
            colors = ['#0066ff', '#00cc88', '#ffcc00']
            
            if other > 0:
                labels.append('其他資產')
                values.append(other)
                colors.append('#ff6b6b')
            
            fig = px.pie(
                values=values,
                names=labels,
                color_discrete_sequence=colors,
                title=f"投資組合配置 - {risk_level}風險"
            )
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=50, r=50, t=80, b=50),
            )
            
            # 生成唯一的文件名
            chart_id = secrets.token_hex(8)
            chart_path = f"static/charts/portfolio_{chart_id}.html"
            fig.write_html(chart_path)
            
            return jsonify({
                'success': True,
                'suggestion': portfolio_suggestion,
                'chart_path': chart_path,
                'allocation': {
                    'stocks': stocks,
                    'bonds': bonds,
                    'cash': cash,
                    'other': other
                }
            })
        except Exception as chart_error:
            logging.error(f"生成投資組合圖表時發生錯誤: {chart_error}")
            # 即使圖表生成失敗，仍返回文字建議
            return jsonify({
                'success': True,
                'suggestion': portfolio_suggestion,
                'chart_error': str(chart_error)
            })
    except Exception as e:
        logging.error(f"優化投資組合時發生錯誤: {e}")
        return jsonify({'error': f"優化投資組合時發生錯誤: {str(e)}"}), 500

@app.route('/api/settings', methods=['GET'])
def get_settings():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '無法連接資料庫'}), 500
            
        cursor = conn.cursor(dictionary=True)
        
        # 暫時使用固定user_id=1，未來應整合登入系統
        user_id = 1
        
        cursor.execute('''
            SELECT dark_mode, font_size, price_alert, market_summary, data_source
            FROM settings WHERE user_id = %s
        ''', (user_id,))
        
        settings = cursor.fetchone()
        cursor.close()
        conn.close()
        
        # 如果沒有設定，使用預設值
        if not settings:
            settings = {
                'dark_mode': True,
                'font_size': 'medium',
                'price_alert': False,
                'market_summary': True,
                'data_source': 'default'
            }
        
        return jsonify({'settings': settings})
    except Exception as e:
        logging.error(f"獲取設定時發生錯誤: {e}")
        return jsonify({'error': f"獲取設定時發生錯誤: {str(e)}"}), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    try:
        data = request.get_json()
        
        # 驗證設定
        if not isinstance(data.get('dark_mode'), bool) or \
           not data.get('font_size') in ['small', 'medium', 'large'] or \
           not isinstance(data.get('price_alert'), bool) or \
           not isinstance(data.get('market_summary'), bool) or \
           not data.get('data_source') in ['default', 'yahoo', 'alpha']:
            return jsonify({'error': '無效的設定值'}), 400
        
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '無法連接資料庫'}), 500
            
        cursor = conn.cursor()
        
        # 暫時使用固定user_id=1，未來應整合登入系統
        user_id = 1
        
        # 檢查用戶是否已有設定
        cursor.execute('SELECT id FROM settings WHERE user_id = %s', (user_id,))
        if cursor.fetchone():
            # 更新現有設定
            cursor.execute('''
                UPDATE settings 
                SET dark_mode = %s, font_size = %s, price_alert = %s, 
                    market_summary = %s, data_source = %s
                WHERE user_id = %s
            ''', (
                data['dark_mode'], 
                data['font_size'], 
                data['price_alert'], 
                data['market_summary'], 
                data['data_source'], 
                user_id
            ))
        else:
            # 創建新設定
            cursor.execute('''
                INSERT INTO settings 
                (user_id, dark_mode, font_size, price_alert, market_summary, data_source)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (
                user_id, 
                data['dark_mode'], 
                data['font_size'], 
                data['price_alert'], 
                data['market_summary'], 
                data['data_source']
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': '設定已更新'})
    except Exception as e:
        logging.error(f"更新設定時發生錯誤: {e}")
        return jsonify({'error': f"更新設定時發生錯誤: {str(e)}"}), 500

@app.route('/api/market_news', methods=['GET'])
def get_market_news():
    try:
        market = request.args.get('market', 'TW')
        category = request.args.get('category', 'general')
        
        news_list = fetch_market_news(market, category)
        
        return jsonify({'news': news_list})
    except Exception as e:
        logging.error(f"獲取市場新聞時發生錯誤: {e}")
        return jsonify({'error': f"獲取市場新聞時發生錯誤: {str(e)}"}), 500

def fetch_market_news(market, category):
    """獲取市場新聞"""
    try:
        # 設定搜尋關鍵字
        if market == 'TW':
            search_term = "台股"
            if category == 'tech':
                search_term += " 科技"
            elif category == 'finance':
                search_term += " 金融"
            elif category == 'industry':
                search_term += " 產業"
        else:
            search_term = "US stock market"
            if category == 'tech':
                search_term += " tech"
            elif category == 'finance':
                search_term += " finance"
            elif category == 'industry':
                search_term += " industry"
                
        # 使用 Google News RSS
        rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(search_term)}&hl={'en-US' if market == 'US' else 'zh-TW'}&gl={'US' if market == 'US' else 'TW'}&ceid={'US:en' if market == 'US' else 'TW:zh-Hant'}"
        
        feed = feedparser.parse(rss_url)
        news_list = []
        
        sentiment_analyzer = pipeline('sentiment-analysis', model='yiyanghkust/finbert-tone')
        
        for entry in feed.entries[:15]:  # 只取前15條新聞
            try:
                published_time = dt.datetime(*entry.published_parsed[:6])
                
                # 使用 FinBERT 進行情緒分析
                result = sentiment_analyzer(entry.title)[0]
                
                # 提取新聞摘要
                summary = entry.summary if hasattr(entry, 'summary') else ""
                # 清理HTML標籤
                summary = re.sub(r'<[^>]+>', '', summary)
                summary = summary[:150] + '...' if len(summary) > 150 else summary
                
                news_entry = {
                    'title': entry.title,
                    'link': entry.link,
                    'date': published_time.strftime('%Y-%m-%d'),
                    'source': entry.source.title if hasattr(entry, 'source') else 'Google News',
                    'summary': summary,
                    'sentiment': result['label'],
                    'sentiment_score': result['score']
                }
                
                news_list.append(news_entry)
            except Exception as inner_e:
                logging.error(f"處理新聞條目時發生錯誤: {inner_e}")
                continue
                
        return news_list
    except Exception as e:
        logging.error(f"獲取市場新聞時發生錯誤: {e}")
        return []

@app.route('/api/market_summary', methods=['GET'])
def get_market_summary():
    try:
        market = request.args.get('market', 'TW')
        
        # 獲取主要指數數據
        if market == 'TW':
            indices = ['^TWII', '0050.TW', '0056.TW']
            index_names = ['台灣加權指數', '元大台灣50', '元大高股息']
        else:
            indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']
            index_names = ['S&P 500', '道瓊工業', '納斯達克', '恐慌指數']
        
        market_data = []
        for i, index in enumerate(indices):
            try:
                data = yf.Ticker(index)
                hist = data.history(period="2d")
                
                if not hist.empty and len(hist) >= 2:
                    current = hist.iloc[-1]
                    prev = hist.iloc[-2]
                    
                    price = current['Close']
                    change = ((current['Close'] - prev['Close']) / prev['Close']) * 100
                    
                    market_data.append({
                        'symbol': index,
                        'name': index_names[i],
                        'price': price,
                        'change': change
                    })
            except Exception as inner_e:
                logging.error(f"獲取指數 {index} 數據時出錯: {inner_e}")
                market_data.append({
                    'symbol': index,
                    'name': index_names[i],
                    'price': 0,
                    'change': 0,
                    'error': True
                })
        
        # 獲取市場新聞摘要
        news = fetch_market_news(market, 'general')[:5]  # 只取前5條新聞
        
        # 獲取市場情緒
        sentiment_scores = [item['sentiment_score'] for item in news if item['sentiment'] == 'Positive']
        positive_ratio = len(sentiment_scores) / len(news) if news else 0
        
        market_sentiment = "中性"
        if positive_ratio > 0.6:
            market_sentiment = "樂觀"
        elif positive_ratio < 0.4:
            market_sentiment = "謹慎"
        
        return jsonify({
            'market': market,
            'indices': market_data,
            'news': news,
            'sentiment': market_sentiment,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logging.error(f"獲取市場摘要時發生錯誤: {e}")
        return jsonify({'error': f"獲取市場摘要時發生錯誤: {str(e)}"}), 500

@app.route('/static/charts/<path:filename>')
def serve_chart(filename):
    return send_from_directory('static/charts', filename)

@app.route('/static/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('static/data', filename)

# ------------------ 主程式 ------------------

if __name__ == '__main__':
    # 初始化資料庫
    init_database()
    app.run(debug=True, host='0.0.0.0', port=5000)


