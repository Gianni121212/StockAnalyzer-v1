from flask import Flask, render_template, request, jsonify, session, redirect
import yfinance as yf
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import warnings
import os
import secrets
import datetime as dt
import feedparser
import urllib.parse
from transformers import pipeline
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# Ensure necessary directories exist (保留 charts 目錄，data 目錄若僅用於 JSON 可保留)
for path in ['static/charts', 'static/data']:
    os.makedirs(path, exist_ok=True)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

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

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-2.0-flash")
        self.sentiment_analyzer = pipeline('sentiment-analysis', model='yiyanghkust/finbert-tone')

        self._get_data()
        self._get_financial_data()
        self._calculate_indicators()

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

            try:
                financials = self.stock.financials
                balance_sheet = self.stock.balance_sheet
                net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0
                equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else 0
                self.roe = net_income / equity if equity != 0 else 'N/A'
            except Exception as inner_e:
                logging.error("計算 ROE 時發生錯誤: %s", inner_e)
                self.roe = 'N/A'
            logging.info("成功取得 %s 的財務資料", self.ticker)
        except Exception as e:
            logging.error("取得財務資料時發生錯誤: %s", e)
            raise

    def _calculate_indicators(self):
        try:
            df = self.data.copy()
            df['MA5'] = df['Close'].rolling(window=5, min_periods=5).mean()
            df['MA20'] = df['Close'].rolling(window=20, min_periods=20).mean()
            df['MA120'] = df['Close'].rolling(window=120, min_periods=120).mean()
            df['MA240'] = df['Close'].rolling(window=240, min_periods=240).mean()

            delta = df['Close'].diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            ema_up = up.ewm(span=14, adjust=False, min_periods=14).mean()
            ema_down = down.ewm(span=14, adjust=False, min_periods=14).mean()
            rs = ema_up / ema_down
            df['RSI'] = 100 - (100 / (1 + rs))

            short_ema = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
            long_ema = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
            df['MACD'] = short_ema - long_ema
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()

            low_min = df['Low'].rolling(window=9, min_periods=1).min()
            high_max = df['High'].rolling(window=9, min_periods=1).max()
            rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
            df['K'] = rsv.ewm(span=3, adjust=False, min_periods=1).mean()
            df['D'] = df['K'].ewm(span=3, adjust=False, min_periods=1).mean()
            df['J'] = 3 * df['K'] - 2 * df['D']

            df['BB_middle'] = df['MA20']
            bb_std = df['Close'].rolling(window=20, min_periods=20).std()
            df['BB_upper'] = df['BB_middle'] + 2 * bb_std
            df['BB_lower'] = df['BB_middle'] - 2 * bb_std

            self.data = df.copy()
            logging.info("技術指標計算完成: %s", self.ticker)
        except Exception as e:
            logging.error("計算技術指標時發生錯誤: %s", e)
            raise

    def calculate_fibonacci_levels(self, window=60):
        try:
            recent_data = self.data.tail(window)
            max_price = recent_data['High'].max()
            min_price = recent_data['Low'].min()
            diff = max_price - min_price
            levels = {
                '0.0': min_price,
                '0.236': min_price + 0.236 * diff,
                '0.382': min_price + 0.382 * diff,
                '0.5': min_price + 0.5 * diff,
                '0.618': min_price + 0.618 * diff,
                '1.0': max_price
            }
            return levels
        except Exception as e:
            logging.error("計算 Fibonacci 水平時發生錯誤: %s", e)
            return {}

    def identify_patterns(self):
        patterns = []
        try:
            if len(self.data) >= 2:
                last = self.data.iloc[-1]
                prev = self.data.iloc[-2]
                if last['MA5'] > last['MA20'] and prev['MA5'] <= prev['MA20']:
                    patterns.append("MA5向上穿越MA20 (黃金交叉)")
                elif last['MA5'] < last['MA20'] and prev['MA5'] >= prev['MA20']:
                    patterns.append("MA5向下穿越MA20 (死亡交叉)")

            if len(self.data) >= 5:
                recent_highs = self.data['High'].tail(5)
                recent_lows = self.data['Low'].tail(5)
                if (recent_highs.iloc[1] < recent_highs.iloc[2] > recent_highs.iloc[3] and
                    recent_highs.iloc[0] < recent_highs.iloc[2] and
                    recent_highs.iloc[4] < recent_highs.iloc[2] and
                    recent_lows.iloc[1] > recent_lows.iloc[3]):
                    patterns.append("潛在頭肩頂形態")

            if len(self.data) >= 5:
                closes = self.data['Close'].tail(5)
                if (closes.iloc[0] > closes.iloc[1] < closes.iloc[2] and
                    closes.iloc[2] > closes.iloc[3] < closes.iloc[4] and
                    closes.iloc[1] < closes.iloc[3] and closes.iloc[4] > closes.iloc[2]):
                    patterns.append("潛在雙底 (W 底)")

            if len(self.data) >= 5:
                closes = self.data['Close'].tail(5)
                if (closes.iloc[0] < closes.iloc[1] > closes.iloc[2] and
                    closes.iloc[2] < closes.iloc[3] > closes.iloc[4] and
                    closes.iloc[1] > closes.iloc[3] and closes.iloc[4] < closes.iloc[2]):
                    patterns.append("潛在雙頂 (M 頂)")

            if len(self.data) >= 10:
                recent_highs = self.data['High'].tail(10)
                recent_lows = self.data['Low'].tail(10)
                high_mean = recent_highs.mean()
                high_std = recent_highs.std()
                if (high_std < 0.02 * high_mean and
                    recent_lows.iloc[-1] > recent_lows.iloc[-3] > recent_lows.iloc[-5]):
                    patterns.append("潛在上升三角形")

            if len(self.data) >= 10:
                recent_highs = self.data['High'].tail(10)
                recent_lows = self.data['Low'].tail(10)
                low_mean = recent_lows.mean()
                low_std = recent_lows.std()
                if (low_std < 0.02 * low_mean and
                    recent_highs.iloc[-1] < recent_highs.iloc[-3] < recent_highs.iloc[-5]):
                    patterns.append("潛在下降三角形")
        except Exception as e:
            logging.error("識別技術形態時發生錯誤: %s", e)
        return patterns

    def get_recent_news(self, days=30, num_news=10):
        try:
            if self.market == "TW":
                query = self.ticker.replace('.TW', '')
            else:
                query = self.company_name if self.company_name else self.ticker
            encoded_query = urllib.parse.quote(query)
            if self.market == "TW":
                rss_url = f"https://news.google.com/rss/search?q={encoded_query}+stock&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            else:
                rss_url = f"https://news.google.com/rss/search?q={encoded_query}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            recent_news = []
            now = dt.datetime.now()
            for entry in feed.entries:
                try:
                    published_time = dt.datetime(*entry.published_parsed[:6])
                except Exception:
                    published_time = now
                if (now - published_time).days <= days:
                    news_entry = {
                        'title': entry.title,
                        'link': entry.link,
                        'date': published_time.strftime('%Y-%m-%d'),
                        'source': entry.get('source', {}).get('title', 'Google News')
                    }
                    try:
                        sentiment = self.sentiment_analyzer(entry.title)[0]
                        news_entry['sentiment'] = sentiment['label']
                        news_entry['sentiment_score'] = sentiment['score']
                    except Exception as e:
                        logging.error("情緒分析失敗: %s", e)
                        news_entry['sentiment'] = 'N/A'
                        news_entry['sentiment_score'] = 0
                    recent_news.append(news_entry)
                    if len(recent_news) >= num_news:
                        break
            if not recent_news:
                return self._get_fallback_news()
            return recent_news
        except Exception as e:
            logging.error("取得近期新聞時發生錯誤: %s", e)
            return self._get_fallback_news()

    def _get_fallback_news(self):
        try:
            now_str = dt.datetime.now().strftime('%Y-%m-%d')
            if self.market == "TW":
                return [{
                    'title': f'請訪問 Yahoo 股市或工商時報查看 {self.ticker} 的最新新聞',
                    'date': now_str,
                    'source': '系統訊息',
                    'link': f'https://tw.stock.yahoo.com/quote/{self.ticker.replace(".TW", "")}/news',
                    'sentiment': 'N/A',
                    'sentiment_score': 0
                }]
            else:
                return [{
                    'title': f'請訪問 Yahoo Finance 或 MarketWatch 查看 {self.ticker} 的最新新聞',
                    'date': now_str,
                    'source': '系統訊息',
                    'link': f'https://finance.yahoo.com/quote/{self.ticker}/news',
                    'sentiment': 'N/A',
                    'sentiment_score': 0
                }]
        except Exception as e:
            logging.error("備用新聞方法發生錯誤: %s", e)
            return [{
                'title': '暫時無法獲取相關新聞',
                'date': dt.datetime.now().strftime('%Y-%m-%d'),
                'source': '系統訊息',
                'link': '#',
                'sentiment': 'N/A',
                'sentiment_score': 0
            }]

    def generate_strategy(self):
        try:
            last_row = self.data.iloc[-1]
            sentiment_summary = [news['sentiment'] for news in self.get_recent_news()]
            positive_count = sentiment_summary.count('Positive')
            negative_count = sentiment_summary.count('Negative')
            total_news = len(sentiment_summary)
            sentiment_ratio = (positive_count - negative_count) / total_news if total_news > 0 else 0

            if last_row['RSI'] < 30 and last_row['MACD'] > last_row['MACD_signal'] and sentiment_ratio > 0:
                return "Buy"
            elif last_row['RSI'] > 70 and sentiment_ratio < 0:
                return "Sell"
            else:
                return "Hold"
        except Exception as e:
            logging.error("生成策略時發生錯誤: %s", e)
            return "Hold"

    def plot_analysis(self, days=180, ma_lines=['MA5', 'MA20']):
        try:
            plot_data = self.data.tail(days).copy()
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                vertical_spacing=0.05,
                                subplot_titles=("價格與均線", "RSI", "MACD", "KDJ"))

            fig.add_trace(go.Candlestick(x=plot_data.index,
                                         open=plot_data['Open'],
                                         high=plot_data['High'],
                                         low=plot_data['Low'],
                                         close=plot_data['Close'],
                                         name='蠟燭圖'),
                          row=1, col=1)

            for ma in ma_lines:
                if ma in plot_data.columns:
                    color = {'MA5': 'blue', 'MA20': 'orange', 'MA120': 'purple', 'MA240': 'brown'}.get(ma, 'black')
                    label = ma if ma not in ['MA120', 'MA240'] else (f"{ma} (半年線)" if ma=='MA120' else f"{ma} (年線)")
                    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data[ma],
                                             mode='lines', name=label,
                                             line=dict(color=color, width=1)),
                                  row=1, col=1)

            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['BB_upper'],
                                     mode='lines', name='BB 上軌',
                                     line=dict(color='grey', width=1)),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['BB_lower'],
                                     mode='lines', name='BB 下軌',
                                     line=dict(color='grey', width=1)),
                          row=1, col=1)

            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['RSI'],
                                     mode='lines', name='RSI',
                                     line=dict(color='purple', width=1)),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=plot_data.index, y=[70]*len(plot_data),
                                     mode='lines', name='超買區',
                                     line=dict(color='red', dash='dash')),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=plot_data.index, y=[30]*len(plot_data),
                                     mode='lines', name='超賣區',
                                     line=dict(color='green', dash='dash')),
                          row=2, col=1)

            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MACD'],
                                     mode='lines', name='MACD',
                                     line=dict(color='blue', width=1)),
                          row=3, col=1)
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MACD_signal'],
                                     mode='lines', name='MACD Signal',
                                     line=dict(color='orange', width=1)),
                          row=3, col=1)
            macd_hist = plot_data['MACD'] - plot_data['MACD_signal']
            fig.add_trace(go.Bar(x=plot_data.index, y=macd_hist,
                                 name='MACD Histogram',
                                 marker_color='grey', opacity=0.5),
                          row=3, col=1)

            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['K'],
                                     mode='lines', name='K',
                                     line=dict(color='blue', width=1)),
                          row=4, col=1)
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['D'],
                                     mode='lines', name='D',
                                     line=dict(color='orange', width=1)),
                          row=4, col=1)
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['J'],
                                     mode='lines', name='J',
                                     line=dict(color='green', width=1)),
                          row=4, col=1)

            fib_levels = self.calculate_fibonacci_levels()
            for level, value in fib_levels.items():
                fig.add_hline(y=value, line_width=1, line_dash="dash", line_color="grey",
                              annotation_text=f"Fib {level}", annotation_position="right",
                              row=1, col=1)

            fig.update_layout(
                title=f'{self.company_name} ({self.ticker}) 技術分析圖表',
                height=1100,
                width=900,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_xaxes(rangeslider_visible=False)
            fig.update_yaxes(title_text=f"價格 ({self.currency})", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_yaxes(title_text="KDJ", row=4, col=1)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"{self.ticker}_analysis_{timestamp}.html"
            html_path = os.path.join('static', 'charts', file_name)
            fig.write_html(html_path)
            logging.info("圖表生成並儲存至 %s", html_path)
            return html_path
        except Exception as e:
            logging.error("生成圖表時發生錯誤: %s", e)
            raise

    def generate_ai_analysis(self, days=180):
        try:
            if len(self.data) < 2:
                raise ValueError("數據不足，無法計算價格變動。")
            last_price = self.data['Close'].iloc[-1]
            prev_price = self.data['Close'].iloc[-2]
            price_change = ((last_price - prev_price) / prev_price * 100) if prev_price else 0

            recent_news = self.get_recent_news()
            news_summary = "\n".join([f"- [{news['date']}] {news['title']} (來源: {news['source']}, 情緒: {news['sentiment']})" 
                                      for news in recent_news])

            technical_status = {
                'last_price': last_price,
                'price_change': price_change,
                'rsi': self.data['RSI'].iloc[-1],
                'ma5': self.data['MA5'].iloc[-1],
                'ma20': self.data['MA20'].iloc[-1],
                'ma120': self.data['MA120'].iloc[-1],
                'ma240': self.data['MA240'].iloc[-1],
                'bb_upper': self.data['BB_upper'].iloc[-1],
                'bb_lower': self.data['BB_lower'].iloc[-1],
                'macd': self.data['MACD'].iloc[-1],
                'macd_signal': self.data['MACD_signal'].iloc[-1],
                'kdj_k': self.data['K'].iloc[-1],
                'kdj_d': self.data['D'].iloc[-1],
                'kdj_j': self.data['J'].iloc[-1],
                'patterns': self.identify_patterns(),
                'fib_levels': self.calculate_fibonacci_levels()
            }

            prompt = f"""
你是一位擁有十年經驗的專業股市分析師，請根據以下 {self.company_name} ({self.ticker}) 的數據和技術指標進行深入的股市分析，並使用專業術語（如動能、趨勢、波動性）解釋其影響：

【基本資訊】
- 公司名稱: {self.company_name}
- 股票代碼: {self.ticker}
- 最新收盤價: {last_price:.2f} {self.currency}
- 日漲跌: {price_change:.2f}%

【財務數據】
- 市盈率 (Trailing P/E): {self.pe_ratio}
- 市值: {self.market_cap}
- 預期市盈率 (Forward P/E): {self.forward_pe}
- 利潤率: {self.profit_margins}
- EPS: {self.eps}
- ROE: {self.roe}

【技術指標】
- RSI(14): {technical_status['rsi']:.2f}
- MA5: {technical_status['ma5']:.2f}
- MA20: {technical_status['ma20']:.2f}
- MA120 (半年線): {technical_status['ma120']:.2f}
- MA240 (年線): {technical_status['ma240']:.2f}
- 布林帶上軌: {technical_status['bb_upper']:.2f}
- 布林帶下軌: {technical_status['bb_lower']:.2f}
- MACD: {technical_status['macd']:.2f}
- MACD Signal: {technical_status['macd_signal']:.2f}
- KDJ (K/D/J): {technical_status['kdj_k']:.2f}/{technical_status['kdj_d']:.2f}/{technical_status['kdj_j']:.2f}

【技術形態】
- {', '.join(technical_status['patterns']) if technical_status['patterns'] else '無明顯技術形態'}

【Fibonacci 回檔水平】
- {', '.join([f"{k}: {v:.2f}" for k, v in technical_status['fib_levels'].items()])}

【近期新聞】
{news_summary}

請按以下結構提供分析：
1. 近期價格走勢評估（分析動能與趨勢）
2. 支撐與壓力位分析（解釋 Fibonacci 和布林帶的作用）
3. 短期走勢預測（基於 MACD、RSI 和 KDJ）
4. 策略建議與風險分析（提供買入、賣出或持有建議，並說明理由）
5. 該股票近期相關新聞的影響分析（結合情緒分析）

具體要求：
- 解釋 RSI 的超買超賣情況
- 分析 MACD 的趨勢信號
- 評估 KDJ 的買賣信號
- 描述布林帶的價格波動性

最後，請在分析報告的結尾加入以下免責聲明：
"本分析報告僅供參考，不構成投資建議。投資者應自行承擔投資風險。"
            """
            response = self.model.generate_content(prompt)
            response_text = response.text.replace('\n', '<br>')
            return response_text
        except Exception as e:
            logging.error("生成 AI 分析時發生錯誤: %s", e)
            return f"生成 AI 分析時發生錯誤: {str(e)}"

    def run_full_analysis(self, days_to_analyze=180, ma_lines=['MA5', 'MA20']):
        try:
            ai_analysis = self.generate_ai_analysis(days_to_analyze)
            chart_path = self.plot_analysis(days_to_analyze, ma_lines)
            strategy = self.generate_strategy()

            last_row = self.data.iloc[-1]
            summary = {
                "company_name": self.company_name,
                "ticker": self.ticker,
                "close_price": float(last_row['Close']),
                "currency": self.currency,
                "rsi": float(last_row['RSI']),
                "ma5": float(last_row['MA5']),
                "ma20": float(last_row['MA20']),
                "ma120": float(last_row['MA120']),
                "ma240": float(last_row['MA240']),
                "bb_upper": float(last_row['BB_upper']),
                "bb_lower": float(last_row['BB_lower']),
                "macd": float(last_row['MACD']),
                "macd_signal": float(last_row['MACD_signal']),
                "kdj_k": float(last_row['K']),
                "kdj_d": float(last_row['D']),
                "kdj_j": float(last_row['J']),
                "patterns": self.identify_patterns(),
                "fib_levels": {k: float(v) for k, v in self.calculate_fibonacci_levels().items()},
                "strategy": strategy
            }

            logging.info("財務資料: PE={0}, 市值={1}, Forward P/E={2}, 利潤率={3}, EPS={4}, ROE={5}".format(
                self.pe_ratio, self.market_cap, self.forward_pe, self.profit_margins, self.eps, self.roe))
            recent_news = self.get_recent_news()
            for i, news in enumerate(recent_news, 1):
                logging.info("新聞 %d: 標題: %s, 日期: %s, 來源: %s, 情緒: %s, 連結: %s",
                             i, news['title'], news['date'], news['source'], news['sentiment'], news['link'])

            timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = os.path.join('static', 'data', f"{self.ticker}_analysis_{timestamp}.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'analysis': ai_analysis,
                    'summary': summary,
                    'chart_path': chart_path,
                    'timestamp': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, f, ensure_ascii=False, indent=4)
            logging.info("JSON 結果儲存至 %s", result_file)

            return {
                'analysis': ai_analysis,
                'summary': summary,
                'chart_path': chart_path,
                'timestamp': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logging.error("完整分析流程發生錯誤: %s", e)
            raise

@app.route('/')
def cover():
    return render_template('cover.html')

@app.route('/index')
def index():
    return render_template('index.html')

load_dotenv()

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.form
        ticker = data.get('ticker', '').strip()
        market = data.get('market', 'TW')
        period = '10y'
        days = int(data.get('days', 180))
        ma_lines = data.getlist('ma_lines')

        api_key = os.getenv("GEMINI_API_KEY")
        if not ticker:
            return jsonify({'error': '請輸入股票代碼'}), 400
        if not api_key:
            return jsonify({'error': 'API 金鑰無效，請檢查 .env 設定'}), 500

        analyzer = StockAnalyzer(ticker=ticker, api_key=api_key, period=period, market=market)
        results = analyzer.run_full_analysis(days_to_analyze=days, ma_lines=ma_lines or ['MA5', 'MA20'])

        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = f"static/data/{ticker}_analysis_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        session['last_analysis_file'] = result_file
        logging.info("Session 更新，分析檔案: %s", result_file)
        return jsonify({'status': 'success', 'redirect': '/results'})
    except Exception as e:
        logging.error("分析流程發生錯誤: %s", e)
        return jsonify({'error': f"分析過程發生錯誤: {str(e)}"}), 500

@app.route('/results')
def results():
    result_file = session.get('last_analysis_file')
    if not result_file or not os.path.exists(result_file):
        logging.warning("Session 中無分析檔案，或檔案不存在")
        return redirect('/')

    with open(result_file, 'r', encoding='utf-8') as f:
        analysis_results = json.load(f)

    os.remove(result_file)
    session.pop('last_analysis_file', None)
    return render_template('results.html', results=analysis_results)

@app.route('/get_stock_change/<ticker>', methods=['GET'])
def get_stock_change(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        change_percent = info.get('regularMarketChangePercent', 0)
        return jsonify({'change': change_percent})
    except Exception as e:
        logging.error("取得股票漲跌幅錯誤: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
