from flask import Flask, render_template, request, jsonify, session, redirect
import yfinance as yf
import pandas as pd
import numpy as np
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

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# 建立儲存生成檔案的目錄
os.makedirs('static/charts', exist_ok=True)
os.makedirs('static/data', exist_ok=True)

class StockAnalyzer:
    def __init__(self, ticker, api_key, period="10y", market="TW"):
        """
        初始化股票分析器

        Parameters:
        ticker (str): 股票代碼，如 '2330.TW' 或 'AAPL'
        api_key (str): Gemini API 金鑰
        period (str): 分析期間，預設為 "10y"
        market (str): 市場，"TW"代表台灣股市，"US"代表美國股市
        """
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
        # 使用穩定版本
        self.model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

        # 初始化情緒分析器（使用 FinBERT）
        self.sentiment_analyzer = pipeline('sentiment-analysis', model='yiyanghkust/finbert-tone')

        self._get_data()
        self._get_financial_data()
        self._calculate_indicators()

    def _get_data(self):
        """獲取股票歷史數據與基本資訊"""
        self.data = self.stock.history(period=self.period)
        if self.data.empty:
            raise ValueError(f"無法取得 {self.ticker} 的資料，請確認股票代碼是否正確")
        company_info = self.stock.info
        self.company_name = company_info.get('longName', self.ticker)
        self.currency = company_info.get('currency', 'TWD' if self.market == 'TW' else 'USD')

    def _get_financial_data(self):
        """獲取基本的財務數據，包括 EPS 和 ROE"""
        info = self.stock.info
        self.pe_ratio = info.get('trailingPE', 'N/A')
        self.market_cap = info.get('marketCap', 'N/A')
        self.forward_pe = info.get('forwardPE', 'N/A')
        self.profit_margins = info.get('profitMargins', 'N/A')
        self.eps = info.get('trailingEps', 'N/A')

        # 計算 ROE（淨收入 / 股東權益）
        try:
            financials = self.stock.financials
            balance_sheet = self.stock.balance_sheet
            net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0
            equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else 0
            self.roe = net_income / equity if equity != 0 else 'N/A'
        except Exception as e:
            print(f"無法計算 ROE: {e}")
            self.roe = 'N/A'

    def _calculate_indicators(self):
        """計算技術指標：移動平均線、RSI、布林帶、MACD 和 KDJ"""
        df = self.data.copy()

        # 移動平均線
        df['MA5'] = df['Close'].rolling(window=5, min_periods=5).mean()
        df['MA20'] = df['Close'].rolling(window=20, min_periods=20).mean()

        # RSI
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ema_up = up.ewm(span=14, adjust=False, min_periods=14).mean()
        ema_down = down.ewm(span=14, adjust=False, min_periods=14).mean()
        rs = ema_up / ema_down
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        short_ema = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
        long_ema = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
        df['MACD'] = short_ema - long_ema
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()

        # KDJ
        low_min = df['Low'].rolling(window=9, min_periods=1).min()
        high_max = df['High'].rolling(window=9, min_periods=1).max()
        rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
        df['K'] = rsv.ewm(span=3, adjust=False, min_periods=1).mean()
        df['D'] = df['K'].ewm(span=3, adjust=False, min_periods=1).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']

        # 布林帶
        df['BB_middle'] = df['MA20']  # 共享 MA20
        bb_std = df['Close'].rolling(window=20, min_periods=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * bb_std
        df['BB_lower'] = df['BB_middle'] - 2 * bb_std

        self.data = df.copy()

    def calculate_fibonacci_levels(self, window=60):
        """計算最近 window 日內的 Fibonacci 回檔水平"""
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

    def identify_patterns(self):
        """識別技術形態：黃金交叉、死亡交叉和頭肩頂"""
        patterns = []
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

        return patterns

    def get_recent_news(self, days=30, num_news=10):
        """
        利用 feedparser 從 Google News RSS 抓取近期與股票相關的新聞，並進行情緒分析
        """
        # 台灣市場：使用股市代號（去除 .TW），其他市場使用公司名稱
        if self.market == "TW":
            query = self.ticker.replace('.TW', '')  # 例如 "2330.TW" 變成 "2330"
        else:
            query = self.company_name if self.company_name else self.ticker
        encoded_query = urllib.parse.quote(query)
        
        # 構建 RSS URL
        if self.market == "TW":
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}+股市&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
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
                # 情緒分析
                try:
                    sentiment = self.sentiment_analyzer(entry.title)[0]
                    news_entry['sentiment'] = sentiment['label']
                    news_entry['sentiment_score'] = sentiment['score']
                except Exception as e:
                    print(f"情緒分析失敗: {e}")
                    news_entry['sentiment'] = 'N/A'
                    news_entry['sentiment_score'] = 0
                recent_news.append(news_entry)
                if len(recent_news) >= num_news:
                    break

        if not recent_news:
            return self._get_fallback_news()
        return recent_news

    def _get_fallback_news(self):
        """當無法獲取新聞時的備用方法"""
        try:
            if self.market == "TW":
                return [{
                    'title': f'請訪問 Yahoo 股市或工商時報查看 {self.ticker} 的最新新聞',
                    'date': dt.datetime.now().strftime('%Y-%m-%d'),
                    'source': '系統訊息',
                    'link': f'https://tw.stock.yahoo.com/quote/{self.ticker.replace(".TW", "")}/news',
                    'sentiment': 'N/A',
                    'sentiment_score': 0
                }]
            else:
                return [{
                    'title': f'請訪問 Yahoo Finance 或 MarketWatch 查看 {self.ticker} 的最新新聞',
                    'date': dt.datetime.now().strftime('%Y-%m-%d'),
                    'source': '系統訊息',
                    'link': f'https://finance.yahoo.com/quote/{self.ticker}/news',
                    'sentiment': 'N/A',
                    'sentiment_score': 0
                }]
        except Exception as e:
            print(f"備用新聞獲取方法發生錯誤: {str(e)}")
            return [{
                'title': '暫時無法獲取相關新聞',
                'date': dt.datetime.now().strftime('%Y-%m-%d'),
                'source': '系統訊息',
                'link': '#',
                'sentiment': 'N/A',
                'sentiment_score': 0
            }]

    def generate_strategy(self):
        """根據技術指標和情緒分析生成買賣建議"""
        last_row = self.data.iloc[-1]
        sentiment_summary = [news['sentiment'] for news in self.get_recent_news()]
        positive_count = sentiment_summary.count('Positive')
        negative_count = sentiment_summary.count('Negative')
        total_news = len(sentiment_summary)

        if total_news > 0:
            sentiment_ratio = (positive_count - negative_count) / total_news
        else:
            sentiment_ratio = 0

        if last_row['RSI'] < 30 and last_row['MACD'] > last_row['MACD_signal'] and sentiment_ratio > 0:
            return "Buy"
        elif last_row['RSI'] > 70 and sentiment_ratio < 0:
            return "Sell"
        else:
            return "Hold"

    def plot_analysis(self, days=180):
        """繪製價格、均線、RSI、MACD（含柱狀圖）和 KDJ 的圖表"""
        plot_data = self.data.tail(days).copy()
        # 建立 4 行子圖：價格與均線、RSI、MACD、KDJ
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=("價格與均線", "RSI", "MACD", "KDJ"))

        # 第 1 行：價格與均線、布林帶
        fig.add_trace(go.Candlestick(x=plot_data.index,
                                    open=plot_data['Open'],
                                    high=plot_data['High'],
                                    low=plot_data['Low'],
                                    close=plot_data['Close'],
                                    name='蠟燭圖'),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA5'], mode='lines', name='MA5',
                                line=dict(color='blue', width=1)),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MA20'], mode='lines', name='MA20',
                                line=dict(color='orange', width=1)),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['BB_upper'], mode='lines', name='BB 上軌',
                                line=dict(color='grey', width=1)),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['BB_lower'], mode='lines', name='BB 下軌',
                                line=dict(color='grey', width=1)),
                    row=1, col=1)

        # 第 2 行：RSI 指標
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['RSI'], mode='lines', name='RSI',
                                line=dict(color='purple', width=1)),
                    row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=[70] * len(plot_data), mode='lines', name='超買區',
                                line=dict(color='red', dash='dash')),
                    row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=[30] * len(plot_data), mode='lines', name='超賣區',
                                line=dict(color='green', dash='dash')),
                    row=2, col=1)

        # 第 3 行：MACD 指標
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MACD'], mode='lines', name='MACD',
                                line=dict(color='blue', width=1)),
                    row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['MACD_signal'], mode='lines', name='MACD Signal',
                                line=dict(color='orange', width=1)),
                    row=3, col=1)
        # MACD 柱狀圖：計算直方圖 (MACD - MACD_signal)
        macd_hist = plot_data['MACD'] - plot_data['MACD_signal']
        fig.add_trace(go.Bar(x=plot_data.index, y=macd_hist, name='MACD Histogram',
                            marker_color='grey', opacity=0.5),
                    row=3, col=1)

        # 第 4 行：KDJ 指標
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['K'], mode='lines', name='K',
                                line=dict(color='blue', width=1)),
                    row=4, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['D'], mode='lines', name='D',
                                line=dict(color='orange', width=1)),
                    row=4, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['J'], mode='lines', name='J',
                                line=dict(color='green', width=1)),
                    row=4, col=1)

        # 加入 Fibonacci 水平線（在第一行圖上）
        fib_levels = self.calculate_fibonacci_levels()
        for level, value in fib_levels.items():
            fig.add_hline(y=value, line_width=1, line_dash="dash", line_color="grey",
                        annotation_text=f"Fib {level}", annotation_position="right",
                        row=1, col=1)

        fig.update_layout(
            title=f'{self.company_name} ({self.ticker}) 技術分析圖表',
            height=1100,  # 調整高度以適應 4 個子圖
            width=900,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_yaxes(title_text=f"價格 ({self.currency})", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="KDJ", row=4, col=1)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{self.ticker}_analysis_{timestamp}"
        html_path = os.path.join('static', 'charts', f"{file_name}.html")
        fig.write_html(html_path)
        
        return html_path

    def generate_ai_analysis(self, days=180):
        """利用 Gemini API 生成 AI 分析報告，包含近期新聞和情緒分析"""
        recent_data = self.data.tail(days).copy()
        if len(self.data) < 2:
            raise ValueError("數據不足，無法計算價格變動。")
        last_price = self.data['Close'].iloc[-1]
        prev_price = self.data['Close'].iloc[-2]
        price_change = ((last_price - prev_price) / prev_price * 100) if prev_price else 0

        recent_news = self.get_recent_news()
        news_summary = "\n".join([f"- [{news['date']}] {news['title']} (來源: {news['source']}, 情緒: {news['sentiment']})" for news in recent_news])

        technical_status = {
            'last_price': last_price,
            'price_change': price_change,
            'rsi': self.data['RSI'].iloc[-1],
            'ma5': self.data['MA5'].iloc[-1],
            'ma20': self.data['MA20'].iloc[-1],
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
        """
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.replace('\n', '<br>')
            return response_text
        except Exception as e:
            return f"生成 AI 分析時發生錯誤: {str(e)}"

    def run_full_analysis(self, days_to_analyze=180):
        """執行完整的分析流程，包括 AI 報告、圖表繪製、指標輸出與資料匯出"""
        ai_analysis = self.generate_ai_analysis(days=days_to_analyze)
        chart_path = self.plot_analysis(days=days_to_analyze)
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

        # 列印財務數據
        financial_data = {
            "pe_ratio": self.pe_ratio,
            "market_cap": self.market_cap,
            "forward_pe": self.forward_pe,
            "profit_margins": self.profit_margins,
            "eps": self.eps,
            "roe": self.roe
        }
        print(f"\n=== {self.ticker} 的財務數據 ===")
        for key, value in financial_data.items():
            print(f"{key}: {value}")

        # 列印新聞
        recent_news = self.get_recent_news()
        print(f"\n=== {self.ticker} 的近期新聞 ===")
        for i, news in enumerate(recent_news, 1):
            print(f"新聞 {i}:")
            print(f"  標題: {news['title']}")
            print(f"  日期: {news['date']}")
            print(f"  來源: {news['source']}")
            print(f"  情緒: {news['sentiment']} (分數: {news['sentiment_score']:.2f})")
            print(f"  連結: {news['link']}")
            print("---")

        # 匯出數據為 CSV
        recent_data = self.data.tail(days_to_analyze).copy()
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join('static', 'data', f"{self.ticker}_data_{timestamp}.csv")
        recent_data.to_csv(csv_path)

        return {
            'analysis': ai_analysis,
            'summary': summary,
            'chart_path': chart_path,
            'csv_path': csv_path,
            'timestamp': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
@app.route('/')
def cover():
    return render_template('cover.html')
@app.route('/index')
def index():
    return render_template('index.html')
# 載入 .env 檔案
load_dotenv()

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.form
        ticker = data.get('ticker', '').strip()
        market = data.get('market', 'TW')
        period = '10y'
        days = int(data.get('days', 180))

        api_key = os.getenv("GEMINI_API_KEY")
        
        if not ticker:
            return jsonify({'error': '請輸入股票代碼'}), 400
        if not api_key:
            return jsonify({'error': 'API 金鑰無效，請檢查 .env 設定'}), 500
        
        analyzer = StockAnalyzer(
            ticker=ticker,
            api_key=api_key,
            period=period,
            market=market
        )
        
        results = analyzer.run_full_analysis(days_to_analyze=days)

        if "error" in results:
            return jsonify({'error': results['error']}), 400
        
        session['last_analysis'] = results
        
        return jsonify({
            'status': 'success',
            'redirect': '/results'
        })
    except Exception as e:
        return jsonify({'error': f"分析過程發生錯誤: {str(e)}"}), 500

@app.route('/results')
def results():
    if 'last_analysis' not in session:
        return redirect('/')
    
    analysis_results = session['last_analysis']
    return render_template('results.html', results=analysis_results)

if __name__ == "__main__":
    app.run(debug=True)
