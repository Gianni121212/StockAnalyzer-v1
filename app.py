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

        # 設定 Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")

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
        """獲取基本的財務數據"""
        info = self.stock.info
        self.pe_ratio = info.get('trailingPE', 'N/A')
        self.market_cap = info.get('marketCap', 'N/A')
        self.forward_pe = info.get('forwardPE', 'N/A')
        self.profit_margins = info.get('profitMargins', 'N/A')

    def _calculate_indicators(self):
        """計算技術指標：移動平均線、RSI 和布林帶"""
        df = self.data.copy()

        # 移動平均線
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()

        # RSI (14)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        # 布林帶 (20日)
        df['BB_upper'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()

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
        # 移動平均線交叉
        if len(self.data) >= 2:
            last = self.data.iloc[-1]
            prev = self.data.iloc[-2]
            if last['MA5'] > last['MA20'] and prev['MA5'] <= prev['MA20']:
                patterns.append("MA5向上穿越MA20 (黃金交叉)")
            elif last['MA5'] < last['MA20'] and prev['MA5'] >= prev['MA20']:
                patterns.append("MA5向下穿越MA20 (死亡交叉)")

        # 頭肩頂識別（簡化版本）
        if len(self.data) >= 5:
            recent_highs = self.data['High'].tail(5)
            recent_lows = self.data['Low'].tail(5)
            if (recent_highs.iloc[1] < recent_highs.iloc[2] > recent_highs.iloc[3] and
                recent_highs.iloc[0] < recent_highs.iloc[2] and
                recent_highs.iloc[4] < recent_highs.iloc[2] and
                recent_lows.iloc[1] > recent_lows.iloc[3]):
                patterns.append("潛在頭肩頂形態")

        return patterns

    def get_recent_news(self, days=15, num_news=10):
        """
        利用 feedparser 從 Google News RSS 抓取近期與股票相關的新聞
        
        Parameters:
          days (int): 要獲取最近幾天內的新聞，預設 7 天
          num_news (int): 抓取新聞的數量，預設 3 則
          
        Returns:
          list: 包含新聞資訊的列表，每則新聞包含 title、link、date 與 source
        """
        # 使用公司名稱作為查詢關鍵字；若 company_name 不存在則用股票代碼
        query = self.company_name if self.company_name else self.ticker
        encoded_query = urllib.parse.quote(query)
        # 根據市場決定 RSS URL
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
                recent_news.append(news_entry)
                if len(recent_news) >= num_news:
                    break

        if not recent_news:
            return self._get_fallback_news()
        return recent_news

    def _get_fallback_news(self):
        """
        當無法從 yfinance 或 RSS 獲取新聞時的備用方法，
        返回一則預設新聞提示。
        """
        try:
            if self.market == "TW":
                return [{
                    'title': f'請訪問 Yahoo 股市或工商時報查看 {self.ticker} 的最新新聞',
                    'date': dt.datetime.now().strftime('%Y-%m-%d'),
                    'source': '系統訊息',
                    'link': f'https://tw.stock.yahoo.com/quote/{self.ticker.replace(".TW", "")}/news'
                }]
            else:
                return [{
                    'title': f'請訪問 Yahoo Finance 或 MarketWatch 查看 {self.ticker} 的最新新聞',
                    'date': dt.datetime.now().strftime('%Y-%m-%d'),
                    'source': '系統訊息',
                    'link': f'https://finance.yahoo.com/quote/{self.ticker}/news'
                }]
        except Exception as e:
            print(f"備用新聞獲取方法發生錯誤: {str(e)}")
            return [{
                'title': '暫時無法獲取相關新聞',
                'date': dt.datetime.now().strftime('%Y-%m-%d'),
                'source': '系統訊息',
                'link': '#'
            }]

    def plot_analysis(self, days=180):
        """繪製價格、均線、RSI 和布林帶的圖表"""
        plot_data = self.data.tail(days).copy()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=("價格與均線", "RSI"))

        # 主圖：蠟燭圖、均線和布林帶
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

        # RSI 圖
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['RSI'], mode='lines', name='RSI',
                                 line=dict(color='purple', width=1)),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=[70] * len(plot_data), mode='lines', name='超買區',
                                 line=dict(color='red', dash='dash')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_data.index, y=[30] * len(plot_data), mode='lines', name='超賣區',
                                 line=dict(color='green', dash='dash')),
                      row=2, col=1)

        # 加入 Fibonacci 水平線
        fib_levels = self.calculate_fibonacci_levels()
        for level, value in fib_levels.items():
            fig.add_hline(y=value, line_width=1, line_dash="dash", line_color="grey",
                          annotation_text=f"Fib {level}", annotation_position="right",
                          row=1, col=1)

        fig.update_layout(
            title=f'{self.company_name} ({self.ticker}) 技術分析圖表',
            height=700,
            width=900,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_yaxes(title_text=f"價格 ({self.currency})", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)

        # 生成文件名稱
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{self.ticker}_analysis_{timestamp}"
        html_path = f"static/charts/{file_name}.html"
        fig.write_html(html_path)
        
        return html_path

    def generate_ai_analysis(self, days=180):
        """利用 Gemini API 生成 AI 分析報告，包含近期新聞"""
        recent_data = self.data.tail(days).copy()
        if len(self.data) < 2:
            raise ValueError("數據不足，無法計算價格變動。")
        last_price = self.data['Close'].iloc[-1]
        prev_price = self.data['Close'].iloc[-2]
        price_change = ((last_price - prev_price) / prev_price * 100) if prev_price else 0

        # 獲取近期新聞
        recent_news = self.get_recent_news()
        news_summary = "\n".join([f"- [{news['date']}] {news['title']} (來源: {news['source']})" for news in recent_news])

        technical_status = {
            'last_price': last_price,
            'price_change': price_change,
            'rsi': self.data['RSI'].iloc[-1],
            'ma5': self.data['MA5'].iloc[-1],
            'ma20': self.data['MA20'].iloc[-1],
            'bb_upper': self.data['BB_upper'].iloc[-1],
            'bb_lower': self.data['BB_lower'].iloc[-1],
            'patterns': self.identify_patterns(),
            'fib_levels': self.calculate_fibonacci_levels()
        }

        prompt = f"""
請根據以下 {self.company_name} ({self.ticker}) 的數據和技術指標進行深入的股市分析：

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

【技術指標】
- RSI(14): {technical_status['rsi']:.2f}
- MA5: {technical_status['ma5']:.2f}
- MA20: {technical_status['ma20']:.2f}
- 布林帶上軌: {technical_status['bb_upper']:.2f}
- 布林帶下軌: {technical_status['bb_lower']:.2f}

【技術形態】
- {', '.join(technical_status['patterns']) if technical_status['patterns'] else '無明顯技術形態'}

【Fibonacci 回檔水平】
- {', '.join([f"{k}: {v:.2f}" for k, v in technical_status['fib_levels'].items()])}

【近期新聞】
{news_summary}

請提供以下分析:
1. 近期價格走勢評估
2. 支撐與壓力位分析
3. 短期走勢預測
4. 策略建議與風險分析
5. 該股票近期相關新聞的影響分析
        """
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.replace('\n', '<br>')  # 確保換行在 HTML 中顯示
            return response_text
        except Exception as e:
            return f"生成 AI 分析時發生錯誤: {str(e)}"

    def run_full_analysis(self, days_to_analyze=180):
        """執行完整的分析流程，包括 AI 報告、圖表繪製、指標輸出與資料匯出"""
        ai_analysis = self.generate_ai_analysis(days=days_to_analyze)
        chart_path = self.plot_analysis(days=days_to_analyze)

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
            "patterns": self.identify_patterns(),
            "fib_levels": {k: float(v) for k, v in self.calculate_fibonacci_levels().items()}
        }

        # 匯出最近的數據為 CSV
        recent_data = self.data.tail(days_to_analyze).copy()
        timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = f"static/data/{self.ticker}_data_{timestamp}.csv"
        recent_data.to_csv(csv_path)

        return {
            'analysis': ai_analysis,
            'summary': summary,
            'chart_path': chart_path,
            'csv_path': csv_path,
            'timestamp': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

@app.route('/')
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
        # 分析期間這裡固定使用 10 年，您可根據需求調整
        period = '10y'
        days = int(data.get('days', 180))

        # 從 .env 讀取 API Key
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
        
        # 儲存分析結果到 session
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
