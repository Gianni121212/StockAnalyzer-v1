from flask import Flask, render_template, request, jsonify, session
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import google.generativeai as genai
import warnings
import json
import os
import secrets

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Create directories for storing generated files
os.makedirs('static/charts', exist_ok=True)
os.makedirs('static/data', exist_ok=True)

class StockAnalyzer:
    def __init__(self, ticker, api_key, period="3y", market="TW"):
        """
        初始化股票分析器

        Parameters:
        ticker (str): 股票代碼，如 '2330.TW' 或 'AAPL'
        api_key (str): Gemini API 金鑰
        period (str): 分析期間，預設為 "3y"
        market (str): 市場，"TW"代表台灣股市，"US"代表美國股市
        """
        self.ticker = ticker.strip()
        if market == "TW" and "." not in self.ticker:
            self.ticker = f"{self.ticker}.TW"
        self.period = period
        self.market = market
        self.stock = yf.Ticker(self.ticker)
        self.data = None

        # 設定 Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-pro")

        self._get_data()
        self._calculate_indicators()

    def _get_data(self):
        """獲取股票歷史數據與基本資訊"""
        self.data = self.stock.history(period=self.period)
        if self.data.empty:
            raise ValueError(f"無法取得 {self.ticker} 的資料，請確認股票代碼是否正確")
        company_info = self.stock.info
        self.company_name = company_info.get('longName', self.ticker)
        self.currency = company_info.get('currency', 'TWD' if self.market == 'TW' else 'USD')

    def _calculate_indicators(self):
        """計算簡單的技術指標：移動平均線與 RSI"""
        df = self.data.copy()

        # 計算移動平均線 (MA5, MA20)
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()

        # RSI (14)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

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
        """簡單的黃金交叉與死亡交叉判斷"""
        patterns = []
        if len(self.data) >= 2:
            last = self.data.iloc[-1]
            prev = self.data.iloc[-2]
            if last['MA5'] > last['MA20'] and prev['MA5'] <= prev['MA20']:
                patterns.append("MA5向上穿越MA20 (黃金交叉)")
            elif last['MA5'] < last['MA20'] and prev['MA5'] >= prev['MA20']:
                patterns.append("MA5向下穿越MA20 (死亡交叉)")
        return patterns

    def plot_analysis(self, days=180):
        """繪製價格、均線與 RSI 的圖表"""
        plot_data = self.data.tail(days).copy()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=("價格與均線", "RSI"))

        # 主圖：蠟燭圖與移動平均線
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
        """利用 Gemini API 生成 AI 分析報告"""
        recent_data = self.data.tail(days).copy()
        if len(self.data) < 2:
            raise ValueError("數據不足，無法計算價格變動。")
        last_price = self.data['Close'].iloc[-1]
        prev_price = self.data['Close'].iloc[-2]
        price_change = ((last_price - prev_price) / prev_price * 100) if prev_price else 0

        technical_status = {
            'last_price': last_price,
            'price_change': price_change,
            'rsi': self.data['RSI'].iloc[-1],
            'ma5': self.data['MA5'].iloc[-1],
            'ma20': self.data['MA20'].iloc[-1],
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

【技術指標】
- RSI(14): {technical_status['rsi']:.2f}
- MA5: {technical_status['ma5']:.2f}
- MA20: {technical_status['ma20']:.2f}

【技術形態】
- {', '.join(technical_status['patterns']) if technical_status['patterns'] else '無明顯技術形態'}

【Fibonacci 回檔水平】
- {', '.join([f"{k}: {v:.2f}" for k, v in technical_status['fib_levels'].items()])}

請提供以下分析:
1. 近期價格走勢評估
2. 支撐與壓力位分析
3. 短期走勢預測
4. 策略建議與風險分析
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
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
            "patterns": self.identify_patterns(),
            "fib_levels": {k: float(v) for k, v in self.calculate_fibonacci_levels().items()}
        }

        # 匯出最近的數據為 CSV
        recent_data = self.data.tail(days_to_analyze).copy()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = f"static/data/{self.ticker}_data_{timestamp}.csv"
        recent_data.to_csv(csv_path)

        return {
            'analysis': ai_analysis,
            'summary': summary,
            'chart_path': chart_path,
            'csv_path': csv_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.form
        ticker = data.get('ticker', '')
        market = data.get('market', 'TW')
        period = data.get('period', '3y')
        days = int(data.get('days', 180))
        api_key = data.get('api_key', '')
        
        if not ticker or not api_key:
            return jsonify({'error': '請輸入股票代碼與 API 金鑰'}), 400
        
        analyzer = StockAnalyzer(
            ticker=ticker,
            api_key=api_key,
            period=period,
            market=market
        )
        
        results = analyzer.run_full_analysis(days_to_analyze=days)
        
        # 儲存分析結果到 session
        session['last_analysis'] = results
        
        return jsonify({
            'status': 'success',
            'redirect': '/results'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/results')
def results():
    if 'last_analysis' not in session:
        return redirect('/')
    
    analysis_results = session['last_analysis']
    return render_template('results.html', results=analysis_results)


if __name__ == "__main__":
    app.run(debug=True)