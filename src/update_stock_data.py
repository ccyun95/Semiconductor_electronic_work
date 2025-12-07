# -*- coding: utf-8 -*-
"""
KRX 일별 데이터 수집/업데이트 → per-ticker JSON 생성 → 표 + 2개 차트가 포함된 index.html 생성

폴더 구조(권장)
.
├─ data/
│  └─ company_list.txt           # "영문기업명,티커" 한 줄씩
├─ src/
│  └─ update_stock_data.py       # 본 파일
├─ requirements.txt
└─ .github/workflows/krx_data.yml

CSV 스키마:
일자,시가,고가,저가,종가,거래량,등락률,기관 합계,기타법인,개인,외국인 합계,전체,공매도,공매도비중,공매도잔고,공매도잔고비중

생성 산출물:
- data/<영문기업명>_stock_data.csv
- docs/api/<영문기업명>_<티커>.json
- docs/index.html (표 + 차트 2개/종목)
"""
import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from dateutil import tz
import time
import pandas as pd

from pykrx import stock

# ============================================================
# 설정
# ============================================================
DATA_DIR = Path(os.getenv("GITHUB_WORKSPACE", ".")) / "data"
DOCS_DIR = Path(os.getenv("GITHUB_WORKSPACE", ".")) / "docs"
API_DIR = DOCS_DIR / "api"

OUTPUT_SUFFIX = "_stock_data.csv"
ENCODING = "utf-8-sig"     # 엑셀 호환
SLEEP_SEC = 0.3            # API 과도 호출 방지
WINDOW_DAYS_INIT = 370     # 신규 생성시 과거 1년+α
SORT_DESC = True           # CSV 및 JSON 정렬: 최신일자 -> 과거

REQ_COLS = [
    "일자","시가","고가","저가","종가","거래량","등락률",
    "기관 합계","기타법인","개인","외국인 합계","전체",
    "공매도","공매도비중","공매도잔고","공매도잔고비중"
]

KST = tz.gettz("Asia/Seoul")

# pykrx 쪽 잘못된 logging.format 호출을 묵음 처리
for name in ["pykrx", "pykrx.website", "pykrx.website.comm", "pykrx.website.comm.util"]:
    logging.getLogger(name).disabled = True


# ============================================================
# 유틸
# ============================================================
def kst_now():
    return datetime.now(tz=KST)

def kst_today_date():
    return kst_now().date()

def yyyymmdd(d):
    return d.strftime("%Y%m%d")

def empty_with_cols(cols):
    """지정한 컬럼을 가진 '빈 DF'(merge 안전)를 반환."""
    data = {}
    for c in cols:
        data[c] = pd.Series(dtype="object" if c == "일자" else "float64")
    return pd.DataFrame(data)

def read_company_list(path: Path):
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"기업 리스트 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                name, ticker = [x.strip() for x in line.split(",", 1)]
            else:
                parts = line.replace("\t", " ").split()
                if len(parts) < 2:
                    logging.warning("기업 라인 파싱 불가: %s", line)
                    continue
                name, ticker = parts[0], parts[1]
            ticker = ticker.zfill(6)
            rows.append((name, ticker))
    return rows

def csv_path_for(eng_name: str, ticker: str) -> Path:
    return DATA_DIR / f"{eng_name}{OUTPUT_SUFFIX}"

def last_trading_day_by_ohlcv(ticker: str, today: datetime.date):
    """최근 구간의 마지막 거래일을 OHLCV로 판정."""
    start = today - timedelta(days=30)
    df = stock.get_market_ohlcv(yyyymmdd(start), yyyymmdd(today), ticker)
    if df is None or df.empty:
        start = today - timedelta(days=90)
        df = stock.get_market_ohlcv(yyyymmdd(start), yyyymmdd(today), ticker)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker} : 최근 거래 자료가 없습니다.")
    return pd.to_datetime(df.index.max()).date()

def normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """pykrx 데이터프레임을 '일자' 컬럼(YYYY-MM-DD)으로 정규화. 빈 DF여도 '일자' 보장."""
    if df is None or df.empty:
        return empty_with_cols(["일자"])
    df = df.copy()
    if df.index.name is None:
        df.index.name = "일자"
    idx = pd.to_datetime(df.index, errors="coerce")
    df.index = idx
    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: "일자"}, inplace=True)
    df["일자"] = pd.to_datetime(df["일자"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df

def rename_investor_cols(df: pd.DataFrame) -> pd.DataFrame:
    """투자자별 거래실적 컬럼 표준화(빈 DF 안전)."""
    if df is None or df.empty or "일자" not in df.columns:
        return empty_with_cols(["일자","기관 합계","기타법인","개인","외국인 합계","전체"])
    mapping = {
        "기관합계": "기관 합계",
        "외국인합계": "외국인 합계",
        "전체": "전체",
        "개인": "개인",
        "기타법인": "기타법인",
        "기관 합계": "기관 합계",
        "외국인 합계": "외국인 합계",
    }
    df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
    for need in ["기관 합계","기타법인","개인","외국인 합계","전체"]:
        if need not in df.columns:
            df[need] = 0
    keep = ["일자","기관 합계","기타법인","개인","외국인 합계","전체"]
    return df[keep]

def rename_short_cols(df: pd.DataFrame, is_balance=False) -> pd.DataFrame:
    """
    공매도 거래량/비중 또는 잔고/비중을 표준 컬럼으로 정리.
    입력 df는 normalize_date_index를 거쳐 '일자'가 항상 존재(빈 DF 가능).
    """
    if df is None or df.empty or "일자" not in df.columns:
        return empty_with_cols(["일자"] + (["공매도잔고","공매도잔고비중"] if is_balance else ["공매도","공매도비중"]))
    dfc = df.copy()

    if is_balance:
        amt = next((c for c in dfc.columns if any(k in c for k in ["공매도잔고","잔고","BAL_QTY"])), None)
        rto = next((c for c in dfc.columns if any(k in c for k in ["공매도잔고비중","잔고비중","BAL_RTO"])), None)
        dfc["공매도잔고"] = pd.to_numeric(dfc[amt], errors="coerce") if amt else 0
        dfc["공매도잔고비중"] = pd.to_numeric(dfc[rto], errors="coerce") if rto else 0.0
        keep = ["일자","공매도잔고","공매도잔고비중"]
        return dfc[keep]
    else:
        amt = next((c for c in dfc.columns if any(k in c for k in ["공매도","공매도거래량","거래량"])), None)
        rto = next((c for c in dfc.columns if any(k in c for k in ["공매도비중","비중"])), None)
        dfc["공매도"] = pd.to_numeric(dfc[amt], errors="coerce") if amt else 0
        dfc["공매도비중"] = pd.to_numeric(dfc[rto], errors="coerce") if rto else 0.0
        keep = ["일자","공매도","공매도비중"]
        return dfc[keep]

def ensure_all_cols(df: pd.DataFrame) -> pd.DataFrame:
    """최종 스키마(REQ_COLS)를 강제합니다."""
    for col in REQ_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[REQ_COLS]

def fetch_block(ticker: str, start_d: datetime.date, end_d: datetime.date) -> pd.DataFrame:
    s, e = yyyymmdd(start_d), yyyymmdd(end_d)

    # 1) OHLCV
    ohlcv = stock.get_market_ohlcv(s, e, ticker)
    df1 = normalize_date_index(ohlcv)

    # 2) 투자자별 거래실적
    inv = stock.get_market_trading_volume_by_date(s, e, ticker)
    df2 = rename_investor_cols(normalize_date_index(inv))

    # 3) 공매도 거래량/비중 (예외 안전)
    try:
        sv = stock.get_shorting_volume_by_date(s, e, ticker)
    except Exception:
        sv = pd.DataFrame()
    df3 = rename_short_cols(normalize_date_index(sv), is_balance=False)

    # 4) 공매도 잔고/비중 (예외 안전)
    try:
        sb = stock.get_shorting_balance_by_date(s, e, ticker)
    except Exception:
        sb = pd.DataFrame()
    df4 = rename_short_cols(normalize_date_index(sb), is_balance=True)

    # 안전 머지 (모두 '일자' 보유)
    df = df1.merge(df2, on="일자", how="left") \
            .merge(df3, on="일자", how="left") \
            .merge(df4, on="일자", how="left")

    # 스키마/형 변환/정렬
    df = ensure_all_cols(df)
    for c in [c for c in df.columns if c != "일자"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # 최신 → 과거 정렬
    df = df.sort_values("일자", ascending=not SORT_DESC)
    return df

def upsert_company(eng_name: str, ticker: str, run_on_holiday: bool):
    out_path = csv_path_for(eng_name, ticker)

    today = kst_today_date()
    end_date = last_trading_day_by_ohlcv(ticker, today)

    if out_path.exists():
        base = pd.read_csv(out_path, encoding=ENCODING)
        if base.empty:
            last_have = None
        else:
            base["일자"] = pd.to_datetime(base["일자"], errors="coerce").dt.date
            last_have = base["일자"].max()
        start_date = (last_have + timedelta(days=1)) if last_have else (end_date - timedelta(days=WINDOW_DAYS_INIT))
    else:
        start_date = end_date - timedelta(days=WINDOW_DAYS_INIT)

    # 휴장일이며 run_on_holiday=False이고 신규라면 스킵
    if (end_date < today) and (not run_on_holiday) and (not out_path.exists()):
        logging.info("[%s] 휴장일(run_on_holiday=False) → 신규 생성 건 스킵", eng_name)
        return False

    if start_date > end_date:
        logging.info("[%s] 최신 상태 (추가할 데이터 없음).", eng_name)
        return False

    logging.info("[%s] 수집 구간: %s ~ %s (티커 %s)", eng_name, start_date, end_date, ticker)
    df = fetch_block(ticker, start_date, end_date)

    if out_path.exists():
        base = pd.read_csv(out_path, encoding=ENCODING)
        merged = pd.concat([base, df], ignore_index=True)
        merged.drop_duplicates(subset=["일자"], keep="last", inplace=True)
        merged = merged.sort_values("일자", ascending=not SORT_DESC)
        merged.to_csv(out_path, index=False, encoding=ENCODING, lineterminator="\n")
        logging.info("[%s] 업데이트 완료 → %s", eng_name, out_path)
    else:
        df = df.sort_values("일자", ascending=not SORT_DESC)
        df.to_csv(out_path, index=False, encoding=ENCODING, lineterminator="\n")
        logging.info("[%s] 신규 생성 완료 → %s", eng_name, out_path)
    return True


# ============================================================
# 산출물: per-ticker JSON & index.html
# ============================================================
def emit_per_ticker_json(companies, rows_limit=None):
    API_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for name, ticker in companies:
        csv_path = csv_path_for(name, ticker)
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, encoding=ENCODING)
        except Exception:
            df = pd.read_csv(csv_path)

        if df.empty:
            continue

        # 필요 시 상위 rows만
        if rows_limit:
            df_use = df.head(int(rows_limit))
        else:
            df_use = df

        # JSON 스키마
        payload = {
            "name": name,
            "ticker": str(ticker).zfill(6),
            "columns": list(df_use.columns),
            "rows": df_use.astype(object).where(pd.notna(df_use), "").values.tolist(),
            "generated_at": kst_now().strftime("%Y-%m-%d %H:%M:%S %Z")
        }
        out = API_DIR / f"{name}_{str(ticker).zfill(6)}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        count += 1
    logging.info("per-ticker JSON 생성 완료: %d files → %s", count, API_DIR)


def emit_index_html(companies, rows_limit=None):
    import html as _html
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    sections = []
    generated = kst_now().strftime("%Y-%m-%d %H:%M:%S %Z")

    for name, ticker in companies:
        csv_path = csv_path_for(name, ticker)
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, encoding=ENCODING)
        except Exception:
            df = pd.read_csv(csv_path)
        if df.empty:
            continue
        if rows_limit:
            df = df.head(int(rows_limit))

        columns = [str(c) for c in df.columns]
        rows = df.astype(str).values.tolist()

        # 표 HTML
        thead = "".join(f"<th>{_html.escape(c)}</th>" for c in columns)
        tbody = "\n".join(
            "<tr>" + "".join(f"<td>{_html.escape(v)}</td>" for v in row) + "</tr>" for row in rows
        )
        sec_id = f"{name}_{str(ticker).zfill(6)}"

        # 표 + 차트 두 개의 컨테이너(표 아래)
        sections.append(f"""
<section id="{_html.escape(sec_id)}">
  <h2>{_html.escape(name)} ({str(ticker).zfill(6)})</h2>
  <div class="grid">
    <div class="scroll">
      <table>
        <thead><tr>{thead}</tr></thead>
        <tbody>
        {tbody}
        </tbody>
      </table>
      <p class="meta">rows: {len(rows)} · source: data/{_html.escape(csv_path.name)} · json: api/{_html.escape(sec_id)}.json</p>
    </div>
    <div class="charts">
      <div class="chart-row">
        <div id="chart-price-{_html.escape(sec_id)}" class="chart"></div>
        <div id="chart-flow-{_html.escape(sec_id)}" class="chart"></div>
      </div>
    </div>
  </div>
</section>""")

    # 섹션 네비
    def _id_from(sec_html: str) -> str:
        try:
            return sec_html.split('id="', 1)[1].split('"', 1)[0]
        except Exception:
            return "section"

    nav = "".join(f'<a href="#{_id_from(s)}">{_id_from(s)}</a>' for s in sections)

    # HTML + Plotly 로딩 + JS 지표 계산 & 그리기
    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KRX 기업별 데이터 테이블</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
  header {{ margin-bottom: 20px; }}
  .meta-top {{ color:#666; font-size:14px; }}
  .nav {{ display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:8px; }}
  .nav a {{ font-size:13px; text-decoration:none; color:#2563eb; }}
  section {{ margin: 32px 0; }}
  h2 {{ font-size: 18px; margin: 12px 0; }}
  .grid {{ display: flex; flex-direction: column; gap: 12px; }}
  .scroll {{ overflow:auto; max-height: 50vh; border:1px solid #e5e7eb; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; white-space: nowrap; }}
  thead th {{ position: sticky; top:0; background:#fafafa; }}
  .meta {{ color:#666; font-size:12px; }}
  .charts {{ width: 100%; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .chart {{ width: 100%; height: 420px; border:1px solid #e5e7eb; }}
  @media (max-width: 1000px) {{
    .chart-row {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<header>
  <h1>KRX 기업별 데이터 테이블</h1>
  <div class="meta-top">생성 시각: {generated} · 타임존: Asia/Seoul</div>
  <nav class="nav">{nav}</nav>
</header>

{''.join(sections) if sections else '<p>표시할 데이터가 없습니다.</p>'}

<script>
/* ---------- 지표 계산 유틸 ---------- */
function SMA(arr, n) {{
  const out = Array(arr.length).fill(null);
  let sum = 0; let q = [];
  for (let i=0;i<arr.length;i++) {{
    const v = Number(arr[i]) || 0;
    q.push(v); sum += v;
    if (q.length > n) sum -= q.shift();
    if (q.length === n) out[i] = sum / n;
  }}
  return out;
}}
function EMA(arr, n) {{
  const out = Array(arr.length).fill(null);
  const k = 2/(n+1); let prev = null;
  for (let i=0;i<arr.length;i++) {{
    const v = Number(arr[i]) || 0;
    if (prev == null) prev = v;
    else prev = v*k + prev*(1-k);
    out[i] = prev;
  }}
  return out;
}}
function STD(arr, n) {{
  const out = Array(arr.length).fill(null);
  let q=[];
  for (let i=0;i<arr.length;i++) {{
    const v = Number(arr[i]) || 0;
    q.push(v);
    if (q.length>n) q.shift();
    if (q.length===n) {{
      const m = q.reduce((a,b)=>a+b,0)/n;
      const s2 = q.reduce((a,b)=>a+(b-m)*(b-m),0)/n;
      out[i] = Math.sqrt(s2);
    }}
  }}
  return out;
}}
function RSI(close, n=14) {{
  const out = Array(close.length).fill(null);
  let gains=0, losses=0;
  for (let i=1;i<close.length;i++) {{
    const ch = close[i]-close[i-1];
    const g = ch>0?ch:0, l = ch<0?(-ch):0;
    if (i<=n) {{ gains+=g; losses+=l; if (i===n) {{
      const rs = (gains/n)/((losses/n)||1e-9);
      out[i]=100-100/(1+rs);
    }}}}
    else {{
      gains = (gains*(n-1)+g)/n;
      losses= (losses*(n-1)+l)/n;
      const rs = (gains)/((losses)||1e-9);
      out[i]=100-100/(1+rs);
    }}
  }}
  return out;
}}
function MACD(close, fast=12, slow=26, sig=9) {{
  const emaFast = EMA(close, fast);
  const emaSlow = EMA(close, slow);
  const macd = emaFast.map((v,i)=> (v!=null && emaSlow[i]!=null) ? v-emaSlow[i] : null);
  const signal = EMA(macd.map(v=>v??0), sig);
  const hist = macd.map((v,i)=> (v!=null && signal[i]!=null)? v - signal[i] : null);
  return {{ macd, signal, hist }};
}}
function bbBands(close, n=20, k=2) {{
  const ma = SMA(close, n);
  const sd = STD(close, n);
  const upper = ma.map((m,i)=> (m!=null && sd[i]!=null)? m + k*sd[i] : null);
  const lower = ma.map((m,i)=> (m!=null && sd[i]!=null)? m - k*sd[i] : null);
  return {{ ma, upper, lower }};
}}

/* ---------- 데이터 파싱 & 차트 ---------- */
async function renderOne(secId) {{
  const url = 'api/'+secId+'.json';
  const res = await fetch(url, {{cache:'no-store'}});
  if (!res.ok) return;
  const j = await res.json();
  const cols = j.columns;
  const idx = (name)=> cols.indexOf(name);

  // 컬럼 인덱스
  const iDate = idx('일자'),
        iOpen = idx('시가'), iHigh=idx('고가'), iLow=idx('저가'), iClose=idx('종가'), iVol=idx('거래량'),
        iFor = idx('외국인 합계'), iInst = idx('기관 합계'),
        iShortR = idx('공매도비중'), iShortBR = idx('공매도잔고비중');

  const rows = j.rows; // 최신→과거 순 (CSV 저장 로직 기준)
  const date = rows.map(r=>r[iDate]);
  const open = rows.map(r=>Number((r[iOpen]||'0').replace(/,/g,'')));
  const high = rows.map(r=>Number((r[iHigh]||'0').replace(/,/g,'')));
  const low  = rows.map(r=>Number((r[iLow] ||'0').replace(/,/g,'')));
  const close= rows.map(r=>Number((r[iClose]||'0').replace(/,/g,'')));
  const vol  = rows.map(r=>Number((r[iVol]  ||'0').replace(/,/g,'')));

  const foreign = rows.map(r=>Number((r[iFor]   ||'0').replace(/,/g,'')));
  const inst    = rows.map(r=>Number((r[iInst]  ||'0').replace(/,/g,'')));
  const shortR  = rows.map(r=>parseFloat(String(r[iShortR] || '0').toString().replace('%','')));   // 공매도비중(%)
  const shortBR = rows.map(r=>parseFloat(String(r[iShortBR]|| '0').toString().replace('%','')));   // 잔고비중(%)

  // 지표
  const ma20 = SMA(close, 20), ma60 = SMA(close, 60), ma120 = SMA(close, 120);
  const bb = bbBands(close, 20, 2);
  const rsi = RSI(close, 14);
  const {{ macd, signal, hist }} = MACD(close, 12, 26, 9);

  // ---------- 차트 1: 가격(캔들) + MA + 볼린저 + RSI + MACD ----------
  // 서브플롯 도메인: 가격(60%), RSI(15%), MACD(25%)
  const layout1 = {{
    grid: {{rows:3, columns:1, pattern:'independent', roworder:'top to bottom'}},
    xaxis:  {{domain:[0,1]}},
    yaxis:  {{domain:[0.40,1.00], title:'Price'}},
    xaxis2: {{anchor:'y2'}},
    yaxis2: {{domain:[0.25,0.40], title:'RSI', range:[0,100]}},
    xaxis3: {{anchor:'y3'}},
    yaxis3: {{domain:[0.00,0.25], title:'MACD'}},
    legend: {{orientation:'h', y:1.08}},
    margin: {{t:30,l:50,r:50,b:30}},
    hovermode:'x unified',
  }};

  const traces1 = [
    {{type:'candlestick', x:date, open, high, low, close, name:'OHLC', xaxis:'x', yaxis:'y'}},
    {{type:'scatter', mode:'lines', x:date, y:ma20, name:'MA20', xaxis:'x', yaxis:'y'}},
    {{type:'scatter', mode:'lines', x:date, y:ma60, name:'MA60', xaxis:'x', yaxis:'y'}},
    {{type:'scatter', mode:'lines', x:date, y:ma120, name:'MA120', xaxis:'x', yaxis:'y'}},
    {{type:'scatter', mode:'lines', x:date, y:bb.upper, name:'BB upper', xaxis:'x', yaxis:'y', line:{{dash:'dot'}} }},
    {{type:'scatter', mode:'lines', x:date, y:bb.ma,    name:'BB mid',   xaxis:'x', yaxis:'y', line:{{dash:'dot'}} }},
    {{type:'scatter', mode:'lines', x:date, y:bb.lower, name:'BB lower', xaxis:'x', yaxis:'y', line:{{dash:'dot'}} }},
    // RSI
    {{type:'scatter', mode:'lines', x:date, y:rsi, name:'RSI(14)', xaxis:'x2', yaxis:'y2'}},
    // MACD
    {{type:'bar', x:date, y:hist, name:'MACD hist', xaxis:'x3', yaxis:'y3', opacity:0.5}},
    {{type:'scatter', mode:'lines', x:date, y:macd, name:'MACD', xaxis:'x3', yaxis:'y3'}},
    {{type:'scatter', mode:'lines', x:date, y:signal, name:'Signal', xaxis:'x3', yaxis:'y3'}},
  ];
  Plotly.newPlot('chart-price-'+secId, traces1, layout1, {{responsive:true, displaylogo:false}});

  // ---------- 차트 2: 외국인/기관 수급 + 공매도비중/잔고비중 ----------
  const layout2 = {{
    barmode:'relative',
    yaxis: {{title:'Net (Shares/Vol)'}},
    yaxis2: {{title:'Short %', overlaying:'y', side:'right'}},
    margin: {{t:30,l:50,r:50,b:30}},
    hovermode:'x unified',
    legend: {{orientation:'h', y:1.08}},
  }};
  const traces2 = [
    {{type:'bar', x:date, y:inst, name:'기관 합계'}},
    {{type:'bar', x:date, y:foreign, name:'외국인 합계'}},
    {{type:'scatter', mode:'lines', x:date, y:shortR, name:'공매도비중(%)', yaxis:'y2'}},
    {{type:'scatter', mode:'lines', x:date, y:shortBR, name:'공매도잔고비중(%)', yaxis:'y2'}},
  ];
  Plotly.newPlot('chart-flow-'+secId, traces2, layout2, {{responsive:true, displaylogo:false}});
}}

(async function main(){{
  const ids = Array.from(document.querySelectorAll('section[id]')).map(s=>s.id);
  for (const id of ids) {{
    try {{ await renderOne(id); }} catch(e) {{ console.error(id, e); }}
  }}
}})();
</script>

<footer style="margin-top:40px;color:#666;font-size:12px">
  Published via GitHub Pages · Per-ticker JSON: /api/*.json
</footer>
</body>
</html>"""
    (DOCS_DIR / "index.html").write_text(html_doc, encoding="utf-8")
    logging.info("index.html 생성 완료 → %s", DOCS_DIR / "index.html")


# ============================================================
# 엔트리포인트
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="KRX 일별 데이터 수집 & CSV 업데이트 & 웹문서 생성")
    parser.add_argument("--company-list", default=str(DATA_DIR / "company_list.txt"))
    parser.add_argument("--run-on-holiday", default="true",
                        help="휴장일에도 실행(전 영업일 데이터 사용) (true/false)")
    parser.add_argument("--rows-limit", default=None,
                        help="index.html 및 JSON에 포함할 최대 행 수(최신 상위). 기본: 전체")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    run_on_holiday = str(args.run_on_holiday).lower() in ("1","true","yes","y")
    rows_limit = int(args.rows_limit) if args.rows_limit else None

    # 준비
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    API_DIR.mkdir(parents=True, exist_ok=True)

    try:
        companies = read_company_list(Path(args.company_list))
    except Exception as e:
        logging.exception("기업 리스트 로딩 실패: %s", e)
        return

    if not companies:
        logging.warning("수집 대상 기업이 없습니다.")
        return

    # 수집/업데이트
    changed = False
    for name, ticker in companies:
        try:
            time.sleep(SLEEP_SEC)
            updated = upsert_company(name, ticker, run_on_holiday)
            changed = changed or updated
        except Exception as e:
            logging.exception("[%s,%s] 처리 중 오류: %s", name, ticker, e)

    # 산출물
    emit_per_ticker_json(companies, rows_limit=rows_limit)
    emit_index_html(companies, rows_limit=rows_limit)

    if changed:
        logging.info("변경사항이 있습니다. Git 커밋 단계에서 반영됩니다.")
    else:
        logging.info("변경사항 없음.")


if __name__ == "__main__":
    main()
