# -*- coding: utf-8 -*-
"""
KRX 수집/업데이트 → per-ticker JSON → 표 + 2개 차트(index.html)
- 차트 데이터는 섹션마다 inline JSON으로 넣어 CORS 없이 file://에서도 동작
- GitHub Pages에서도 동작(외부 fetch 없이 inline만으로 렌더)

CSV 스키마:
일자,시가,고가,저가,종가,거래량,등락률,기관 합계,기타법인,개인,외국인 합계,전체,공매도,공매도비중,공매도잔고,공매도잔고비중
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

# =========================
# 설정
# =========================
DATA_DIR = Path(os.getenv("GITHUB_WORKSPACE", ".")) / "data"
DOCS_DIR = Path(os.getenv("GITHUB_WORKSPACE", ".")) / "docs"
API_DIR = DOCS_DIR / "api"

OUTPUT_SUFFIX = "_stock_data.csv"
ENCODING = "utf-8-sig"
SLEEP_SEC = 0.3
WINDOW_DAYS_INIT = 370
SORT_DESC = True  # 최신→과거

REQ_COLS = [
    "일자","시가","고가","저가","종가","거래량","등락률",
    "기관 합계","기타법인","개인","외국인 합계","전체",
    "공매도","공매도비중","공매도잔고","공매도잔고비중"
]

KST = tz.gettz("Asia/Seoul")
for name in ["pykrx", "pykrx.website", "pykrx.website.comm", "pykrx.website.comm.util"]:
    logging.getLogger(name).disabled = True

# =========================
# 유틸
# =========================
def kst_now():
    return datetime.now(tz=KST)

def kst_today_date():
    return kst_now().date()

def yyyymmdd(d):
    return d.strftime("%Y%m%d")

def empty_with_cols(cols):
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

def csv_path_for(eng_name: str, _ticker: str) -> Path:
    return DATA_DIR / f"{eng_name}{OUTPUT_SUFFIX}"

def last_trading_day_by_ohlcv(ticker: str, today: datetime.date):
    start = today - timedelta(days=30)
    df = stock.get_market_ohlcv(yyyymmdd(start), yyyymmdd(today), ticker)
    if df is None or df.empty:
        start = today - timedelta(days=90)
        df = stock.get_market_ohlcv(yyyymmdd(start), yyyymmdd(today), ticker)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker} : 최근 거래 자료가 없습니다.")
    return pd.to_datetime(df.index.max()).date()

def normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
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
    if df is None or df.empty or "일자" not in df.columns:
        return empty_with_cols(["일자","기관 합계","기타법인","개인","외국인 합계","전체"])
    mapping = {
        "기관합계": "기관 합계",
        "외국인합계": "외국인 합계",
        "전체": "전체", "개인": "개인", "기타법인": "기타법인",
        "기관 합계": "기관 합계", "외국인 합계": "외국인 합계",
    }
    df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
    for need in ["기관 합계","기타법인","개인","외국인 합계","전체"]:
        if need not in df.columns:
            df[need] = 0
    keep = ["일자","기관 합계","기타법인","개인","외국인 합계","전체"]
    return df[keep]

def rename_short_cols(df: pd.DataFrame, is_balance=False) -> pd.DataFrame:
    if df is None or df.empty or "일자" not in df.columns:
        return empty_with_cols(["일자"] + (["공매도잔고","공매도잔고비중"] if is_balance else ["공매도","공매도비중"]))
    dfc = df.copy()
    if is_balance:
        amt = next((c for c in dfc.columns if any(k in c for k in ["공매도잔고","잔고","BAL_QTY"])), None)
        rto = next((c for c in dfc.columns if any(k in c for k in ["공매도잔고비중","잔고비중","BAL_RTO"])), None)
        dfc["공매도잔고"] = pd.to_numeric(dfc[amt], errors="coerce") if amt else 0
        dfc["공매도잔고비중"] = pd.to_numeric(dfc[rto], errors="coerce") if rto else 0.0
        return dfc[["일자","공매도잔고","공매도잔고비중"]]
    else:
        amt = next((c for c in dfc.columns if any(k in c for k in ["공매도","공매도거래량","거래량"])), None)
        rto = next((c for c in dfc.columns if any(k in c for k in ["공매도비중","비중"])), None)
        dfc["공매도"] = pd.to_numeric(dfc[amt], errors="coerce") if amt else 0
        dfc["공매도비중"] = pd.to_numeric(dfc[rto], errors="coerce") if rto else 0.0
        return dfc[["일자","공매도","공매도비중"]]

def ensure_all_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQ_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[REQ_COLS]

def fetch_block(ticker: str, start_d: datetime.date, end_d: datetime.date) -> pd.DataFrame:
    s, e = yyyymmdd(start_d), yyyymmdd(end_d)
    ohlcv = stock.get_market_ohlcv(s, e, ticker); df1 = normalize_date_index(ohlcv)
    inv   = stock.get_market_trading_volume_by_date(s, e, ticker); df2 = rename_investor_cols(normalize_date_index(inv))
    try: sv = stock.get_shorting_volume_by_date(s, e, ticker)
    except Exception: sv = pd.DataFrame()
    df3 = rename_short_cols(normalize_date_index(sv), is_balance=False)
    try: sb = stock.get_shorting_balance_by_date(s, e, ticker)
    except Exception: sb = pd.DataFrame()
    df4 = rename_short_cols(normalize_date_index(sb), is_balance=True)

    df = df1.merge(df2, on="일자", how="left").merge(df3, on="일자", how="left").merge(df4, on="일자", how="left")
    df = ensure_all_cols(df)
    for c in [c for c in df.columns if c != "일자"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
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

    if (end_date < today) and (not run_on_holiday) and (not out_path.exists()):
        logging.info("[%s] 휴장일(run_on_holiday=False) → 신규 생성 건 스킵", eng_name)
        return False
    if start_date > end_date:
        logging.info("[%s] 최신 상태 (추가할 데이터 없음).", eng_name); return False

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

# =========================
# 산출물 (JSON + index.html)
# =========================
def emit_per_ticker_json(companies, rows_limit=None):
    API_DIR.mkdir(parents=True, exist_ok=True)
    cnt = 0
    for name, ticker in companies:
        csv_path = csv_path_for(name, ticker)
        if not csv_path.exists():
            continue
        try: df = pd.read_csv(csv_path, encoding=ENCODING)
        except Exception: df = pd.read_csv(csv_path)
        if df.empty: continue
        df_use = df.head(int(rows_limit)) if rows_limit else df
        payload = {
            "name": name, "ticker": str(ticker).zfill(6),
            "columns": list(df_use.columns),
            "rows": df_use.astype(object).where(pd.notna(df_use), "").values.tolist(),
            "generated_at": kst_now().strftime("%Y-%m-%d %H:%M:%S %Z")
        }
        (API_DIR / f"{name}_{str(ticker).zfill(6)}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        cnt += 1
    logging.info("per-ticker JSON 생성 완료: %d files → %s", cnt, API_DIR)

def emit_index_html(companies, rows_limit=None):
    import html as _html
    from string import Template

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
        rows = df.astype(object).where(pd.notna(df), "").astype(str).values.tolist()

        thead = "".join(f"<th>{_html.escape(c)}</th>" for c in columns)
        tbody = "\n".join(
            "<tr>" + "".join(f"<td>{_html.escape(v)}</td>" for v in row) + "</tr>" for row in rows
        )
        sec_id = f"{name}_{str(ticker).zfill(6)}"

        # 표 + 차트 컨테이너 + inline JSON
        payload = {
            "name": name,
            "ticker": str(ticker).zfill(6),
            "columns": columns,
            "rows": rows,
        }
        sections.append(
            f"""
<section id="{_html.escape(sec_id)}">
  <h2>{_html.escape(name)} ({str(ticker).zfill(6)})</h2>
  <div class="grid">
    <div class="scroll">
      <table>
        <thead><tr>{thead}</tr></thead>
        <tbody>{tbody}</tbody>
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
  <script id="data-{_html.escape(sec_id)}" type="application/json">{_html.escape(json.dumps(payload, ensure_ascii=False))}</script>
</section>"""
        )

    # 섹션 네비
    def _id_from(sec_html: str) -> str:
        try:
            return sec_html.split('id="', 1)[1].split('"', 1)[0]
        except Exception:
            return "section"

    nav = "".join(f'<a href="#{_id_from(s)}">{_id_from(s)}</a>' for s in sections)

    # ⚠️ f-string 금지! Template로 치환 (중괄호 충돌 방지)
    html_template = Template("""<!doctype html>
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
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
  header { margin-bottom: 20px; }
  .meta-top { color:#666; font-size:14px; }
  .nav { display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:8px; }
  .nav a { font-size:13px; text-decoration:none; color:#2563eb; }
  section { margin: 32px 0; }
  h2 { font-size: 18px; margin: 12px 0; }
  .grid { display: flex; flex-direction: column; gap: 12px; }
  .scroll { overflow:auto; max-height: 50vh; border:1px solid #e5e7eb; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }
  th:first-child, td:first-child { text-align: left; white-space: nowrap; }
  thead th { position: sticky; top:0; background:#fafafa; }
  .meta { color:#666; font-size:12px; }
  .charts { width: 100%; }
  .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .chart { width: 100%; height: 420px; border:1px solid #e5e7eb; }
  @media (max-width: 1000px) {
    .chart-row { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<header>
  <h1>KRX 기업별 데이터 테이블</h1>
  <div class="meta-top">생성 시각: $generated · 타임존: Asia/Seoul</div>
  <nav class="nav">$nav</nav>
</header>

$sections

<script>
/* ===== 유틸 ===== */
function SMA(arr,n){const o=Array(arr.length).fill(null);let s=0,q=[];for(let i=0;i<arr.length;i++){const v=+arr[i]||0;q.push(v);s+=v;if(q.length>n)s-=q.shift();if(q.length===n)o[i]=s/n}return o}
function EMA(arr,n){const o=Array(arr.length).fill(null);const k=2/(n+1);let p=null;for(let i=0;i<arr.length;i++){const v=+arr[i]||0;p=(p==null)?v:v*k+p*(1-k);o[i]=p}return o}
function STD(arr,n){const o=Array(arr.length).fill(null);let q=[];for(let i=0;i<arr.length;i++){const v=+arr[i]||0;q.push(v);if(q.length>n)q.shift();if(q.length===n){const m=q.reduce((a,b)=>a+b,0)/n;const s2=q.reduce((a,b)=>a+(b-m)*(b-m),0)/n;o[i]=Math.sqrt(s2)}}return o}
function RSI(close,n=14){const o=Array(close.length).fill(null);let g=0,l=0;for(let i=1;i<close.length;i++){const ch=close[i]-close[i-1],G=ch>0?ch:0,L=ch<0?-ch:0;if(i<=n){g+=G;l+=L;if(i===n){const rs=(g/n)/((l/n)||1e-9);o[i]=100-100/(1+rs)}}else{g=(g*(n-1)+G)/n;l=(l*(n-1)+L)/n;const rs=g/(l||1e-9);o[i]=100-100/(1+rs)}}return o}
function MACD(close,f=12,s=26,sg=9){const ef=EMA(close,f),es=EMA(close,s),m=ef.map((v,i)=>v!=null&&es[i]!=null?v-es[i]:null),signal=EMA(m.map(v=>v??0),sg),h=m.map((v,i)=>v!=null&&signal[i]!=null?v-signal[i]:null);return{macd:m,signal,hist:h}}
function bbBands(close,n=20,k=2){const ma=SMA(close,n),sd=STD(close,n),u=ma.map((m,i)=>m!=null&&sd[i]!=null?m+k*sd[i]:null),l=ma.map((m,i)=>m!=null&&sd[i]!=null?m-k*sd[i]:null);return{ma,upper:u,lower:l}}
function nnum(x){if(x==null)return 0;return +String(x).replace(/,/g,'').replace(/\\s+/g,'').replace(/%/g,'')||0}
function str(x){return (x==null)?'':String(x)}
function showError(secId,msg){const l=document.getElementById('chart-price-'+secId);const r=document.getElementById('chart-flow-'+secId);if(l)l.innerHTML='<div style="padding:12px;color:#b91c1c;font-size:13px">'+msg+'</div>';if(r)r.innerHTML='<div style="padding:12px;color:#b91c1c;font-size:13px">'+msg+'</div>'}

/* ===== 안전한 컬럼 인덱싱 ===== */
function idxOf(cols, primary, alts=[]){const i=cols.indexOf(primary);if(i>-1)return i;for(const a of alts){const j=cols.indexOf(a);if(j>-1)return j;}return -1;}

/* ===== 렌더링 ===== */
function renderOne(secId){
  const tag=document.getElementById('data-'+secId);
  if(!tag){ showError(secId,'섹션 데이터가 없습니다.'); return; }
  let j=null; try{ j=JSON.parse(tag.textContent); }catch(e){ showError(secId,'섹션 데이터 파싱 실패: '+e); return; }

  const cols=j.columns||[];
  const iDate=idxOf(cols,'일자',['\\ufeff일자','DATE','date']),
        iOpen=idxOf(cols,'시가',['Open','open']),
        iHigh=idxOf(cols,'고가',['High','high']),
        iLow =idxOf(cols,'저가',['Low','low']),
        iClose=idxOf(cols,'종가',['Close','close']),
        iVol =idxOf(cols,'거래량',['Volume','volume']),
        iFor =idxOf(cols,'외국인 합계',['외국인합계','외인합계']),
        iInst=idxOf(cols,'기관 합계',['기관합계']),
        iShortR=idxOf(cols,'공매도비중',['공매도 거래량 비중','비중']),
        iShortBR=idxOf(cols,'공매도잔고비중',['잔고비중']);

  if([iDate,iOpen,iHigh,iLow,iClose].some(i=>i<0)){ showError(secId,'필수 컬럼 누락(일자/시가/고가/저가/종가)'); return; }
  const rows=j.rows||[]; if(!rows.length){ showError(secId,'시계열 행이 없습니다.'); return; }

  const date   = rows.map(r=>str(r[iDate]));
  const open   = rows.map(r=>nnum(r[iOpen]));
  const high   = rows.map(r=>nnum(r[iHigh]));
  const low    = rows.map(r=>nnum(r[iLow]));
  const close  = rows.map(r=>nnum(r[iClose]));
  const vol    = (iVol>=0)? rows.map(r=>nnum(r[iVol])): rows.map(_=>0);

  const foreign= (iFor   >=0)? rows.map(r=>nnum(r[iFor]))  : rows.map(_=>0);
  const inst   = (iInst  >=0)? rows.map(r=>nnum(r[iInst])) : rows.map(_=>0);
  const shortR = (iShortR>=0)? rows.map(r=>nnum(r[iShortR])): rows.map(_=>0);
  const shortBR= (iShortBR>=0)? rows.map(r=>nnum(r[iShortBR])): rows.map(_=>0);

  const ma20=SMA(close,20), ma60=SMA(close,60), ma120=SMA(close,120);
  const bb=bbBands(close,20,2);
  const rsi=RSI(close,14);
  const {macd,signal,hist}=MACD(close,12,26,9);

  const layout1={grid:{rows:3,columns:1,pattern:'independent',roworder:'top to bottom'},
    xaxis:{domain:[0,1]}, yaxis:{domain:[0.40,1.00],title:'Price'},
    xaxis2:{anchor:'y2'}, yaxis2:{domain:[0.25,0.40],title:'RSI',range:[0,100]},
    xaxis3:{anchor:'y3'}, yaxis3:{domain:[0.00,0.25],title:'MACD'},
    legend:{orientation:'h',y:1.08}, margin:{t:30,l:50,r:50,b:30}, hovermode:'x unified'};
  const traces1=[
    {type:'candlestick',x:date,open,high,low,close,name:'OHLC',xaxis:'x',yaxis:'y'},
    {type:'scatter',mode:'lines',x:date,y:ma20,name:'MA20',xaxis:'x',yaxis:'y'},
    {type:'scatter',mode:'lines',x:date,y:ma60,name:'MA60',xaxis:'x',yaxis:'y'},
    {type:'scatter',mode:'lines',x:date,y:ma120,name:'MA120',xaxis:'x',yaxis:'y'},
    {type:'scatter',mode:'lines',x:date,y:bb.upper,name:'BB upper',xaxis:'x',yaxis:'y',line:{dash:'dot'}},
    {type:'scatter',mode:'lines',x:date,y:bb.ma,   name:'BB mid',  xaxis:'x',yaxis:'y',line:{dash:'dot'}},
    {type:'scatter',mode:'lines',x:date,y:bb.lower,name:'BB lower',xaxis:'x',yaxis:'y',line:{dash:'dot'}},
    {type:'scatter',mode:'lines',x:date,y:rsi,name:'RSI(14)',xaxis:'x2',yaxis:'y2'},
    {type:'bar',x:date,y:hist,name:'MACD hist',xaxis:'x3',yaxis:'y3',opacity:0.5},
    {type:'scatter',mode:'lines',x:date,y:macd,name:'MACD',xaxis:'x3',yaxis:'y3'},
    {type:'scatter',mode:'lines',x:date,y:signal,name:'Signal',xaxis:'x3',yaxis:'y3'},
  ];
  Plotly.newPlot('chart-price-'+secId,traces1,layout1,{responsive:true,displaylogo:false});

  const layout2={barmode:'relative', yaxis:{title:'Net (Shares/Vol)'},
                  yaxis2:{title:'Short %',overlaying:'y',side:'right'},
                  margin:{t:30,l:50,r:50,b:30}, hovermode:'x unified',
                  legend:{orientation:'h',y:1.08}};
  const traces2=[
    {type:'bar',x:date,y:inst,name:'기관 합계'},
    {type:'bar',x:date,y:foreign,name:'외국인 합계'},
    {type:'scatter',mode:'lines',x:date,y:shortR,name:'공매도비중(%)',yaxis:'y2'},
    {type:'scatter',mode:'lines',x:date,y:shortBR,name:'공매도잔고비중(%)',yaxis:'y2'},
  ];
  Plotly.newPlot('chart-flow-'+secId,traces2,layout2,{responsive:true,displaylogo:false});
}

(function main(){
  const ids=Array.from(document.querySelectorAll('section[id]')).map(s=>s.id);
  for(const id of ids){ try{ renderOne(id); }catch(e){ showError(id,'렌더링 오류: '+e); } }
})();
</script>

<footer style="margin-top:40px;color:#666;font-size:12px">
  Published via GitHub Pages · Inline JSON rendering (CORS-free)
</footer>
</body>
</html>""")

    html_doc = html_template.substitute(
        generated=generated,
        nav=nav,
        sections="".join(sections),
    )

    (DOCS_DIR / "index.html").write_text(html_doc, encoding="utf-8")
    logging.info("index.html 생성 완료 → %s", DOCS_DIR / "index.html")

# =========================
# 엔트리포인트
# =========================
def main():
    parser = argparse.ArgumentParser(description="KRX 수집 & CSV 업데이트 & 웹문서 생성")
    parser.add_argument("--company-list", default=str(DATA_DIR / "company_list.txt"))
    parser.add_argument("--run-on-holiday", default="true",
                        help="휴장일에도 실행(전 영업일 데이터 사용) (true/false)")
    parser.add_argument("--rows-limit", default=None,
                        help="index.html/JSON에 포함할 최대 행 수(최신 상위). 기본: 전체")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    run_on_holiday = str(args.run_on_holiday).lower() in ("1","true","yes","y")
    rows_limit = int(args.rows_limit) if args.rows_limit else None

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

    changed = False
    for name, ticker in companies:
        try:
            time.sleep(SLEEP_SEC)
            updated = upsert_company(name, ticker, run_on_holiday)
            changed = changed or updated
        except Exception as e:
            logging.exception("[%s,%s] 처리 중 오류: %s", name, ticker, e)

    emit_per_ticker_json(companies, rows_limit=rows_limit)
    emit_index_html(companies, rows_limit=rows_limit)

    if changed:
        logging.info("변경사항이 있습니다. Git 커밋 단계에서 반영됩니다.")
    else:
        logging.info("변경사항 없음.")

if __name__ == "__main__":
    main()
