import argparse
import logging
import os
import json
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
OUTPUT_SUFFIX = "_stock_data.csv"
ENCODING = "utf-8-sig"     # 엑셀 호환
SLEEP_SEC = 0.3            # API 과호출 방지
WINDOW_DAYS_INIT = 370     # 신규 생성 시 과거 1년+α

REQ_COLS = [
    "일자","시가","고가","저가","종가","거래량","등락률",
    "기관 합계","기타법인","개인","외국인 합계","전체",
    "공매도","공매도비중","공매도잔고","공매도잔고비중"
]

KST = tz.gettz("Asia/Seoul")

# pykrx 내부 로그 묵음
for name in ["pykrx", "pykrx.website", "pykrx.website.comm", "pykrx.website.comm.util"]:
    logging.getLogger(name).disabled = True

# =========================
# 유틸
# =========================
def kst_today_date():
    return datetime.now(tz=KST).date()

def yyyymmdd(d):
    return d.strftime("%Y%m%d")

def empty_with_cols(cols):
    data = {}
    for c in cols:
        data[c] = pd.Series(dtype="object") if c == "일자" else pd.Series(dtype="float64")
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
            rows.append((name, ticker.zfill(6)))
    return rows

def last_trading_day_by_ohlcv(ticker: str, today):
    start = today - timedelta(days=30)
    df = stock.get_market_ohlcv(yyyymmdd(start), yyyymmdd(today), ticker)
    if df is None or df.empty:
        start = today - timedelta(days=90)
        df = stock.get_market_ohlcv(yyyymmdd(start), yyyymmdd(today), ticker)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: 최근 거래 자료 없음")
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
        "기관합계":"기관 합계", "외국인합계":"외국인 합계",
        "기관 합계":"기관 합계", "외국인 합계":"외국인 합계",
        "개인":"개인", "기타법인":"기타법인", "전체":"전체"
    }
    df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
    for need in ["기관 합계","기타법인","개인","외국인 합계","전체"]:
        if need not in df.columns:
            df[need] = 0
    return df[["일자","기관 합계","기타법인","개인","외국인 합계","전체"]]

def rename_short_cols(df: pd.DataFrame, is_balance=False) -> pd.DataFrame:
    if df is None or df.empty or "일자" not in df.columns:
        base = ["공매도잔고","공매도잔고비중"] if is_balance else ["공매도","공매도비중"]
        return empty_with_cols(["일자"] + base)
    dfc = df.copy()
    if is_balance:
        amt = next((c for c in dfc.columns if any(k in c for k in ["공매도잔고","잔고","BAL_QTY"])), None)
        rto = next((c for c in dfc.columns if any(k in c for k in ["공매도잔고비중","잔고비중","BAL_RTO"])), None)
        dfc["공매도잔고"] = pd.to_numeric(dfc[amt], errors="coerce") if amt else 0
        dfc["공매도잔고비중"] = pd.to_numeric(dfc[rto], errors="coerce") if rto else 0.0
        keep = ["일자","공매도잔고","공매도잔고비중"]
    else:
        amt = next((c for c in dfc.columns if any(k in c for k in ["공매도","공매도거래량","거래량"])), None)
        rto = next((c for c in dfc.columns if any(k in c for k in ["공매도비중","비중"])), None)
        dfc["공매도"] = pd.to_numeric(dfc[amt], errors="coerce") if amt else 0
        dfc["공매도비중"] = pd.to_numeric(dfc[rto], errors="coerce") if rto else 0.0
        keep = ["일자","공매도","공매도비중"]
    return dfc[keep]

def ensure_all_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQ_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[REQ_COLS]

# ---------- CSV 파일명 규칙: <이름>_<6자리티커>_stock_data.csv ----------
def csv_path_for(eng_name: str, ticker: str) -> Path:
    return DATA_DIR / f"{eng_name}_{str(ticker).zfill(6)}{OUTPUT_SUFFIX}"

def fetch_block(ticker: str, start_d, end_d) -> pd.DataFrame:
    s, e = yyyymmdd(start_d), yyyymmdd(end_d)
    ohlcv = stock.get_market_ohlcv(s, e, ticker)
    df1 = normalize_date_index(ohlcv)

    inv = stock.get_market_trading_volume_by_date(s, e, ticker)
    df2 = rename_investor_cols(normalize_date_index(inv))

    try:
        sv = stock.get_shorting_volume_by_date(s, e, ticker)
    except Exception:
        sv = pd.DataFrame()
    df3 = rename_short_cols(normalize_date_index(sv), is_balance=False)

    try:
        sb = stock.get_shorting_balance_by_date(s, e, ticker)
    except Exception:
        sb = pd.DataFrame()
    df4 = rename_short_cols(normalize_date_index(sb), is_balance=True)

    df = df1.merge(df2, on="일자", how="left") \
            .merge(df3, on="일자", how="left") \
            .merge(df4, on="일자", how="left")
    df = ensure_all_cols(df)
    for c in [c for c in df.columns if c != "일자"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df.sort_values("일자", ascending=False)

# =========================
# (신규) 공매도 잔고/비중 '2거래일 전 값' 반영 함수
# =========================
def apply_short_balance_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    최신 정렬(내림차순) 기준으로,
    0행(현거래일), 1행(1거래일 전)의 '공매도잔고/공매도잔고비중'을
    2행(2거래일 전)의 값으로 덮어씁니다.
    - 2거래일 전 데이터가 없으면(행<3) 변경하지 않음.
    """
    cols = ["공매도잔고", "공매도잔고비중"]
    if not all(c in df.columns for c in cols):
        return df
    if df.empty:
        return df

    df = df.copy()
    # '일자' 기준 내림차순 보장
    try:
        df["__dt__"] = pd.to_datetime(df["일자"], errors="coerce")
        df.sort_values("__dt__", ascending=False, inplace=True)
        df.drop(columns="__dt__", inplace=True)
    except Exception:
        df.sort_values("일자", ascending=False, inplace=True)

    if len(df) >= 3:
        ref = df.iloc[2][cols].values  # 2거래일 전
        df.iloc[0, df.columns.get_indexer(cols)] = ref
        df.iloc[1, df.columns.get_indexer(cols)] = ref
    return df

def upsert_company(eng_name: str, ticker: str, run_on_holiday: bool):
    out_path = csv_path_for(eng_name, ticker)
    today = kst_today_date()
    end_date = last_trading_day_by_ohlcv(ticker, today)

    if out_path.exists():
        base = pd.read_csv(out_path, encoding=ENCODING)
        last_have = None if base.empty else pd.to_datetime(base["일자"], errors="coerce").dt.date.max()
        start_date = (last_have + timedelta(days=1)) if last_have else (end_date - timedelta(days=WINDOW_DAYS_INIT))
    else:
        start_date = end_date - timedelta(days=WINDOW_DAYS_INIT)

    if (end_date < today) and (not run_on_holiday) and (not out_path.exists()):
        logging.info("[%s] 휴장일(run_on_holiday=False) → 신규 생성 스킵", eng_name)
        return False

    if start_date > end_date:
        logging.info("[%s] 최신 상태 (추가 데이터 없음)", eng_name)
        return False

    logging.info("[%s] 수집: %s ~ %s (티커 %s)", eng_name, start_date, end_date, ticker)
    df = fetch_block(ticker, start_date, end_date)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        base = pd.read_csv(out_path, encoding=ENCODING)
        merged = pd.concat([base, df], ignore_index=True)
        merged.drop_duplicates(subset=["일자"], keep="last", inplace=True)
        merged.sort_values("일자", ascending=False, inplace=True)

        # === 여기서 최신/전일의 공매도잔고/비중을 2거래일 전 값으로 치환 ===
        merged = apply_short_balance_lag(merged)

        merged.to_csv(out_path, index=False, encoding=ENCODING, lineterminator="\n")
        logging.info("[%s] 업데이트 → %s", eng_name, out_path)
    else:
        # 신규 파일 생성 직전에도 동일 규칙 적용
        df = apply_short_balance_lag(df)
        df.to_csv(out_path, index=False, encoding=ENCODING, lineterminator="\n")
        logging.info("[%s] 신규 생성 → %s", eng_name, out_path)
    return True

# =========================
# 기업별 JSON + index.html 생성
#  - 단일 index.json 생성 없음
# =========================
def emit_per_ticker_json(companies, rows_limit=None):
    api_dir = Path(os.getenv("GITHUB_WORKSPACE", ".")) / "docs" / "api"
    api_dir.mkdir(parents=True, exist_ok=True)
    cnt = 0
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

        item = {
            "name": name,
            "ticker": str(ticker).zfill(6),
            "columns": [str(c) for c in df.columns],
            "rows": df.astype(str).values.tolist(),
            "row_count": int(len(df)),
        }
        out = api_dir / f"{name}_{str(ticker).zfill(6)}.json"
        out.write_text(json.dumps(item, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        cnt += 1
    logging.info("기업별 JSON 생성: %d개", cnt)

def emit_index_html(companies, rows_limit=None):
    import html as _html
    docs_dir = Path(os.getenv("GITHUB_WORKSPACE", ".")) / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    sections = []
    generated = datetime.now(tz=KST).strftime("%Y-%m-%d %H:%M:%S %Z")

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

        thead = "".join(f"<th>{_html.escape(c)}</th>" for c in columns)
        tbody = "\n".join(
            "<tr>" + "".join(f"<td>{_html.escape(v)}</td>" for v in row) + "</tr>" for row in rows
        )
        sec_id = f"{name}_{str(ticker).zfill(6)}"
        sections.append(f"""
<section id="{_html.escape(sec_id)}">
  <h2>{_html.escape(name)} ({str(ticker).zfill(6)})</h2>
  <div class="scroll">
    <table>
      <thead><tr>{thead}</tr></thead>
      <tbody>
      {tbody}
      </tbody>
    </table>
  </div>
  <p class="meta">rows: {len(rows)} · source: data/{_html.escape(csv_path.name)} · json: api/{_html.escape(sec_id)}.json</p>
</section>""")

    def _id_from(sec_html: str) -> str:
        try:
            return sec_html.split('id="', 1)[1].split('"', 1)[0]
        except Exception:
            return "section"

    nav = "".join(f'<a href="#{_id_from(s)}">{_id_from(s)}</a>' for s in sections)

    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KRX 기업별 데이터 테이블</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
  header {{ margin-bottom: 20px; }}
  .meta-top {{ color:#666; font-size:14px; }}
  .nav {{ display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:8px; }}
  .nav a {{ font-size:13px; text-decoration:none; color:#2563eb; }}
  section {{ margin: 32px 0; }}
  h2 {{ font-size: 18px; margin: 12px 0; }}
  .scroll {{ overflow:auto; max-height: 60vh; border:1px solid #e5e7eb; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; white-space: nowrap; }}
  thead th {{ position: sticky; top:0; background:#fafafa; }}
  .meta {{ color:#666; font-size:12px; }}
</style>
</head>
<body>
<header>
  <h1>KRX 기업별 데이터 테이블</h1>
  <div class="meta-top">생성 시각: {generated} · 타임존: Asia/Seoul</div>
  <nav class="nav">{nav}</nav>
</header>
{''.join(sections) if sections else '<p>표시할 데이터가 없습니다.</p>'}
<footer style="margin-top:40px;color:#666;font-size:12px">
  Published via GitHub Pages · Per-ticker JSON: /api/*.json
</footer>
</body>
</html>"""
    (docs_dir / "index.html").write_text(html_doc, encoding="utf-8")
    logging.info("index.html 생성 완료 → %s", docs_dir / "index.html")

# =========================
# 엔트리포인트
# =========================
def main():
    parser = argparse.ArgumentParser(description="KRX 일별 데이터 수집 & CSV 업데이트")
    parser.add_argument("--company-list", default=str(DATA_DIR / "company_list.txt"))
    parser.add_argument("--run-on-holiday", default="true", help="휴장일에도 실행 (true/false)")
    parser.add_argument("--rows-limit", default=os.getenv("ROWS_LIMIT", "").strip(),
                        help="HTML/JSON 포함 최대 행 수 (빈 값이면 전량)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    run_on_holiday = str(args.run_on_holiday).lower() in ("1","true","yes","y")
    rows_limit = None if args.rows_limit in ("", "0", "none", "None") else int(args.rows_limit)

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

    if changed:
        logging.info("변경사항 존재 → 커밋 단계에서 반영됩니다.")
    else:
        logging.info("변경사항 없음.")

    # 단일 index.json은 만들지 않음 → 기업별 JSON + index.html만 생성
    emit_per_ticker_json(companies, rows_limit=rows_limit)
    emit_index_html(companies, rows_limit=rows_limit)

if __name__ == "__main__":
    main()
