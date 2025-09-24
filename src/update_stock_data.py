import argparse
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
OUTPUT_SUFFIX = "_stock_data.csv"
ENCODING = "utf-8-sig"     # 엑셀 호환
SLEEP_SEC = 0.3            # API 과도 호출 방지
WINDOW_DAYS_INIT = 370     # 신규 생성시 과거 1년+α

REQ_COLS = [
    "일자","시가","고가","저가","종가","거래량","등락률",
    "기관 합계","기타법인","개인","외국인 합계","전체",
    "공매도","공매도비중","공매도잔고","공매도잔고비중"
]

KST = tz.gettz("Asia/Seoul")


# ============================================================
# 로거 설정 (pykrx 잘못된 logging 호출 묵음 처리)
# ============================================================
for name in ["pykrx", "pykrx.website", "pykrx.website.comm", "pykrx.website.comm.util"]:
    logging.getLogger(name).disabled = True


# ============================================================
# 유틸 함수
# ============================================================
def kst_today_date():
    return datetime.now(tz=KST).date()

def yyyymmdd(d):
    return d.strftime("%Y%m%d")

def empty_with_cols(cols):
    """지정한 컬럼을 가진 '빈 DF'를 반환합니다(merge 키 보장)."""
    data = {}
    for c in cols:
        # '일자'는 문자열로, 나머지는 float 기본
        if c == "일자":
            data[c] = pd.Series(dtype="object")
        else:
            data[c] = pd.Series(dtype="float64")
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

def last_trading_day_by_ohlcv(ticker: str, today: datetime.date):
    """최근 구간의 마지막 거래일을 OHLCV로 판정"""
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
    # 컬럼 순서 고정
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

    # 4) 공매도 잔고/비중 (예외 안전) ← 문제 발생 지점
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
    df = df.sort_values("일자", ascending=False)
    return df

def upsert_company(eng_name: str, ticker: str, run_on_holiday: bool):
    out_path = DATA_DIR / f"{eng_name}{OUTPUT_SUFFIX}"

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
        merged = merged.sort_values("일자", ascending=False)
        merged.to_csv(out_path, index=False, encoding=ENCODING, lineterminator="\r\n")
        logging.info("[%s] 업데이트 완료 → %s", eng_name, out_path)
    else:
        df.to_csv(out_path, index=False, encoding=ENCODING, lineterminator="\r\n")
        logging.info("[%s] 신규 생성 완료 → %s", eng_name, out_path)
    return True


# ============================================================
# 엔트리포인트
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="KRX 일별 데이터 수집 & CSV 업데이트")
    parser.add_argument("--company-list", default=str(DATA_DIR / "company_list.txt"))
    parser.add_argument("--run-on-holiday", default="true",
                        help="휴장일에도 실행(전 영업일 데이터 사용) (true/false)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    run_on_holiday = str(args.run_on_holiday).lower() in ("1","true","yes","y")

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
        logging.info("변경사항이 있습니다. Git 커밋 단계에서 반영됩니다.")
    else:
        logging.info("변경사항 없음.")


if __name__ == "__main__":
    main()
