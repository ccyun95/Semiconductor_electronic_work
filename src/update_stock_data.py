import argparse
import csv
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from dateutil import tz
import time
import pandas as pd

from pykrx import stock

# -----------------------------
# 설정
# -----------------------------
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


# -----------------------------
# 유틸
# -----------------------------
def kst_today_date():
    return datetime.now(tz=KST).date()

def yyyymmdd(d):
    return d.strftime("%Y%m%d")

def read_company_list(path: Path):
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"기업 리스트 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 허용: "Name,005930" 또는 "Name 005930" 또는 "Name\t005930"
            if "," in line:
                name, ticker = [x.strip() for x in line.split(",", 1)]
            else:
                parts = line.replace("\t", " ").split()
                if len(parts) < 2:
                    logging.warning(f"파싱 불가: {line}")
                    continue
                name, ticker = parts[0], parts[1]
            # 티커 6자리 보정
            ticker = ticker.zfill(6)
            rows.append((name, ticker))
    return rows

def last_trading_day_by_ohlcv(ticker: str, today: datetime.date):
    # 최근 21일 구간에서 마지막 거래일 찾기
    start = today - timedelta(days=21)
    df = stock.get_market_ohlcv(yyyymmdd(start), yyyymmdd(today), ticker)
    if df is None or df.empty:
        # 극단적 케이스: 더 넓혀서 시도
        start = today - timedelta(days=60)
        df = stock.get_market_ohlcv(yyyymmdd(start), yyyymmdd(today), ticker)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker} : 최근 60일 내 거래 자료가 없습니다.")
    return df.index.max().date()

def normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # pykrx는 index에 날짜가 들어옵니다.
    if df.index.name is None:
        df.index.name = "일자"
    # 인덱스가 Timestamp/Datetime 인 경우
    try:
        idx = pd.to_datetime(df.index)
    except Exception:
        # 혹시 문자열(YYYYMMDD)인 경우
        idx = pd.to_datetime(df.index.astype(str))
    df.index = idx
    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: "일자"}, inplace=True)
    df["일자"] = df["일자"].dt.strftime("%Y-%m-%d")
    return df

def rename_investor_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    mapping = {
        "기관합계": "기관 합계",
        "외국인합계": "외국인 합계",
        "전체": "전체",
        "개인": "개인",
        "기타법인": "기타법인",
        # 어떤 환경에서는 이미 공백 포함으로 올 수 있어 pass
        "기관 합계": "기관 합계",
        "외국인 합계": "외국인 합계",
    }
    cols = {c: mapping.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)
    # 없는 컬럼 보강
    for need in ["기관 합계","기타법인","개인","외국인 합계","전체"]:
        if need not in df.columns:
            df[need] = 0
    keep = ["일자","기관 합계","기타법인","개인","외국인 합계","전체"]
    return df[keep]

def rename_short_cols(df: pd.DataFrame, is_balance=False) -> pd.DataFrame:
    """공매도/잔고 데이터 컬럼을 표준화"""
    if df is None or df.empty:
        return pd.DataFrame()
    dfc = df.copy()
    # 가능한 컬럼명 후보들 매핑
    vol_candidates = ["공매도","공매도거래량","공매도 거래량","거래량"]
    vol_ratio_candidates = ["공매도비중","공매도 거래량 비중","비중"]
    bal_candidates = ["공매도잔고","잔고","공매도잔고수량","잔고수량"]
    bal_ratio_candidates = ["공매도잔고비중","잔고비중"]

    if is_balance:
        # 잔고
        c_amt, c_ratio = None, None
        for c in dfc.columns:
            if c_amt is None and any(k in c for k in bal_candidates):
                c_amt = c
            if c_ratio is None and any(k in c for k in bal_ratio_candidates):
                c_ratio = c
        if c_amt is None:
            # 어떤 환경에선 '잔고'만 있을 수도
            c_amt = next((c for c in dfc.columns if "잔고" in c), None)
        if c_ratio is None:
            # 비중 누락 시 0으로 채움
            dfc["공매도잔고비중"] = 0.0
        else:
            dfc["공매도잔고비중"] = pd.to_numeric(dfc[c_ratio], errors="coerce").fillna(0.0)
        if c_amt is None:
            dfc["공매도잔고"] = 0
        else:
            dfc["공매도잔고"] = pd.to_numeric(dfc[c_amt], errors="coerce").fillna(0)
        return dfc[["일자","공매도잔고","공매도잔고비중"]]
    else:
        # 공매도 거래량/비중
        c_amt, c_ratio = None, None
        for c in dfc.columns:
            if c_amt is None and any(k in c for k in vol_candidates):
                c_amt = c
            if c_ratio is None and any(k in c for k in vol_ratio_candidates):
                c_ratio = c
        if c_ratio is None:
            dfc["공매도비중"] = 0.0
        else:
            dfc["공매도비중"] = pd.to_numeric(dfc[c_ratio], errors="coerce").fillna(0.0)
        if c_amt is None:
            dfc["공매도"] = 0
        else:
            dfc["공매도"] = pd.to_numeric(dfc[c_amt], errors="coerce").fillna(0)
        return dfc[["일자","공매도","공매도비중"]]

def ensure_all_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQ_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[REQ_COLS]

def fetch_block(ticker: str, start_d: datetime.date, end_d: datetime.date) -> pd.DataFrame:
    s, e = yyyymmdd(start_d), yyyymmdd(end_d)

    # 1) OHLCV
    ohlcv = stock.get_market_ohlcv(s, e, ticker)
    df1 = normalize_date_index(ohlcv)
    # 2) 투자자별 거래실적(거래량)
    inv = stock.get_market_trading_volume_by_date(s, e, ticker)
    df2 = normalize_date_index(inv)
    df2 = rename_investor_cols(df2)
    # 3) 공매도 거래량/비중
    sv = stock.get_shorting_volume_by_date(s, e, ticker)
    df3 = normalize_date_index(sv)
    df3 = rename_short_cols(df3, is_balance=False)
    # 4) 공매도 잔고/비중
    sb = stock.get_shorting_balance_by_date(s, e, ticker)
    df4 = normalize_date_index(sb)
    df4 = rename_short_cols(df4, is_balance=True)

    # 머지
    df = df1.merge(df2, on="일자", how="left")
    df = df.merge(df3, on="일자", how="left")
    df = df.merge(df4, on="일자", how="left")

    # 결측 보강 + 정렬
    df = ensure_all_cols(df)
    # 숫자형 변환(가능한 컬럼)
    num_cols = [c for c in df.columns if c != "일자"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = df.sort_values("일자")
    return df

def upsert_company(eng_name: str, ticker: str, run_on_holiday: bool):
    out_path = DATA_DIR / f"{eng_name}{OUTPUT_SUFFIX}"

    today = kst_today_date()
    # 휴장일에도 실행: end_date는 "마지막 거래일"
    end_date = last_trading_day_by_ohlcv(ticker, today)

    if out_path.exists():
        # 증분 업데이트
        base = pd.read_csv(out_path, encoding=ENCODING)
        if base.empty:
            last_have = None
        else:
            base["일자"] = pd.to_datetime(base["일자"]).dt.date
            last_have = base["일자"].max()
        start_date = (last_have + timedelta(days=1)) if last_have else (end_date - timedelta(days=WINDOW_DAYS_INIT))
    else:
        # 신규: 과거 1년 치
        start_date = end_date - timedelta(days=WINDOW_DAYS_INIT)

    # 만약 오늘이 휴장이고 run_on_holiday=False 라면: 그냥 스킵 (원하실 경우)
    if (end_date < today) and (not run_on_holiday) and (not out_path.exists()):
        logging.info(f"[{eng_name}] 휴장일이며 run_on_holiday=False → 신규 생성 건 스킵")
        return False

    if start_date > end_date:
        logging.info(f"[{eng_name}] 최신 상태 (추가할 데이터 없음).")
        return False

    logging.info(f"[{eng_name}] 수집 구간: {start_date} ~ {end_date} (티커 {ticker})")
    df = fetch_block(ticker, start_date, end_date)

    if out_path.exists():
        base = pd.read_csv(out_path, encoding=ENCODING)
        merged = pd.concat([base, df], ignore_index=True)
        merged.drop_duplicates(subset=["일자"], keep="last", inplace=True)
        merged = merged.sort_values("일자")
        merged.to_csv(out_path, index=False, encoding=ENCODING)
        logging.info(f"[{eng_name}] 업데이트 완료 → {out_path}")
    else:
        # 신규 저장
        df.to_csv(out_path, index=False, encoding=ENCODING)
        logging.info(f"[{eng_name}] 신규 생성 완료 → {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="KRX 일별 데이터 수집 & CSV 업데이트")
    parser.add_argument("--company-list", default=str(DATA_DIR / "company_list.txt"))
    parser.add_argument("--run-on-holiday", default="true", help="휴장일에도 실행(전 영업일 데이터 사용) (true/false)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    run_on_holiday = str(args.run_on_holiday).lower() in ("1","true","yes","y")

    companies = read_company_list(Path(args.company_list))
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
            logging.exception(f"[{name},{ticker}] 처리 중 오류: {e}")

    if changed:
        logging.info("변경사항이 있습니다. Git 커밋 단계에서 반영하세요.")
    else:
        logging.info("변경사항 없음.")


if __name__ == "__main__":
    main()
