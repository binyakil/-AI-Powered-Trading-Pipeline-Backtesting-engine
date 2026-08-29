from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import os
import importlib.util
import logging
from pathlib import Path
from datetime import datetime
import json
import shutil

RUNS_DIR = Path("outputs/sweep_outputs")
RUNS_DIR.mkdir(exist_ok=True)
import time
from ta.volatility import AverageTrueRange
from ta.trend import PSARIndicator

app = Flask(__name__)
app.secret_key = 'supersecret123'
BASE_DIR = Path(__file__).resolve().parent
CURRENT_RUN_PATH = BASE_DIR / "current_run.txt"
UPLOADED_DATA_PATH = BASE_DIR / "uploaded_data.csv"
app.config['UPLOAD_FOLDER'] = str(BASE_DIR)

STOPPX_METHOD_MAP = {
    'ATR×2': 'ATR2',
    'ATR×3': 'ATR3',
    'Fixed 1R': 'Fixed1R',
    'PSAR': 'PSAR'
}

SWEEP_R_VALUES = [1.0, 1.5, 2.0]
SWEEP_R_VALUES_TP3 = [1.0, 1.5, 2.0, 3.0]
SWEEP_PCT_SPLITS_2 = [[50, 50], [60, 40]]
SWEEP_PCT_SPLITS_3 = [[50, 30, 20], [40, 30, 30], [60, 25, 15]]
STOP_OPTIONS = ['Static', 'Break-Even', 'trail-0.5R']
CONFLICT_MODES = ['hedged', 'flat-before-entry', 'net-reverse', 'ignore-opposite']
CLOSED_PCT_EPS = 1e-6

def validate_tp_splits(splits, label):
    for split in splits:
        if sum(split) != 100:
            raise ValueError(f"{label} split must sum to 100: {split}")

TRADE_EXPORT_COLUMNS = [
    'trade_id', 'run_id', 'side', 'entry_time', 'entry_px',
    'stop_px_initial', 'stop_px_final', 'tp1_px', 'tp2_px', 'tp3_px',
    'exit_time', 'exit_px', 'exit_reason', 'R_result'
]
TRADE_EXPORT_DIR = "trade_lists"

for d in RUNS_DIR.glob("run_*"):
    status = d / "status.txt"
    if status.exists() and status.read_text().strip() == "RUNNING":
        status.write_text("ABORTED")

# Google Sheets
#SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
#SPREADSHEET_ID = '1cuW3cODx66gczD6kwhbT-F9PF_vKxJMRxhgKjMgu28M'
#CREDS_FILE = '2credentials.json'
#TOKEN_FILE = 'token.json'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#def upload_sheet_in_chunks(service, spreadsheet_id, range_start, values, chunk_size=1000):
#    for i in range(0, len(values), chunk_size):
#        chunk = values[i:i + chunk_size]
#        service.spreadsheets().values().update(
#            spreadsheetId=spreadsheet_id,
#            range=f"{range_start}{i+1}",
#            valueInputOption='RAW',
#            body={'values': chunk}
#        ).execute()
#        time.sleep(1.1)

#def ensure_sheet_exists(service, spreadsheet_id, sheet_name, min_rows=1000, min_cols=26):
#    ss = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
#    sheets = ss.get('sheets', [])

#    sheet = next((s for s in sheets if s['properties']['title'] == sheet_name), None)

    # create if missing
#    if sheet is None:
#        resp = service.spreadsheets().batchUpdate(
#            spreadsheetId=spreadsheet_id,
#            body={'requests': [{'addSheet': {'properties': {'title': sheet_name}}}]}
#        ).execute()
#        sheet_id = resp['replies'][0]['addSheet']['properties']['sheetId']
#        current_rows = resp['replies'][0]['addSheet']['properties'].get('gridProperties', {}).get('rowCount', 1000)
#        current_cols = resp['replies'][0]['addSheet']['properties'].get('gridProperties', {}).get('columnCount', 26)
#    else:
#        props = sheet['properties']
#        sheet_id = props['sheetId']
#        grid = props.get('gridProperties', {})
#        current_rows = grid.get('rowCount', 1000)
#        current_cols = grid.get('columnCount', 26)

    # resize if too small
#    target_rows = max(current_rows, min_rows)
#    target_cols = max(current_cols, min_cols)

#    if target_rows != current_rows or target_cols != current_cols:
#        service.spreadsheets().batchUpdate(
#            spreadsheetId=spreadsheet_id,
#            body={'requests': [{
#                'updateSheetProperties': {
#                    'properties': {
#                        'sheetId': sheet_id,
#                        'gridProperties': {'rowCount': target_rows, 'columnCount': target_cols}
#                    },
#                    'fields': 'gridProperties.rowCount,gridProperties.columnCount'
#                }
#            }]}
#        ).execute()

def create_run_context():
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "status.txt").write_text("RUNNING")

    return {
        "run_id": run_id,
        "run_dir": run_dir
    }

def normalize_ohlcv_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw OHLCV CSVs (headered or headerless):
    Epoch (seconds) or EpochMs, Open, High, Low, Close, Volume, ...
    """
    required = {'Open', 'High', 'Low', 'Close', 'Volume'}
    required_lower = {c.lower() for c in required}
    df = df.copy()

    def normalize_epoch_columns(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        if 'Epoch' not in frame.columns and 'EpochMs' in frame.columns:
            frame['Epoch'] = frame['EpochMs']
            frame = frame.drop(columns=['EpochMs'])
        if 'Epoch' in frame.columns:
            frame['Epoch'] = pd.to_numeric(frame['Epoch'], errors='coerce')
            epoch_median = frame['Epoch'].dropna().median()
            if pd.notna(epoch_median) and epoch_median > 1e11:
                frame['Epoch'] = (frame['Epoch'] // 1000).astype('Int64')
        return frame

    # Header row accidentally included when reading with header=None
    if all(isinstance(c, (int, np.integer)) for c in df.columns) and len(df) > 0:
        first_row = df.iloc[0].astype(str).str.strip().str.lower().tolist()
        if required_lower.issubset(set(first_row)):
            df = df.iloc[1:].reset_index(drop=True)

    # Normalize named columns (case-insensitive)
    col_map = {}
    for c in df.columns:
        c_lower = str(c).strip().lower()
        if c_lower in ('epoch', 'time', 'timestamp'):
            col_map[c] = 'Epoch'
        elif c_lower in ('epochms', 'epoch_ms', 'timestampms', 'timestamp_ms', 'time_ms'):
            col_map[c] = 'EpochMs'
        elif c_lower == 'open':
            col_map[c] = 'Open'
        elif c_lower == 'high':
            col_map[c] = 'High'
        elif c_lower == 'low':
            col_map[c] = 'Low'
        elif c_lower == 'close':
            col_map[c] = 'Close'
        elif c_lower == 'volume':
            col_map[c] = 'Volume'
    if col_map:
        df = df.rename(columns=col_map)

    # Already correct → normalize Epoch variant and return
    if required.issubset(df.columns):
        return normalize_epoch_columns(df)

    # Headerless numeric columns → map by position
    if all(isinstance(c, (int, np.integer)) for c in df.columns):
        if df.shape[1] < 6:
            raise ValueError("CSV must contain at least 6 columns (Epoch + OHLCV)")

        df.rename(columns={
            df.columns[0]: 'Epoch',
            df.columns[1]: 'Open',
            df.columns[2]: 'High',
            df.columns[3]: 'Low',
            df.columns[4]: 'Close',
            df.columns[5]: 'Volume',
        }, inplace=True)

        return normalize_epoch_columns(df)

    raise ValueError(f"Unrecognized CSV format: {list(df.columns)}")

def resample_ohlcv(df: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
    """
    Robust 1m → higher TF OHLCV resampling (Kraken-safe).
    """

    if 'Epoch' not in df.columns:
        raise ValueError("Epoch column required")

    df = df.copy()

    # ensure numeric
    for c in ['Epoch','Open','High','Low','Close','Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['Epoch'])

    # ---------- STEP 1: normalize to clean 1-minute bars ----------
    agg_1m = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }

    df_1m = (
        df
        .groupby('Epoch', as_index=False)
        .agg(agg_1m)
    )

    # ---------- STEP 2: datetime index ----------
    df_1m['dt'] = pd.to_datetime(df_1m['Epoch'], unit='s', utc=True)
    df_1m = df_1m.set_index('dt')

    # ---------- STEP 3: resample ----------
    rule = f'{target_minutes}T'

    df_tf = (
        df_1m
        .resample(rule, label='left', closed='left')
        .agg(agg_1m)
        .dropna(subset=['Open'])   # keep non-empty candles
    )

    # restore Epoch from datetime index (start of each resampled candle)
    df_tf = df_tf.reset_index()              # brings back 'dt'
    epoch_vals = df_tf['dt'].astype('int64')
    if len(epoch_vals):
        if epoch_vals.max() > 1_000_000_000_000:
            epoch_vals = epoch_vals // 1_000_000_000
    df_tf['Epoch'] = epoch_vals.astype('int64')
    df_tf = df_tf.drop(columns=['dt'])

    # optional: consistent column order
    df_tf = df_tf[['Epoch', 'Open', 'High', 'Low', 'Close', 'Volume']]

    return df_tf

#def get_sheets_service():
#    creds = None
#    if os.path.exists(TOKEN_FILE):
#        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
#    if not creds or not creds.valid:
#        if creds and creds.expired and creds.refresh_token:
#            creds.refresh(Request())
#        else:
#            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
#            creds = flow.run_local_server(port=0)
#        with open(TOKEN_FILE, 'w') as f:
#            f.write(creds.to_json())
#    return build('sheets', 'v4', credentials=creds)

def safe_exec_indicator(df, run_dir=None):
    if run_dir is None:
        run_dir = Path(CURRENT_RUN_PATH.read_text().strip())
    spec = importlib.util.spec_from_file_location(
        "indicator",
        run_dir / "indicator.py"
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'generate_signals'):
        raise ValueError("indicator.py must contain def generate_signals(df):")
    return module.generate_signals(df.copy())

def load_processed_dataframe(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        run_dir / "input_ohlcv.csv",
        header=None,
        engine='python',
        on_bad_lines='skip'
    )
    df = normalize_ohlcv_csv(df)
    df = resample_ohlcv(df, target_minutes=3)
    df = safe_exec_indicator(df, run_dir)
    return add_stopppx_methods(df)

def add_stopppx_methods(df):
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    psar = PSARIndicator(high=df['High'], low=df['Low'], close=df['Close']).psar()
    
    df['StopPx_ATR2'] = np.where(df['Side'] == 1, df['Open'] - 2*atr, df['Open'] + 2*atr)
    df['StopPx_ATR3'] = np.where(df['Side'] == 1, df['Open'] - 3*atr, df['Open'] + 3*atr)
    df['StopPx_Fixed1R'] = np.where(df['Side'] == 1, df['Open'] * 0.99, df['Open'] * 1.01)
    df['StopPx_PSAR'] = psar.fillna(method='ffill').fillna(df['Open'])  # fill NaN safely
    
    return df

def build_signals_export(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Epoch' not in df.columns:
        # fallback: create a synthetic epoch-like index
        df['Epoch'] = np.arange(len(df))

    df_raw = df[['Epoch', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_side = df[['Side']].copy()
    spacer = pd.Series([''] * len(df_raw), name='')
    df_export = pd.concat([df_raw, spacer, df_side], axis=1)
    return df_export.replace([np.nan, np.inf, -np.inf], '')

def processTrades(df, params, stoppx_col, collect_details=False):
    df = df.copy()
    df['StopPx'] = df[stoppx_col]
    # Initialize required columns to prevent KeyError
    required_cols = ['EntryPx', 'RiskPts', 'TP1Px', 'TP2Px', 'TP3Px', 'ExitPx', 'ExitTime', 'R_Result']
    for col in required_cols:
        if col not in df.columns:
            df[col] = ''

    tradeId = 0
    executed_trades = set()
    openTrades = []
    tradeExits = {}
    trade_logs = {} if collect_details else None
    for row in range(len(df)):
        # Safe Epoch handling
        try:
            epoch = df.iloc[row]['Epoch']
        except KeyError:
            epoch = row  # fallback if no Epoch column

        high = df.iloc[row]['High']
        low = df.iloc[row]['Low']
        side = df.iloc[row]['Side']
        if side in [1, -1]:
            stopPx = df.iloc[row]['StopPx']
            if not pd.isna(stopPx):
                conflict_logs = trade_logs if collect_details else None
                if handleConflictMode(params, side, openTrades, tradeExits, df, epoch, row, conflict_logs):
                    # Kausalitäts-Check: Sind wir an der allerletzten Kerze? Dann können wir nicht in t+1 einsteigen.
                    if row + 1 >= len(df):
                        continue
                    
                    tradeId += 1
                    executed_trades.add(tradeId)
                    
                    # Strikt t+1 Ausführung
                    entryPx = float(df.iloc[row + 1]['Open'])
                    execution_time = df.iloc[row + 1]['Epoch']
                    riskPts = abs(entryPx - stopPx)
                    tp2_r = params.get('tp2_r', 0)
                    tp2_pct = params.get('tp2_pct', 0)
                    tp3_r = params.get('tp3_r', 0)
                    tp3_pct = params.get('tp3_pct', 0)
                    df.at[row, 'EntryPx'] = entryPx
                    df.at[row, 'RiskPts'] = riskPts
                    tp1Px = entryPx + riskPts * params['tp1_r'] if side == 1 else entryPx - riskPts * params['tp1_r']
                    df.at[row, 'TP1Px'] = tp1Px
                    tp2Px = np.nan
                    if tp2_r > 0 and tp2_pct > 0:
                        tp2Px = entryPx + riskPts * tp2_r if side == 1 else entryPx - riskPts * tp2_r
                    df.at[row, 'TP2Px'] = tp2Px
                    tp3Px = np.nan
                    if tp3_r > 0 and tp3_pct > 0:
                        tp3Px = entryPx + riskPts * tp3_r if side == 1 else entryPx - riskPts * tp3_r
                    df.at[row, 'TP3Px'] = tp3Px

                    base_id = str(tradeId)
                    tradeExits[base_id] = {'exitPx': [], 'exitParts': [], 'exitTime': None, 'closedPct': 0, 'last_exit_reason': ''}
                    if trade_logs is not None:
                        trade_logs[base_id] = {
                            'trade_id': tradeId,
                            'side': side,
                            'entry_time': execution_time,
                            'entry_px': entryPx,
                            'stop_px_initial': stopPx,
                            'stop_px_final': stopPx,
                            'tp1_px': tp1Px,
                            'tp2_px': tp2Px if not pd.isna(tp2Px) else '',
                            'tp3_px': tp3Px if not pd.isna(tp3Px) else '',
                            'exit_time': '',
                            'exit_px': '',
                            'exit_reason': '',
                            'R_result': '',
                            'entry_row': row
                        }

                    openTrades.append(createTrade(tradeId, row, row, side, entryPx, stopPx, params['tp1_r'], riskPts, params['tp1_pct'], 'TP1'))
                    if tp2_r > 0 and tp2_pct > 0:
                        openTrades.append(createTrade(tradeId, row, row, side, entryPx, stopPx, tp2_r, riskPts, tp2_pct, 'TP2'))
                    if tp3_r > 0 and tp3_pct > 0:
                        openTrades.append(createTrade(tradeId, row, row, side, entryPx, stopPx, tp3_r, riskPts, tp3_pct, 'TP3'))
        tradesToCheck = openTrades.copy()
        for trade in tradesToCheck:
            if trade['closed']:
                continue

#            highNum, lowNum, tpNum, slNum = float(high), float(low), float(trade['tpPx']), float(trade['stopPx'])
#            hitTP = (trade['side'] == 1 and highNum >= tpNum) or (trade['side'] == -1 and lowNum <= tpNum)
#            hitSL = (trade['side'] == 1 and lowNum <= slNum) or (trade['side'] == -1 and highNum >= slNum)
#            if not hitTP and not hitSL:
#                continue
            highNum, lowNum = float(high), float(low)
            tpNum, slNum = float(trade['tpPx']), float(trade['stopPx'])

            # Intrabar execution order (NO ambiguity)
            if trade['side'] == 1:  # LONG: Low → High
                if lowNum <= slNum:
                    exit_reason = 'SL'
                elif highNum >= tpNum:
                    exit_reason = 'TP'
                else:
                    continue
            else:  # SHORT: High → Low
                if highNum >= slNum:
                    exit_reason = 'SL'
                elif lowNum <= tpNum:
                    exit_reason = 'TP'
                else:
                    continue

            trade['exitRow'] = row
            trade['exitTime'] = epoch
            trade['exitPx'] = trade['tpPx'] if exit_reason == 'TP' else trade['stopPx']
            trade['closed'] = True
            base_trade_id = trade['id'].split('_')[0]
            tradeExits[base_trade_id]['exitPx'].append(trade['exitPx'])
            tradeExits[base_trade_id]['exitParts'].append({
                'px': trade['exitPx'],
                'pct': trade['sizePct']
            })
            tradeExits[base_trade_id]['closedPct'] += trade['sizePct']
            tradeExits[base_trade_id]['last_exit_reason'] = exit_reason
            if trade_logs is not None:
                log = trade_logs.get(base_trade_id)
                if log:
                    log['stop_px_final'] = trade['stopPx']
                    log['exit_reason'] = exit_reason

            if trade['id'].endswith('TP1') and params['stop_option'] in ['break-even', 'trail-0.5r']:
                tradeExits[trade['id'].split('_')[0]]['pending_trail'] = True


#            if trade['id'].endswith('TP1') and params['stop_option'] in ['break-even', 'trail-0.5r']:
#                for t in openTrades:
#                    if t['id'].endswith('TP2') and t['id'].split('_')[0] == trade['id'].split('_')[0] and not t['closed']:
#                        t['stopPx'] = trade['entryPx'] if params['stop_option'] == 'break-even' else \
#                                      trade['entryPx'] + 0.5 * t['riskPts'] if t['side'] == 1 else trade['entryPx'] - 0.5 * t['riskPts']

            openTrades = [t for t in openTrades if not t['closed']]
            if tradeExits[base_trade_id]['closedPct'] >= 1 - CLOSED_PCT_EPS:
                tradeExits[base_trade_id]['exitTime'] = epoch
            writeExit(df, trade, tradeExits[base_trade_id])
            if trade_logs is not None and tradeExits[base_trade_id]['closedPct'] >= 1 - CLOSED_PCT_EPS:
                log = trade_logs.get(base_trade_id)
                if log:
                    log['exit_time'] = tradeExits[base_trade_id]['exitTime']
                    log['exit_px'] = ';'.join(map(str, tradeExits[base_trade_id]['exitPx']))
                    log['R_result'] = df.at[log['entry_row'], 'R_Result']

        # ✅ APPLY TRAILING STOPS AT END OF BAR (NO LOOKAHEAD)
        for trade_id, meta in tradeExits.items():
            if meta.get('pending_trail'):
                for t in openTrades:
                    if t['id'].startswith(trade_id) and t['label'] in ('TP2', 'TP3') and not t['closed']:
                        if params['stop_option'] == 'break-even':
                            t['stopPx'] = t['entryPx']
                        elif params['stop_option'] == 'trail-0.5r':
                            t['stopPx'] = (
                                t['entryPx'] + 0.5 * t['riskPts']
                                if t['side'] == 1
                                else t['entryPx'] - 0.5 * t['riskPts']
                            )
                        if trade_logs is not None:
                            log = trade_logs.get(trade_id)
                            if log:
                                log['stop_px_final'] = t['stopPx']
                meta['pending_trail'] = False
    if not collect_details:
        return df, executed_trades
    finalized = []
    for log in trade_logs.values():
        if log['exit_px']:
            log.pop('entry_row', None)
            finalized.append(log)
    return df, executed_trades, finalized

def runSweep(df_processed, stoppx_method, is_test_mode=False):
    df = df_processed.copy()
    if stoppx_method not in STOPPX_METHOD_MAP:
        raise ValueError(f"Unknown stoppx_method: {stoppx_method}")

    stop_col = f"StopPx_{STOPPX_METHOD_MAP[stoppx_method]}"
    df['StopPx'] = df[stop_col]
    result_data = []
    for run_id, params in iterate_sweep_configs():
        if is_test_mode and run_id > 2:
            logger.info("Test mode enabled: Stopping sweep early.")
            break
        df_temp = df.copy()
        df_temp, trades = processTrades(df_temp, params, stop_col)
        rResults = df_temp['R_Result'].replace('', np.nan).dropna().astype(float).tolist()
        expectancy = sum(rResults) / len(rResults) if rResults else 0
        gains = sum(r for r in rResults if r > 0)
        losses = abs(sum(r for r in rResults if r < 0))
        profitFactor = gains / losses if losses != 0 else (float('inf') if gains > 0 else 0)
        equity = maxEquity = maxDD = 0
        for r in rResults:
            equity += r
            maxEquity = max(maxEquity, equity)
            maxDD = max(maxDD, maxEquity - equity)
        maxDDPercent = (maxDD / maxEquity * 100) if maxEquity != 0 else 0
        mar = expectancy / (maxDDPercent / 100) if maxDDPercent != 0 else (expectancy if expectancy > 0 else 0)

#        trade_count = len(
#            df_temp['EntryPx'].replace('', np.nan).dropna()
#        )
        trade_count = len(trades)

        tp_r_values = [params['tp1_r'], params['tp2_r']]
        tp_pct_values = [int(round(params['tp1_pct'] * 100)), int(round(params['tp2_pct'] * 100))]
        if params.get('tp3_pct', 0) > 0 and params.get('tp3_r', 0) > 0:
            tp_r_values.append(params['tp3_r'])
            tp_pct_values.append(int(round(params['tp3_pct'] * 100)))
        tp_r_label = ';'.join(str(r) for r in tp_r_values)
        tp_pct_label = ';'.join(str(p) for p in tp_pct_values)

        stop_label = params.get('stop_option_label', params['stop_option'])
        conflict_label = params.get('conflict_mode_label', params['conflict_mode'])
        result_data.append([run_id, tp_r_label, tp_pct_label, stop_label, conflict_label, stoppx_method, trade_count,
                            round(expectancy, 3), round(profitFactor, 2), round(maxDDPercent, 2), round(mar, 2)])
        logger.info(f"Completed run {run_id}")
    return pd.DataFrame(result_data, columns=['RunID','TP_Rs','TP_%s','StopHandling','ConflictMode','StopPx_Method','Trades','Expectancy','PF','MaxDD%','MAR'])

def load_available_runs(run_dir: Path):
    results_path = run_dir / "sweep_results.csv"
    if not results_path.exists():
        return []
    df = pd.read_csv(results_path)
    display_cols = ['RunID', 'TP_Rs', 'TP_%s', 'StopHandling', 'ConflictMode', 'Expectancy', 'PF', 'Trades']
    existing_cols = [c for c in display_cols if c in df.columns]
    return df[existing_cols].to_dict('records')

def iterate_sweep_configs():
    validate_tp_splits(SWEEP_PCT_SPLITS_2, "SWEEP_PCT_SPLITS_2")
    validate_tp_splits(SWEEP_PCT_SPLITS_3, "SWEEP_PCT_SPLITS_3")
    run_id = 1
    for r1 in SWEEP_R_VALUES:
        for r2 in SWEEP_R_VALUES:
            if r1 >= r2:
                continue
            for tp1Pct, tp2Pct in SWEEP_PCT_SPLITS_2:
                for stopOption in STOP_OPTIONS:
                    for conflictMode in CONFLICT_MODES:
                        yield run_id, {
                            'tp1_r': r1,
                            'tp1_pct': tp1Pct / 100,
                            'tp2_r': r2,
                            'tp2_pct': tp2Pct / 100,
                            'tp3_r': 0,
                            'tp3_pct': 0,
                            'stop_option': stopOption.lower(),
                            'stop_option_label': stopOption,
                            'conflict_mode': conflictMode.lower(),
                            'conflict_mode_label': conflictMode
                        }
                        run_id += 1
    for r1 in SWEEP_R_VALUES:
        for r2 in SWEEP_R_VALUES:
            for r3 in SWEEP_R_VALUES_TP3:
                if r1 >= r2 or r2 >= r3:
                    continue
                for tp1Pct, tp2Pct, tp3Pct in SWEEP_PCT_SPLITS_3:
                    for stopOption in STOP_OPTIONS:
                        for conflictMode in CONFLICT_MODES:
                            yield run_id, {
                                'tp1_r': r1,
                                'tp1_pct': tp1Pct / 100,
                                'tp2_r': r2,
                                'tp2_pct': tp2Pct / 100,
                                'tp3_r': r3,
                                'tp3_pct': tp3Pct / 100,
                                'stop_option': stopOption.lower(),
                                'stop_option_label': stopOption,
                                'conflict_mode': conflictMode.lower(),
                                'conflict_mode_label': conflictMode
                            }
                            run_id += 1


def collect_trades_for_run(df_processed, stoppx_method, target_run_id):
    if stoppx_method not in STOPPX_METHOD_MAP:
        raise ValueError(f"Unknown stoppx_method: {stoppx_method}")
    stop_col = f"StopPx_{STOPPX_METHOD_MAP[stoppx_method]}"
    params = None
    for run_id, candidate in iterate_sweep_configs():
        if run_id == target_run_id:
            params = candidate
            break
    if params is None:
        raise ValueError(f"RunID {target_run_id} not found")
    df_temp = df_processed.copy()
    df_temp, _, trade_details = processTrades(df_temp, params, stop_col, collect_details=True)
    trades = []
    for log in trade_details:
        trades.append({
            'trade_id': log['trade_id'],
            'run_id': target_run_id,
            'side': log['side'],
            'entry_time': log['entry_time'],
            'entry_px': log['entry_px'],
            'stop_px_initial': log['stop_px_initial'],
            'stop_px_final': log['stop_px_final'],
            'tp1_px': log['tp1_px'],
            'tp2_px': log['tp2_px'],
            'tp3_px': log.get('tp3_px', ''),
            'exit_time': log.get('exit_time', ''),
            'exit_px': log['exit_px'],
            'exit_reason': log['exit_reason'] or 'n/a',
            'R_result': log['R_result'] if log['R_result'] != '' else 0
        })
    return trades


def export_trades_for_run_ids(run_dir, run_ids):
    if not run_ids:
        raise ValueError("run_ids list required")
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError("Missing config.json in run directory")
    config = json.loads(config_path.read_text())
    stoppx_method = config.get('stop_method')
    if not stoppx_method:
        raise ValueError("Sweep configuration missing stop method")
    df_processed = load_processed_dataframe(run_dir)
    export_dir = run_dir / TRADE_EXPORT_DIR
    export_dir.mkdir(exist_ok=True)
    exported = []
    seen = set()
    for run_id in run_ids:
        if run_id in seen:
            continue
        seen.add(run_id)
        trade_rows = collect_trades_for_run(df_processed, stoppx_method, run_id)
        if not trade_rows:
            continue
        output_path = export_dir / f"trades_run_{run_id}.csv"
        df_export = pd.DataFrame(trade_rows, columns=TRADE_EXPORT_COLUMNS)
        if 'entry_time' in df_export.columns:
            df_export['entry_time'] = pd.to_datetime(df_export['entry_time'], unit='s')
            df_export['exit_time'] = pd.to_datetime(df_export['exit_time'], unit='s')
        df_export.to_csv(output_path, index=False)
        exported.append(output_path.name)
    return exported

def get_trade_csv_path(run_dir: Path, run_id: int) -> Path:
    export_dir = run_dir / TRADE_EXPORT_DIR
    csv_path = export_dir / f"trades_run_{run_id}.csv"
    if csv_path.exists():
        return csv_path
    export_trades_for_run_ids(run_dir, [run_id])
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"Trades for RunID {run_id} not found")

def build_price_trade_chart_data(price_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    if price_df is None or price_df.empty:
        return {}
    price = price_df[['Epoch', 'Open', 'High', 'Low', 'Close']].copy()
    for col in ['Epoch', 'Open', 'High', 'Low', 'Close']:
        price[col] = pd.to_numeric(price[col], errors='coerce')
    price = price.dropna(subset=['Epoch', 'Open', 'High', 'Low', 'Close'])
    if price.empty:
        return {}
    price = price.sort_values('Epoch').reset_index(drop=True)
    price_epochs = price['Epoch'].astype('int64').tolist()
    epoch_array = np.array(price_epochs, dtype='int64')

    labels = pd.to_datetime(price['Epoch'], unit='s', utc=True, errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
    labels = labels.fillna('')

    def map_epoch(epoch_value):
        if epoch_value is None or pd.isna(epoch_value):
            return None
        try:
            epoch_int = int(epoch_value)
        except (TypeError, ValueError):
            return None
        if len(epoch_array) == 0:
            return None
        idx = int(np.searchsorted(epoch_array, epoch_int))
        if idx < 0:
            return 0
        if idx >= len(epoch_array):
            return len(epoch_array) - 1
        return idx

    def safe_float(value):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def parse_exit_prices(raw_value):
        if raw_value is None:
            return []
        if isinstance(raw_value, (int, float, np.number)):
            if pd.isna(raw_value):
                return []
            return [float(raw_value)]
        raw_text = str(raw_value).strip()
        if not raw_text:
            return []
        parts = [p.strip() for p in raw_text.split(';') if p.strip()]
        prices = []
        for part in parts:
            try:
                prices.append(float(part))
            except (TypeError, ValueError):
                continue
        return prices

    entries_long = []
    entries_short = []
    exits_win = []
    exits_loss = []
    exits_flat = []
    win_lines = []
    loss_lines = []
    flat_lines = []
    stop_lines = []

    if trades_df is not None and not trades_df.empty:
        trades = trades_df.copy()
        for col in ['entry_time', 'exit_time', 'entry_px', 'stop_px_final', 'stop_px_initial', 'R_result', 'side']:
            if col in trades.columns:
                trades[col] = pd.to_numeric(trades[col], errors='coerce')
        for _, row in trades.iterrows():
            entry_time = row.get('entry_time')
            exit_time = row.get('exit_time') if 'exit_time' in trades.columns else None
            entry_idx = map_epoch(entry_time)
            if entry_idx is None:
                continue
            exit_idx = map_epoch(exit_time) if exit_time is not None and not pd.isna(exit_time) else entry_idx
            entry_px = safe_float(row.get('entry_px'))
            if entry_px is None:
                continue
            exit_prices = parse_exit_prices(row.get('exit_px'))
            exit_px = exit_prices[-1] if exit_prices else safe_float(row.get('exit_px'))
            if exit_px is None:
                exit_px = entry_px
            side = row.get('side')
            side = int(side) if side is not None and not pd.isna(side) else 0
            r_result = row.get('R_result')
            try:
                r_result = float(r_result)
            except (TypeError, ValueError):
                r_result = 0.0
            exit_reason = str(row.get('exit_reason', '') or '').strip()
            trade_id = row.get('trade_id')
            try:
                trade_id = int(trade_id)
            except (TypeError, ValueError):
                pass

            meta = {
                'trade_id': trade_id,
                'side': side,
                'r_result': round(r_result, 4),
                'entry_time': int(entry_time) if entry_time is not None and not pd.isna(entry_time) else None,
                'exit_time': int(exit_time) if exit_time is not None and not pd.isna(exit_time) else None,
                'exit_reason': exit_reason,
                'exit_px_list': exit_prices
            }

            entry_point = {'x': entry_idx, 'y': entry_px, **meta}
            if side == 1:
                entries_long.append(entry_point)
            elif side == -1:
                entries_short.append(entry_point)
            else:
                entries_long.append(entry_point)

            exit_point = {'x': exit_idx, 'y': exit_px, **meta}
            if r_result > 0:
                exits_win.append(exit_point)
            elif r_result < 0:
                exits_loss.append(exit_point)
            else:
                exits_flat.append(exit_point)

            line_target = win_lines if r_result > 0 else loss_lines if r_result < 0 else flat_lines
            line_target.extend([
                {'x': entry_idx, 'y': entry_px},
                {'x': exit_idx, 'y': exit_px},
                {'x': None, 'y': None}
            ])

            stop_px = safe_float(row.get('stop_px_final'))
            if stop_px is None:
                stop_px = safe_float(row.get('stop_px_initial'))
            if stop_px is not None:
                stop_lines.extend([
                    {'x': entry_idx, 'y': stop_px},
                    {'x': exit_idx, 'y': stop_px},
                    {'x': None, 'y': None}
                ])

    return {
        'price_labels': labels.tolist(),
        'price_epochs': price_epochs,
        'price_open': price['Open'].round(6).tolist(),
        'price_high': price['High'].round(6).tolist(),
        'price_low': price['Low'].round(6).tolist(),
        'price_close': price['Close'].round(6).tolist(),
        'entries_long': entries_long,
        'entries_short': entries_short,
        'exits_win': exits_win,
        'exits_loss': exits_loss,
        'exits_flat': exits_flat,
        'trade_lines_win': win_lines,
        'trade_lines_loss': loss_lines,
        'trade_lines_flat': flat_lines,
        'stop_lines': stop_lines
    }

def build_trade_chart_data(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {'summary': {}}
    df = trades_df.copy()
    df['R_result'] = pd.to_numeric(df['R_result'], errors='coerce').fillna(0.0)
    df['side'] = pd.to_numeric(df['side'], errors='coerce').fillna(0).astype(int)
    df['entry_time'] = pd.to_numeric(df['entry_time'], errors='coerce')
    df = df.sort_values(by=['entry_time', 'trade_id']).reset_index(drop=True)
    df['trade_index'] = np.arange(1, len(df) + 1)
    df['equity'] = df['R_result'].cumsum()
    df['max_equity'] = df['equity'].cummax()
    df['drawdown'] = df['equity'] - df['max_equity']
    df['underwater'] = np.where(df['max_equity'] != 0, df['equity'] / df['max_equity'] - 1, 0.0)
    df['rolling_expectancy'] = df['R_result'].rolling(window=20, min_periods=1).mean()
    df['long_equity'] = (df['R_result'].where(df['side'] == 1, 0)).cumsum()
    df['short_equity'] = (df['R_result'].where(df['side'] == -1, 0)).cumsum()
    streaks = []
    current = 0
    for val in df['R_result']:
        if val > 0:
            current = current + 1 if current >= 0 else 1
        elif val < 0:
            current = current - 1 if current <= 0 else -1
        else:
            current = 0
        streaks.append(current)
    entry_labels = pd.to_datetime(df['entry_time'], unit='s', errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    entry_labels = entry_labels.fillna(df['trade_index'].astype(str))
    summary = {
        'trades': int(len(df)),
        'total_r': round(df['R_result'].sum(), 3),
        'expectancy': round(df['R_result'].mean(), 3) if len(df) else 0.0,
        'win_rate': round((df['R_result'] > 0).mean() * 100, 1) if len(df) else 0.0,
        'avg_win': round(df.loc[df['R_result'] > 0, 'R_result'].mean(), 3) if (df['R_result'] > 0).any() else 0.0,
        'avg_loss': round(df.loc[df['R_result'] < 0, 'R_result'].mean(), 3) if (df['R_result'] < 0).any() else 0.0,
        'max_drawdown': round(df['drawdown'].min(), 3),
        'long_total': round(df.loc[df['side'] == 1, 'R_result'].sum(), 3),
        'short_total': round(df.loc[df['side'] == -1, 'R_result'].sum(), 3),
        'best_streak': int(max(streaks)) if streaks else 0,
        'worst_streak': int(min(streaks)) if streaks else 0
    }
    return {
        'trade_labels': df['trade_index'].tolist(),
        'entry_labels': entry_labels.tolist(),
        'equity': df['equity'].round(3).tolist(),
        'drawdown': df['drawdown'].round(3).tolist(),
        'underwater': df['underwater'].round(4).tolist(),
        'long_equity': df['long_equity'].round(3).tolist(),
        'short_equity': df['short_equity'].round(3).tolist(),
        'rolling_expectancy': df['rolling_expectancy'].round(3).tolist(),
        'streaks': streaks,
        'summary': summary
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'csv_file' not in request.files or 'indicator_file' not in request.files:
            flash("Both files required")
            return redirect(request.url)
        
        csv_file = request.files['csv_file']
        indicator_file = request.files['indicator_file']
        csv_label = (request.form.get('csv_label') or '').strip()
        indicator_label = (request.form.get('indicator_label') or '').strip()
        
        if csv_file.filename == '' or indicator_file.filename == '':
            flash("Select both files")
            return redirect(request.url)
        if not csv_label or not indicator_label:
            flash("Add descriptions for both the OHLCV data and the indicator.")
            return redirect(request.url)
        
        ctx = create_run_context()

        csv_path = ctx["run_dir"] / "input_ohlcv.csv"
        strat_path = ctx["run_dir"] / "indicator.py"

        csv_file.save(csv_path)
        indicator_file.save(strat_path)
        meta = {
            "csv_label": csv_label,
            "indicator_label": indicator_label,
            "csv_filename": csv_file.filename,
            "indicator_filename": indicator_file.filename,
            "created_at": ctx["run_id"]
        }
        (ctx["run_dir"] / "run_meta.json").write_text(json.dumps(meta, indent=2))

        # remember run path for later
        CURRENT_RUN_PATH.write_text(str(ctx["run_dir"]))
        
        return redirect(url_for('processing'))
    
    return render_template('index.html')

@app.route('/processing')
def processing():
    return render_template('processing.html')

@app.route('/sweep', methods=['GET', 'POST'])
def sweep():
    if not UPLOADED_DATA_PATH.exists():
        return redirect(url_for('index'))
    
    run_dir = Path(CURRENT_RUN_PATH.read_text().strip())

    df = load_processed_dataframe(run_dir)
    
    df_signals = build_signals_export(df)
    df_signals.to_csv(run_dir / "signals.csv", index=False)
    available_runs = load_available_runs(run_dir)
    
    if request.method == 'POST':
        stoppx_method = request.form['stoppx_method']
        is_test_mode = request.form.get('test_mode') == 'on'

        run_id = run_dir.name.replace("run_", "")
        meta_path = run_dir / "run_meta.json"
        run_meta = {}
        if meta_path.exists():
            try:
                run_meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                run_meta = {}

        config = {
            "run_id": run_id,
            "rows": len(df),
            "indicator_file": "indicator.py",
            "input_file": "input_ohlcv.csv",
            "csv_label": run_meta.get("csv_label", ""),
            "indicator_label": run_meta.get("indicator_label", ""),
            "csv_filename": run_meta.get("csv_filename", ""),
            "indicator_filename": run_meta.get("indicator_filename", ""),
            "stop_method": stoppx_method,
            "created_at": run_id
        }

        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        results_df = runSweep(df, stoppx_method, is_test_mode=is_test_mode)
        
        # Write results
#        service.spreadsheets().values().clear(spreadsheetId=SPREADSHEET_ID, range='Results+Evaluation!A:Z').execute()
#        results_upload = results_df.replace([np.nan, np.inf, -np.inf], '')
#        values = [results_upload.columns.tolist()] + results_upload.values.tolist()
#        upload_sheet_in_chunks(service, SPREADSHEET_ID, 'Results+Evaluation!A', values)
        results_df.to_csv(run_dir / "sweep_results.csv", index=False)
        (run_dir / "status.txt").write_text("COMPLETED")
        flash("Sweep completed – check sweep_results.csv inside run_… folder")
        available_runs = load_available_runs(run_dir)
    
    return render_template('sweep.html', available_runs=available_runs)

@app.route('/export_trades', methods=['POST'])
def export_trades_endpoint():
    # Load the current run directory
    try:
        run_dir = Path(CURRENT_RUN_PATH.read_text().strip())
    except FileNotFoundError:
        return {"error": "No active run found"}, 400
    
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        run_ids = payload.get('run_ids') if isinstance(payload, dict) else None
        if not isinstance(run_ids, list) or not run_ids:
            return {"error": "Provide run_ids as a non-empty list"}, 400
        try:
            run_ids = [int(r) for r in run_ids]
        except (TypeError, ValueError):
            return {"error": "run_ids must contain integers"}, 400
        try:
            exported = export_trades_for_run_ids(run_dir, run_ids)
        except Exception as exc:
            logger.exception("Failed to export trades")
            return {"error": str(exc)}, 400
        return {"exported_files": exported}, 200

    # HTML form submission path
    form_run_ids = request.form.getlist('run_ids')
    if not form_run_ids:
        flash("Select at least one RunID before exporting.")
        return redirect(url_for('sweep'))
    try:
        parsed_ids = [int(r) for r in form_run_ids]
    except ValueError:
        flash("Invalid RunID selection.")
        return redirect(url_for('sweep'))
    try:
        exported = export_trades_for_run_ids(run_dir, parsed_ids)
    except Exception as exc:
        logger.exception("Failed to export trades")
        flash(f"Export failed: {exc}")
    else:
        label = ", ".join(exported) if exported else "No trades generated."
        flash(f"Export complete: {label}")
    return redirect(url_for('sweep'))

@app.route('/analysis/<int:run_id>')
def trade_analysis(run_id):
    try:
        run_dir = Path(CURRENT_RUN_PATH.read_text().strip())
    except FileNotFoundError:
        flash("No active run found.")
        return redirect(url_for('sweep'))
    try:
        csv_path = get_trade_csv_path(run_dir, run_id)
    except Exception as exc:
        flash(f"Unable to load trades for RunID {run_id}: {exc}")
        return redirect(url_for('sweep'))
    trades_df = pd.read_csv(csv_path)
    needs_refresh = False
    if 'exit_time' not in trades_df.columns:
        needs_refresh = True
    if 'entry_time' in trades_df.columns:
        entry_times = pd.to_numeric(trades_df['entry_time'], errors='coerce')
        if entry_times.notna().any() and entry_times.max() < 1_000_000_000:
            needs_refresh = True
    if needs_refresh:
        try:
            export_trades_for_run_ids(run_dir, [run_id])
            trades_df = pd.read_csv(csv_path)
        except Exception:
            logger.exception("Failed to refresh trade export for exit_time column")
    if trades_df.empty:
        flash("No trades available for this RunID.")
        return redirect(url_for('sweep'))
    df_processed = load_processed_dataframe(run_dir)
    chart_payload = build_trade_chart_data(trades_df)
    chart_payload.update(build_price_trade_chart_data(df_processed, trades_df))
    summary = chart_payload.pop('summary', {})
    return render_template(
        'analysis.html',
        run_id=run_id,
        summary=summary,
        chart_data=json.dumps(chart_payload)
    )

# Your full backtest functions (from backtest_flask.py)
def handleConflictMode(params, side, openTrades, tradeExits, df, epoch, row, trade_logs=None):
    def finalize_conflict_log(trade):
        if trade_logs is None:
            return
        base_id = trade['id'].split('_')[0]
        log = trade_logs.get(base_id)
        if not log:
            return
        log['stop_px_final'] = trade['stopPx']
        log['exit_reason'] = 'conflict'
        log['exit_time'] = tradeExits[base_id].get('exitTime')
        log['exit_px'] = ';'.join(map(str, tradeExits[base_id]['exitPx']))
        log['R_result'] = df.at[log['entry_row'], 'R_Result']

    if params['conflict_mode'] == 'hedged': return True
    oppositeTrades = [t for t in openTrades if t['side'] != side and not t['closed']]
    if not oppositeTrades: return True
    if params['conflict_mode'] == 'ignore-opposite': return False
    if params['conflict_mode'] == 'flat-before-entry':
        for t in openTrades:
            if not t['closed']:
                t['exitPx'] = df.iloc[row]['Low'] if t['side'] == 1 else df.iloc[row]['High']
                t['exitTime'] = epoch
                t['closed'] = True
                base_id = t['id'].split('_')[0]
                tradeExits[base_id]['exitPx'].append(t['exitPx'])
                tradeExits[base_id]['exitParts'].append({
                    'px': t['exitPx'],
                    'pct': t['sizePct']
                })
                tradeExits[base_id]['closedPct'] += t['sizePct']
                if tradeExits[base_id]['closedPct'] >= 1 - CLOSED_PCT_EPS:
                    tradeExits[base_id]['exitTime'] = epoch
                writeExit(df, t, tradeExits[base_id])
                finalize_conflict_log(t)
        openTrades.clear()
        return True
    if params['conflict_mode'] == 'net-reverse':
        totalLongPct = sum(t['sizePct'] for t in openTrades if t['side'] == 1 and not t['closed'])
        totalShortPct = sum(t['sizePct'] for t in openTrades if t['side'] == -1 and not t['closed'])
        for t in oppositeTrades:
            t['exitPx'] = df.iloc[row]['Low'] if t['side'] == 1 else df.iloc[row]['High']
            t['exitTime'] = epoch
            t['closed'] = True
            base_id = t['id'].split('_')[0]
            tradeExits[base_id]['exitPx'].append(t['exitPx'])
            tradeExits[base_id]['exitParts'].append({
                'px': t['exitPx'],
                'pct': t['sizePct']
            })
            tradeExits[base_id]['closedPct'] += t['sizePct']
            if tradeExits[base_id]['closedPct'] >= 1 - CLOSED_PCT_EPS:
                tradeExits[base_id]['exitTime'] = epoch
            writeExit(df, t, tradeExits[base_id])
            finalize_conflict_log(t)
        openTrades[:] = [t for t in openTrades if not t['closed']]
        netPct = (totalLongPct - totalShortPct) if side == 1 else (totalShortPct - totalLongPct)
        return netPct < 0
    return True

def createTrade(id, signalRow, execRow, side, entryPx, stopPx, rr, riskPts, pct, label):
    tpPx = entryPx + riskPts * rr if side == 1 else entryPx - riskPts * rr
    return {'id': f"{id}_{label}", 'entryRow': signalRow, 'execRow': execRow, 'side': side,
            'entryPx': entryPx, 'stopPx': stopPx, 'tpPx': tpPx, 'riskPts': riskPts, 'sizePct': pct,
            'closed': False, 'exitPx': '', 'exitTime': '', 'label': label}

def writeExit(df, trade, tradeExit):
    r = trade['entryRow']
    df.at[r, 'ExitPx'] = ';'.join(map(str, tradeExit['exitPx']))
    if tradeExit['exitTime']:
        df.at[r, 'ExitTime'] = tradeExit['exitTime']
    side = trade['side']
    entryPx = float(trade['entryPx'])
    riskPts = float(df.iloc[r]['RiskPts'] or 0)
    totalGain = 0
    for part in tradeExit.get('exitParts', []):
        numericPx = float(part['px'])
        pct = float(part['pct'])
        totalGain += pct * (numericPx - entryPx if side == 1 else entryPx - numericPx)
    rResult = totalGain / riskPts if riskPts != 0 else 0.0
    df.at[r, 'R_Result'] = rResult

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)