import io
import base64
import shutil
from pathlib import Path
from typing import Optional

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    send_file,
    abort,
)
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server
import matplotlib.pyplot as plt

from src.pipeline import WellAnalysisPipeline
from config.config import OUTPUT_DIR

app = Flask(__name__)
app.secret_key = 'replace-this-with-a-secure-random-secret'

UPLOAD_DIR = Path('data') / 'uploaded'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')


def _sync_prod_data(prod_path: Path) -> None:
    """Copy uploaded production data to project root so pipeline matches CLI behavior."""
    try:
        project_root = Path(__file__).resolve().parent
        dest = project_root / 'prod_data.csv'
        shutil.copy(prod_path, dest)
    except Exception as exc:
        print(f"[sync_prod_data] warning: {exc}")


def _calculate_indicator(slope_val):
    """Calculate indicator arrow based on slope value."""
    if pd.isna(slope_val) or slope_val == 0:
        return ""
    elif slope_val > 0:
        return "↑"
    else:
        return "↓"


def _build_indicator_string(row, columns=['A', 'IP', 'DP', 'R']):
    """Build indicator string like 'A↓ IP↑ DP↓ R↓' from slope values."""
    indicators = []
    for col in columns:
        if col in row and pd.notna(row[col]):
            arrow = _calculate_indicator(row[col])
            if arrow:
                indicators.append(f"{col}{arrow}")
    return " ".join(indicators) if indicators else ""


def _well_output_dir(well_name: str) -> Path:
    """Return the per-well output directory under OUTPUT_DIR."""
    path = OUTPUT_DIR / well_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dataset_paths(well_name: str) -> dict:
    out_dir = _well_output_dir(well_name)
    return {
        '30min': out_dir / f"{well_name}_failure_prediction_30min.csv",
        '30min_indicator': out_dir / f"{well_name}_indicator_30min.csv",
        '3hour': out_dir / "result_df_3 jam.csv",
        '3hour_indicator': out_dir / "result_df_3jam_with_indicator.csv",
        'table_30min_display': out_dir / f"{well_name}_table_30min_display.csv",
        'table_3hour_display': out_dir / f"{well_name}_table_3hour_display.csv",
    }


def _find_dataset_file(well_name: str, dataset: str) -> Path:
    """Return existing dataset path, falling back to legacy flat files when needed."""
    canonical = _dataset_paths(well_name).get(dataset)
    if canonical and canonical.exists():
        return canonical

    fallback_map = {
        '30min': OUTPUT_DIR / f"{well_name}_failure_prediction_30min.csv",
        '30min_indicator': OUTPUT_DIR / f"{well_name}_indicator_30min.csv",
        '3hour': OUTPUT_DIR / "result_df_3 jam.csv",
        '3hour_indicator': OUTPUT_DIR / "result_df_3jam_with_indicator.csv",
        'table_30min_display': OUTPUT_DIR / f"{well_name}_table_30min_display.csv",
        'table_3hour_display': OUTPUT_DIR / f"{well_name}_table_3hour_display.csv",
    }
    fallback = fallback_map.get(dataset)
    if fallback and fallback.exists():
        return fallback

    return canonical


def _render_results(pipeline: WellAnalysisPipeline, well_name: str, zoom_start: Optional[str] = None, zoom_end: Optional[str] = None):
    """Shared renderer for results to support both POST and GET flows."""
    out_dir = _well_output_dir(well_name)
    dataset_files = {key: _find_dataset_file(well_name, key) for key in ['30min', '30min_indicator', '3hour', '3hour_indicator']}

    # Build slopes and resampled dataset for visualization
    slopes_df = pipeline._compute_window_slopes_30min(pipeline.data)
    df_all = pipeline._build_df_all_30min(pipeline.data)

    # Plot: slopes over time for A, IP, DP, IT, MT, V, R (per minute)
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    if not slopes_df.empty:
        slopes_df = slopes_df.sort_values('Window_Start_Time')
        x = slopes_df['Window_Start_Time']
        for col, color in zip(['A','IP','DP','IT','MT','V','R'], ['#4C78A8','#F58518','#54A24B','#E45756','#72B7B2','#B279A2','#FF9DA6']):
            if col in slopes_df.columns and slopes_df[col].notna().any():
                ax1.plot(x, slopes_df[col] * 60.0, label=col, linewidth=1, color=color)
    ax1.set_title(f"{well_name} - 30-minute window slopes")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Slope (per minute)')
    ax1.legend(ncol=4, fontsize=9)
    slope_plot = fig_to_base64(fig1)

    # Plot: line rate (Virtual Rate) over time
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    if 'Virtual Rate (BFPD) (Raw)' in pipeline.data.columns:
        ax2.plot(pd.to_datetime(pipeline.data['Reading Time'], errors='coerce'),
                 pipeline.data['Virtual Rate (BFPD) (Raw)'],
                 label='Virtual Rate', color='#4C78A8', linewidth=1)
    elif 'predicted_virtual_rate' in pipeline.data.columns:
        ax2.plot(pd.to_datetime(pipeline.data['Reading Time'], errors='coerce'),
                 pipeline.data['predicted_virtual_rate'],
                 label='Virtual Rate (predicted)', color='#4C78A8', linewidth=1)
    ax2.set_title(f"{well_name} - Virtual Rate")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('BFPD')
    ax2.legend()
    rate_plot = fig_to_base64(fig2)

    # Load failure prediction table for 30-minute view
    # Prefer the indicator-enriched file so 'Indicator' matches pipeline output
    pred_csv = (
        _find_dataset_file(well_name, '30min_indicator')
        or _find_dataset_file(well_name, '30min')
        or _dataset_paths(well_name)['30min']
    )
    pred_df = None
    pdf = pd.DataFrame()
    aggregated_df = None
    aggregated_lookup = {}
    latest_failure = None
    latest_failure_30min = None
    latest_failure_3hour = None
    latest_report = None
    
    # Load latest_report.json if exists (for new UI format)
    # Try per-well directory first, then fallback to root OUTPUT_DIR
    latest_report_json = out_dir / f"{well_name}_latest_report.json"
    if not latest_report_json.exists():
        latest_report_json = OUTPUT_DIR / f"{well_name}_latest_report.json"
    
    if latest_report_json.exists():
        try:
            with open(latest_report_json, 'r', encoding='utf-8') as f:
                latest_report = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load latest_report.json: {e}")
    
    if pred_csv.exists():
        pred_df = pd.read_csv(pred_csv)
        pdf = pred_df.copy()

        aggregated_path = dataset_files['3hour_indicator'] or _dataset_paths(well_name)['3hour_indicator']
        if aggregated_path.exists():
            try:
                aggregated_df = pd.read_csv(aggregated_path)
                aggregated_df['Window_Start_Time'] = pd.to_datetime(
                    aggregated_df['Window_Start_Time'], errors='coerce'
                )
                aggregated_df = aggregated_df.dropna(subset=['Window_Start_Time']).sort_values('Window_Start_Time')
                if not aggregated_df.empty:
                    aggregated_lookup = aggregated_df.set_index('Window_Start_Time').to_dict('index')
            except Exception:
                aggregated_df = None
                aggregated_lookup = {}

        # Ensure slopes exist for aggregation logic; preserve pipeline-provided Indicator values
        if pred_df is not None and not pred_df.empty:
            # Add slope cols if missing (indicator_30min.csv already has them)
            need_slopes = not set(['A', 'IP', 'DP', 'R']).issubset(pred_df.columns)
            if need_slopes and not slopes_df.empty:
                pred_df['Window_Start_Time'] = pd.to_datetime(pred_df['Window_Start_Time'], errors='coerce')
                slopes_df['Window_Start_Time'] = pd.to_datetime(slopes_df['Window_Start_Time'], errors='coerce')
                pred_df = pred_df.merge(
                    slopes_df[['Window_Start_Time', 'A', 'IP', 'DP', 'R']],
                    on='Window_Start_Time',
                    how='left'
                )
            # Only compute Indicator if not provided by pipeline
            if 'Indicator' not in pred_df.columns:
                pred_df['Indicator'] = pred_df.apply(
                    lambda row: _build_indicator_string(row, ['A', 'IP', 'DP', 'R']), axis=1
                )

            # Normalize indicator text for special statuses to match pipeline semantics
            try:
                pred_df['Status'] = pred_df['Status'].astype(str).str.strip()
                pred_df.loc[pred_df['Status'] == '100% Watercut', 'Indicator'] = '100% WC in Prod'
                pred_df.loc[pred_df['Status'] == 'Electrical Downhole Problem', 'Indicator'] = 'A and Freq 0, others constant'
                pred_df.loc[pred_df['Status'].isin(['Running', 'Shut-in', 'Start-up Phase']), 'Indicator'] = ''
                if 'Recommendation' in pred_df.columns:
                    pred_df['Recommendation'] = (
                        pred_df['Recommendation']
                        .astype(str)
                        .str.replace('NOTIFICATIONS FOR ENGINEER!', '', regex=False)
                        .str.strip()
                    )
            except Exception:
                pass

    # Build 3-hour grouped summaries according to rules
    group_summaries = []
    detected_events = []
    pie_all_b64 = None
    pie_nonrun_b64 = None
    nonrun_distribution = []
    summary_text = None
    zoom_links = []
    events_plot = None
    slope_data = None
    daily_bar_data = {'dates': [], 'counts': [], 'statuses': []}
    if pred_df is not None and not pred_df.empty and 'Window_Start_Time' in pred_df.columns and 'Status' in pred_df.columns:
        pdf = pred_df.copy()
        pdf['Window_Start_Time'] = pd.to_datetime(pdf['Window_Start_Time'], errors='coerce')
        pdf = pdf.dropna(subset=['Window_Start_Time']).sort_values('Window_Start_Time')
        pdf['Status'] = pdf['Status'].astype(str).str.strip()
        pdf_valid = pdf[~pdf['Status'].str.lower().isin(['', 'nan'])]
        non_running = pdf_valid[pdf_valid['Status'].str.lower() != 'running']
        if not non_running.empty:
            latest = non_running.sort_values('Window_Start_Time').iloc[-1]
            latest_failure_30min = {
                'timestamp': latest['Window_Start_Time'].strftime('%Y-%m-%d %H:%M:%S'),
                'status': latest['Status'],
                'recommendation': str(latest.get('Recommendation', '') or '').strip(),
                'indicator': str(latest.get('Indicator', '') or '').strip(),
            }
            # Keep old format for backward compatibility
            latest_failure = latest_failure_30min
        pdf['date'] = pdf['Window_Start_Time'].dt.normalize()
        pdf['group_start'] = pdf['Window_Start_Time'].dt.floor('3H')
        pdf['group_end'] = pdf['group_start'] + pd.Timedelta(hours=3)

        # Dominant status logic (include Running)
        severity_priority = {
            'Increase in Watercut': 3,
            'Shut-in': 2,
            'Electrical Downhole Problem': 1,
            'Running': 0,
        }

        def pick_dominant_status(group: pd.DataFrame) -> str:
            grp_counts = group['Status'].value_counts()
            maxc = grp_counts.max()
            top_statuses = grp_counts[grp_counts == maxc].index.tolist()
            if len(top_statuses) == 1:
                return top_statuses[0]
            day = group['date'].iloc[0]
            day_rows = pdf[pdf['date'] == day]
            day_counts = day_rows['Status'].value_counts()
            day_counts = day_counts[day_counts.index.isin(top_statuses)] if not day_counts.empty else pd.Series(dtype=int)
            if not day_counts.empty:
                top_day = day_counts[day_counts == day_counts.max()].index.tolist()
            else:
                top_day = top_statuses
            if len(top_day) == 1:
                return top_day[0]
            return max(top_day, key=lambda s: severity_priority.get(s, 0))

        groups = pdf.groupby(['date', 'group_start', 'group_end'], as_index=False)
        dom_rows = []
        for (d, gs, ge), g in groups:
            dominant = pick_dominant_status(g)
            non_run_count = (g['Status'] != 'Running').sum()

            # Calculate aggregated indicator for 3-hour window
            agg_indicator_parts = []
            for param in ['A', 'IP', 'DP', 'R']:
                if param in g.columns:
                    slopes = g[param].dropna()
                    if len(slopes) > 0:
                        has_up = (slopes > 0).any()
                        has_down = (slopes < 0).any()
                        if has_up and has_down:
                            agg_indicator_parts.append(f"{param}↑↓")
                        elif has_up:
                            agg_indicator_parts.append(f"{param}↑")
                        elif has_down:
                            agg_indicator_parts.append(f"{param}↓")
            agg_indicator = " ".join(agg_indicator_parts)

            dom_rows.append({
                'date': d,
                'group_start': gs,
                'group_end': ge,
                'Dominant Status': dominant,
                'non_running_count': int(non_run_count),
                'indicator': agg_indicator,
            })
        result_df = pd.DataFrame(dom_rows)

        # Summaries for rendering
        for _, r in result_df.iterrows():
            gs = pd.to_datetime(r['group_start'])
            ge = pd.to_datetime(r['group_end'])
            dominant_status = r['Dominant Status']
            indicator_value = r.get('indicator', '')
            recommendation_value = str(r.get('recommendation', '') or '').strip()

            agg_info = aggregated_lookup.get(gs) if aggregated_lookup else None
            if agg_info:
                agg_status = str(agg_info.get('Dominant Status', '') or '').strip()
                if agg_status:
                    dominant_status = agg_status
                agg_indicator = str(agg_info.get('Indicator', '') or '').strip()
                if agg_indicator:
                    indicator_value = agg_indicator
                agg_rec = agg_info.get('Recommendation', '')
                if isinstance(agg_rec, str):
                    agg_rec = agg_rec.strip()
                elif pd.notna(agg_rec):
                    agg_rec = str(agg_rec).strip()
                else:
                    agg_rec = ''
                if agg_rec:
                    recommendation_value = agg_rec

            summary = {
                'date': pd.to_datetime(r['date']).strftime('%Y-%m-%d') if not pd.isna(r['date']) else '',
                'group_start': gs.strftime('%Y-%m-%d %H:%M:%S'),
                'group_end': ge.strftime('%Y-%m-%d %H:%M:%S'),
                'non_running_count': int(r['non_running_count']),
                'dominant_status': dominant_status,
                'indicator': indicator_value,
                'recommendation': recommendation_value,
            }
            group_summaries.append(summary)

            # Hanya treat sebagai "failure event" kalau status dominan final (setelah
            # override dari aggregated_lookup) bukan Running. Ini mencegah hari yang
            # akhirnya menjadi Running ikut muncul di Daily Activity.
            if dominant_status != 'Running':
                detected_events.append(summary)
                # Store latest 3-hour failure
                if latest_failure_3hour is None or pd.to_datetime(summary['group_start']) > pd.to_datetime(latest_failure_3hour['timestamp']):
                    latest_failure_3hour = {
                        'timestamp': summary['group_start'],
                        'status': summary['dominant_status'],
                        'recommendation': summary.get('recommendation', ''),
                        'indicator': summary['indicator'],
                    }

        # Zoom links for non-Running bands
        for ev in detected_events:
            label = f"{ev['date']} {ev['group_start'].split(' ')[1]}–{ev['group_end'].split(' ')[1]}: {ev['dominant_status']}"
            zoom_links.append({
                'label': label,
                'href': url_for('results', well=well_name, zoom_start=ev['group_start'], zoom_end=ev['group_end']),
                'start': ev['group_start'],
                'end': ev['group_end'],
            })

        # Pie charts with improved styling
        try:
            status_counts = result_df['Dominant Status'].value_counts()
            if not status_counts.empty:
                figp1, axp1 = plt.subplots(figsize=(7, 7))
                colors = plt.cm.Set3(range(len(status_counts)))
                wedges, texts, autotexts = axp1.pie(
                    status_counts,
                    labels=status_counts.index,
                    autopct='%1.1f%%',
                    startangle=140,
                    colors=colors,
                    textprops={'fontsize': 10, 'weight': 'normal'}
                )
                for autotext in autotexts:
                    autotext.set_color('black')
                    autotext.set_fontsize(10)
                    autotext.set_weight('bold')
                axp1.set_title('Distribution of Dominant Status (per 3-hour window)', fontsize=12, pad=20)
                axp1.axis('equal')
                pie_all_b64 = fig_to_base64(figp1)
            if not status_counts.empty:
                total_all = status_counts.sum()
                if total_all > 0:
                    nonrun_distribution = [
                        {
                            'status': status,
                            'percentage': round((count / total_all) * 100, 1),
                        }
                        for status, count in status_counts.items()
                    ]
            status_counts_non = result_df[result_df['Dominant Status'] != 'Running']['Dominant Status'].value_counts()
            if not status_counts_non.empty:
                figp2, axp2 = plt.subplots(figsize=(7, 7))
                colors2 = plt.cm.Set3(range(len(status_counts_non)))
                wedges2, texts2, autotexts2 = axp2.pie(
                    status_counts_non,
                    labels=status_counts_non.index,
                    autopct='%1.1f%%',
                    startangle=140,
                    colors=colors2,
                    textprops={'fontsize': 10, 'weight': 'normal'}
                )
                for autotext in autotexts2:
                    autotext.set_color('black')
                    autotext.set_fontsize(10)
                    autotext.set_weight('bold')
                axp2.set_title('Dominant Status Distribution (Non-Running Only)', fontsize=12, pad=20)
                axp2.axis('equal')
                pie_nonrun_b64 = fig_to_base64(figp2)
        except Exception:
            pass

        # Summary & Recommendations
        try:
            nonrun = result_df[result_df['Dominant Status'] != 'Running']
            if not nonrun.empty:
                counts = nonrun['Dominant Status'].value_counts()
                lines = []
                rec_map = {
                    'Low PI': (
                        "Low PI: Check fluid level and BHP. If acceptable, adjust tubing WHP to bring pump within design rate; check for possible restricted pump."
                    ),
                    'Shut-in': (
                        "Shut-in: Verify operating schedule and surface conditions; ensure Amps/Frequency are expected to be zero."
                    ),
                }
                recs = [f"- {rec_map[st]}" for st in counts.index if st in rec_map]
                if recs:
                    lines.append('Recommendations:')
                    lines.extend(recs)
                summary_text = "\n".join(lines)
        except Exception:
            pass

        # Prepare embedded slope data for client-side Plotly (times/series/bands)
        try:
            sd0 = slopes_df.copy()
            sd0['Window_Start_Time'] = pd.to_datetime(sd0['Window_Start_Time'], errors='coerce')
            sd0 = sd0.dropna(subset=['Window_Start_Time']).sort_values('Window_Start_Time')
            times0 = sd0['Window_Start_Time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
            series_cols0 = ['A','IP','DP','IT','MT','V','R']
            series0 = {c: (sd0[c].astype(float).where(pd.notna(sd0[c]), None).tolist() if c in sd0.columns else []) for c in series_cols0}
            bands0 = []
            for _, row in result_df[result_df['Dominant Status'] != 'Running'].iterrows():
                bands0.append({
                    'start': pd.to_datetime(row['group_start']).strftime('%Y-%m-%d %H:%M:%S'),
                    'end': pd.to_datetime(row['group_end']).strftime('%Y-%m-%d %H:%M:%S'),
                    'label': row['Dominant Status'],
                })
            slope_data = {'times': times0, 'series': series0, 'bands': bands0}
        except Exception:
            slope_data = None

        # Slope overlay plot with shaded bands and optional zoom (server-generated static as fallback) - per minute
        try:
            fig3, ax3 = plt.subplots(figsize=(14, 6))
            sd = slopes_df.copy()
            sd['Window_Start_Time'] = pd.to_datetime(sd['Window_Start_Time'], errors='coerce')
            sd = sd.dropna(subset=['Window_Start_Time']).sort_values('Window_Start_Time')
            if zoom_start and zoom_end:
                z0 = pd.to_datetime(zoom_start, errors='coerce')
                z1 = pd.to_datetime(zoom_end, errors='coerce')
                sd = sd[(sd['Window_Start_Time'] >= z0) & (sd['Window_Start_Time'] <= z1)]
            for col, color in zip(['A','IP','DP','IT','MT','V','R'], ['#4C78A8','#F58518','#54A24B','#E45756','#72B7B2','#B279A2','#FF9DA6']):
                if col in sd.columns and sd[col].notna().any():
                    ax3.plot(sd['Window_Start_Time'], sd[col] * 60.0, label=col, linewidth=1, color=color)
            # Shade bands
            nonrun_bands = result_df[result_df['Dominant Status'] != 'Running']
            color_map = {'Low PI': '#E45756', 'Shut-in': '#000000'}
            for _, row in nonrun_bands.iterrows():
                gs = pd.to_datetime(row['group_start'])
                ge = pd.to_datetime(row['group_end'])
                if zoom_start and zoom_end:
                    if ge < z0 or gs > z1:
                        continue
                label = row['Dominant Status']
                col = color_map.get(label, '#B279A2')
                ax3.axvspan(gs, ge, color=col, alpha=0.12, lw=0)
                mid = gs + (ge - gs)/2
                ymax = np.nanmax(sd.drop(columns=['Window_Start_Time']).to_numpy(dtype=float)) * 60.0 if not sd.empty else 0
                ax3.text(mid, ymax, label, ha='center', va='bottom', fontsize=8, color=col)
            if zoom_start and zoom_end:
                ax3.set_xlim(pd.to_datetime(zoom_start), pd.to_datetime(zoom_end))
            ax3.set_title(f"{well_name} - 30-min Slopes with Events{' (zoomed)' if zoom_start else ''}")
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Slope (per minute)')
            ax3.legend(ncol=4, fontsize=9)
            events_plot = fig_to_base64(fig3)
        except Exception:
            events_plot = None

    # Build daily bar data from detected_events (3-hour non-running windows)
    try:
        if detected_events:
            events_df = pd.DataFrame(detected_events)
            events_df['date'] = pd.to_datetime(events_df['date']).dt.date

            # Count events per day and find dominant status for that day
            daily_summary = events_df.groupby('date').agg(
                count=('date', 'size'),
                status=('dominant_status', lambda s: s.value_counts().idxmax())
            ).reset_index()

            daily_bar_data = {
                'dates': [d.strftime('%Y-%m-%d') for d in daily_summary['date']],
                'counts': daily_summary['count'].tolist(),
                'statuses': daily_summary['status'].tolist(),
            }
    except Exception:
        daily_bar_data = {'dates': [], 'counts': [], 'statuses': []}

    # If slope_data not prepared above (e.g., no pred_df), still provide base series without bands
    if slope_data is None:
        try:
            sd0 = slopes_df.copy()
            sd0['Window_Start_Time'] = pd.to_datetime(sd0['Window_Start_Time'], errors='coerce')
            sd0 = sd0.dropna(subset=['Window_Start_Time']).sort_values('Window_Start_Time')
            times0 = sd0['Window_Start_Time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
            series_cols0 = ['A','IP','DP','IT','MT','V','R']
            series0 = {c: (sd0[c].astype(float).where(pd.notna(sd0[c]), None).tolist() if c in sd0.columns else []) for c in series_cols0}
            slope_data = {'times': times0, 'series': series0, 'bands': []}
        except Exception:
            slope_data = {'times': [], 'series': {}, 'bands': []}

    # Show top N rows to keep page light; restrict 30-min table to requested columns
    table_preview = None
    table_df_for_download = None
    if pred_df is not None and not pred_df.empty:
        table_df = pred_df.copy()
        # Keep only the requested columns if available
        requested_cols = ['Window_Start_Time', 'Indicator', 'Status', 'Recommendation', 'Date']
        keep_cols = [c for c in requested_cols if c in table_df.columns]
        if keep_cols:
            table_df = table_df[keep_cols]
        else:
            # Fallback: drop known extra columns
            drop_cols = ['Reason', 'Prediction', 'A','IP','DP','IT','MT','V','R']
            table_df = table_df.drop(columns=[c for c in drop_cols if c in table_df.columns], errors='ignore')

        # Store filtered table for download
        table_df_for_download = table_df.copy()
        
        table_preview = table_df.head(500).to_dict(orient='records')
        table_columns = list(table_df.columns)
        status_options = sorted({
            str(row.get('Status', '')).strip()
            for row in table_preview
            if str(row.get('Status', '')).strip()
        })
        indicator_options = sorted({
            str(row.get('Indicator', '')).strip()
            for row in table_preview
            if str(row.get('Indicator', '')).strip()
        })
    else:
        table_preview = []
        table_columns = []
        status_options = []
        indicator_options = []

    event_status_options = sorted({
        summary['dominant_status']
        for summary in group_summaries
        if summary.get('dominant_status')
    }) if group_summaries else []

    # Download links using existing files with Indicator
    download_links = {}
    # Provide download for the displayed 30-min table (filtered columns only)
    path_30min = dataset_files.get('30min_indicator') or dataset_files.get('30min')
    if path_30min and path_30min.exists():
        download_links['table_30min'] = {
            'csv': url_for('download_dataset', well=well_name, dataset='table_30min', fmt='csv'),
            'excel': url_for('download_dataset', well=well_name, dataset='table_30min', fmt='excel'),
        }
    
    # Use 3hour_indicator file (already has Indicator column)
    path_3hour = dataset_files.get('3hour_indicator')
    if path_3hour and path_3hour.exists():
        download_links['table_3hour'] = {
            'csv': url_for('download_dataset', well=well_name, dataset='3hour_indicator', fmt='csv'),
            'excel': url_for('download_dataset', well=well_name, dataset='3hour_indicator', fmt='excel'),
        }

    return render_template(
        'dbfieldmgm.web.id/dblpo_results.html',
        well_name=well_name,
        slope_plot=slope_plot,
        rate_plot=rate_plot,
        events_plot=events_plot if pred_df is not None else None,
        summary_text=summary_text,
        table_columns=table_columns,
        table_preview=table_preview,
        pred_csv_path=str(pred_csv),
        group_summaries=group_summaries,
        detected_events=detected_events,
        pie_all_b64=pie_all_b64,
        pie_nonrun_b64=pie_nonrun_b64,
        zoom_links=zoom_links,
        slope_json=json.dumps(slope_data or {}),
        latest_failure=latest_failure,
        latest_failure_30min=latest_failure_30min,
        latest_failure_3hour=latest_failure_3hour,
        latest_report=latest_report,  # New: side-by-side comparison data
        daily_bar_data=daily_bar_data,
        nonrun_distribution=nonrun_distribution,
        statuses=status_options,
        indicators=indicator_options,
        event_statuses=event_status_options,
        download_links=download_links,
    )


def _build_result_df_for_events(pipeline: WellAnalysisPipeline, well_name: str) -> Optional[pd.DataFrame]:
    pred_csv = _find_dataset_file(well_name, '30min') or _dataset_paths(well_name)['30min']
    if not pred_csv.exists():
        return None
    pred_df = pd.read_csv(pred_csv)
    if pred_df.empty or 'Window_Start_Time' not in pred_df.columns or 'Status' not in pred_df.columns:
        return None
    pdf = pred_df.copy()
    pdf['Window_Start_Time'] = pd.to_datetime(pdf['Window_Start_Time'], errors='coerce')
    pdf = pdf.dropna(subset=['Window_Start_Time']).sort_values('Window_Start_Time')
    pdf['date'] = pdf['Window_Start_Time'].dt.normalize()
    pdf['group_start'] = pdf['Window_Start_Time'].dt.floor('3H')
    pdf['group_end'] = pdf['group_start'] + pd.Timedelta(hours=3)
    groups = pdf.groupby(['date', 'group_start', 'group_end'], as_index=False)
    dom_rows = []
    for (d, gs, ge), g in groups:
        # Simple dominant including Running
        st = g['Status'].value_counts().idxmax()
        dom_rows.append({'date': d, 'group_start': gs, 'group_end': ge, 'Dominant Status': st})
    return pd.DataFrame(dom_rows)


def _make_slope_overlay_b64(pipeline: WellAnalysisPipeline, well_name: str, zoom_start: Optional[str], zoom_end: Optional[str]) -> Optional[str]:
    try:
        slopes_df = pipeline._compute_window_slopes_30min(pipeline.data)
        result_df = _build_result_df_for_events(pipeline, well_name)
        if slopes_df is None or slopes_df.empty or result_df is None or result_df.empty:
            return None
        fig, ax = plt.subplots(figsize=(14, 6))
        sd = slopes_df.copy()
        sd['Window_Start_Time'] = pd.to_datetime(sd['Window_Start_Time'], errors='coerce')
        sd = sd.dropna(subset=['Window_Start_Time']).sort_values('Window_Start_Time')
        if zoom_start and zoom_end:
            z0 = pd.to_datetime(zoom_start, errors='coerce')
            z1 = pd.to_datetime(zoom_end, errors='coerce')
            sd = sd[(sd['Window_Start_Time'] >= z0) & (sd['Window_Start_Time'] <= z1)]
        for col, color in zip(['A','IP','DP','IT','MT','V','R'], ['#4C78A8','#F58518','#54A24B','#E45756','#72B7B2','#B279A2','#FF9DA6']):
            if col in sd.columns and sd[col].notna().any():
                ax.plot(sd['Window_Start_Time'], sd[col] * 60.0, label=col, linewidth=1, color=color)
        nonrun_bands = result_df[result_df['Dominant Status'] != 'Running']
        color_map = {'Low PI': '#E45756', 'Shut-in': '#000000'}
        for _, row in nonrun_bands.iterrows():
            gs = pd.to_datetime(row['group_start'])
            ge = pd.to_datetime(row['group_end'])
            if zoom_start and zoom_end:
                if ge < z0 or gs > z1:
                    continue
            label = row['Dominant Status']
            col = color_map.get(label, '#B279A2')
            ax.axvspan(gs, ge, color=col, alpha=0.12, lw=0)
            mid = gs + (ge - gs)/2
            ymax = np.nanmax(sd.drop(columns=['Window_Start_Time']).to_numpy(dtype=float)) * 60.0 if not sd.empty else 0
            ax.text(mid, ymax, label, ha='center', va='bottom', fontsize=8, color=col)
        if zoom_start and zoom_end:
            ax.set_xlim(pd.to_datetime(zoom_start), pd.to_datetime(zoom_end))
        ax.set_title(f"{well_name} - 30-min Slopes with Events{' (zoomed)' if zoom_start else ''}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Slope (per minute)')
        ax.legend(ncol=4, fontsize=9)
        return fig_to_base64(fig)
    except Exception:
        return None


@app.route('/overlay', methods=['GET'])
def overlay():
    try:
        well = request.args.get('well', 'WELL')
        zoom_start = request.args.get('zoom_start')
        zoom_end = request.args.get('zoom_end')
        sensor_path = UPLOAD_DIR / f"{well}_sensor.csv"
        if not sensor_path.exists():
            return jsonify({"error": "No sensor file"}), 400
        pipeline = WellAnalysisPipeline(well)
        # do not rerun full pipeline heavy parts; assume last run created outputs; we still need data loaded
        pipeline.load_data(input_file=sensor_path)
        img_b64 = _make_slope_overlay_b64(pipeline, well, zoom_start, zoom_end)
        if not img_b64:
            return jsonify({"error": "No overlay"}), 400
        return jsonify({"img": img_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/slopes', methods=['GET'])
def slopes_api():
    """Return JSON payload with 30-min slopes and event bands for Plotly overlay.
    Query: well=NAME
    Optional: zoom_start, zoom_end (ISO strings)
    """
    try:
        well = request.args.get('well', 'WELL')
        sensor_path = UPLOAD_DIR / f"{well}_sensor.csv"
        if not sensor_path.exists():
            return jsonify({"error": "No sensor file"}), 400
        pipeline = WellAnalysisPipeline(well)
        pipeline.load_data(input_file=sensor_path)
        slopes_df = pipeline._compute_window_slopes_30min(pipeline.data)
        if slopes_df is None or slopes_df.empty:
            return jsonify({"error": "No slopes"}), 400

        # Build bands from final failure CSV if present
        bands = []
        pred_csv = OUTPUT_DIR / f"{well}_failure_prediction_30min.csv"
        if pred_csv.exists():
            pdf = pd.read_csv(pred_csv)
            if 'Window_Start_Time' in pdf.columns and 'Status' in pdf.columns:
                pdf['Window_Start_Time'] = pd.to_datetime(pdf['Window_Start_Time'], errors='coerce')
                pdf = pdf.dropna(subset=['Window_Start_Time']).sort_values('Window_Start_Time')
                pdf['date'] = pdf['Window_Start_Time'].dt.normalize()
                pdf['group_start'] = pdf['Window_Start_Time'].dt.floor('3H')
                pdf['group_end'] = pdf['group_start'] + pd.Timedelta(hours=3)
                groups = pdf.groupby(['date', 'group_start', 'group_end'], as_index=False)
                for (d, gs, ge), g in groups:
                    # dominant incl Running
                    dom = g['Status'].value_counts().idxmax()
                    if dom != 'Running':
                        bands.append({
                            'start': gs.strftime('%Y-%m-%d %H:%M:%S'),
                            'end': ge.strftime('%Y-%m-%d %H:%M:%S'),
                            'label': dom,
                        })

        # Prepare series
        sd = slopes_df.copy()
        sd = sd.dropna(subset=['Window_Start_Time']).sort_values('Window_Start_Time')
        times = sd['Window_Start_Time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        series_cols = ['A','IP','DP','IT','MT','V','R']
        series = {c: (sd[c].astype(float).fillna(None).tolist() if c in sd.columns else []) for c in series_cols}

        return jsonify({
            'times': times,
            'series': series,
            'bands': bands,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        well_name = request.form.get('well_name', 'WELL')
        sensor_file = request.files.get('sensor_csv')
        prod_file = request.files.get('prod_csv')

        if not sensor_file or sensor_file.filename == '':
            flash('Please upload the sensor CSV file.')
            return redirect(url_for('index'))

        # Save uploads
        sensor_path = UPLOAD_DIR / f"{well_name}_sensor.csv"
        sensor_file.save(sensor_path)
        prod_path: Optional[Path] = None
        if prod_file and prod_file.filename:
            prod_path = UPLOAD_DIR / f"{well_name}_prod.csv"
            prod_file.save(prod_path)

        # Run pipeline
        pipeline = WellAnalysisPipeline(well_name)
        # If production data provided, load and override df_wc
        if prod_path and prod_path.exists():
            try:
                df_wc = pd.read_csv(prod_path)
                # normalize date
                if 'Date' in df_wc.columns:
                    df_wc['Date'] = pd.to_datetime(df_wc['Date'], errors='coerce').dt.normalize()
                if 'WC' in df_wc.columns:
                    df_wc['WC'] = pd.to_numeric(df_wc['WC'], errors='coerce')
                pipeline.df_wc = df_wc
                _sync_prod_data(prod_path)
            except Exception:
                pass

        # Run pipeline end-to-end
        pipeline.run_full_analysis(input_file=sensor_path)

        # Support optional zoom via query params if provided
        zoom_start = request.args.get('zoom_start')
        zoom_end = request.args.get('zoom_end')

        return _render_results(pipeline, well_name, zoom_start, zoom_end)

    except Exception as e:
        flash(f"Error during analysis: {e}")
        return redirect(url_for('index'))


@app.route('/results', methods=['GET'])
def results():
    try:
        well_name = request.args.get('well', 'WELL')
        if not well_name:
            return redirect(url_for('index'))
        # Rebuild pipeline from last uploaded files
        sensor_path = UPLOAD_DIR / f"{well_name}_sensor.csv"
        prod_path = UPLOAD_DIR / f"{well_name}_prod.csv"
        if not sensor_path.exists():
            flash('No prior analysis found for this well. Please upload files again.')
            return redirect(url_for('index'))
        pipeline = WellAnalysisPipeline(well_name)
        # If production data provided previously, load and override df_wc
        if prod_path.exists():
            try:
                df_wc = pd.read_csv(prod_path)
                if 'Date' in df_wc.columns:
                    df_wc['Date'] = pd.to_datetime(df_wc['Date'], errors='coerce').dt.normalize()
                if 'WC' in df_wc.columns:
                    df_wc['WC'] = pd.to_numeric(df_wc['WC'], errors='coerce')
                pipeline.df_wc = df_wc
                _sync_prod_data(prod_path)
            except Exception:
                pass
        # Run pipeline and render with zoom
        pipeline.run_full_analysis(input_file=sensor_path)
        zoom_start = request.args.get('zoom_start')
        zoom_end = request.args.get('zoom_end')
        return _render_results(pipeline, well_name, zoom_start, zoom_end)
    except Exception as e:
        flash(f"Error loading results: {e}")
        return redirect(url_for('index'))


@app.route('/download/<well>/<dataset>.<fmt>', methods=['GET'])
def download_dataset(well: str, dataset: str, fmt: str):
    allowed_fmt = {'csv', 'excel'}
    if fmt not in allowed_fmt:
        abort(404)

    if dataset not in {'30min', '30min_indicator', '3hour', '3hour_indicator', 'table_30min'}:
        abort(404)

    # Resolve source file
    if dataset == 'table_30min':
        file_path = (
            _find_dataset_file(well, '30min_indicator')
            or _find_dataset_file(well, '30min')
            or _dataset_paths(well)['30min']
        )
    else:
        file_path = _find_dataset_file(well, dataset)
    if file_path is None or not file_path.exists():
        abort(404)

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        abort(500, description=f"Failed to read dataset: {exc}")

    # Filter columns for special virtual dataset
    if dataset == 'table_30min':
        keep = [c for c in ['Window_Start_Time', 'Indicator', 'Status', 'Recommendation', 'Date'] if c in df.columns]
        if keep:
            df = df[keep]

    if fmt == 'csv':
        output = io.BytesIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name=file_path.name,
            mimetype='text/csv; charset=utf-8'
        )

    # Excel: serve as xlsx in-memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='data')
    output.seek(0)
    xlsx_name = file_path.with_suffix('.xlsx').name
    return send_file(
        output,
        as_attachment=True,
        download_name=xlsx_name,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


if __name__ == '__main__':
    # For local development
    app.run(host='0.0.0.0', port=5000, debug=True)
