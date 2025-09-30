import io
import base64
from pathlib import Path
from typing import Optional

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
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


def _render_results(pipeline: WellAnalysisPipeline, well_name: str, zoom_start: Optional[str] = None, zoom_end: Optional[str] = None):
    """Shared renderer for results to support both POST and GET flows."""
    # Build slopes and resampled dataset for visualization
    slopes_df = pipeline._compute_window_slopes_30min(pipeline.data)
    df_all = pipeline._build_df_all_30min(pipeline.data)

    # Plot: slopes over time for A, IP, DP, IT, MT, V, R
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    if not slopes_df.empty:
        slopes_df = slopes_df.sort_values('Window_Start_Time')
        x = slopes_df['Window_Start_Time']
        for col, color in zip(['A','IP','DP','IT','MT','V','R'], ['#4C78A8','#F58518','#54A24B','#E45756','#72B7B2','#B279A2','#FF9DA6']):
            if col in slopes_df.columns and slopes_df[col].notna().any():
                ax1.plot(x, slopes_df[col], label=col, linewidth=1, color=color)
    ax1.set_title(f"{well_name} - 30-minute window slopes")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Slope (per second)')
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

    # Load failure prediction table from saved CSV
    pred_csv = OUTPUT_DIR / f"{well_name}_failure_prediction_30min.csv"
    pred_df = None
    if pred_csv.exists():
        pred_df = pd.read_csv(pred_csv)

    # Build 3-hour grouped summaries according to rules
    group_summaries = []
    detected_events = []
    pie_all_b64 = None
    pie_nonrun_b64 = None
    summary_text = None
    zoom_links = []
    events_plot = None
    slope_data = None
    daily_bar_data = {'dates': [], 'counts': [], 'statuses': []}
    if pred_df is not None and not pred_df.empty and 'Window_Start_Time' in pred_df.columns and 'Status' in pred_df.columns:
        pdf = pred_df.copy()
        pdf['Window_Start_Time'] = pd.to_datetime(pdf['Window_Start_Time'], errors='coerce')
        pdf = pdf.dropna(subset=['Window_Start_Time']).sort_values('Window_Start_Time')
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
            dom_rows.append({
                'date': d,
                'group_start': gs,
                'group_end': ge,
                'Dominant Status': dominant,
                'non_running_count': int(non_run_count),
            })
        result_df = pd.DataFrame(dom_rows)

        # Summaries for rendering
        for _, r in result_df.iterrows():
            summary = {
                'date': pd.to_datetime(r['date']).strftime('%Y-%m-%d'),
                'group_start': pd.to_datetime(r['group_start']).strftime('%Y-%m-%d %H:%M:%S'),
                'group_end': pd.to_datetime(r['group_end']).strftime('%Y-%m-%d %H:%M:%S'),
                'non_running_count': int(r['non_running_count']),
                'dominant_status': r['Dominant Status'],
            }
            group_summaries.append(summary)
            if r['Dominant Status'] != 'Running':
                detected_events.append(summary)

        # Zoom links for non-Running bands
        for ev in detected_events:
            label = f"{ev['date']} {ev['group_start'].split(' ')[1]}–{ev['group_end'].split(' ')[1]}: {ev['dominant_status']}"
            zoom_links.append({
                'label': label,
                'href': url_for('results', well=well_name, zoom_start=ev['group_start'], zoom_end=ev['group_end']),
                'start': ev['group_start'],
                'end': ev['group_end'],
            })

        # Pie charts
        try:
            status_counts = result_df['Dominant Status'].value_counts()
            if not status_counts.empty:
                figp1, axp1 = plt.subplots(figsize=(6, 6))
                axp1.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140)
                axp1.set_title('Distribution of Dominant Status (per 3-hour window)')
                axp1.axis('equal')
                pie_all_b64 = fig_to_base64(figp1)
            status_counts_non = result_df[result_df['Dominant Status'] != 'Running']['Dominant Status'].value_counts()
            if not status_counts_non.empty:
                figp2, axp2 = plt.subplots(figsize=(6, 6))
                axp2.pie(status_counts_non, labels=status_counts_non.index, autopct='%1.1f%%', startangle=140)
                axp2.set_title('Dominant Status Distribution (Non-Running Only)')
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

        # Slope overlay plot with shaded bands and optional zoom (server-generated static as fallback)
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
                    ax3.plot(sd['Window_Start_Time'], sd[col], label=col, linewidth=1, color=color)
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
                ymax = np.nanmax(sd.drop(columns=['Window_Start_Time']).to_numpy(dtype=float)) if not sd.empty else 0
                ax3.text(mid, ymax, label, ha='center', va='bottom', fontsize=8, color=col)
            if zoom_start and zoom_end:
                ax3.set_xlim(pd.to_datetime(zoom_start), pd.to_datetime(zoom_end))
            ax3.set_title(f"{well_name} - 30-min Slopes with Events{' (zoomed)' if zoom_start else ''}")
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Slope (per second)')
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

    # Show top N rows to keep page light; full CSV is available on disk
    table_preview = None
    if pred_df is not None and not pred_df.empty:
        table_df = pred_df.copy()
        if 'Reason' in table_df.columns:
            table_df = table_df.drop(columns=['Reason'])
        table_preview = table_df.head(500).to_dict(orient='records')
        table_columns = list(table_df.columns)
    else:
        table_preview = []
        table_columns = []

    return render_template(
        'results.html',
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
        daily_bar_data=daily_bar_data,
    )


def _build_result_df_for_events(pipeline: WellAnalysisPipeline, well_name: str) -> Optional[pd.DataFrame]:
    pred_csv = OUTPUT_DIR / f"{well_name}_failure_prediction_30min.csv"
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
                ax.plot(sd['Window_Start_Time'], sd[col], label=col, linewidth=1, color=color)
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
            ymax = np.nanmax(sd.drop(columns=['Window_Start_Time']).to_numpy(dtype=float)) if not sd.empty else 0
            ax.text(mid, ymax, label, ha='center', va='bottom', fontsize=8, color=col)
        if zoom_start and zoom_end:
            ax.set_xlim(pd.to_datetime(zoom_start), pd.to_datetime(zoom_end))
        ax.set_title(f"{well_name} - 30-min Slopes with Events{' (zoomed)' if zoom_start else ''}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Slope (per second)')
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
                pipeline.df_wc = df_wc
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
                pipeline.df_wc = df_wc
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


if __name__ == '__main__':
    # For local development
    app.run(host='0.0.0.0', port=5000, debug=True)
