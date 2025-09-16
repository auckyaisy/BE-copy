import io
import base64
from pathlib import Path
from typing import Optional

from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
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

        results = pipeline.run_full_analysis(input_file=sensor_path)

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

        # Plot: line rate (Virtual Rate predicted) over time
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        if 'Virtual Rate (BFPD) (Raw)' in pipeline.data.columns:
            ax2.plot(pd.to_datetime(pipeline.data['Reading Time'], errors='coerce'),
                     pipeline.data['Virtual Rate (BFPD) (Raw)'],
                     label='Virtual Rate (predicted)', color='#4C78A8', linewidth=1)
        ax2.set_title(f"{well_name} - Virtual Rate (predicted)")
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
        if pred_df is not None and not pred_df.empty and 'Window_Start_Time' in pred_df.columns and 'Status' in pred_df.columns:
            pdf = pred_df.copy()
            pdf['Window_Start_Time'] = pd.to_datetime(pdf['Window_Start_Time'], errors='coerce')
            pdf = pdf.dropna(subset=['Window_Start_Time']).sort_values('Window_Start_Time')
            pdf['date'] = pdf['Window_Start_Time'].dt.normalize()
            pdf['group_start'] = pdf['Window_Start_Time'].dt.floor('3H')
            pdf['group_end'] = pdf['group_start'] + pd.Timedelta(hours=3)

            # Function to pick dominant status per group (include 'Running')
            # Tie-breaking: (1) global frequency within day; (2) severity priority; (3) default to 'Running' last
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
                # Tie-break by global (same day) frequency
                day = group['date'].iloc[0]
                day_rows = pdf[pdf['date'] == day]
                day_counts = day_rows['Status'].value_counts()
                # Reduce to tied statuses only
                day_counts = day_counts[day_counts.index.isin(top_statuses)] if not day_counts.empty else pd.Series(dtype=int)
                if not day_counts.empty:
                    top_day = day_counts[day_counts == day_counts.max()].index.tolist()
                else:
                    top_day = top_statuses
                if len(top_day) == 1:
                    return top_day[0]
                # Final tie-break by severity priority (default 0)
                winner = max(top_day, key=lambda s: severity_priority.get(s, 0))
                return winner

            # Apply per (date, group_start)
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
            # Build a DataFrame for charts and summaries
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

        # Show top N rows to keep page light; full CSV is available on disk
        table_preview = None
        if pred_df is not None and not pred_df.empty:
            table_preview = pred_df.head(500).to_dict(orient='records')
            table_columns = list(pred_df.columns)
        else:
            table_preview = []
            table_columns = []

        return render_template(
            'results.html',
            well_name=well_name,
            slope_plot=slope_plot,
            rate_plot=rate_plot,
            table_columns=table_columns,
            table_preview=table_preview,
            pred_csv_path=str(pred_csv),
            group_summaries=group_summaries,
            detected_events=detected_events,
            pie_all_b64=pie_all_b64,
            pie_nonrun_b64=pie_nonrun_b64,
        )

    except Exception as e:
        flash(f"Error during analysis: {e}")
        return redirect(url_for('index'))


if __name__ == '__main__':
    # For local development
    app.run(host='0.0.0.0', port=5000, debug=True)
