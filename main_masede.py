#!/usr/bin/env python3
"""
Main entry point for the Well Analysis Pipeline.

This script demonstrates how to use the WellAnalysisPipeline to process well data
and generate predictions using pre-trained models.
"""
from src.utils import setup_logging
from src.pipeline import WellAnalysisPipeline
import argparse
import logging
from pathlib import Path
import sys
import pandas as pd

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Well Analysis Pipeline")

    parser.add_argument(
        "--well-name",
        type=str,
        required=True,
        help="Name of the well (used for input/output file naming)",
    )

    parser.add_argument(
        "--input-sensor",
        type=str,
        help="Path to the input CSV file. If not provided, looks in data/input/{well_name}_sensor.csv",
    )

    parser.add_argument(
        "--input-prod",
        type=str,
        help="Path to the input CSV file. If not provided, looks in data/input/{well_name}_prod.csv",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "all",
            "discharge_pressure",
            "virtual_rate",
            "slope",
            "failure_prediction",
        ],
        default="all",
        help="Which model(s) to run (default: all)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory. Defaults to data/output under the project root.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def _calculate_indicator(slope_val):
    """Calculate indicator arrow based on slope value."""
    if pd.isna(slope_val) or slope_val == 0:
        return ""
    elif slope_val > 0:
        return "↑"
    else:
        return "↓"


def _build_indicator_string(row, columns=["A", "IP", "DP", "R"]):
    """Build indicator string like 'A↓ IP↑ DP↓ R↓' from slope values."""
    indicators = []
    for col in columns:
        if col in row and pd.notna(row[col]):
            arrow = _calculate_indicator(row[col])
            if arrow:
                indicators.append(f"{col}{arrow}")
    return " ".join(indicators) if indicators else ""


def _dataset_paths(out_dir: Path, well_name: str) -> dict:
    return {
        "30min": out_dir / f"{well_name}_failure_prediction_30min.csv",
        "30min_indicator": out_dir / f"{well_name}_indicator_30min.csv",
        "3hour": out_dir / "result_df_3 jam.csv",
        "3hour_indicator": out_dir / "result_df_3jam_with_indicator.csv",
        "table_30min_display": out_dir / f"{well_name}_table_30min_display.csv",
        "table_3hour_display": out_dir / f"{well_name}_table_3hour_display.csv",
    }


def _find_dataset_file(out_dir: Path, well_name: str, dataset: str) -> Path:
    """Return existing dataset path, falling back to legacy flat files when needed."""
    canonical = _dataset_paths(out_dir, well_name).get(dataset)
    if canonical and canonical.exists():
        return canonical

    fallback_map = {
        "30min": out_dir / f"{well_name}_failure_prediction_30min.csv",
        "30min_indicator": out_dir / f"{well_name}_indicator_30min.csv",
        "3hour": out_dir / "result_df_3 jam.csv",
        "3hour_indicator": out_dir / "result_df_3jam_with_indicator.csv",
        "table_30min_display": out_dir / f"{well_name}_table_30min_display.csv",
        "table_3hour_display": out_dir / f"{well_name}_table_3hour_display.csv",
    }
    fallback = fallback_map.get(dataset)
    if fallback and fallback.exists():
        return fallback

    return canonical


def main():
    """Main function to run the well analysis pipeline."""
    # Parse command line arguments
    args = parse_arguments()

    # Set up logging
    log_level = getattr(logging, args.log_level)
    log_file = Path("logs") / f"{args.well_name}_analysis.log"
    setup_logging(log_file=log_file, level=log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting analysis for well: {args.well_name}")

    try:
        # Initialize the pipeline with optional custom output directory
        output_dir = Path(args.output_dir) if args.output_dir else None

        pipeline = WellAnalysisPipeline(args.well_name, output_dir=output_dir)
        prod_path = (
            Path(args.input_prod)
            if args.input_prod
            else Path("data") / "input" / f"{args.well_name}_prod.csv"
        )
        sensor_path = (
            Path(args.input_sensor)
            if args.input_sensor
            else Path("data") / "input" / f"{args.well_name}_sensor.csv"
        )
        if prod_path.exists():
            try:
                df_wc = pd.read_csv(prod_path)
                if "Date" in df_wc.columns:
                    df_wc["Date"] = pd.to_datetime(
                        df_wc["Date"], errors="coerce"
                    ).dt.normalize()
                pipeline.df_wc = df_wc
            except Exception:
                pass
        # Run pipeline and render with zoom
        results = pipeline.run_full_analysis(input_file=sensor_path)
        # Build slopes and resampled dataset for visualization
        slopes_df = pipeline._compute_window_slopes_30min(pipeline.data)
        df_all = pipeline._build_df_all_30min(pipeline.data)

        # Generate and save plots
        # pipeline.plot_results(results)
        # print(results)

        # Load failure prediction table from saved CSV
        # read file from Path output_dir /
        dataset_files = {
            key: _find_dataset_file(output_dir, args.well_name, key)
            for key in ["30min", "30min_indicator", "3hour", "3hour_indicator"]
        }
        print(dataset_files)

        pred_csv = (
            _find_dataset_file(output_dir, args.well_name, "30min_indicator")
            or _find_dataset_file(output_dir, args.well_name, "30min")
            or _dataset_paths(output_dir, args.well_name)["30min"]
        )
        pred_df = None
        aggregated_df = None
        aggregated_lookup = {}
        latest_failure_30min = None
        latest_failure_3hour = None
        # latest_report = {
        #     "30_minutes": None,
        #     "3_hours": None,
        # }

        # Load latest_report.json if exists (for new UI format)
        # latest_report_json = dataset_files["latest_report"]
        # if latest_report_json.exists():
        #     try:
        #         with open(latest_report_json, "r", encoding="utf-8") as f:
        #             latest_report = json.load(f)
        #     except Exception as e:
        #         print(f"Warning: Could not load latest_report.json: {e}")

        if pred_csv.exists():
            pred_df = pd.read_csv(pred_csv)
            pdf = pred_df.copy()

            aggregated_path = (
                dataset_files["3hour_indicator"]
                or _dataset_paths(args.well_name)["3hour_indicator"]
            )
            if aggregated_path.exists():
                try:
                    aggregated_df = pd.read_csv(aggregated_path)
                    aggregated_df["Window_Start_Time"] = pd.to_datetime(
                        aggregated_df["Window_Start_Time"], errors="coerce"
                    )
                    aggregated_df = aggregated_df.dropna(
                        subset=["Window_Start_Time"]
                    ).sort_values("Window_Start_Time")
                    if not aggregated_df.empty:
                        aggregated_lookup = aggregated_df.set_index(
                            "Window_Start_Time"
                        ).to_dict("index")
                except Exception:
                    aggregated_df = None
                    aggregated_lookup = {}

            # Ensure slopes exist for aggregation logic; preserve pipeline-provided Indicator values
            if pred_df is not None and not pred_df.empty:
                # Add slope cols if missing (indicator_30min.csv already has them)
                need_slopes = not set(["A", "IP", "DP", "R"]).issubset(pred_df.columns)
                if need_slopes and not slopes_df.empty:
                    pred_df["Window_Start_Time"] = pd.to_datetime(
                        pred_df["Window_Start_Time"], errors="coerce"
                    )
                    slopes_df["Window_Start_Time"] = pd.to_datetime(
                        slopes_df["Window_Start_Time"], errors="coerce"
                    )
                    pred_df = pred_df.merge(
                        slopes_df[["Window_Start_Time", "A", "IP", "DP", "R"]],
                        on="Window_Start_Time",
                        how="left",
                    )
                # Only compute Indicator if not provided by pipeline
                if "Indicator" not in pred_df.columns:
                    pred_df["Indicator"] = pred_df.apply(
                        lambda row: _build_indicator_string(
                            row, ["A", "IP", "DP", "R"]
                        ),
                        axis=1,
                    )

                # Normalize indicator text for special statuses to match pipeline semantics
                try:
                    pred_df["Status"] = pred_df["Status"].astype(str).str.strip()
                    pred_df.loc[pred_df["Status"] == "100% Watercut", "Indicator"] = (
                        "100% WC in Prod"
                    )
                    pred_df.loc[
                        pred_df["Status"] == "Electrical Downhole Problem", "Indicator"
                    ] = "A and Freq 0, others constant"
                    pred_df.loc[
                        pred_df["Status"].isin(
                            ["Running", "Shut-in", "Start-up Phase"]
                        ),
                        "Indicator",
                    ] = ""
                except Exception:
                    pass

        group_summaries = []
        detected_events = []
        running_events = []
        pie_all_b64 = None
        pie_nonrun_b64 = None
        nonrun_distribution = []
        summary_text = None
        zoom_links = []
        events_plot = None
        slope_data = None
        daily_bar_data = {"dates": [], "counts": [], "statuses": []}
        if (
            pred_df is not None
            and not pred_df.empty
            and "Window_Start_Time" in pred_df.columns
            and "Status" in pred_df.columns
        ):
            pdf = pred_df.copy()
            pdf["Window_Start_Time"] = pd.to_datetime(
                pdf["Window_Start_Time"], errors="coerce"
            )
            pdf = pdf.dropna(subset=["Window_Start_Time"]).sort_values(
                "Window_Start_Time"
            )
            pdf["Status"] = pdf["Status"].astype(str).str.strip()
            pdf_valid = pdf[~pdf["Status"].str.lower().isin(["", "nan"])]
            non_running = pdf_valid[pdf_valid["Status"].str.lower() != "running"]
            if not non_running.empty:
                latest = non_running.sort_values("Window_Start_Time").iloc[-1]
                latest_failure_30min = {
                    "timestamp": latest["Window_Start_Time"].strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "status": latest["Status"],
                    "recommendation": str(
                        latest.get("Recommendation", "") or ""
                    ).strip(),
                    "indicator": str(latest.get("Indicator", "") or "").strip(),
                }
                # Keep old format for backward compatibility
                latest_failure = latest_failure_30min
            pdf["date"] = pdf["Window_Start_Time"].dt.normalize()
            pdf["group_start"] = pdf["Window_Start_Time"].dt.floor("3H")
            pdf["group_end"] = pdf["group_start"] + pd.Timedelta(hours=3)

            # Dominant status logic (include Running)
            severity_priority = {
                "Increase in Watercut": 3,
                "Shut-in": 2,
                "Electrical Downhole Problem": 1,
                "Running": 0,
            }

            def pick_dominant_status(group: pd.DataFrame) -> str:
                grp_counts = group["Status"].value_counts()
                maxc = grp_counts.max()
                top_statuses = grp_counts[grp_counts == maxc].index.tolist()
                if len(top_statuses) == 1:
                    return top_statuses[0]
                day = group["date"].iloc[0]
                day_rows = pdf[pdf["date"] == day]
                day_counts = day_rows["Status"].value_counts()
                day_counts = (
                    day_counts[day_counts.index.isin(top_statuses)]
                    if not day_counts.empty
                    else pd.Series(dtype=int)
                )
                if not day_counts.empty:
                    top_day = day_counts[day_counts == day_counts.max()].index.tolist()
                else:
                    top_day = top_statuses
                if len(top_day) == 1:
                    return top_day[0]
                return max(top_day, key=lambda s: severity_priority.get(s, 0))

            groups = pdf.groupby(["date", "group_start", "group_end"], as_index=False)
            dom_rows = []
            for (d, gs, ge), g in groups:
                dominant = pick_dominant_status(g)
                non_run_count = (g["Status"] != "Running").sum()

                # Calculate aggregated indicator for 3-hour window
                agg_indicator_parts = []
                for param in ["A", "IP", "DP", "R"]:
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

                dom_rows.append(
                    {
                        "date": d,
                        "group_start": gs,
                        "group_end": ge,
                        "Dominant Status": dominant,
                        "non_running_count": int(non_run_count),
                        "indicator": agg_indicator,
                    }
                )
            result_df = pd.DataFrame(dom_rows)

            # Summaries for rendering
            for _, r in result_df.iterrows():
                gs = pd.to_datetime(r["group_start"])
                ge = pd.to_datetime(r["group_end"])
                dominant_status = r["Dominant Status"]
                indicator_value = r.get("indicator", "")
                recommendation_value = str(r.get("recommendation", "") or "").strip()

                agg_info = aggregated_lookup.get(gs) if aggregated_lookup else None
                if agg_info:
                    agg_status = str(agg_info.get("Dominant Status", "") or "").strip()
                    if agg_status:
                        dominant_status = agg_status
                    agg_indicator = str(agg_info.get("Indicator", "") or "").strip()
                    if agg_indicator:
                        indicator_value = agg_indicator
                    agg_rec = agg_info.get("Recommendation", "")
                    if isinstance(agg_rec, str):
                        agg_rec = agg_rec.strip()
                    elif pd.notna(agg_rec):
                        agg_rec = str(agg_rec).strip()
                    else:
                        agg_rec = ""
                    if agg_rec:
                        recommendation_value = agg_rec

                summary = {
                    "date": (
                        pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
                        if not pd.isna(r["date"])
                        else ""
                    ),
                    "group_start": gs.strftime("%Y-%m-%d %H:%M:%S"),
                    "group_end": ge.strftime("%Y-%m-%d %H:%M:%S"),
                    "non_running_count": int(r["non_running_count"]),
                    "dominant_status": dominant_status,
                    "indicator": indicator_value,
                    "recommendation": recommendation_value,
                }
                group_summaries.append(summary)
                if r["Dominant Status"] != "Running":
                    detected_events.append(summary)
                    # Store latest 3-hour failure
                    if latest_failure_3hour is None or pd.to_datetime(
                        summary["group_start"]
                    ) > pd.to_datetime(latest_failure_3hour["timestamp"]):
                        latest_failure_3hour = {
                            "timestamp": summary["group_start"],
                            "status": summary["dominant_status"],
                            "recommendation": summary.get("recommendation", ""),
                            "indicator": summary["indicator"],
                        }
                else:
                    running_events.append(summary)

        # get min, max date
        dmin_date = result_df["group_start"].min()
        dmax_date = result_df["group_end"].max()
        # get days between min and max date
        dtotal_days = (
            (dmax_date - dmin_date).days
            if pd.notna(dmin_date) and pd.notna(dmax_date)
            else None
        )

        # Pie charts
        status_counts = None
        status_counts_non = None
        try:
            status_counts = result_df["Dominant Status"].value_counts()
            status_counts_non = result_df[result_df["Dominant Status"] != "Running"][
                "Dominant Status"
            ].value_counts()
        except Exception:
            pass

        # Build daily bar data from detected_events (3-hour non-running windows)
        daily_bar_data = {"dates": [], "counts": [], "statuses": []}
        try:
            # merge detected_events and running_events
            #  to get all events for daily summary
            allevents = detected_events + running_events
            if allevents:
                events_df = pd.DataFrame(allevents)
                events_df["date"] = pd.to_datetime(events_df["date"]).dt.date

                # Count events per day and find dominant status for that day
                daily_summary = (
                    events_df.groupby("date")
                    .agg(
                        count=("date", "size"),
                        status=("dominant_status", lambda s: s.value_counts().idxmax()),
                    )
                    .reset_index()
                )

                daily_bar_data = {
                    "dates": [d.strftime("%Y-%m-%d") for d in daily_summary["date"]],
                    "counts": daily_summary["count"].tolist(),
                    "statuses": daily_summary["status"].tolist(),
                }
        except Exception:
            daily_bar_data = {"dates": [], "counts": [], "statuses": []}

        # Prepare embedded slope data for client-side Plotly (times/series/bands)
        try:
            sd0 = slopes_df.copy()
            sd0["Window_Start_Time"] = pd.to_datetime(
                sd0["Window_Start_Time"], errors="coerce"
            )
            sd0 = sd0.dropna(subset=["Window_Start_Time"]).sort_values(
                "Window_Start_Time"
            )
            times0 = sd0["Window_Start_Time"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
            series_cols0 = ["A", "IP", "DP", "IT", "MT", "V", "R"]
            series0 = {
                c: (
                    sd0[c].astype(float).where(pd.notna(sd0[c]), None).tolist()
                    if c in sd0.columns
                    else []
                )
                for c in series_cols0
            }
            bands0 = []
            for _, row in result_df[
                result_df["Dominant Status"] != "Running"
            ].iterrows():
                bands0.append(
                    {
                        "start": pd.to_datetime(row["group_start"]).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "end": pd.to_datetime(row["group_end"]).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "label": row["Dominant Status"],
                    }
                )
            slope_data = {"times": times0, "series": series0, "bands": bands0}
        except Exception:
            slope_data = None

        # If slope_data not prepared above (e.g., no pred_df), still provide base series without bands
        if slope_data is None:
            try:
                sd0 = slopes_df.copy()
                sd0["Window_Start_Time"] = pd.to_datetime(
                    sd0["Window_Start_Time"], errors="coerce"
                )
                sd0 = sd0.dropna(subset=["Window_Start_Time"]).sort_values(
                    "Window_Start_Time"
                )
                times0 = (
                    sd0["Window_Start_Time"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
                )
                series_cols0 = ["A", "IP", "DP", "IT", "MT", "V", "R"]
                series0 = {
                    c: (
                        sd0[c].astype(float).where(pd.notna(sd0[c]), None).tolist()
                        if c in sd0.columns
                        else []
                    )
                    for c in series_cols0
                }
                slope_data = {"times": times0, "series": series0, "bands": []}
            except Exception:
                slope_data = {"times": [], "series": {}, "bands": []}

        # line rate (Virtual Rate) over time
        virtualRate = None
        x = None
        y = None
        if "Virtual Rate (BFPD) (Raw)" in pipeline.data.columns:
            x = pd.to_datetime(pipeline.data["Reading Time"], errors="coerce")
            y = pipeline.data["Virtual Rate (BFPD) (Raw)"]
        elif "predicted_virtual_rate" in pipeline.data.columns:
            x = pd.to_datetime(pipeline.data["Reading Time"], errors="coerce")
            y = pipeline.data["predicted_virtual_rate"]
        if x is not None and y is not None:
            df = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
            result = {
                "x": df["x"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
                "y": df["y"].tolist(),
            }
            virtualRate = {
                "data": result,
                "label": {
                    "x": "Time",
                    "y": "Virtual Rate (BFPD)",
                    "title": f"{args.well_name} - Virtual Rate ({'Raw' if 'Virtual Rate (BFPD) (Raw)' in pipeline.data.columns else 'predicted'})",
                },
            }

        # Load failure prediction table from saved CSV
        pred_csv = Path(output_dir) / f"{args.well_name}_failure_prediction_30min.csv"
        if not pred_csv.exists():
            pred_csv = None

        resultESP = {
            "well_name": args.well_name,
            "recommendations": summary_text,
            "status_counts": {
                "3-hour windows": (
                    status_counts.to_dict() if status_counts is not None else None
                ),
                "Non-Running only": (
                    status_counts_non.to_dict()
                    if status_counts_non is not None
                    else None
                ),
            },
            # "latest_failure": latest_report,
            "daily_bar_data": daily_bar_data,
            "total_reading": {
                "start_date": (
                    dmin_date.strftime("%Y-%m-%d") if pd.notna(dmin_date) else None
                ),
                "end_date": (
                    dmax_date.strftime("%Y-%m-%d") if pd.notna(dmax_date) else None
                ),
                "total_days": dtotal_days if dtotal_days is not None else None,
            },
            # "detected_events": detected_events,
            # "group_summaries": group_summaries,
        }
        result_file = Path(output_dir) / f"{args.well_name}_esp_summary.json"
        import json

        with open(result_file, "w") as f:
            json.dump(resultESP, f, indent=4)

        # save slope_data to json
        slope_file = Path(output_dir) / f"{args.well_name}_esp_slope_data.json"
        with open(slope_file, "w") as f:
            json.dump(slope_data, f, indent=4)

        # save virtualRate to json
        vr_file = Path(output_dir) / f"{args.well_name}_esp_virtual_rate.json"
        with open(vr_file, "w") as f:
            json.dump(virtualRate, f, indent=4)

        table_preview = None
        if pred_df is not None and not pred_df.empty:
            table_df = pred_df.copy()
            # Keep only the requested columns if available
            requested_cols = [
                "Window_Start_Time",
                "Indicator",
                "Status",
                "Recommendation",
                "Date",
            ]
            keep_cols = [c for c in requested_cols if c in table_df.columns]
            if keep_cols:
                table_df = table_df[keep_cols]
            else:
                # Fallback: drop known extra columns
                drop_cols = [
                    "Reason",
                    "Prediction",
                    "A",
                    "IP",
                    "DP",
                    "IT",
                    "MT",
                    "V",
                    "R",
                ]
                table_df = table_df.drop(
                    columns=[c for c in drop_cols if c in table_df.columns],
                    errors="ignore",
                )

            # Store filtered table for download
            # table_df_for_download = table_df.copy()

            table_preview = table_df.to_dict(orient="records")
            # table_columns = list(table_df.columns)
            # status_options = sorted({
            #     str(row.get('Status', '')).strip()
            #     for row in table_preview
            #     if str(row.get('Status', '')).strip()
            # })
            # indicator_options = sorted({
            #     str(row.get('Indicator', '')).strip()
            #     for row in table_preview
            #     if str(row.get('Indicator', '')).strip()
            # })
        else:
            table_preview = []
            # table_columns = []
            # status_options = []
            # indicator_options = []

        if table_preview is not None:
            # save sqlite db
            import sqlite3

            db_file = Path(output_dir) / f"{args.well_name}_esp_failure.sqlite"
            # remove if db file exists
            if db_file.exists():
                db_file.unlink()
            try:
                conn = sqlite3.connect(db_file)
                # save table_preview to sql table failure_prediction
                pd.DataFrame(table_preview).to_sql(
                    "failure_prediction", conn, if_exists="replace", index=False
                )
                # add index on Window_Start_Time
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_window_start ON failure_prediction (Window_Start_Time)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_status ON failure_prediction (Status)"
                )
                conn.close()
            except Exception:
                # remove db file if exists
                if db_file.exists():
                    db_file.unlink()

        # save group_summaries to sqlite db
        summary_db_file = Path(output_dir) / f"{args.well_name}_esp_grp_summary.sqlite"
        # remove if db file exists
        if summary_db_file.exists():
            summary_db_file.unlink()
        try:
            conn = sqlite3.connect(summary_db_file)
            # save result_df to sql table group_summary
            # convert to dataframe
            df_group_summaries = pd.DataFrame(group_summaries)
            # replace nan text with None in column indicator
            df_group_summaries["indicator"] = df_group_summaries["indicator"].replace(
                "nan", None
            )

            df_group_summaries.to_sql(
                "group_summary", conn, if_exists="replace", index=False
            )
            # add index on date
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON group_summary (date)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_group_start ON group_summary (group_start)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_group_end ON group_summary (group_end)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_non_running_count ON group_summary (non_running_count)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dominant_status ON group_summary (dominant_status)"
            )

            conn.close()
        except Exception:
            print("Error saving group summaries to database")
            # remove db file if exists
            if summary_db_file.exists():
                summary_db_file.unlink()

        print("Analysis completed successfully!")

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
