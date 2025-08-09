import io
import uuid
import time
import threading
from datetime import datetime
from typing import Dict, Any

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Flask app config
# ---------------------------
app = Flask(__name__)
app.secret_key = "change-me-in-prod"  # for flashing messages
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit
ALLOWED_EXTENSIONS = {"csv"}

# ---------------------------
# Simple in-memory cache for report data (token -> dict)
# (For production, use Redis or a DB)
# ---------------------------
REPORT_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_LOCK = threading.Lock()
CACHE_TTL_SECONDS = 30 * 60  # 30 minutes


def _cleanup_cache() -> None:
    """Remove stale report entries."""
    now = time.time()
    with CACHE_LOCK:
        to_delete = [k for k, v in REPORT_CACHE.items() if now - v.get("_ts", 0) > CACHE_TTL_SECONDS]
        for k in to_delete:
            REPORT_CACHE.pop(k, None)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------
# Core processing
# ---------------------------
REQUIRED_COLUMNS = ["Date", "Product", "Quantity", "Unit Price", "Salesperson"]

def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and clean data types."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    # Normalize columns we care about
    out = df.copy()

    # Parse dates (day-first friendly; falls back to month-first if needed)
    # Coerce errors to NaT so we can drop them explicitly.
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=True)

    # Coerce numeric columns
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")
    out["Unit Price"] = pd.to_numeric(out["Unit Price"], errors="coerce")

    # Strip strings
    out["Product"] = out["Product"].astype(str).str.strip()
    out["Salesperson"] = out["Salesperson"].astype(str).str.strip()

    # Drop invalid rows (no date / no quantity / no price / missing product or salesperson)
    out = out.dropna(subset=["Date", "Quantity", "Unit Price", "Product", "Salesperson"])

    # Ensure non-negative quantities and prices
    out = out[(out["Quantity"] >= 0) & (out["Unit Price"] >= 0)]

    # Compute revenue
    out["Revenue"] = out["Quantity"] * out["Unit Price"]

    # Sort by date
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def build_aggregations(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute all required aggregations."""
    per_product = (
        df.groupby("Product", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
    )
    per_salesperson = (
        df.groupby("Salesperson", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
    )

    # Monthly summaries (by year-month)
    monthly = (
        df.assign(YearMonth=df["Date"].dt.to_period("M").astype(str))
          .groupby("YearMonth", as_index=False)["Revenue"].sum()
          .sort_values("YearMonth")
    )

    # Year-to-date -> use current year by system clock (or use df["Date"].max().year if preferred)
    current_year = datetime.now().year
    df_current_year = df[df["Date"].dt.year == current_year]
    ytd_total = float(df_current_year["Revenue"].sum()) if not df_current_year.empty else 0.0

    # Summary stats
    total_revenue = float(df["Revenue"].sum())
    best_month_row = monthly.loc[monthly["Revenue"].idxmax()] if not monthly.empty else None
    best_month = (best_month_row["YearMonth"], float(best_month_row["Revenue"])) if best_month_row is not None else ("N/A", 0.0)
    best_product_row = per_product.loc[per_product["Revenue"].idxmax()] if not per_product.empty else None
    best_product = (best_product_row["Product"], float(best_product_row["Revenue"])) if best_product_row is not None else ("N/A", 0.0)
    top_sp_row = per_salesperson.loc[per_salesperson["Revenue"].idxmax()] if not per_salesperson.empty else None
    top_salesperson = (top_sp_row["Salesperson"], float(top_sp_row["Revenue"])) if top_sp_row is not None else ("N/A", 0.0)

    return {
        "per_product": per_product,
        "per_salesperson": per_salesperson,
        "monthly": monthly,
        "total_revenue": total_revenue,
        "ytd_total": ytd_total,
        "best_month": best_month,
        "best_product": best_product,
        "top_salesperson": top_salesperson,
    }


def build_figures(aggs: Dict[str, Any]) -> Dict[str, str]:
    """Create Plotly figures and return HTML divs (with inline Plotly JS) for embedding."""
    monthly = aggs["monthly"]
    per_product = aggs["per_product"]
    per_sales = aggs["per_salesperson"]

    # Monthly revenue line chart
    if not monthly.empty:
        fig_month = px.line(
            monthly, x="YearMonth", y="Revenue", markers=True,
            title="Monthly Revenue"
        )
        fig_month.update_layout(margin=dict(l=10, r=10, t=40, b=30))
        html_month = fig_month.to_html(full_html=False, include_plotlyjs="inline")
    else:
        html_month = "<p>No monthly data available.</p>"

    # Revenue by Product (bar)
    if not per_product.empty:
        fig_prod = px.bar(
            per_product, x="Product", y="Revenue",
            title="Revenue by Product"
        )
        fig_prod.update_layout(xaxis_tickangle=-30, margin=dict(l=10, r=10, t=40, b=90))
        html_prod = fig_prod.to_html(full_html=False, include_plotlyjs=False)  # already included inline above
    else:
        html_prod = "<p>No product data available.</p>"

    # Revenue by Salesperson (bar)
    if not per_sales.empty:
        fig_sp = px.bar(
            per_sales, x="Salesperson", y="Revenue",
            title="Revenue by Salesperson"
        )
        fig_sp.update_layout(xaxis_tickangle=-30, margin=dict(l=10, r=10, t=40, b=90))
        html_sp = fig_sp.to_html(full_html=False, include_plotlyjs=False)
    else:
        html_sp = "<p>No salesperson data available.</p>"

    return {
        "monthly_chart": html_month,
        "product_chart": html_prod,
        "salesperson_chart": html_sp,
    }


def excel_bytes(df_clean: pd.DataFrame, aggs: Dict[str, Any]) -> bytes:
    """Create an Excel workbook in memory with multiple sheets and return bytes."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Summary sheet
        summary_rows = [
            ["Total Revenue", aggs["total_revenue"]],
            ["YTD Revenue", aggs["ytd_total"]],
            ["Best Month", aggs["best_month"][0]],
            ["Best Month Revenue", aggs["best_month"][1]],
            ["Best Product", aggs["best_product"][0]],
            ["Best Product Revenue", aggs["best_product"][1]],
            ["Top Salesperson", aggs["top_salesperson"][0]],
            ["Top Salesperson Revenue", aggs["top_salesperson"][1]],
        ]
        pd.DataFrame(summary_rows, columns=["Metric", "Value"]).to_excel(writer, sheet_name="Summary", index=False)

        aggs["per_product"].to_excel(writer, sheet_name="By_Product", index=False)
        aggs["per_salesperson"].to_excel(writer, sheet_name="By_Salesperson", index=False)
        aggs["monthly"].to_excel(writer, sheet_name="Monthly", index=False)
        df_clean.to_excel(writer, sheet_name="Cleaned_Data", index=False)

    return output.getvalue()


# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    _cleanup_cache()

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in the request.", "danger")
            return redirect(url_for("index"))

        file = request.files["file"]
        if file.filename == "":
            flash("Please choose a CSV file.", "danger")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Only .csv files are allowed.", "danger")
            return redirect(url_for("index"))

        try:
            # Read CSV; support UTF-8/latin1; handle common separators
            content = file.read()
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception:
                df = pd.read_csv(io.BytesIO(content), sep=";")
        except Exception as e:
            flash(f"Could not read CSV: {e}", "danger")
            return redirect(url_for("index"))

        try:
            df_clean = validate_and_clean(df)
            if df_clean.empty:
                flash("CSV appears to have no valid rows after cleaning.", "warning")
                return redirect(url_for("index"))
        except Exception as e:
            flash(f"Invalid CSV format: {e}", "danger")
            return redirect(url_for("index"))

        # Aggregations & figures
        aggs = build_aggregations(df_clean)
        figs = build_figures(aggs)

        # Build HTML tables (sortable via small JS)
        def tbl(d: pd.DataFrame) -> str:
            return d.to_html(index=False, classes="table table-striped table-hover sortable", border=0)

        # Cache for Excel download
        token = str(uuid.uuid4())
        with CACHE_LOCK:
            REPORT_CACHE[token] = {
                "_ts": time.time(),
                "df_clean": df_clean,
                "aggs": aggs,
            }

        # Summary stats for header cards
        summary = {
            "total_revenue": aggs["total_revenue"],
            "ytd_total": aggs["ytd_total"],
            "best_month": aggs["best_month"][0],
            "best_month_revenue": aggs["best_month"][1],
            "best_product": aggs["best_product"][0],
            "best_product_revenue": aggs["best_product"][1],
            "top_salesperson": aggs["top_salesperson"][0],
            "top_salesperson_revenue": aggs["top_salesperson"][1],
        }

        return render_template(
            "index.html",
            uploaded=True,
            token=token,
            summary=summary,
            table_product=tbl(aggs["per_product"]),
            table_salesperson=tbl(aggs["per_salesperson"]),
            table_monthly=tbl(aggs["monthly"]),
            chart_monthly=figs["monthly_chart"],
            chart_product=figs["product_chart"],
            chart_salesperson=figs["salesperson_chart"],
        )

    # GET
    return render_template("index.html", uploaded=False)


@app.route("/download/<token>", methods=["GET"])
def download_report(token: str):
    _cleanup_cache()
    with CACHE_LOCK:
        bundle = REPORT_CACHE.get(token)
    if not bundle:
        flash("Report session expired. Please re-upload your CSV.", "warning")
        return redirect(url_for("index"))

    df_clean = bundle["df_clean"]
    aggs = bundle["aggs"]
    xlsx = excel_bytes(df_clean, aggs)
    fname = f"sales_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(
        io.BytesIO(xlsx),
        as_attachment=True,
        download_name=fname,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    # For production, use a proper WSGI/ASGI server (gunicorn/uvicorn) and set debug=False
    app.run(host="0.0.0.0", port=8000, debug=False)
