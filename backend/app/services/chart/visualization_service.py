import os
import uuid
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ----------------------------
# Paths / URL mapping
# ----------------------------
STATIC_IMAGE_DIR = os.getenv("STATIC_IMAGE_DIR", "static/images")   # web-served dir
LOCAL_SAVE_DIR   = os.getenv("LOCAL_SAVE_DIR", "charts/images")     # filesystem dir (archive)
PUBLIC_URL_BASE  = os.getenv("PUBLIC_IMAGE_BASE", "/static/images") # URL prefix the FE can reach

# ----------------------------
# Service
# ----------------------------
class VisualizationService:
    """
    Deterministic Plotly charts + robust image saving.
    - Cleans DataFrames (dates, tz-naive, numeric coercion, duplicate cols)
    - Auto-resamples wide time windows for time series
    - Safer fallbacks to table/correlation when needed
    - If PNG export fails (e.g., missing Kaleido), writes a placeholder image (if Pillow available),
      otherwise raises a clear RuntimeError.
    """

    def __init__(self, use_ai_reco: bool = False, theme: str = "plotly_white"):
        self.use_ai_reco = False  # keep OFF for stability
        self.theme = theme

        self._local_dir = Path(LOCAL_SAVE_DIR)
        self._local_dir.mkdir(parents=True, exist_ok=True)

        self._static_dir = Path(STATIC_IMAGE_DIR)
        self._static_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "VIZ_INIT: local_dir=%s static_dir=%s public_url_base=%s",
            self._local_dir.resolve(), self._static_dir.resolve(), PUBLIC_URL_BASE
        )

    # =====================================================
    # Public entrypoint
    # =====================================================
    def create_visualization(
        self,
        df: pd.DataFrame,
        user_query: str = "",
        force_chart_type: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            df = self._prep_dataframe(df)
            if force_chart_type:
                chart_type = force_chart_type
                reasoning = f"Forced: {force_chart_type}"
                alts: List[str] = []
            else:
                chart_type, reasoning, alts = self._fallback_recommendation(df, user_query)

            title = title or (user_query[:120] if user_query else chart_type.replace("_", " ").title())
            fig = self._build_chart(df, chart_type, title)

            image_url, filename = self._save_figure(fig, chart_type)
            logger.info("VIZ_SAVED: file=%s url=%s", filename, image_url)

            # light analysis + insights (agent may call these directly too)
            analysis = self._analyze(df)
            insights = self._generate_insights(df, chart_type, analysis, hints=[])

            return {
                "enabled": True,
                "type": chart_type,
                "title": title,
                "url": image_url,      # UI uses this
                "filename": filename,  # optional
                "reasoning": reasoning,
                "alternatives": alts,
                "insights": insights,
                "data_summary": {
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "column_names": list(map(str, df.columns)),
                },
            }
        except Exception as e:
            logger.exception("Visualization failed: %s", e)
            return {"enabled": False, "reason": f"Visualization error: {e}"}

    # =====================================================
    # DataFrame prep (robust)
    # =====================================================
    def _prep_dataframe(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Make df safe & predictable; handle duplicates, dates, numbers, NaNs/Infs."""
        if df is None:
            return pd.DataFrame()
        dfx = df.copy()

        # Drop fully-empty columns
        try:
            dfx = dfx.loc[:, ~(dfx.isna().all())]
        except Exception:
            pass

        # Deduplicate column labels to avoid hard crashes downstream
        dfx = self._dedupe_columns(dfx)

        # Coerce datetimes by name/value (timezone → naive)
        for c in dfx.columns:
            s = dfx[c]
            name = str(c).lower()
            if pd.api.types.is_datetime64_any_dtype(s):
                dfx[c] = self._strip_tz(s)
                continue
            if any(k in name for k in ("date", "time", "day", "workdate", "startdate", "enddate")):
                dfx[c] = self._coerce_datetime(s)

        # Coerce numeric-ish by name (but avoid ID-like)
        for c in dfx.columns:
            s = dfx[c]
            name = str(c).lower()
            if pd.api.types.is_numeric_dtype(s):
                continue
            if any(k in name for k in ("amount", "qty", "quantity", "hours", "value", "count", "rate", "score", "minutes")):
                coerced = pd.to_numeric(s, errors="coerce")
                if coerced.notna().mean() >= 0.6:
                    dfx[c] = coerced

        # Replace inf/-inf → NaN, then keep rows with something present
        dfx = dfx.replace([np.inf, -np.inf], np.nan)

        # If a single datetime column exists, sort for nicer visuals
        dt_cols = [c for c in dfx.columns if pd.api.types.is_datetime64_any_dtype(dfx[c])]
        if len(dt_cols) == 1:
            try:
                dfx = dfx.sort_values(by=dt_cols[0], ascending=True)
            except Exception:
                pass

        return dfx

    def _dedupe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure unique column names: col, col_1, col_2, ..."""
        seen: Dict[str, int] = {}
        new_cols: List[str] = []
        for c in df.columns:
            base = str(c)
            if base not in seen:
                seen[base] = 0
                new_cols.append(base)
            else:
                seen[base] += 1
                new_cols.append(f"{base}_{seen[base]}")
        if list(df.columns) != new_cols:
            df = df.copy()
            df.columns = new_cols
        return df

    @staticmethod
    def _strip_tz(s: pd.Series) -> pd.Series:
        try:
            if hasattr(s.dt, "tz") and s.dt.tz is not None:
                return s.dt.tz_convert(None)
        except Exception:
            pass
        return s

    @staticmethod
    def _coerce_datetime(s: pd.Series) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(s):
            return VisualizationService._strip_tz(s)
        try:
            out = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=True)
            try:
                out = out.dt.tz_convert(None)
            except Exception:
                pass
            return out
        except Exception:
            return s

    # =====================================================
    # Recommendation (deterministic & simple)
    # =====================================================
    def _fallback_recommendation(self, df: pd.DataFrame, query: str) -> Tuple[str, str, List[str]]:
        if df is None or df.empty:
            return ("table", "No data to visualize", [])

        cols = list(df.columns)
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        datetime_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
        q = (query or "").lower()

        if "trend" in q or "over time" in q or "折線" in q or "趨勢" in q or datetime_cols:
            return ("line_chart", "Time trend", ["bar_chart"])
        if "distribution" in q or "histogram" in q or "分佈" in q:
            return ("histogram", "Numeric distribution", ["box_plot"])
        if len(numeric) >= 2 or "correlation" in q or "相關" in q:
            return ("scatter_plot", "Two+ numeric variables", ["heatmap", "bar_chart"])
        if len(numeric) == 1:
            return ("bar_chart", "Numeric by category", ["pie_chart", "box_plot"])
        return ("table", "Fallback to table", [])

    # =====================================================
    # Chart builders (robust)
    # =====================================================
    def _build_chart(self, df: pd.DataFrame, chart_type: str, title: str) -> go.Figure:
        chart_type = (chart_type or "").lower()
        theme = self.theme

        if df is None or df.empty:
            return self._no_data_figure()

        cols = list(df.columns)
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        datetime_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
        categorical = [c for c in cols if c not in numeric and c not in datetime_cols]

        x = datetime_cols[0] if datetime_cols else (categorical[0] if categorical else (cols[0] if cols else None))
        y = numeric[0] if numeric else (cols[1] if len(cols) > 1 else None)

        # Time-series: auto-resample to keep point count in check
        if x and pd.api.types.is_datetime64_any_dtype(df[x]) and chart_type in {"line_chart", "bar_chart"}:
            df = self._maybe_resample_timeseries(df, x, y)

        # Build
        if chart_type == "line_chart" and x is not None and y is not None and y in df.columns:
            fig = px.line(df, x=x, y=y, title=title, template=theme)
        elif chart_type == "bar_chart" and x is not None:
            if y is None or y not in df.columns or not pd.api.types.is_numeric_dtype(df[y]):
                counts = df[x].value_counts(dropna=False).reset_index()
                counts.columns = [str(x), "Count"]
                fig = px.bar(counts, x=str(x), y="Count", title=title, template=theme)
            else:
                fig = px.bar(df, x=x, y=y, title=title, template=theme)
        elif chart_type == "pie_chart":
            name_col = x or (categorical[0] if categorical else cols[0])
            if y is None or y not in df.columns or not pd.api.types.is_numeric_dtype(df[y]):
                vc = df[name_col].fillna("Unknown").value_counts(dropna=False).reset_index()
                vc.columns = [str(name_col), "Count"]
                fig = px.pie(vc, names=str(name_col), values="Count", title=title)
            else:
                fig = px.pie(df, names=name_col, values=y, title=title)
        elif chart_type == "scatter_plot":
            if len(numeric) >= 2:
                fig = px.scatter(df, x=numeric[0], y=numeric[1], title=title, template=theme)
            elif x is not None and y is not None and y in df.columns:
                fig = px.scatter(df, x=x, y=y, title=title, template=theme)
            else:
                fig = px.scatter(title=title, template=theme)
        elif chart_type == "histogram":
            xx = numeric[0] if numeric else (cols[0] if cols else None)
            if xx and xx in df.columns and pd.api.types.is_numeric_dtype(df[xx]):
                fig = px.histogram(df, x=xx, title=title, template=theme)
            else:
                # fallback table when histogram target is non-numeric
                fig = self._table_figure(df, "Table (non-numeric for histogram)")
        elif chart_type == "heatmap":
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(num_cols) >= 2:
                corr = df[num_cols].corr(numeric_only=True)
                fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Blues"))
                fig.update_layout(title=title, template=theme)
            else:
                fig = self._table_figure(df, "Table (no numeric matrix for heatmap)")
        elif chart_type == "box_plot":
            yy = y or (numeric[0] if numeric else None)
            fig = px.box(df, y=yy, x=(categorical[0] if categorical else None), title=title, template=theme)
        else:
            fig = self._table_figure(df, title)

        # Polishing
        fig.update_layout(
            width=800, height=500, template=theme,
            legend_title_text=None,
            margin=dict(l=40, r=20, t=60, b=40),
            hovermode="x unified" if (x and x in df.columns and pd.api.types.is_datetime64_any_dtype(df[x])) else "closest",
        )
        if x and x in df.columns and pd.api.types.is_datetime64_any_dtype(df[x]):
            fig.update_xaxes(showgrid=True, tickformat="%Y-%m-%d")
        return fig

    def _no_data_figure(self) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template=self.theme, width=800, height=500, title="No Data")
        return fig

    def _table_figure(self, df: pd.DataFrame, title: str) -> go.Figure:
        header_vals = list(map(str, df.columns))
        # Convert each column to plain Python to avoid serialization surprises
        cell_vals = [df[c].tolist() for c in df.columns]
        fig = go.Figure(data=[go.Table(
            header=dict(values=header_vals, fill_color="#e2e8f0", align="left"),
            cells=dict(values=cell_vals, fill_color="#ffffff", align="left"),
        )])
        fig.update_layout(title=title, template=self.theme, width=800, height=500)
        return fig

    # =====================================================
    # Time-series resampling
    # =====================================================
    def _maybe_resample_timeseries(self, df: pd.DataFrame, x_col: str, y_col: Optional[str], *, target_points: int = 400) -> pd.DataFrame:
        """Resample long time series to D/W/M/Q/Y by span to keep visuals readable."""
        if x_col not in df.columns:
            return df
        s = df[x_col].dropna()
        if not pd.api.types.is_datetime64_any_dtype(s) or s.empty:
            return df
        if len(df) <= target_points:
            return df

        # Span-based frequency
        try:
            span_days = (s.max() - s.min()).days
        except Exception:
            return df
        if span_days <= 60:
            freq = "D"
        elif span_days <= 365 * 2:
            freq = "W"
        elif span_days <= 365 * 5:
            freq = "M"
        elif span_days <= 365 * 12:
            freq = "Q"
        else:
            freq = "Y"

        dfy = df.copy().set_index(x_col)
        if y_col and y_col in dfy.columns and pd.api.types.is_numeric_dtype(dfy[y_col]):
            grouped = dfy[y_col].resample(freq).sum().reset_index()
            grouped.columns = [x_col, y_col]
            return grouped
        # Count fallback
        grouped = dfy.resample(freq).size().reset_index(name="Count")
        return grouped

    # =====================================================
    # Save (PNG) with resilient fallback
    # =====================================================
    def _save_figure(self, fig: go.Figure, chart_type: str) -> Tuple[str, str]:
        """
        Save a PNG into LOCAL_SAVE_DIR and mirror to STATIC_IMAGE_DIR.
        Returns (web_url, filename).
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{chart_type}_{ts}_{uuid.uuid4().hex[:8]}.png"
        local_abs = (self._local_dir / fname).resolve()
        static_abs = (self._static_dir / fname).resolve()

        # Try to render PNG via Kaleido
        try:
            fig.write_image(str(local_abs), width=800, height=500)
            logger.info("VIZ_WRITE_IMAGE_OK: %s", local_abs)
        except Exception as e:
            logger.error(
                "Static PNG export failed: %s. Will attempt Pillow placeholder. "
                "Install Kaleido: `pip install -U kaleido`.", e
            )
            if not self._write_placeholder_png(local_abs, str(e)):
                # No placeholder available → escalate clearly
                raise RuntimeError(
                    f"Static PNG export failed: {e}. "
                    "Install Kaleido (`pip install -U kaleido`) or enable image service."
                )

        # Mirror to static dir (so the app can serve it)
        try:
            if local_abs != static_abs:
                shutil.copyfile(local_abs, static_abs)
        except Exception as e:
            logger.warning("Failed to copy chart to static dir: %s", e)

        web_url = f"{PUBLIC_URL_BASE.rstrip('/')}/{fname}"
        return web_url, fname

    def _write_placeholder_png(self, out_path: Path, error_msg: str) -> bool:
        """If Pillow exists, write a simple placeholder PNG explaining the error."""
        try:
            from PIL import Image, ImageDraw, ImageFont  # type: ignore
        except Exception:
            return False

        try:
            img = Image.new("RGB", (800, 500), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)

            title = "Chart rendering unavailable"
            body1 = "Static export failed (Kaleido missing)."
            body2 = "Ask admin to install: pip install -U kaleido"
            body3 = f"Details: {str(error_msg)[:120]}..."

            try:
                font_title = ImageFont.truetype("arial.ttf", 22)
                font_body  = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                font_title = None
                font_body = None

            draw.text((40, 40), title, fill=(20, 20, 20), font=font_title)
            draw.text((40, 100), body1, fill=(40, 40, 40), font=font_body)
            draw.text((40, 130), body2, fill=(40, 40, 40), font=font_body)
            draw.text((40, 170), body3, fill=(70, 70, 70), font=font_body)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(out_path), format="PNG")
            logger.info("Placeholder PNG written: %s", out_path)
            return True
        except Exception as e:
            logger.warning("Failed to write placeholder PNG: %s", e)
            return False

    # =====================================================
    # Lightweight profiling + insights (used by agent)
    # =====================================================
    def _analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "rows": int(df.shape[0]) if df is not None else 0,
            "cols": int(df.shape[1]) if df is not None else 0,
            "numeric_cols": [],
            "datetime_cols": [],
            "categorical_cols": [],
            "time_span_days": None,
            "top_categories": {},
        }
        if df is None or df.empty:
            return out

        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                out["numeric_cols"].append(str(c))
            elif pd.api.types.is_datetime64_any_dtype(df[c]):
                out["datetime_cols"].append(str(c))
            else:
                out["categorical_cols"].append(str(c))

        # time span
        if out["datetime_cols"]:
            s = df[out["datetime_cols"][0]].dropna()
            try:
                if not s.empty:
                    out["time_span_days"] = int((s.max() - s.min()).days)
            except Exception:
                pass

        # top categories for first few categoricals
        for c in out["categorical_cols"][:3]:
            try:
                vc = df[c].fillna("Unknown").value_counts().head(5)
                out["top_categories"][c] = [(str(k), int(v)) for k, v in vc.items()]
            except Exception:
                out["top_categories"][c] = []

        return out

    def _generate_insights(
        self,
        df: pd.DataFrame,
        chart_type: str,
        analysis: Dict[str, Any],
        hints: Optional[List[str]] = None,
    ) -> List[str]:
        """Very lightweight heuristics; safe, deterministic."""
        insights: List[str] = []
        if not df is None and not df.empty:
            insights.append(f"{analysis.get('rows', 0)} rows × {analysis.get('cols', 0)} columns.")
            if analysis.get("datetime_cols"):
                span = analysis.get("time_span_days")
                if span is not None:
                    insights.append(f"Time span ~ {span} days.")
            if analysis.get("top_categories"):
                for col, pairs in analysis["top_categories"].items():
                    if pairs:
                        top = ", ".join([f"{k} ({v})" for k, v in pairs])
                        insights.append(f"Top in {col}: {top}")
        if hints:
            for h in hints:
                if isinstance(h, str) and h.strip():
                    insights.append(h.strip())
        return insights
