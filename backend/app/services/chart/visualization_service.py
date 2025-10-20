import os
import uuid
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ---- Config via env (optional) ----
STATIC_IMAGE_DIR = os.getenv("STATIC_IMAGE_DIR", "charts/images")   # web-served dir
LOCAL_SAVE_DIR   = os.getenv("LOCAL_SAVE_DIR", "charts/images")     # filesystem dir
DEFAULT_REGION   = os.getenv("AWS_REGION", "ap-southeast-2")

# Optional AI (for recommendations only). We keep this tiny & safe.
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    _OPENAI_OK = bool(OPENAI_KEY)
except Exception:
    _OPENAI_OK = False


# ---------------------------
# Utility
# ---------------------------

def _coerce_dtypes_for_viz(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight dtype coercion helpful for charts (dates, numeric)."""
    if df is None or df.empty:
        return df
    df = df.copy()

    # Dates
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ["date", "day", "time", "workdate", "startdate", "enddate", "canusedate", "disableddate"]):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass

    # Numeric strings with commas (avoid deprecated errors="ignore")
    for c in df.columns:
        if df[c].dtype == object:
            try:
                s = df[c].astype(str).str.replace(",", "")
                try:
                    df[c] = pd.to_numeric(s)  # will raise on non-numeric; that's fine
                except Exception:
                    # leave as-is if truly non-numeric
                    pass
            except Exception:
                pass

    return df



# -------------------------------------------------------------------
# Visualization Service
# -------------------------------------------------------------------

class VisualizationService:
    """
    Secure, deterministic chart generator with optional AI-powered recommendation (no AI codegen).
    - Chart recommendations: AI (if available) -> fallback rule-based
    - Chart rendering: deterministic Plotly builders (no exec)
    - Output: saved PNG + 'visualization' dict the UI expects
    """

    def __init__(self, use_ai_reco: bool = True, theme: str = "plotly_white"):
        self.use_ai_reco = use_ai_reco and _OPENAI_OK
        self.theme = theme

        # Ensure save directories exist
        self._local_dir = Path(LOCAL_SAVE_DIR)
        self._local_dir.mkdir(parents=True, exist_ok=True)
        self._static_dir = Path(STATIC_IMAGE_DIR)
        self._static_dir.mkdir(parents=True, exist_ok=True)

        # Optional AI: build a tiny prompt (recommendation only)
        if self.use_ai_reco:
            try:
                self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
                self._reco_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        "You are a data visualization expert. Given data structure and user query, "
                        "recommend the best chart. Return JSON only with keys: "
                        '{"recommended_chart": "...", "reasoning": "...", "alternative_charts": [], '
                        '"insights_to_highlight": []}'
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "Shape: {shape}\n"
                        "Numeric: {numeric}\n"
                        "Categorical: {categorical}\n"
                        "Datetime: {datetime}\n"
                        "User query: {query}\n"
                        "Columns: {columns}\n"
                        "Sample: {sample}\n"
                    ),
                ])
            except Exception as e:
                logger.warning("AI reco disabled: %s", e)
                self.use_ai_reco = False

    # ---------- Public entrypoint ----------

    def create_visualization(
        self,
        df: pd.DataFrame,
        user_query: str = "",
        force_chart_type: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Decide chart -> build chart -> save image -> return 'visualization' dict for the API response.

        Notes:
        - Honors explicit chart keywords in user_query (EN + ZH) if present.
        - If force_chart_type is provided, it overrides both query intent and AI recommendation.
        """
        try:
            if df is None:
                raise ValueError("DataFrame is None")
            df = _coerce_dtypes_for_viz(df)

            analysis = self._analyze(df)

            # 1) Explicit chart intent from user query (if any)
            user_chart = self._extract_chart_intent(user_query)

            # 2) Determine chart type: forced > explicit > AI > fallback
            if force_chart_type:
                chart_type = force_chart_type
                reasoning = f"Forced chart type '{force_chart_type}'."
                alts: List[str] = []
                insights_hint: List[str] = []
            else:
                if user_chart:
                    chart_type = user_chart
                    reasoning = f"User requested chart type '{user_chart}'."
                    alts = []
                    insights_hint = []
                else:
                    reco = self._recommend_chart(df, user_query, analysis)
                    chart_type = reco["recommended_chart"]
                    reasoning = reco.get("reasoning") or "Recommended by system"
                    alts = reco.get("alternative_charts", [])
                    insights_hint = reco.get("insights_to_highlight", [])

            if not title:
                title = self._default_title(chart_type, user_query, df)

            fig = self._build_chart(df, chart_type, title)
            image_url, filename = self._save_figure(fig, chart_type)

            insights = self._generate_insights(df, chart_type, analysis, insights_hint)

            return {
                "enabled": True,
                "type": chart_type,
                "title": title,
                "url": image_url,               # UI uses this
                "filename": filename,           # optional
                "reasoning": reasoning,
                "insights": insights,
                "alternatives": alts,
                "data_summary": {
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "column_names": list(map(str, df.columns)),
                },
            }

        except Exception as e:
            logger.exception("Visualization failed: %s", e)
            return {
                "enabled": False,
                "reason": f"Visualization error: {e}",
            }

    # ---------- Analysis & Recommendation ----------

    def _analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or df.empty:
            return {
                "empty": True,
                "shape": (0, 0),
                "numeric_cols": [],
                "categorical_cols": [],
                "datetime_cols": [],
                "columns": [],
                "sample": [],
            }

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

        sample = df.head(2).to_dict("records")

        return {
            "empty": False,
            "shape": df.shape,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "columns": list(df.columns),
            "sample": sample,
        }

    def _recommend_chart(self, df: pd.DataFrame, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        # Try AI reco
        if self.use_ai_reco and not analysis.get("empty"):
            try:
                msgs = self._reco_prompt.format_messages(
                    shape=f"{analysis['shape'][0]}x{analysis['shape'][1]}",
                    numeric=analysis["numeric_cols"],
                    categorical=analysis["categorical_cols"],
                    datetime=analysis["datetime_cols"],
                    query=query,
                    columns=analysis["columns"],
                    sample=analysis["sample"],
                )
                res = self.llm.invoke(msgs)
                txt = getattr(res, "content", "") or ""
                reco = json.loads(txt)
                rc = reco.get("recommended_chart")
                if rc:
                    return {
                        "recommended_chart": rc,
                        "reasoning": reco.get("reasoning", ""),
                        "alternative_charts": reco.get("alternative_charts", []),
                        "insights_to_highlight": reco.get("insights_to_highlight", []),
                    }
            except Exception as e:
                logger.warning("AI reco failed: %s (falling back)", e)

        # Fallback rules (fast, deterministic)
        return self._fallback_recommendation(analysis, query)

    def _fallback_recommendation(self, a: Dict[str, Any], query: str) -> Dict[str, Any]:
        if a.get("empty"):
            return {
                "recommended_chart": "table",
                "reasoning": "No data to visualize",
                "alternative_charts": [],
                "insights_to_highlight": [],
            }

        num = len(a["numeric_cols"])
        cat = len(a["categorical_cols"])
        dt  = len(a["datetime_cols"])

        q = (query or "").lower()
        if any(k in q for k in ["trend", "over time", "time series", "timeseries"]) or dt >= 1:
            return {"recommended_chart": "line_chart", "reasoning": "Time trend", "alternative_charts": ["bar_chart"], "insights_to_highlight": []}
        if "distribution" in q or "histogram" in q:
            return {"recommended_chart": "histogram", "reasoning": "Distribution requested", "alternative_charts": ["box_plot"], "insights_to_highlight": []}
        if "correlation" in q or num >= 2:
            return {"recommended_chart": "scatter_plot", "reasoning": "Two or more numeric variables", "alternative_charts": ["heatmap","bar_chart"], "insights_to_highlight": []}
        if num == 1 and cat >= 1:
            return {"recommended_chart": "bar_chart", "reasoning": "Numeric by category", "alternative_charts": ["pie_chart","box_plot"], "insights_to_highlight": []}
        if cat >= 1:
            return {"recommended_chart": "bar_chart", "reasoning": "Categorical frequency", "alternative_charts": ["pie_chart","table"], "insights_to_highlight": []}
        if num == 1:
            return {"recommended_chart": "histogram", "reasoning": "Single numeric distribution", "alternative_charts": ["box_plot"], "insights_to_highlight": []}
        return {"recommended_chart": "table", "reasoning": "Fallback to table", "alternative_charts": [], "insights_to_highlight": []}

    # ---------- Query intent parsing (explicit user request) ----------

    def _extract_chart_intent(self, query: str) -> Optional[str]:
        """
        Map common EN + ZH chart keywords to our internal chart types.
        Returns one of:
            line_chart, bar_chart, pie_chart, scatter_plot, histogram, heatmap, box_plot, table, area_chart
        """
        if not query:
            return None
        q = query.strip().lower()

        # English
        mapping = {
            "line": "line_chart",
            "line chart": "line_chart",
            "timeseries": "line_chart",
            "time series": "line_chart",
            "trend": "line_chart",
            "bar": "bar_chart",
            "bar chart": "bar_chart",
            "column": "bar_chart",
            "pie": "pie_chart",
            "donut": "pie_chart",
            "scatter": "scatter_plot",
            "scatterplot": "scatter_plot",
            "hist": "histogram",
            "histogram": "histogram",
            "heatmap": "heatmap",
            "corr": "heatmap",
            "box": "box_plot",
            "boxplot": "box_plot",
            "table": "table",
            "grid": "table",
            "area": "area_chart",
            "area chart": "area_chart",
        }

        # Chinese (Traditional)
        zh_map = {
            "折線": "line_chart",
            "趨勢": "line_chart",
            "長條": "bar_chart",
            "柱狀": "bar_chart",
            "圓餅": "pie_chart",
            "散點": "scatter_plot",
            "直方": "histogram",
            "熱圖": "heatmap",
            "箱型": "box_plot",
            "表格": "table",
            "面積": "area_chart",
        }

        for k, v in mapping.items():
            if k in q:
                return v
        for k, v in zh_map.items():
            if k in q:
                return v
        return None

    def _default_title(self, chart_type: str, query: str, df: pd.DataFrame) -> str:
        if query:
            return query[:120]
        return chart_type.replace("_", " ").title()

    # ---------- Chart Builders (deterministic) ----------

    def _build_chart(self, df: pd.DataFrame, chart_type: str, title: str) -> go.Figure:
        chart_type = (chart_type or "").lower()

        if df is None or df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False, font=dict(size=18, color="gray"))
            fig.update_layout(template=self.theme, width=800, height=500, title="No Data")
            return fig

        # Auto-pick axes
        cols = list(df.columns)
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        datetime_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
        categorical = [c for c in cols if c not in numeric and c not in datetime_cols]

        x = datetime_cols[0] if datetime_cols else (categorical[0] if categorical else (cols[0] if cols else None))
        y = (numeric[0] if numeric else (cols[1] if len(cols) > 1 else None))

        # Helper: choose a series/color column if cardinality is reasonable
        color_col = None
        if categorical:
            # prefer the first categorical with <= 12 unique values
            for cc in categorical:
                try:
                    if df[cc].nunique(dropna=False) <= 12:
                        color_col = cc
                        break
                except Exception:
                    continue

        if chart_type == "area_chart" and x is not None and y is not None:
            if color_col:
                fig = px.area(df, x=x, y=y, color=color_col, title=title, template=self.theme, groupnorm=None)
            else:
                fig = px.area(df, x=x, y=y, title=title, template=self.theme)

        elif chart_type == "line_chart" and x is not None and y is not None:
            if color_col:
                fig = px.line(df, x=x, y=y, color=color_col, title=title, template=self.theme)
            else:
                fig = px.line(df, x=x, y=y, title=title, template=self.theme)

        elif chart_type == "bar_chart" and x is not None:
            if y is None or not pd.api.types.is_numeric_dtype(df[y]):
                counts = df[x].value_counts(dropna=False).reset_index()
                counts.columns = [str(x), "Count"]
                fig = px.bar(counts, x=str(x), y="Count", title=title, template=self.theme)
            else:
                # If we have a color grouping and it's not the x itself, use it for clustered bars
                if color_col and str(color_col) != str(x):
                    fig = px.bar(df, x=x, y=y, color=color_col, barmode="group", title=title, template=self.theme)
                else:
                    fig = px.bar(df, x=x, y=y, title=title, template=self.theme)

        elif chart_type == "pie_chart":
            xx = x if x is not None else (cols[0] if cols else None)
            if xx is None:
                fig = px.pie(title=title)
            elif y is None or not pd.api.types.is_numeric_dtype(df[y]):
                vc = df[xx].value_counts(dropna=False).reset_index()
                vc.columns = [str(xx), "Count"]
                fig = px.pie(vc, names=str(xx), values="Count", title=title)
            else:
                fig = px.pie(df, names=xx, values=y, title=title)

        elif chart_type == "scatter_plot":
            if len(numeric) >= 2:
                fig = px.scatter(df, x=numeric[0], y=numeric[1], color=color_col, title=title, template=self.theme)
            elif x is not None and y is not None:
                fig = px.scatter(df, x=x, y=y, color=color_col, title=title, template=self.theme)
            else:
                fig = px.scatter(title=title, template=self.theme)

        elif chart_type == "histogram":
            xx = numeric[0] if numeric else (cols[0] if cols else None)
            if xx:
                fig = px.histogram(df, x=xx, color=color_col, title=title, template=self.theme)
            else:
                fig = px.histogram(title=title, template=self.theme)

        elif chart_type == "heatmap" and len(numeric) >= 2:
            corr = df[numeric].corr(numeric_only=True)
            fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Blues"))
            fig.update_layout(title=title, template=self.theme)

        elif chart_type == "box_plot":
            yy = numeric[0] if numeric else (cols[0] if cols else None)
            fig = px.box(df, y=yy, x=(categorical[0] if categorical else None), title=title, template=self.theme)

        else:
            # Table fallback
            header_vals = list(map(str, df.columns))
            cell_vals = [df[c] for c in df.columns]
            fig = go.Figure(data=[go.Table(
                header=dict(values=header_vals, fill_color="#e2e8f0", align="left"),
                cells=dict(values=cell_vals, fill_color="#ffffff", align="left"),
            )])
            fig.update_layout(title=title, template=self.theme, width=800, height=500)

        # Make sure dates are sorted for nicer trends
        try:
            if x in df.columns and pd.api.types.is_datetime64_any_dtype(df[x]):
                fig.update_xaxes(type="date")
        except Exception:
            pass

        fig.update_layout(width=800, height=500)
        return fig

    # ---------- Save & Insights ----------

    def _save_figure(self, fig: go.Figure, chart_type: str) -> Tuple[str, str]:
        """
        Save to filesystem (LOCAL_SAVE_DIR) and return a web URL under /{LOCAL_SAVE_DIR}/.
        Requires kaleido installed for static export.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{chart_type}_{ts}_{uuid.uuid4().hex[:8]}.png"

        # Write into the same directory the server exposes (LOCAL_SAVE_DIR)
        fpath = self._local_dir / fname

        try:
            fig.write_image(str(fpath), width=800, height=500)  # needs kaleido
        except Exception as e:
            logger.warning("Static export failed (%s). Retrying write_image()", e)
            fig.write_image(str(fpath), width=800, height=500)

        # Build a URL that matches the served route (seen in your logs)
        # e.g., LOCAL_SAVE_DIR="charts/images" -> "/charts/images/<file>"
        web_prefix = "/" + str(self._local_dir).replace("\\", "/").strip("/")
        web_url = f"{web_prefix}/{fname}"

        return web_url, fname

    def _generate_insights(
        self,
        df: pd.DataFrame,
        chart_type: str,
        analysis: Dict[str, Any],
        hints: Optional[List[str]] = None,
    ) -> List[str]:
        insights: List[str] = []
        if df is None or df.empty:
            insights.append("No data available for analysis.")
            return insights

        rows, cols = analysis.get("shape", (len(df), len(df.columns)))
        insights.append(f"Dataset has {rows} rows and {cols} columns.")

        if chart_type == "line_chart":
            insights.append("Line chart highlights trends across the x-axis progression.")
        elif chart_type == "bar_chart":
            insights.append("Bar chart compares values across categories.")
        elif chart_type == "pie_chart":
            insights.append("Pie chart shows proportional composition.")
        elif chart_type == "scatter_plot":
            insights.append("Scatter plot can reveal correlations and outliers.")
        elif chart_type == "histogram":
            insights.append("Histogram shows the distribution of a numeric variable.")
        elif chart_type == "heatmap":
            insights.append("Heatmap visualizes relationships in a matrix (e.g., correlations).")
        elif chart_type == "box_plot":
            insights.append("Box plot summarizes distribution and outliers.")
        elif chart_type == "area_chart":
            insights.append("Area chart emphasizes cumulative totals over time.")

        if hints:
            insights.extend([h for h in hints if isinstance(h, str)][:3])

        nulls = df.isnull().sum()
        if (nulls > 0).any():
            cols_with_nulls = [c for c, v in nulls.items() if v > 0]
            insights.append(f"Missing values present in: {', '.join(map(str, cols_with_nulls))}")

        return insights
