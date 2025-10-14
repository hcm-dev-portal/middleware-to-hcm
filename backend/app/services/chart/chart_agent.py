import os
import json
import uuid
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pydantic import BaseModel, Field, ValidationError, constr
from enum import Enum

# Reuse your existing VisualizationService for insights + saving
from .visualization_service import VisualizationService  # adjust import to your layout

logger = logging.getLogger(__name__)

# -----------------------------
# LLM (optional) – tiny wrapper
# -----------------------------
_OPENAI_OK = False
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_KEY:
        _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False


# -----------------------------
# Spec Schema (safe + validated)
# -----------------------------
class ChartType(str, Enum):
    line = "line"
    area = "area"
    bar = "bar"
    stacked_bar = "stacked_bar"
    grouped_bar = "grouped_bar"
    pie = "pie"
    scatter = "scatter"
    histogram = "histogram"
    heatmap = "heatmap"
    box = "box"
    table = "table"

class Agg(str, Enum):
    sum = "sum"
    avg = "avg"
    mean = "mean"
    count = "count"
    min = "min"
    max = "max"
    none = "none"

class Orientation(str, Enum):
    v = "v"
    h = "h"

class SortDir(str, Enum):
    asc = "asc"
    desc = "desc"

class ChartSpec(BaseModel):
    chart_type: ChartType
    title: Optional[constr(strip_whitespace=True, min_length=1)] = None # type: ignore
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    # If aggregation is needed (e.g., sum per category)
    aggregate: Agg = Agg.none
    group_by: List[str] = Field(default_factory=list)  # columns to group by before plotting
    orientation: Optional[Orientation] = None
    bins: Optional[int] = Field(default=None, ge=2, le=200)  # for histogram
    sort_by: Optional[str] = None
    sort_dir: Optional[SortDir] = None
    top_n: Optional[int] = Field(default=None, ge=1, le=100)
    facet_row: Optional[str] = None
    facet_col: Optional[str] = None
    trendline: Optional[bool] = False

class ChartReco(BaseModel):
    recommended_spec: ChartSpec
    reasoning: Optional[str] = ""
    alternative_charts: List[str] = Field(default_factory=list)
    insights_to_highlight: List[str] = Field(default_factory=list)


# -----------------------------
# Agent
# -----------------------------
@dataclass
class ChartVisualizationAgentConfig:
    model: str = "gpt-4o-mini"  # small & cheap; change as you like
    temperature: float = 0.0
    max_cols_in_schema: int = 30
    max_uniques_sample: int = 12
    theme: str = "plotly_white"  # default theme for plots
    use_ai: bool = True  # gate to disable AI globally


class ChartVisualizationAgent:
    """
    AI-chart agent that:
      1) extracts a safe schema from df (no raw data beyond tiny samples),
      2) asks LLM for a JSON-only chart spec,
      3) validates spec via Pydantic,
      4) renders deterministically (Plotly),
      5) saves & returns the same 'visualization' dict as VisualizationService.
    """

    def __init__(self, viz_service: Optional[VisualizationService] = None, cfg: Optional[ChartVisualizationAgentConfig] = None):
        self.cfg = cfg or ChartVisualizationAgentConfig()
        self.viz = viz_service or VisualizationService(theme=self.cfg.theme)

        self.use_ai = self.cfg.use_ai and _OPENAI_OK
        if self.use_ai:
            try:
                self.llm = ChatOpenAI(model=self.cfg.model, temperature=self.cfg.temperature)
                self.prompt = ChatPromptTemplate.from_template(self._prompt_template())
            except Exception as e:
                logger.warning("Chart agent LLM disabled: %s", e)
                self.use_ai = False

    # -------------------------
    # Public entrypoint
    # -------------------------
    def generate_chart(
        self,
        df: pd.DataFrame,
        user_query: str,
        sql_meta: Optional[Dict[str, Any]] = None,
        force_chart_type: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns the same response shape as VisualizationService.create_visualization().
        """
        # Empty DF? Defer to existing deterministic service for a friendly response.
        if df is None or df.empty:
            return self.viz.create_visualization(df, user_query=user_query, force_chart_type=force_chart_type, title=title)

        # If caller forces chart type, skip AI, delegate to viz service
        if force_chart_type:
            return self.viz.create_visualization(df, user_query=user_query, force_chart_type=force_chart_type, title=title)

        # 1) Build safe schema/context
        schema = self._schema_summary(df)

        # 2) Ask LLM for spec (or fallback)
        spec, reasoning, alts, insight_hints = self._get_spec(schema, user_query)

        # 3) Render deterministically from spec
        fig = self._build_from_spec(df, spec)

        # 4) Save image and craft response (reuse VisualizationService helpers)
        image_url, filename = self.viz._save_figure(fig, spec.chart_type.value)
        analysis = self.viz._analyze(df)  # reuse for consistent insights
        insights = self.viz._generate_insights(df, self._to_viz_chart_type(spec.chart_type), analysis, hints=insight_hints)

        final_title = title or spec.title or (user_query[:120] if user_query else spec.chart_type.value.title())
        return {
            "enabled": True,
            "type": self._to_viz_chart_type(spec.chart_type),
            "title": final_title,
            "url": image_url,
            "filename": filename,
            "reasoning": reasoning or "AI-generated chart spec (validated)",
            "alternatives": alts,
            "insights": insights,
            "spec": spec.model_dump(),  # helpful for audit/debug
            "data_summary": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "column_names": list(map(str, df.columns)),
            },
        }

    # -------------------------
    # LLM prompting
    # -------------------------
    def _prompt_template(self) -> str:
        # Constrain to JSON; give explicit schema & rules to avoid hallucinations
        return (
            "You are a senior data-visualization agent. Your task is to produce a JSON chart spec only.\n"
            "Follow STRICTLY:\n"
            "1) Respond with **JSON only**, no prose.\n"
            "2) Use this JSON schema (keys are required unless specified as Optional):\n"
            "{\n"
            '  "recommended_spec": {\n'
            '    "chart_type": "line|area|bar|stacked_bar|grouped_bar|pie|scatter|histogram|heatmap|box|table",\n'
            '    "title": "<Optional short title>",\n'
            '    "x": "<Optional column name>",\n'
            '    "y": "<Optional column name>",\n'
            '    "color": "<Optional column name>",\n'
            '    "size": "<Optional numeric column>",\n'
            '    "aggregate": "sum|avg|mean|count|min|max|none",\n'
            '    "group_by": ["<Optional column>", "..."],\n'
            '    "orientation": "v|h|<omit>",\n'
            '    "bins": <Optional integer 2..200>,\n'
            '    "sort_by": "<Optional column>",\n'
            '    "sort_dir": "asc|desc|<omit>",\n'
            '    "top_n": <Optional integer 1..100>,\n'
            '    "facet_row": "<Optional column>",\n'
            '    "facet_col": "<Optional column>",\n'
            '    "trendline": <Optional boolean>\n'
            "  },\n"
            '  "reasoning": "<One sentence max>",\n'
            '  "alternative_charts": ["<Optional chart_type>", "..."],\n'
            '  "insights_to_highlight": ["<bullets>", "..."]\n'
            "}\n\n"
            "Context:\n"
            "User query: {user_query}\n"
            "Table schema (column -> type): {schema_columns}\n"
            "Column samples (truncated): {schema_samples}\n"
            "Guidelines:\n"
            "- If a time column exists and the query implies trend, prefer line/area.\n"
            "- If categorical + single numeric, prefer bar; consider stacked/grouped if a second category exists (use 'color').\n"
            "- Use 'aggregate' and 'group_by' if the chart requires aggregation (e.g., sum by category).\n"
            "- Use 'top_n' for long-tail categories; sort then keep top_n.\n"
            "- For histogram, set a reasonable 'bins'.\n"
            "- For heatmap, it means correlation heatmap across numeric columns.\n"
            "- For pie, ensure categories are few; otherwise switch to bar.\n"
            "- Prefer concise titles.\n"
            "- NEVER invent columns; only use provided column names.\n"
            "- Output valid JSON only."
        )

    def _get_spec(self, schema: Dict[str, Any], user_query: str) -> Tuple[ChartSpec, str, List[str], List[str]]:
        # Try AI
        if self.use_ai:
            try:
                messages = self.prompt.format_messages(
                    user_query=user_query,
                    schema_columns=json.dumps(schema["columns"], ensure_ascii=False),
                    schema_samples=json.dumps(schema["samples"], ensure_ascii=False),
                )
                result = self.llm.invoke(messages)
                raw = getattr(result, "content", "") or "{}"
                data = json.loads(raw)
                reco = ChartReco.model_validate(data)

                # Minor normalization (map 'mean' to 'avg' for consistency)
                if reco.recommended_spec.aggregate == Agg.mean:
                    reco.recommended_spec.aggregate = Agg.avg

                return (
                    reco.recommended_spec,
                    reco.reasoning or "",
                    reco.alternative_charts or [],
                    reco.insights_to_highlight or [],
                )
            except Exception as e:
                logger.warning("Chart LLM spec failed (%s). Falling back.", e)

        # Fallback: convert your VisualizationService recommendation to a spec
        # Build a tiny temp VizService to reuse its recommendation rules
        rec = self.viz._fallback_recommendation(self.viz._analyze(schema["__df__"]), user_query)
        vt = rec["recommended_chart"]  # e.g., "bar_chart"
        spec = self._fallback_to_spec(vt, schema)
        return (spec, rec.get("reasoning", "fallback"), rec.get("alternative_charts", []), [])

    # -------------------------
    # Schema extraction (safe)
    # -------------------------
    def _schema_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        # capture dtypes
        cols = []
        samples = {}
        for i, c in enumerate(df.columns[: self.cfg.max_cols_in_schema]):
            dtype = "numeric" if pd.api.types.is_numeric_dtype(df[c]) else (
                "datetime" if pd.api.types.is_datetime64_any_dtype(df[c]) else "categorical"
            )
            cols.append({"name": str(c), "type": dtype})
            # tiny sample for context; no large data exposure
            try:
                uniques = df[c].dropna().unique()
                if len(uniques) > self.cfg.max_uniques_sample:
                    uniques = uniques[: self.cfg.max_uniques_sample]
                samples[str(c)] = list(map(self._to_prim, uniques))
            except Exception:
                samples[str(c)] = []

        return {"columns": cols, "samples": samples, "__df__": df}  # keep df for fallback only

    @staticmethod
    def _to_prim(v: Any) -> Any:
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return str(v)

    # -------------------------
    # Spec → Plotly rendering
    # -------------------------
    def _build_from_spec(self, df: pd.DataFrame, spec: ChartSpec) -> go.Figure:
        chart = spec.chart_type
        dfx = df.copy()

        # Optional aggregation/grouping
        if spec.aggregate != Agg.none and (spec.y or spec.size or spec.aggregate == Agg.count):
            group_cols = [c for c in [spec.x, spec.color, spec.facet_row, spec.facet_col] if c]
            if len(group_cols) == 0 and spec.aggregate != Agg.count:
                # if no group_by provided but aggregation requested, treat x as group if present
                if spec.x:
                    group_cols = [spec.x]
            if len(group_cols) > 0:
                if spec.aggregate == Agg.count:
                    dfx = dfx.groupby(group_cols, dropna=False).size().reset_index(name="__value__")
                    y_col = "__value__"
                else:
                    target = spec.y or spec.size
                    fn = {"sum": "sum", "avg": "mean", "mean": "mean", "min": "min", "max": "max"}[spec.aggregate.value]
                    dfx = dfx.groupby(group_cols, dropna=False)[target].agg(fn).reset_index(name="__value__")
                    y_col = "__value__"
                # If y not set, point it to aggregation result
                if not spec.y:
                    spec.y = y_col

        # Sorting & Top-N
        if spec.sort_by and spec.sort_by in dfx.columns:
            ascending = (spec.sort_dir or SortDir.asc) == SortDir.asc
            dfx = dfx.sort_values(by=spec.sort_by, ascending=ascending)
        if spec.top_n and spec.x and spec.x in dfx.columns:
            # keep top_n categories by y (or counts)
            y_col = spec.y or (spec.x if spec.chart_type == ChartType.pie else None)
            if y_col and y_col in dfx.columns:
                dfx = dfx.sort_values(by=y_col, ascending=False).head(spec.top_n)

        # Figure builders
        title = spec.title or chart.value.title()
        theme = self.cfg.theme

        if chart == ChartType.line:
            fig = px.line(dfx, x=spec.x, y=spec.y, color=spec.color, title=title, template=theme)
        elif chart == ChartType.area:
            fig = px.area(dfx, x=spec.x, y=spec.y, color=spec.color, title=title, template=theme)
        elif chart in (ChartType.bar, ChartType.grouped_bar, ChartType.stacked_bar):
            barmode = "relative" if chart == ChartType.stacked_bar else ("group" if chart == ChartType.grouped_bar else None)
            fig = px.bar(
                dfx,
                x=spec.x if (spec.orientation or Orientation.v) == Orientation.v else spec.y,
                y=spec.y if (spec.orientation or Orientation.v) == Orientation.v else spec.x,
                color=spec.color,
                title=title,
                template=theme,
            )
            if barmode:
                fig.update_layout(barmode=barmode)
        elif chart == ChartType.pie:
            names = spec.x or spec.color
            values = spec.y
            if not names:
                # default to first non-numeric column
                names = next((c for c in dfx.columns if not pd.api.types.is_numeric_dtype(dfx[c])), dfx.columns[0])
            if not values:
                # if no values, use frequency
                counts = dfx[names].value_counts().reset_index()
                counts.columns = [names, "__value__"]
                fig = px.pie(counts, names=names, values="__value__", title=title)
            else:
                fig = px.pie(dfx, names=names, values=values, color=spec.color, title=title)
        elif chart == ChartType.scatter:
            fig = px.scatter(dfx, x=spec.x, y=spec.y, color=spec.color, size=spec.size, title=title, template=theme, trendline=("ols" if spec.trendline else None))
        elif chart == ChartType.histogram:
            fig = px.histogram(dfx, x=spec.x or spec.y, nbins=spec.bins, title=title, template=theme, color=spec.color)
        elif chart == ChartType.heatmap:
            # correlation across numeric columns
            numeric_cols = [c for c in dfx.columns if pd.api.types.is_numeric_dtype(dfx[c])]
            corr = dfx[numeric_cols].corr(numeric_only=True)
            fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Blues"))
            fig.update_layout(title=title, template=theme)
        elif chart == ChartType.box:
            fig = px.box(dfx, x=spec.x, y=spec.y, color=spec.color, title=title, template=theme)
        else:
            # table
            header_vals = list(map(str, dfx.columns))
            cell_vals = [dfx[c] for c in dfx.columns]
            fig = go.Figure(data=[go.Table(
                header=dict(values=header_vals, fill_color="#e2e8f0", align="left"),
                cells=dict(values=cell_vals, fill_color="#ffffff", align="left"),
            )])
            fig.update_layout(title=title, template=theme, width=800, height=500)

        # common layout
        fig.update_layout(width=800, height=500)
        return fig

    # -------------------------
    # Helpers
    # -------------------------
    def _fallback_to_spec(self, vt: str, schema: Dict[str, Any]) -> ChartSpec:
        # maps your viz service chart ids to a safe default spec
        m = {
            "line_chart": ChartType.line,
            "bar_chart": ChartType.bar,
            "pie_chart": ChartType.pie,
            "scatter_plot": ChartType.scatter,
            "histogram": ChartType.histogram,
            "heatmap": ChartType.heatmap,
            "box_plot": ChartType.box,
            "table": ChartType.table,
        }
        ct = m.get(vt, ChartType.table)
        # Try picking reasonable x/y from schema for a minimal spec
        cols = [c["name"] for c in schema["columns"]]
        numerics = [c["name"] for c in schema["columns"] if c["type"] == "numeric"]
        dates = [c["name"] for c in schema["columns"] if c["type"] == "datetime"]
        cats = [c["name"] for c in schema["columns"] if c["type"] == "categorical"]

        x = (dates[0] if dates else (cats[0] if cats else (cols[0] if cols else None)))
        y = (numerics[0] if numerics else (cols[1] if len(cols) > 1 else None))

        return ChartSpec(chart_type=ct, x=x, y=y)

    def _to_viz_chart_type(self, ct: ChartType) -> str:
        # normalize to your VisualizationService naming to keep UI consistent
        return {
            ChartType.line: "line_chart",
            ChartType.area: "area_chart",
            ChartType.bar: "bar_chart",
            ChartType.stacked_bar: "bar_chart",     # your UI shows 'bar_chart'
            ChartType.grouped_bar: "bar_chart",
            ChartType.pie: "pie_chart",
            ChartType.scatter: "scatter_plot",
            ChartType.histogram: "histogram",
            ChartType.heatmap: "heatmap",
            ChartType.box: "box_plot",
            ChartType.table: "table",
        }[ct]
