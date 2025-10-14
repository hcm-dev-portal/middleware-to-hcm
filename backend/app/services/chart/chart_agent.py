import os
import json
import uuid
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
    title: Optional[constr(strip_whitespace=True, min_length=1)] = None  # type: ignore
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    aggregate: Agg = Agg.none
    group_by: List[str] = Field(default_factory=list)
    orientation: Optional[Orientation] = None
    bins: Optional[int] = Field(default=None, ge=2, le=200)
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
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_cols_in_schema: int = 30
    max_uniques_sample: int = 12
    theme: str = "plotly_white"
    use_ai: bool = True
    # Safety thresholds
    max_categories: int = 30
    default_hist_bins: int = 24
    topn_for_long_tail: int = 12
    # Time-series controls
    max_timeseries_points: int = 400   # resample/aggregate to at most this many points
    default_target_bins: int = 120     # for choosing D/W/M/Q/Y grouping


class ChartVisualizationAgent:
    """
    AI-chart agent with strong safety rails:
      1) builds a safe schema,
      2) asks LLM for JSON spec (optional),
      3) validates & auto-repairs spec against df (robust date handling),
      4) renders deterministically (Plotly),
      5) saves PNG via VisualizationService.
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
        # Empty → defer to viz (friendly “No data” figure)
        if df is None or df.empty:
            return self.viz.create_visualization(df, user_query=user_query, force_chart_type=force_chart_type, title=title)

        # Clean & normalize the dataframe up front (handles dates, tz, dtypes, dup cols)
        df_clean, clean_warnings = self._clean_dataframe(df)

        # Build schema AFTER cleaning so types reflect reality
        schema = self._schema_summary(df_clean)
        spec_warnings: List[str] = clean_warnings[:]  # carry over

        # Forced chart type → skip AI; still validate below
        if force_chart_type:
            base_spec = self._fallback_to_spec(self._map_to_viz(force_chart_type), schema)
            spec = self._validate_and_repair_spec(df_clean, base_spec, spec_warnings, user_query)
            ai_reasoning, ai_alts, ai_insights = "", [], []
        else:
            # Get spec from LLM or rules, then validate/repair
            spec_ai, reasoning, alts, insight_hints = self._get_spec(schema, user_query)
            spec = self._validate_and_repair_spec(df_clean, spec_ai, spec_warnings, user_query)
            ai_reasoning, ai_alts, ai_insights = reasoning, alts, insight_hints

        # If validation downgraded to table, render table with reason
        if spec.chart_type == ChartType.table:
            fig = self._build_from_spec(df_clean, spec)
            image_url, filename = self.viz._save_figure(fig, "table")
            analysis = self.viz._analyze(df_clean)
            insights = self.viz._generate_insights(df_clean, "table", analysis, hints=[])
            final_title = title or spec.title or (user_query[:120] if user_query else "Table")
            reason = " ; ".join(spec_warnings) or "Incompatible data for requested chart"
            return {
                "enabled": True,
                "type": "table",
                "title": final_title,
                "url": image_url,
                "filename": filename,
                "reasoning": reason,
                "alternatives": [],
                "insights": insights,
                "spec": spec.model_dump(),
                "data_summary": {"rows": int(df_clean.shape[0]), "columns": int(df_clean.shape[1]), "column_names": list(map(str, df_clean.columns))},
            }

        # Render deterministically
        fig = self._build_from_spec(df_clean, spec)

        # Save + insights
        image_url, filename = self.viz._save_figure(fig, spec.chart_type.value)
        analysis = self.viz._analyze(df_clean)
        insights = self.viz._generate_insights(df_clean, self._to_viz_chart_type(spec.chart_type), analysis, hints=(ai_insights if not force_chart_type else []))

        # Title + reasoning
        final_title = title or spec.title or (user_query[:120] if user_query else spec.chart_type.value.title())
        reasoning = []
        if not force_chart_type and ai_reasoning:
            reasoning.append(ai_reasoning)
        if spec_warnings:
            reasoning.append("Auto-repairs: " + "; ".join(spec_warnings))

        return {
            "enabled": True,
            "type": self._to_viz_chart_type(spec.chart_type),
            "title": final_title,
            "url": image_url,
            "filename": filename,
            "reasoning": " | ".join(reasoning) or "Validated chart spec",
            "alternatives": (ai_alts if not force_chart_type else []),
            "insights": insights,
            "spec": spec.model_dump(),
            "data_summary": {
                "rows": int(df_clean.shape[0]),
                "columns": int(df_clean.shape[1]),
                "column_names": list(map(str, df_clean.columns)),
            },
        }

    # -------------------------
    # LLM prompting
    # -------------------------
    def _prompt_template(self) -> str:
        return (
            "You are a senior data-visualization agent. Produce a JSON chart spec only.\n"
            "Rules:\n"
            "1) Respond with JSON only.\n"
            "2) JSON schema:\n"
            "{"
            '  "recommended_spec": {'
            '    "chart_type": "line|area|bar|stacked_bar|grouped_bar|pie|scatter|histogram|heatmap|box|table",'
            '    "title": "<Optional short title>",'
            '    "x": "<Optional column name>",'
            '    "y": "<Optional column name>",'
            '    "color": "<Optional column name>",'
            '    "size": "<Optional numeric column>",'
            '    "aggregate": "sum|avg|mean|count|min|max|none",'
            '    "group_by": ["<Optional column>", "..."],'
            '    "orientation": "v|h|<omit>",'
            '    "bins": <Optional integer 2..200>,'
            '    "sort_by": "<Optional column>",'
            '    "sort_dir": "asc|desc|<omit>",'
            '    "top_n": <Optional integer 1..100>,'
            '    "facet_row": "<Optional column>",'
            '    "facet_col": "<Optional column>",'
            '    "trendline": <Optional boolean>'
            "  },"
            '  "reasoning": "<One sentence max>",'
            '  "alternative_charts": ["<Optional chart_type>", "..."],'
            '  "insights_to_highlight": ["<bullets>", "..."]'
            "}\n"
            "Context:\n"
            "User query: {user_query}\n"
            "Table schema: {schema_columns}\n"
            "Column samples: {schema_samples}\n"
            "Guidelines: prefer time-series for trends; aggregate when needed; NEVER invent columns.\n"
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

        # Fallback: reuse viz rules
        rec = self.viz._fallback_recommendation(self.viz._analyze(schema["__df__"]), user_query)
        vt = rec["recommended_chart"]
        spec = self._fallback_to_spec(vt, schema)
        return (spec, rec.get("reasoning", "fallback"), rec.get("alternative_charts", []), [])

    # -------------------------
    # Schema extraction (safe)
    # -------------------------
    def _schema_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        cols = []
        samples = {}
        for c in df.columns[: self.cfg.max_cols_in_schema]:
            dtype = (
                "numeric" if pd.api.types.is_numeric_dtype(df[c]) else
                ("datetime" if pd.api.types.is_datetime64_any_dtype(df[c]) else "categorical")
            )
            cols.append({"name": str(c), "type": dtype})
            try:
                uniques = df[c].dropna().unique()
                if len(uniques) > self.cfg.max_uniques_sample:
                    uniques = uniques[: self.cfg.max_uniques_sample]
                samples[str(c)] = [self._to_prim(u) for u in uniques]
            except Exception:
                samples[str(c)] = []
        return {"columns": cols, "samples": samples, "__df__": df}

    @staticmethod
    def _to_prim(v: Any) -> Any:
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        return v if isinstance(v, (str, int, float, bool)) or v is None else str(v)

    # -------------------------
    # DataFrame cleaning (robust to dates, tz, dtypes, dups)
    # -------------------------
    def _clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        warnings: List[str] = []
        dfx = df.copy()

        # 0) Normalize column labels to strings, deduplicate if needed
        new_cols = []
        seen = {}
        for c in dfx.columns:
            name = str(c)
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[str(c)]}"
                warnings.append(f"Duplicate column renamed to {name}")
            else:
                seen[name] = 0
            new_cols.append(name)
        dfx.columns = new_cols

        # 1) Heuristic coercions by name
        for col in dfx.columns:
            low = col.lower()
            series = dfx[col]

            # timezone-safe datetime coerce
            if any(k in low for k in ["date", "day", "time", "workdate", "startdate", "enddate"]):
                dfx[col] = self._coerce_datetime(series)
                if pd.api.types.is_datetime64_any_dtype(dfx[col]) and series.dtype != dfx[col].dtype:
                    warnings.append(f"Coerced {col} to datetime")

            # numeric-ish strings (but avoid ID-like columns)
            elif any(k in low for k in ["amount", "qty", "quantity", "hours", "value", "count", "rate", "score", "minutes"]):
                coerced = pd.to_numeric(series, errors="coerce")
                if coerced.notna().mean() >= 0.6 and not pd.api.types.is_numeric_dtype(series):
                    dfx[col] = coerced
                    warnings.append(f"Coerced {col} to numeric")

        return dfx, warnings

    @staticmethod
    def _coerce_datetime(s: pd.Series) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(s):
            # unify to naive (strip tz) for Plotly consistency
            try:
                if hasattr(s.dt, "tz"):
                    return s.dt.tz_convert(None) if s.dt.tz is not None else s
            except Exception:
                pass
            return s
        try:
            out = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=True)
            try:
                out = out.dt.tz_convert(None)
            except Exception:
                pass
            return out
        except Exception:
            return s

    # -------------------------
    # Safety: spec validation & auto-repair
    # -------------------------
    def _validate_and_repair_spec(self, df: pd.DataFrame, spec: ChartSpec, warnings: List[str], user_query: str) -> ChartSpec:
        dfx = df.copy()

        # 1) Coerce types by signal + actual content
        def _coerce_types(dfx: pd.DataFrame, col: Optional[str]) -> None:
            if not col or col not in dfx.columns:
                return
            cl = str(col).lower()
            s = dfx[col]
            if pd.api.types.is_object_dtype(s) and any(k in cl for k in ["date", "day", "time", "workdate", "startdate", "enddate"]):
                dfx[col] = self._coerce_datetime(s)
            if pd.api.types.is_object_dtype(s) and any(k in cl for k in ["hour", "hours", "duration", "value", "count", "qty", "amount", "minutes"]):
                dfx[col] = pd.to_numeric(s, errors="coerce")

        for c in [spec.x, spec.y, spec.color, spec.size, spec.facet_row, spec.facet_col, spec.sort_by]:
            _coerce_types(dfx, c)

        # 2) Ensure columns exist; drop invalid references
        def _exists(c: Optional[str]) -> bool:
            return bool(c) and c in dfx.columns

        dropped = []
        for attr in ["x", "y", "color", "size", "facet_row", "facet_col", "sort_by"]:
            cname = getattr(spec, attr)
            if cname and not _exists(cname):
                dropped.append((attr, cname))
                setattr(spec, attr, None)
        if dropped:
            warnings.append("Removed unknown columns: " + ", ".join([f"{a}={n}" for a, n in dropped]))

        # 3) If nothing left to map, table fallback
        if not any([spec.x, spec.y]) and spec.chart_type not in (ChartType.heatmap, ChartType.table, ChartType.histogram):
            warnings.append("No valid x/y columns after validation; downgraded to table")
            return ChartSpec(chart_type=ChartType.table, title=spec.title)

        # 4) Cardinality control (categorical axes)
        def _is_categorical(c: Optional[str]) -> bool:
            return bool(c) and (not pd.api.types.is_numeric_dtype(dfx[c])) and (not pd.api.types.is_datetime64_any_dtype(dfx[c]))

        if _is_categorical(spec.x):
            nunique = dfx[spec.x].nunique(dropna=False)
            if nunique > self.cfg.max_categories:
                spec.top_n = spec.top_n or self.cfg.topn_for_long_tail
                warnings.append(f"High cardinality on x={spec.x} ({nunique}); applying top_n={spec.top_n}")

        if spec.color and _is_categorical(spec.color) and dfx[spec.color].nunique(dropna=False) > self.cfg.max_categories:
            warnings.append(f"color={spec.color} too granular; dropping color")
            spec.color = None

        # 5) Numeric Y requirements for certain charts
        needs_numeric_y = spec.chart_type in (
            ChartType.line, ChartType.area, ChartType.bar, ChartType.grouped_bar, ChartType.stacked_bar, ChartType.scatter, ChartType.box
        )
        if needs_numeric_y and (not spec.y or not pd.api.types.is_numeric_dtype(dfx[spec.y])):
            if spec.chart_type in (ChartType.bar, ChartType.grouped_bar, ChartType.stacked_bar) and _exists(spec.x):
                warnings.append("Non-numeric y for bar; switching to frequency by x")
                spec.y = None
                spec.aggregate = Agg.count
            elif spec.chart_type in (ChartType.line, ChartType.area) and _exists(spec.x) and pd.api.types.is_datetime64_any_dtype(dfx[spec.x]):
                warnings.append("Missing numeric y for time series; using count by time")
                spec.y = None
                spec.aggregate = Agg.count
            elif spec.chart_type == ChartType.scatter and spec.x and pd.api.types.is_numeric_dtype(dfx[spec.x]):
                warnings.append("Non-numeric y for scatter; downgrading to histogram on x")
                spec = ChartSpec(chart_type=ChartType.histogram, x=spec.x, bins=spec.bins or self.cfg.default_hist_bins, title=spec.title)
            else:
                warnings.append("No numeric y available; downgraded to table")
                return ChartSpec(chart_type=ChartType.table, title=spec.title)

        # 6) Aggregation auto-fill
        if spec.chart_type in (ChartType.bar, ChartType.grouped_bar, ChartType.stacked_bar, ChartType.line, ChartType.area) and spec.x:
            if spec.aggregate == Agg.none and (spec.y or spec.aggregate == Agg.count):
                if spec.y and not self._is_one_row_per_category(dfx, spec.x):
                    spec.aggregate = Agg.sum if pd.api.types.is_numeric_dtype(dfx[spec.y]) else Agg.count
                    warnings.append(f"Auto-aggregate={spec.aggregate.value} by x={spec.x}")
            if spec.aggregate != Agg.none and not spec.group_by:
                spec.group_by = [spec.x] + ([spec.color] if spec.color else [])

        # 7) Histogram defaults
        if spec.chart_type == ChartType.histogram and not spec.bins:
            spec.bins = self.cfg.default_hist_bins

        # 8) Orientation sanity for bar
        if spec.chart_type in (ChartType.bar, ChartType.grouped_bar, ChartType.stacked_bar) and spec.orientation is None:
            spec.orientation = Orientation.v

        # 9) Trendline only for scatter
        if spec.trendline and spec.chart_type != ChartType.scatter:
            spec.trendline = False

        return spec

    def _is_one_row_per_category(self, dfx: pd.DataFrame, cat: str) -> bool:
        try:
            return dfx[cat].value_counts(dropna=False).max() <= 1
        except Exception:
            return False

    # -------------------------
    # Spec → Plotly rendering
    # -------------------------
    def _apply_aggregation(self, dfx: pd.DataFrame, spec: ChartSpec) -> Tuple[pd.DataFrame, Optional[str]]:
        if spec.aggregate == Agg.none:
            return dfx, spec.y

        group_cols = list(dict.fromkeys([c for c in [spec.x, spec.color, spec.facet_row, spec.facet_col] if c]))
        # When only counting, we don't need a target column
        if spec.aggregate == Agg.count:
            out = dfx.groupby(group_cols, dropna=False).size().reset_index(name="__value__")
            return out, "__value__"

        # Otherwise we need a numeric target
        target = spec.y or spec.size
        if not target or target not in dfx.columns:
            # If user omitted y, count fallback
            out = dfx.groupby(group_cols, dropna=False).size().reset_index(name="__value__")
            return out, "__value__"

        fn_map = {"sum": "sum", "avg": "mean", "mean": "mean", "min": "min", "max": "max"}
        fn = fn_map.get(spec.aggregate.value, "sum")
        out = dfx.groupby(group_cols, dropna=False)[target].agg(fn).reset_index(name="__value__")
        return out, "__value__"

    def _apply_topn(self, dfx: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
        if not spec.top_n:
            return dfx
        metric_col = spec.y if (spec.y and spec.y in dfx.columns) else ("__value__" if "__value__" in dfx.columns else None)
        if not metric_col or not spec.x or spec.x not in dfx.columns:
            return dfx

        # Apply a global top-n by metric; bucket the rest into "Other".
        tmp = dfx.sort_values(by=metric_col, ascending=False)
        head = tmp.head(spec.top_n)
        if len(tmp) > len(head):
            other_val = tmp.iloc[spec.top_n:][metric_col].sum()
            # Build a single "Other" row – keep color/facets neutral if present
            other = {}
            for c in head.columns:
                if c == spec.x:
                    other[c] = "Other"
                elif c == metric_col:
                    other[c] = other_val
                else:
                    # preserve the mode to keep shape consistent
                    try:
                        other[c] = head[c].mode(dropna=False).iloc[0] if not head[c].empty else None
                    except Exception:
                        other[c] = None
            head = pd.concat([head, pd.DataFrame([other])], ignore_index=True)
        return head

    def _resample_timeseries_if_needed(self, dfx: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
        """Resample/aggregate time series to a reasonable bin count."""
        if not spec.x or spec.x not in dfx.columns:
            return dfx
        if not pd.api.types.is_datetime64_any_dtype(dfx[spec.x]):
            return dfx

        # If already small enough, keep as-is
        if len(dfx) <= self.cfg.max_timeseries_points:
            return dfx

        # Choose a frequency based on span and a target bin count
        s = dfx[spec.x].dropna()
        if s.empty:
            return dfx
        span_days = (s.max() - s.min()).days if isinstance(s.max(), pd.Timestamp) else None
        target = self.cfg.default_target_bins

        freq = "D"
        try:
            if span_days is None:
                freq = "D"
            elif span_days <= 60:
                freq = "D"
            elif span_days <= 365 * 2:
                freq = "W"
            elif span_days <= 365 * 5:
                freq = "M"
            elif span_days <= 365 * 12:
                freq = "Q"
            else:
                freq = "Y"
        except Exception:
            freq = "M"

        # Determine metric to aggregate
        metric_col = spec.y if (spec.y and spec.y in dfx.columns and pd.api.types.is_numeric_dtype(dfx[spec.y])) else None
        agg = spec.aggregate

        dfy = dfx.copy()
        dfy = dfy.set_index(spec.x)

        if agg == Agg.count or metric_col is None:
            # Count rows per bin
            grouped = dfy.resample(freq).size().reset_index(name="__value__")
            res = grouped.rename(columns={spec.x: spec.x})
            if metric_col is None:
                # ensure the spec uses our computed column
                if spec.y != "__value__":
                    spec.y = "__value__"
                    spec.aggregate = Agg.none
            return res

        fn_map = {"sum": "sum", "avg": "mean", "mean": "mean", "min": "min", "max": "max", "none": "sum"}
        fn = fn_map.get(agg.value, "sum")

        grouped = dfy[metric_col].resample(freq).agg(fn).reset_index().rename(columns={metric_col: "__value__"})
        if spec.y != "__value__":
            spec.y = "__value__"
            spec.aggregate = Agg.none
        return grouped

    def _post_sort(self, dfx: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
        # Sorting preference: explicit spec.sort_by > time on x (asc) > y (desc for bars)
        try:
            if spec.sort_by and spec.sort_by in dfx.columns:
                ascending = (spec.sort_dir == SortDir.asc)
                return dfx.sort_values(by=spec.sort_by, ascending=ascending)
            if spec.x and spec.x in dfx.columns and pd.api.types.is_datetime64_any_dtype(dfx[spec.x]):
                return dfx.sort_values(by=spec.x, ascending=True)
            if spec.chart_type in (ChartType.bar, ChartType.grouped_bar, ChartType.stacked_bar) and spec.y and spec.y in dfx.columns:
                return dfx.sort_values(by=spec.y, ascending=False)
        except Exception:
            pass
        return dfx

    def _build_from_spec(self, df: pd.DataFrame, spec: ChartSpec) -> go.Figure:
        dfx = df.copy()

        # Aggregation
        dfx, y_col = self._apply_aggregation(dfx, spec)
        if y_col and (not spec.y or spec.aggregate != Agg.none):
            spec.y = y_col

        # Resample wide date ranges (line/area; also for bar if x is datetime)
        if spec.chart_type in (ChartType.line, ChartType.area, ChartType.bar) and spec.x and spec.x in dfx.columns:
            dfx = self._resample_timeseries_if_needed(dfx, spec)

        # Top-N bucketing (after aggregation/resample)
        dfx = self._apply_topn(dfx, spec)

        # Drop rows with missing required fields
        needed = [c for c in [spec.x, spec.y] if c]
        if needed:
            before = len(dfx)
            dfx = dfx.dropna(subset=needed)
            if len(dfx) < before:
                logger.debug("Dropped %d rows with NaN in %s", before - len(dfx), needed)

        # Final sorting preference
        dfx = self._post_sort(dfx, spec)

        title = spec.title or spec.chart_type.value.title()
        theme = self.cfg.theme
        chart = spec.chart_type

        # Facets setup
        facet_kwargs = {}
        if spec.facet_row and spec.facet_row in dfx.columns:
            facet_kwargs["facet_row"] = spec.facet_row
        if spec.facet_col and spec.facet_col in dfx.columns:
            facet_kwargs["facet_col"] = spec.facet_col

        # Build figures
        if chart == ChartType.line:
            fig = px.line(dfx, x=spec.x, y=spec.y, color=spec.color, title=title, template=theme, **facet_kwargs)
        elif chart == ChartType.area:
            fig = px.area(dfx, x=spec.x, y=spec.y, color=spec.color, title=title, template=theme, **facet_kwargs)
        elif chart in (ChartType.bar, ChartType.grouped_bar, ChartType.stacked_bar):
            barmode = "relative" if chart == ChartType.stacked_bar else ("group" if chart == ChartType.grouped_bar else None)
            orient = spec.orientation or Orientation.v
            xx = spec.x if orient == Orientation.v else spec.y
            yy = spec.y if orient == Orientation.v else spec.x
            fig = px.bar(dfx, x=xx, y=yy, color=spec.color, title=title, template=theme, **facet_kwargs)
            if barmode:
                fig.update_layout(barmode=barmode)
        elif chart == ChartType.pie:
            # Choose names and values safely
            names = spec.x or spec.color or next((c for c in dfx.columns if not pd.api.types.is_numeric_dtype(dfx[c])), dfx.columns[0])
            values = spec.y if (spec.y and spec.y in dfx.columns and pd.api.types.is_numeric_dtype(dfx[spec.y])) else None
            if not values:
                counts = dfx[names].fillna("Unknown").value_counts(dropna=False).reset_index()
                counts.columns = [names, "__value__"]
                fig = px.pie(counts, names=names, values="__value__", title=title, template=theme)
            else:
                fig = px.pie(dfx, names=names, values=values, color=spec.color, title=title, template=theme)
        elif chart == ChartType.scatter:
            # Trendline requires statsmodels; if missing, ignore
            trendline = "ols" if (spec.trendline is True) else None
            try:
                fig = px.scatter(dfx, x=spec.x, y=spec.y, color=spec.color, size=spec.size, title=title, template=theme, trendline=trendline, **facet_kwargs)
            except Exception:
                fig = px.scatter(dfx, x=spec.x, y=spec.y, color=spec.color, size=spec.size, title=title, template=theme, **facet_kwargs)
        elif chart == ChartType.histogram:
            target_col = spec.x or spec.y
            if not target_col or target_col not in dfx.columns or not pd.api.types.is_numeric_dtype(dfx[target_col]):
                # Try to coerce once
                if target_col and target_col in dfx.columns:
                    cand = pd.to_numeric(dfx[target_col], errors="coerce")
                    if cand.notna().any():
                        dfx[target_col] = cand
                if not target_col or not pd.api.types.is_numeric_dtype(dfx[target_col]):
                    # fallback table
                    header_vals = list(map(str, dfx.columns))
                    cell_vals = [dfx[c] for c in dfx.columns]
                    fig = go.Figure(data=[go.Table(
                        header=dict(values=header_vals, fill_color="#e2e8f0", align="left"),
                        cells=dict(values=cell_vals, fill_color="#ffffff", align="left"),
                    )])
                    fig.update_layout(title="Table (non-numeric for histogram)", template=theme, width=800, height=500)
                else:
                    fig = px.histogram(dfx, x=target_col, nbins=spec.bins or self.cfg.default_hist_bins, title=title, template=theme, color=spec.color)
            else:
                fig = px.histogram(dfx, x=target_col, nbins=spec.bins or self.cfg.default_hist_bins, title=title, template=theme, color=spec.color)
        elif chart == ChartType.heatmap:
            # Try matrix from two categoricals + numeric, else correlation of numerics
            num_cols = [c for c in dfx.columns if pd.api.types.is_numeric_dtype(dfx[c])]
            cat_cols = [c for c in dfx.columns if not pd.api.types.is_numeric_dtype(dfx[c])]
            fig = None
            if len(cat_cols) >= 2 and (spec.y or "__value__" in dfx.columns):
                yv = spec.y if (spec.y in dfx.columns) else "__value__"
                pivot = dfx.pivot_table(index=cat_cols[0], columns=cat_cols[1], values=yv, aggfunc="sum", fill_value=0)
                fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns.astype(str), y=pivot.index.astype(str), colorscale="Blues"))
                fig.update_layout(title=title, template=theme)
            if fig is None:
                if len(num_cols) < 2:
                    header_vals = list(map(str, dfx.columns))
                    cell_vals = [dfx[c] for c in dfx.columns]
                    fig = go.Figure(data=[go.Table(
                        header=dict(values=header_vals, fill_color="#e2e8f0", align="left"),
                        cells=dict(values=cell_vals, fill_color="#ffffff", align="left"),
                    )])
                    fig.update_layout(title="Table (no numeric matrix for heatmap)", template=theme, width=800, height=500)
                else:
                    corr = dfx[num_cols].corr(numeric_only=True)
                    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Blues"))
                    fig.update_layout(title=title, template=theme)
        elif chart == ChartType.box:
            fig = px.box(dfx, x=spec.x, y=spec.y, color=spec.color, title=title, template=theme, **facet_kwargs)
        else:
            header_vals = list(map(str, dfx.columns))
            cell_vals = [dfx[c] for c in dfx.columns]
            fig = go.Figure(data=[go.Table(
                header=dict(values=header_vals, fill_color="#e2e8f0", align="left"),
                cells=dict(values=cell_vals, fill_color="#ffffff", align="left"),
            )])
            fig.update_layout(title=title, template=theme, width=800, height=500)

        # Common layout polish (safe defaults)
        fig.update_layout(
            width=800,
            height=500,
            legend_title_text=None,
            margin=dict(l=40, r=20, t=60, b=40),
            hovermode="x unified" if (spec.x and spec.x in dfx.columns and pd.api.types.is_datetime64_any_dtype(dfx[spec.x])) else "closest",
        )
        # Date axis formatting when present
        if spec.x and spec.x in dfx.columns and pd.api.types.is_datetime64_any_dtype(dfx[spec.x]):
            fig.update_xaxes(showgrid=True, tickformat="%Y-%m-%d")

        return fig

    # -------------------------
    # Helpers
    # -------------------------
    def _fallback_to_spec(self, vt: str, schema: Dict[str, Any]) -> ChartSpec:
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
        cols = [c["name"] for c in schema["columns"]]
        numerics = [c["name"] for c in schema["columns"] if c["type"] == "numeric"]
        dates = [c["name"] for c in schema["columns"] if c["type"] == "datetime"]
        cats = [c["name"] for c in schema["columns"] if c["type"] == "categorical"]
        x = (dates[0] if dates else (cats[0] if cats else (cols[0] if cols else None)))
        y = (numerics[0] if numerics else (cols[1] if len(cols) > 1 else None))
        return ChartSpec(chart_type=ct, x=x, y=y)

    def _to_viz_chart_type(self, ct: ChartType) -> str:
        return {
            ChartType.line: "line_chart",
            ChartType.area: "area_chart",
            ChartType.bar: "bar_chart",
            ChartType.stacked_bar: "bar_chart",
            ChartType.grouped_bar: "bar_chart",
            ChartType.pie: "pie_chart",
            ChartType.scatter: "scatter_plot",
            ChartType.histogram: "histogram",
            ChartType.heatmap: "heatmap",
            ChartType.box: "box_plot",
            ChartType.table: "table",
        }[ct]

    def _map_to_viz(self, vt_or_ct: str) -> str:
        # Accept both your viz ids and ChartType strings
        s = (vt_or_ct or "").lower()
        mapping = {
            "line_chart": "line_chart", "line": "line_chart",
            "bar_chart": "bar_chart", "bar": "bar_chart",
            "pie_chart": "pie_chart", "pie": "pie_chart",
            "scatter_plot": "scatter_plot", "scatter": "scatter_plot",
            "histogram": "histogram",
            "heatmap": "heatmap",
            "box_plot": "box_plot", "box": "box_plot",
            "table": "table",
        }
        return mapping.get(s, "table")
