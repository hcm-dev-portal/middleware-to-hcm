import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .visualization_service import VisualizationService, _coerce_dtypes_for_viz

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    _OPENAI = True
except Exception:
    _OPENAI = False


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

@dataclass
class ChartVisualizationAgentConfig:
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    theme: str = "plotly_white"
    use_ai: bool = True  # if False, we strictly use rule-based fallbacks


# -------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------

class ChartVisualizationAgent:
    """
    Builds a chart using:
      1) explicit chart type in user query (if present),
      2) AI-designed chart spec (if enabled),
      3) deterministic fallback.

    Always operates on RAW df (no enrichment).
    """

    def __init__(self, viz_service: VisualizationService, cfg: ChartVisualizationAgentConfig):
        self.viz_service = viz_service
        self.cfg = cfg
        self.theme = cfg.theme

        self.llm = None
        self._prompt = None
        if self.cfg.use_ai and _OPENAI:
            try:
                self.llm = ChatOpenAI(model=self.cfg.model, temperature=self.cfg.temperature)
                # Strict JSON chart-spec prompt
                self._prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        "You are a careful visualization planner. "
                        "Given the user query and dataset schema, produce a JSON object describing the chart to build. "
                        "Return ONLY valid JSON, no preamble or commentary. "
                        "Schema:\n"
                        "{\n"
                        '  "chart_type": "line_chart | bar_chart | pie_chart | scatter_plot | histogram | heatmap | box_plot | table | area_chart",\n'
                        '  "x": "name of x column or null",\n'
                        '  "y": "name of y column or null",\n'
                        '  "color": "categorical column or null",\n'
                        '  "facet_col": "categorical column to facet by, or null",\n'
                        '  "agg": "sum | mean | count | median | min | max | none",\n'
                        '  "bins": "integer or null (for histogram)",\n'
                        '  "top_n": "integer or null (apply to categories after aggregation)",\n'
                        '  "sort_by": "x | y | none",\n'
                        '  "sort_order": "asc | desc | none",\n'
                        '  "reasoning": "short explanation"\n'
                        "}\n"
                        "Rules:\n"
                        "- Pick existing columns only. If unsure, set the field to null.\n"
                        "- Prefer a time column on x for trends. Use numeric on y. "
                        "  If the user asks for distribution, choose histogram; correlation -> scatter. "
                        "- If only one numeric column is available, prefer histogram or bar by category.\n"
                        "- Keep it simple. Do not invent columns not listed."
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "User query: {query}\n"
                        "Columns: {columns}\n"
                        "Types: {types}\n"
                        "Sample (first 2 rows): {sample}\n"
                    ),
                ])
            except Exception as e:
                logger.warning("Chart agent AI disabled: %s", e)
                self.cfg.use_ai = False

    # -------------------------------------------------------------------

    def generate_chart(
        self,
        df: pd.DataFrame,
        user_query: str,
        sql_meta: Optional[Dict[str, Any]] = None,
        force_chart_type: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entrypoint: returns a 'visualization' payload.
        """
        try:
            if df is None:
                raise ValueError("DataFrame is None")
            df = _coerce_dtypes_for_viz(df)

            analysis = self.viz_service._analyze(df)  # reuse
            chart_from_query = self.viz_service._extract_chart_intent(user_query)
            chosen_type = force_chart_type or chart_from_query

            # 1) If user forced a type, build with default axes
            if chosen_type:
                fig = self.viz_service._build_chart(df, chosen_type, self._auto_title(user_query, chosen_type, title))
                url, fname = self.viz_service._save_figure(fig, chosen_type)
                return self._pack_payload(chosen_type, title or self._auto_title(user_query, chosen_type, title), url, fname,
                                          reasoning=f"Chart type chosen: {('forced' if force_chart_type else 'from query')}.",
                                          df=df, analysis=analysis)

            # 2) AI chart spec
            spec = None
            if self.cfg.use_ai and self.llm and self._prompt and not analysis.get("empty"):
                spec = self._ask_llm_for_spec(df, user_query)

            # 3) Build from spec (validated), else fallback
            if spec:
                fig = self._build_from_spec(df, spec, title or self._auto_title(user_query, spec.get("chart_type"), title))
                if fig:
                    url, fname = self.viz_service._save_figure(fig, spec.get("chart_type") or "table")
                    return self._pack_payload(
                        spec.get("chart_type") or "table",
                        title or self._auto_title(user_query, spec.get("chart_type"), title),
                        url, fname,
                        reasoning=spec.get("reasoning", "AI-designed chart."),
                        df=df, analysis=analysis
                    )

            # 4) Deterministic fallback
            reco = self.viz_service._fallback_recommendation(analysis, user_query)
            fallback_type = reco["recommended_chart"]
            fig = self.viz_service._build_chart(df, fallback_type, self._auto_title(user_query, fallback_type, title))
            url, fname = self.viz_service._save_figure(fig, fallback_type)
            return self._pack_payload(
                fallback_type,
                title or self._auto_title(user_query, fallback_type, title),
                url, fname,
                reasoning=reco.get("reasoning", "Fallback recommendation."),
                df=df, analysis=analysis
            )

        except Exception as e:
            logger.warning("Visualization agent failed: %s", e, exc_info=True)
            return {"enabled": False, "reason": f"Visualization error: {e}"}

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------

    def _ask_llm_for_spec(self, df: pd.DataFrame, user_query: str) -> Optional[Dict[str, Any]]:
        try:
            cols = list(map(str, df.columns))
            types = {c: str(df[c].dtype) for c in df.columns}
            sample = df.head(2).to_dict("records")

            msgs = self._prompt.format_messages(
                query=user_query,
                columns=cols,
                types=types,
                sample=sample,
            )
            res = self.llm.invoke(msgs)
            txt = getattr(res, "content", "") or ""
            spec = json.loads(txt)
            return self._validate_and_normalize_spec(spec, cols)
        except Exception as e:
            logger.info("LLM spec parse/validate failed: %s", e)
            return None

    def _validate_and_normalize_spec(self, spec: Dict[str, Any], cols: List[str]) -> Optional[Dict[str, Any]]:
        if not isinstance(spec, dict):
            return None

        allowed_types = {
            "line_chart", "bar_chart", "pie_chart", "scatter_plot",
            "histogram", "heatmap", "box_plot", "table", "area_chart"
        }
        ct = str(spec.get("chart_type") or "").lower()
        if ct not in allowed_types:
            # lenient: if absent, delay to fallback
            ct = "table"

        def _pick(col_name: Optional[str]) -> Optional[str]:
            if not col_name:
                return None
            low = col_name.lower()
            for c in cols:
                if c.lower() == low:
                    return c
            # soft fuzzy: startswith
            for c in cols:
                if c.lower().startswith(low):
                    return c
            return None

        out = {
            "chart_type": ct,
            "x": _pick(spec.get("x")),
            "y": _pick(spec.get("y")),
            "color": _pick(spec.get("color")),
            "facet_col": _pick(spec.get("facet_col")),
            "agg": (spec.get("agg") or "none").lower(),
            "bins": int(spec["bins"]) if str(spec.get("bins", "")).isdigit() else None,
            "top_n": int(spec["top_n"]) if str(spec.get("top_n", "")).isdigit() else None,
            "sort_by": (spec.get("sort_by") or "none").lower(),
            "sort_order": (spec.get("sort_order") or "none").lower(),
            "reasoning": str(spec.get("reasoning") or ""),
        }
        return out

    def _build_from_spec(self, df: pd.DataFrame, spec: Dict[str, Any], title: str) -> Optional[go.Figure]:
        ct = spec.get("chart_type", "table")
        x = spec.get("x")
        y = spec.get("y")
        color = spec.get("color")
        facet_col = spec.get("facet_col")
        agg = spec.get("agg", "none")
        bins = spec.get("bins")
        top_n = spec.get("top_n")
        sort_by = spec.get("sort_by", "none")
        sort_order = spec.get("sort_order", "none")

        # Pre-aggregation if requested
        work = df.copy()
        if agg and agg != "none" and (x or color) and y and y in work.columns:
            group_keys = [c for c in [x, color] if c]
            if group_keys:
                try:
                    if agg == "sum":
                        work = work.groupby(group_keys, dropna=False)[y].sum().reset_index()
                    elif agg == "mean":
                        work = work.groupby(group_keys, dropna=False)[y].mean().reset_index()
                    elif agg == "count":
                        work = work.groupby(group_keys, dropna=False)[y].count().reset_index()
                        y = y  # count column stays y
                    elif agg == "median":
                        work = work.groupby(group_keys, dropna=False)[y].median().reset_index()
                    elif agg == "min":
                        work = work.groupby(group_keys, dropna=False)[y].min().reset_index()
                    elif agg == "max":
                        work = work.groupby(group_keys, dropna=False)[y].max().reset_index()
                except Exception as e:
                    logger.info("Aggregation failed, continue without aggregation: %s", e)

        # Top-N after aggregation (by y)
        if top_n and y in work.columns and (x in work.columns or color in work.columns):
            sort_key = y
            try:
                work = work.sort_values(by=sort_key, ascending=False).head(int(top_n))
            except Exception:
                pass

        # Sorting
        if sort_by in ("x", "y") and ((sort_by == "x" and x in work.columns) or (sort_by == "y" and y in work.columns)):
            asc = (sort_order != "desc")
            try:
                work = work.sort_values(by=(x if sort_by == "x" else y), ascending=asc)
            except Exception:
                pass

        # Plotly builders from spec
        try:
            if ct == "line_chart":
                if x and y:
                    fig = px.line(work, x=x, y=y, color=color, facet_col=facet_col, title=title, template=self.theme)
                else:
                    fig = px.line(title=title, template=self.theme)
            elif ct == "area_chart":
                if x and y:
                    fig = px.area(work, x=x, y=y, color=color, facet_col=facet_col, title=title, template=self.theme)
                else:
                    fig = px.area(title=title, template=self.theme)
            elif ct == "bar_chart":
                if x and y:
                    fig = px.bar(work, x=x, y=y, color=color, facet_col=facet_col, title=title, template=self.theme)
                elif x:
                    vc = work[x].value_counts(dropna=False).reset_index()
                    vc.columns = [x, "Count"]
                    fig = px.bar(vc, x=x, y="Count", color=color, title=title, template=self.theme)
                else:
                    fig = px.bar(title=title, template=self.theme)
            elif ct == "pie_chart":
                names = x or color
                if names and y and y in work.columns:
                    fig = px.pie(work, names=names, values=y, title=title)
                elif names:
                    vc = work[names].value_counts(dropna=False).reset_index()
                    vc.columns = [names, "Count"]
                    fig = px.pie(vc, names=names, values="Count", title=title)
                else:
                    fig = px.pie(title=title)
            elif ct == "scatter_plot":
                if x and y:
                    fig = px.scatter(work, x=x, y=y, color=color, facet_col=facet_col, title=title, template=self.theme)
                else:
                    fig = px.scatter(title=title, template=self.theme)
            elif ct == "histogram":
                hx = x or y
                if hx:
                    fig = px.histogram(work, x=hx, nbins=bins, color=color, title=title, template=self.theme)
                else:
                    fig = px.histogram(title=title, template=self.theme)
            elif ct == "heatmap":
                numeric = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
                if len(numeric) >= 2:
                    corr = work[numeric].corr(numeric_only=True)
                    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Blues"))
                    fig.update_layout(title=title, template=self.theme)
                else:
                    fig = go.Figure()
                    fig.update_layout(title=title, template=self.theme)
            elif ct == "box_plot":
                if y and y in work.columns:
                    fig = px.box(work, y=y, x=color or x, title=title, template=self.theme)
                else:
                    fig = px.box(title=title, template=self.theme)
            else:
                # Table
                header_vals = list(map(str, work.columns))
                cell_vals = [work[c] for c in work.columns]
                fig = go.Figure(data=[go.Table(
                    header=dict(values=header_vals, fill_color="#e2e8f0", align="left"),
                    cells=dict(values=cell_vals, fill_color="#ffffff", align="left"),
                )])
                fig.update_layout(title=title, template=self.theme, width=800, height=500)
        except Exception as e:
            logger.info("Build from spec failed, fallback to service builder: %s", e)
            return self.viz_service._build_chart(df, ct, title)

        fig.update_layout(width=800, height=500)
        return fig

    def _pack_payload(
        self,
        chart_type: str,
        title: str,
        url: str,
        filename: str,
        reasoning: str,
        df: pd.DataFrame,
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "enabled": True,
            "type": chart_type,
            "title": title,
            "url": url,
            "filename": filename,
            "reasoning": reasoning,
            "insights": self.viz_service._generate_insights(df, chart_type, analysis, []),
            "alternatives": [],
            "data_summary": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "column_names": list(map(str, df.columns)),
            },
        }

    def _auto_title(self, user_query: str, chart_type: Optional[str], existing: Optional[str]) -> str:
        if existing:
            return existing
        if user_query:
            return user_query[:120]
        return (chart_type or "Chart").replace("_", " ").title()
