from __future__ import annotations

import inspect
from collections import defaultdict
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
    Type,
)
import subprocess

"""
pipeline_printer.py – quick visualisation helpers for the Agent‑Framework.

Usage
-----
>>> printer = PipelinePrinter()
>>> printer.print_ascii(scheduler.agents)
>>> dot_str = printer.to_dot(scheduler.agents)
>>> Path('pipeline.dot').write_text(dot_str)

Self‑contained – it only relies on an agent exposing:

Legacy (single output)
  _output_port : ToolPort      # or output_port
  output_schema : Type[BaseModel] | None

New (multi output)
  _output_ports : Dict[Type[BaseModel], ToolPort]   # or output_ports
  output_schemas : List[Type[BaseModel]] | None     # (len==1 for compat)

Input‑side (unchanged)
  _input_ports : Dict[str, ToolPort]
  input_schema : Type[BaseModel] | None
"""

if TYPE_CHECKING:
    from AgentFramework.ConnectedAgent import ConnectedAgent
    from AgentFramework.ToolPort import ToolPort


class PipelinePrinter:
    """Generate human‑readable or GraphViz views of a list of agents."""

    # ------------------------------------------------------------------ construction

    def __init__(
        self,
        is_ortho: bool = False,
        direction: str = "TB",
        fillcolor: Optional[str] = None,
        *,
        show_schemas: bool = False,
        schema_fillcolor: Optional[str] = None,
    ) -> None:
        """
        Args
        ----
        is_ortho
            If *True*, use orthogonal (right‑angled) edges.
        direction
            Rank direction for GraphViz ("TB", "LR", …).
        fillcolor
            Base colour for *agent* nodes. If *None*, no fill.
        show_schemas
            If *True*, insert schema‑type “note” nodes between agents.
        schema_fillcolor
            Fill colour for those schema nodes (inherit default when *None*).
        """
        self.is_ortho = is_ortho
        self.direction = direction

        # agent‑node colours -------------------------------------------------
        self.fillcolor = fillcolor
        if fillcolor:
            self.node_fill = (
                fillcolor if fillcolor.startswith("light") else f"light{fillcolor}"
            )
            self.node_border = fillcolor
            self.node_fontcolor = "black"
        else:
            self.node_fill = None
            self.node_border = None
            self.node_fontcolor = None

        # schema‑node settings ----------------------------------------------
        self.show_schemas = show_schemas
        self.schema_fillcolor = schema_fillcolor

    # ------------------------------------------------------------------ public helpers

    def print_ascii(self, agents: Iterable["ConnectedAgent"]) -> None:
        """Pretty‑print an ASCII overview to *stdout*."""
        edges = self._collect_edges(list(agents))
        self._ascii(edges)

    def to_dot(self, agents: Iterable["ConnectedAgent"]) -> str:
        """Return a GraphViz *dot* string describing the pipeline."""
        edges = self._collect_edges(list(agents))
        return self._dot(edges)

    def save_as_png(self, agents: Iterable["ConnectedAgent"], png_path: Union[str, Path]):
        """Render the graph directly to a PNG (requires *dot* on PATH)."""
        png_path = Path(png_path)
        dot_str = self.to_dot(agents)
        try:
            subprocess.run(
                ["dot", "-Tpng", "-Gdpi=300", "-o", str(png_path)],
                input=dot_str.encode("utf-8"),
                check=True,
            )
            print(f"PNG generated: {png_path}")
        except FileNotFoundError:
            print("Error: Graphviz 'dot' command not found. Is it installed and on your PATH?")
        except subprocess.CalledProcessError as e:
            print(f"dot command failed: {e}")

    # ------------------------------------------------------------------ internal logic

    def _collect_edges(
        self, agents: List["ConnectedAgent"]
    ) -> Dict[str, List[Tuple[str, str, Optional[type], Optional[type]]]]:
        """
        Walk all output‑port connections, returning:

            {src_label: [(tgt_label,
                          tgt_port_suffix,
                          output_schema (per‑port),
                          input_schema), …]}
        """
        # --- compact labels -------------------------------------------------
        names: Dict["ConnectedAgent", str] = {}
        cls_counter: Dict[str, int] = defaultdict(int)
        for ag in agents:
            uid = "" if ag.uuid == "default" else f"[{ag.uuid}]"
            cls = ag.__class__.__name__
            cls_counter[cls] += 1
            names[ag] = f"{cls}#{cls_counter[cls]} {uid}"

        edges: Dict[str, List[Tuple[str, str, Optional[type], Optional[type]]]] = defaultdict(list)

        for ag in agents:
            for out_port in self._iter_output_ports(ag):
                out_schema = self._schema_for_port(ag, out_port)

                for tgt_port, _tr, _cond, (_src, tgt_agent) in out_port.connections:
                    src_lbl = names[ag]
                    tgt_lbl = names.get(tgt_agent, tgt_agent.__class__.__name__)
                    tgt_suffix = self._find_port_name(tgt_agent, tgt_port)
                    in_schema = getattr(tgt_agent, "input_schema", None)

                    edges[src_lbl].append((tgt_lbl, tgt_suffix, out_schema, in_schema))

        return edges

    # ------------------------------------------------------------------ low‑level helpers

    @staticmethod
    def _iter_output_ports(agent: "ConnectedAgent"):
        """Yield every *distinct* output `ToolPort` an agent exposes."""
        seen = set()

        # legacy single output ------------------------------------------------
        for attr in ("_output_port", "output_port"):
            if hasattr(agent, attr):
                port = getattr(agent, attr)
                if port and port not in seen:
                    seen.add(port)
                    yield port

        # new multi‑output mapping -------------------------------------------
        for attr in ("_output_ports", "output_ports"):
            if hasattr(agent, attr):
                mapping = getattr(agent, attr) or {}
                for port in mapping.values():
                    if port not in seen:
                        seen.add(port)
                        yield port

    @staticmethod
    def _schema_for_port(agent: "ConnectedAgent", port_obj: "ToolPort") -> Optional[Type]:
        """Resolve the schema *associated* with a given output port."""
        # 1) new mapping {schema_type: port}
        for attr in ("_output_ports", "output_ports"):
            if hasattr(agent, attr):
                for schema_type, p in getattr(agent, attr).items():
                    if p is port_obj:
                        return schema_type

        # 2) legacy list – use only when unambiguous
        if hasattr(agent, "output_schemas"):
            out_schemas = getattr(agent, "output_schemas") or []
            if len(out_schemas) == 1:
                return out_schemas[0]

        # 3) final single‑schema fallback
        return getattr(agent, "output_schema", None)

    @staticmethod
    def _find_port_name(agent: "ConnectedAgent", port_obj: "ToolPort") -> str:
        """Return '@<input_name>' if *agent* is multi‑port and *port_obj* matches."""
        if hasattr(agent, "_input_ports"):
            for name, p in agent._input_ports.items():
                if p is port_obj:
                    return f"@{name}"
        return ""

    # ------------------------------------------------------------------ renderers

    def _ascii(self, edges):
        """Minimal ASCII tree (schema details omitted)."""
        for src, targets in edges.items():
            print(src)
            for i, (tgt, suffix, *_rest) in enumerate(targets):
                connector = "└─▶" if i == len(targets) - 1 else "├─▶"
                print(f"  {connector} {tgt}{suffix}")

    def _dot(self, edges) -> str:
        """Return a GraphViz *dot* string for the graph – honours `show_schemas`."""
        label_type = "xlabel" if self.is_ortho else "label"

        lines: List[str] = ["digraph pipeline {", f"  rankdir={self.direction};", "  bgcolor=transparent;"]
        if self.is_ortho:
            lines.append("  splines=ortho;")

        # node defaults ------------------------------------------------------
        if self.fillcolor:
            lines.append(
                '  node [shape=box, style="rounded,filled", '
                f'fillcolor="{self.node_fill}", color="{self.node_border}", '
                f'fontcolor="{self.node_fontcolor}"];'
            )
        else:
            lines.append("  node [shape=box, style=rounded];")

        # edges (and optional schema notes) ----------------------------------
        for src, targets in edges.items():
            for tgt, suffix, out_schema, in_schema in targets:
                if self.show_schemas:
                    safe = suffix or "@"
                    note_id = f"{src}_{tgt}_{safe}_schema"
                    if out_schema == in_schema:
                        label = out_schema.__name__ if out_schema else "?"
                    else:
                        ol = out_schema.__name__ if out_schema else "?"
                        il = in_schema.__name__ if in_schema else "?"
                        label = f"{ol} →\\n→ {il}"

                    attrs = [
                        f'label="{label}"',
                        "shape=note",
                        "style=filled",
                        "fontsize=8",
                        "margin=0.05",
                        "width=0.0",
                        "height=0.0",
                        'color="black"',
                        "penwidth=0.5",
                    ]
                    if self.schema_fillcolor:
                        attrs.append(f'fillcolor="{self.schema_fillcolor}"')
                    lines.append(f'  "{note_id}" [{", ".join(attrs)}];')

                    colour = "seagreen" if out_schema == in_schema else "blue"
                    lines.append(f'  "{src}" -> "{note_id}" [color={colour}, arrowsize=0.75];')
                    lines.append(f'  "{note_id}" -> "{tgt}" [color={colour}, arrowsize=0.75];')
                else:
                    lbl = suffix[1:] if suffix else ""
                    edge_attr = f' [{label_type}="{lbl}"]' if lbl else ""
                    lines.append(f'  "{src}" -> "{tgt}"{edge_attr};')

        lines.append("}")
        return "\n".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
#  ADD TO PipelinePrinter (place just below _dot, or wherever you group helpers)
# ──────────────────────────────────────────────────────────────────────────────

    # ───────────────────────────  NEW: Mermaid  ────────────────────────────

    def to_mermaid(self, agents: Iterable["ConnectedAgent"]) -> str:
        """
        Return a Mermaid-JS graph definition that mirrors `.to_dot()`.

        Example
        -------
        >>> printer = PipelinePrinter(direction="LR", show_schemas=True)
        >>> mmd = printer.to_mermaid(scheduler.agents)
        >>> Path("pipeline.mmd").write_text(mmd)
        """
        edges = self._collect_edges(list(agents))
        return self._mermaid(edges)

    # ------------------------------------------------------------------ helpers

    def _mermaid(self, edges) -> str:
        """Internal rendering routine – very similar to `_dot()`."""
        # Graph orientation ────────────────────────────────────────────────
        orient_map = {"TB": "TD", "BT": "BT", "LR": "LR", "RL": "RL"}
        orientation = orient_map.get(self.direction, "TD")

        def _safe(node: str) -> str:
            """Mermaid node ids must be bare-word – replace non-word chars."""
            import re
            return re.sub(r"\W+", "_", node)

        # Build text lines ─────────────────────────────────────────────────
        lines: List[str] = [f"graph {orientation}"]

        for src, targets in edges.items():
            src_id = _safe(src)
            lines.append(f'    {src_id}["{src}"]')          # declare once

            for tgt, suffix, out_schema, in_schema in targets:
                tgt_id = _safe(tgt)
                if f'{tgt_id}["' not in " ".join(lines):     # declare once
                    lines.append(f'    {tgt_id}["{tgt}"]')

                # Edge label (input-port suffix)  ────────────────────────
                elabel = f' |{suffix[1:]}| ' if suffix else " "

                # Optional schema “note” nodes  ───────────────────────────
                if self.show_schemas:
                    safe = suffix or "@"
                    note_id = _safe(f"{src}_{tgt}_{safe}_schema")
                    if out_schema == in_schema:
                        label = out_schema.__name__ if out_schema else "?"
                    else:
                        ol = out_schema.__name__ if out_schema else "?"
                        il = in_schema.__name__ if in_schema else "?"
                        label = f"{ol} →<br/>→ {il}"
                    lines.append(f'    {note_id}["{label}"]:::schema')
                    lines.append(f"    {src_id} -->{elabel}{note_id}")
                    lines.append(f"    {note_id} --> {tgt_id}")
                else:
                    lines.append(f"    {src_id} -->{elabel}{tgt_id}")

        lines.append("")

        # Optional styling block for schema notes (only if used) ───────────
        if self.show_schemas:
            fill = self.schema_fillcolor or "#FFE"
            lines.insert(
                2,
                f'classDef schema fill:{fill},stroke:#333,stroke-width:1px,font-size:10px;',
            )

        return "\n".join(lines)


    def save_as_dot(
            self,
            agents: Iterable["ConnectedAgent"],
            dot_path: Union[str, Path],
    ) -> None:
        """
        Render the pipeline to a GraphViz DOT file.

        Args:
            agents: Iterable of ConnectedAgent
            dot_path: Path or filename where the .dot should be written
        """
        dot_path = Path(dot_path)
        dot_str = self.to_dot(agents)
        dot_path.write_text(dot_str, encoding="utf-8")
        print(f"DOT file generated: {dot_path}")

    def save_as_mermaid(
            self,
            agents: Iterable["ConnectedAgent"],
            mmd_path: Union[str, Path],
    ) -> None:
        """
        Render the pipeline to a Mermaid-JS file.

        Args:
            agents: Iterable of ConnectedAgent
            mmd_path: Path or filename where the .mmd should be written
        """
        mmd_path = Path(mmd_path)
        mmd_str = self.to_mermaid(agents)
        mmd_path.write_text(mmd_str, encoding="utf-8")
        print(f"Mermaid file generated: {mmd_path}")
