from __future__ import annotations

import inspect

"""pipeline_printer.py – quick visualisation helpers for the Agent‑Framework.

Usage
-----
>>> printer = PipelinePrinter()
>>> printer.print_ascii(scheduler.agents)
>>> dot_str = printer.to_dot(scheduler.agents)
>>> Path('pipeline.dot').write_text(dot_str)

This module is **self‑contained** – importable without other framework internals
except the public attributes we rely on (`_output_port`, `_input_port`,
`_input_ports`, `output_port`).
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, TYPE_CHECKING, Union, Optional
import subprocess

if TYPE_CHECKING:
    from AgentFramework.ConnectedAgent import ConnectedAgent
    from AgentFramework.ToolPort import ToolPort


class PipelinePrinter:
    """Generate human‑readable or GraphViz views of a list of agents."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        is_ortho: bool = False,
        direction: str = "TB",
        fillcolor: Optional[str] = None,
        *,
        # NEW ----------------------------------------------------------------
        show_schemas: bool = False,
        schema_fillcolor: Optional[str] = None,
    ) -> None:
        """Initialise a PipelinePrinter instance.

        Args:
            is_ortho: If *True*, use orthogonal (right‑angled) edges.
            direction: Rank direction for GraphViz ("TB", "LR", …).
            fillcolor: Base colour for *agent* nodes.  If *None*, no fill.
            show_schemas: If *True*, show *input/output* schema types as
              intermediate *note* nodes between agents.
            schema_fillcolor: Fill colour for schema *note* nodes.  If *None*,
              the schema nodes inherit GraphViz defaults (white).
        """
        self.is_ortho = is_ortho
        self.direction = direction
        # Agent‑node colours -------------------------------------------------
        self.fillcolor = fillcolor
        if fillcolor:
            # Guard against double‑prefix – avoid "lightlightblue" etc.
            self.node_fill = (
                fillcolor if fillcolor.startswith("light") else f"light{fillcolor}"
            )
            self.node_border = fillcolor
            self.node_fontcolor = "black"
        else:
            self.node_fill = None
            self.node_border = None
            self.node_fontcolor = None
        # Schema‑node settings ----------------------------------------------
        self.show_schemas = show_schemas
        self.schema_fillcolor = schema_fillcolor

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def print_ascii(self, agents: Iterable["ConnectedAgent"]) -> None:
        """Pretty‑print an ASCII tree of connections to *stdout*."""
        edges = self._collect_edges(list(agents))
        self._ascii(edges)

    def to_dot(self, agents: Iterable["ConnectedAgent"]) -> str:
        """Return a GraphViz DOT string describing the pipeline graph."""
        edges = self._collect_edges(list(agents))
        return self._dot(edges)

    def to_png(self, agents: Iterable["ConnectedAgent"], png_path: Union[str, Path]):
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_edges(
        self, agents: List["ConnectedAgent"]
    ) -> Dict[str, List[Tuple[str, str, Optional[type], Optional[type]]]]:
        """Walk output‑port connections and return a mapping of *edges*.

        Returned structure
        ------------------
        {src_label: [(tgt_label, port_suffix, output_schema, input_schema), …]}

        This richer edge description allows downstream rendering to show the
        *intermediate* data schemas when ``self.show_schemas`` is enabled.
        """
        # Give each agent a short unique label like "CounterAgent#2"
        names: Dict["ConnectedAgent", str] = {}
        cls_counter: Dict[str, int] = defaultdict(int)
        for ag in agents:
            uuid = ag.uuid
            if uuid == "default":
                uuid = ""
            else:
                uuid = f"[{uuid}]"
            cls = ag.__class__.__name__
            cls_counter[cls] += 1
            names[ag] = f"{cls}#{cls_counter[cls]} {uuid}"

        edges: Dict[str, List[Tuple[str, str, Optional[type], Optional[type]]]] = defaultdict(list)

        for ag in agents:
            # Gather the schema *once* per *agent* – if not present use *None*
            out_schema = getattr(ag, "output_schema", None)

            for out_port in self._iter_output_ports(ag):
                for target_port, _tr, _cond, (_src, tgt_agent) in out_port.connections:
                    src_label = names[ag]
                    tgt_label = names.get(tgt_agent, tgt_agent.__class__.__name__)
                    tgt_port_suffix = self._find_port_name(tgt_agent, target_port)
                    in_schema = getattr(tgt_agent, "input_schema", None)

                    edges[src_label].append(
                        (tgt_label, tgt_port_suffix, out_schema, in_schema)
                    )
        return edges

    # ----------------------------- ascii -------------------------------

    def _ascii(
        self, edges: Dict[str, List[Tuple[str, str, Optional[type], Optional[type]]]]
    ) -> None:
        """Simple ASCII tree.  Ignores schema details for brevity."""
        for src, targets in edges.items():
            print(src)
            for i, item in enumerate(targets):
                tgt, port_suffix = item[0], item[1]
                connector = "└─▶" if i == len(targets) - 1 else "├─▶"
                print(f"  {connector} {tgt}{port_suffix}")

    # ----------------------------- dot ---------------------------------

    def _dot(
            self,
            edges: Dict[str, List[Tuple[str, str, Optional[type], Optional[type]]]],
    ) -> str:
        label_type = "xlabel" if self.is_ortho else "label"

        # ── Graph header ────────────────────────────────────────────────────────────
        lines: List[str] = [
            "digraph pipeline {",
            f"  rankdir={self.direction};",
        ]
        lines.append("  bgcolor=transparent;")
        if self.is_ortho:
            lines.append("  splines=ortho;")

        # ── Default style for “real” nodes ─────────────────────────────────────────
        if self.fillcolor:
            lines.append(
                "  node [shape=box, style=\"rounded,filled\", "
                f"fillcolor=\"{self.node_fill}\", "
                f"color=\"{self.node_border}\", "
                f"fontcolor=\"{self.node_fontcolor}\"];"
            )
        else:
            lines.append("  node [shape=box, style=rounded];")

        # ── Edges & optional schema nodes ──────────────────────────────────────────
        for src, targets in edges.items():
            for tgt, port_suffix, out_schema, in_schema in targets:
                if self.show_schemas:
                    # ----- build / reuse schema node --------------------------------
                    safe_port = port_suffix or "@"
                    schema_node_id = f"{src}_{tgt}_{safe_port}_schema"

                    if out_schema == in_schema:
                        label = out_schema.__name__ if out_schema else "?"
                    else:
                        out_label = out_schema.__name__ if out_schema else "?"
                        in_label = in_schema.__name__ if in_schema else "?"
                        label = f"{out_label} \u2192\\n\u2192 {in_label}"


                    # ---- attribute list (all commas correct) ----------------------
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

                    lines.append(f'  "{schema_node_id}" [{", ".join(attrs)}];')

                    if out_schema == in_schema:
                        c_color = 'seagreen'
                    else:
                        c_color = 'blue'

                    # ---- two orange edges through the schema node -----------------
                    lines.append(f'  "{src}" -> "{schema_node_id}" [color={c_color}, arrowsize=0.75];')
                    lines.append(f'  "{schema_node_id}" -> "{tgt}" [color={c_color}, arrowsize=0.75];')

                else:
                    # ----- plain edge (back‑compat path) ---------------------------
                    label = port_suffix[1:] if port_suffix else ""
                    edge_attr = f' [{label_type}="{label}"]' if label else ""
                    lines.append(f'  "{src}" -> "{tgt}"{edge_attr};')

        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_output_ports(agent: "ConnectedAgent"):
        """Yield all *distinct* output ToolPort instances for a given agent."""
        seen = set()
        # Standard single‑output agent -----------------------------------
        if hasattr(agent, "_output_port") and agent._output_port is not None:
            seen.add(agent._output_port)
            yield agent._output_port
        # Alternative attribute name -------------------------------------
        if (
            hasattr(agent, "output_port")
            and agent.output_port is not None
            and agent.output_port not in seen
        ):
            seen.add(agent.output_port)
            yield agent.output_port
        # Potential multiple outputs -------------------------------------
        if hasattr(agent, "_output_ports"):
            for p in getattr(agent, "_output_ports").values():
                if p not in seen:
                    seen.add(p)
                    yield p

    @staticmethod
    def _find_port_name(agent: "ConnectedAgent", port_obj: "ToolPort") -> str:
        """Return '@<input_name>' if *agent* is multi‑port and *port_obj* matches."""
        if hasattr(agent, "_input_ports"):
            for name, p in agent._input_ports.items():
                if p is port_obj:
                    return f"@{name}"
        return ""
