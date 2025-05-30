from __future__ import annotations

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
pipeline_printer.py – quick visualisation helpers for the Agent-Framework.
"""

if TYPE_CHECKING:
    from AgentFramework.core.ConnectedAgent import ConnectedAgent
    from AgentFramework.core.ToolPort import ToolPort


class PipelinePrinter:
    """Generate human-readable or GraphViz views of a list of agents."""

    # ------------------------------------------------------------------ construction
    def __init__(
        self,
        is_ortho: bool = False,
        direction: str = "TB",
        fillcolor: Optional[str] = None,
        entry_exit_fillcolor: Optional[str] = None,
        *,
        show_schemas: bool = False,
        schema_fillcolor: Optional[str] = None,
    ) -> None:
        self.is_ortho = is_ortho
        self.direction = direction

        # agent-node colours -------------------------------------------------
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

        if entry_exit_fillcolor:
            self.entry_exit_node_fill = (
                entry_exit_fillcolor if entry_exit_fillcolor.startswith("light")
                else f"light{entry_exit_fillcolor}"
            )
            self.entry_exit_node_border = fillcolor
        else:
            self.entry_exit_node_fill = self.node_fill
            self.entry_exit_node_border = self.node_border

        # schema-node settings ----------------------------------------------
        self.show_schemas = show_schemas
        self.schema_fillcolor = schema_fillcolor

        # edge-label settings -----------------------------------------------
        self.edge_label_fontsize = 6
        self.output_label_color = "blue"
        self.input_label_color = "green"

    # ------------------------------------------------------------------ public helpers
    def print_ascii(self, agents: Iterable["ConnectedAgent"]) -> None:
        edges = self._collect_edges(list(agents))
        self._ascii(edges)

    def to_dot(self, agents: Iterable["ConnectedAgent"]) -> str:
        edges = self._collect_edges(list(agents))
        return self._dot(edges)

    def to_mermaid(self, agents: Iterable["ConnectedAgent"]) -> str:
        edges = self._collect_edges(list(agents))
        return self._mermaid(edges)

    def save_as_png(self, agents: Iterable["ConnectedAgent"], png_path: Union[str, Path]):
        png_path = Path(png_path)
        dot_str = self.to_dot(agents)
        subprocess.run(["dot", "-Tpng", "-Gdpi=300", "-o", str(png_path)], input=dot_str.encode("utf-8"),check=True,)
        print(f"PNG generated: {png_path}")

    # ------------------------------------------------------------------ internal logic
    def _collect_edges(
        self, agents: List["ConnectedAgent"]
    ) -> Dict[str, List[Tuple[str, str, str, Optional[type], Optional[type]]]]:
        agents = self._flatten(agents)
        names: Dict["ConnectedAgent", str] = {}
        cls_counter: Dict[str, int] = defaultdict(int)
        for ag in agents:
            uid = "" if ag.uuid == "default" else f"[{ag.uuid}]"
            cls = ag.__class__.__name__
            cls_counter[cls] += 1
            names[ag] = f"{cls}#{cls_counter[cls]} {uid}"
            # Append the agent’s UUID, but only if it isn’t the sentinel
            # value "default", so we don’t clutter every single label.
            uuid_part = getattr(ag, "uuid", "default")
            if uuid_part and uuid_part != "default":
                label = f"{cls}#{cls_counter[cls]}\n[{uuid_part}]"
            else:
                label = f"{cls}#{cls_counter[cls]}\n"
            names[ag] = label


        edges: Dict[str, List[Tuple[str, str, str, Optional[type], Optional[type]]]] = defaultdict(list)

        for ag in agents:
            out_name_by_port: Dict["ToolPort", str] = {}
            for attr in ("_output_ports", "output_ports"):
                if hasattr(ag, attr):
                    mapping = getattr(ag, attr) or {}
                    for key, port in mapping.items():
                        out_name_by_port[port] = self._port_key_to_label(key)

            for out_port in self._iter_output_ports(ag):
                out_schema = self._schema_for_port(ag, out_port)
                src_suffix = (
                    f"@{out_name_by_port[out_port]}" if out_port in out_name_by_port else ""
                )

                for tgt_port, _tr1, _tr2, _cond, (_src, tgt_agent) in out_port.connections:
                    src_lbl = names[ag]
                    tgt_lbl = names.get(tgt_agent, tgt_agent.__class__.__name__)
                    tgt_suffix = self._find_port_name(tgt_agent, tgt_port)
                    in_schema = getattr(tgt_agent, "input_schema", None)

                    edges[src_lbl].append(
                        (tgt_lbl, tgt_suffix, src_suffix, out_schema, in_schema)
                    )
        return edges

    # ------------------------------------------------------------------ helpers
    # pipeline_printer.py  (add inside the class – e.g. right above _collect_edges)

    @staticmethod
    def _flatten(items) -> List["ConnectedAgent"]:
        """
        Recursively expand AgentSchedulers so we end up with only ConnectedAgents.
        Avoids changing external callers – they can still pass mixed lists.
        """
        from AgentFramework.core.AgentScheduler import AgentScheduler
        flat: List["ConnectedAgent"] = []
        for obj in items:
            if isinstance(obj, AgentScheduler):
                flat.extend(PipelinePrinter._flatten(obj.agents))
            else:
                flat.append(obj)
        return flat


    @staticmethod
    def _get_schema_alias(schema_cls: Type["BaseModel"]) -> str:
        import re, inspect
        doc = inspect.cleandoc(schema_cls.__doc__ or "")
        m = re.search(r"^Alias:\s*(.+)$", doc, re.MULTILINE)
        if m:
            return m.group(1)
        name = schema_cls.__name__
        for suf in ("InputSchema", "OutputSchema", "Schema", "Model"):
            if name.endswith(suf):
                return name[:-len(suf)]
        return name

    @staticmethod
    def _iter_output_ports(agent: "ConnectedAgent"):
        seen = set()
        for attr in ("_output_port", "output_port"):
            if hasattr(agent, attr):
                port = getattr(agent, attr)
                if port and port not in seen:
                    seen.add(port)
                    yield port
        for attr in ("_output_ports", "output_ports"):
            if hasattr(agent, attr):
                mapping = getattr(agent, attr) or {}
                for port in mapping.values():
                    if port not in seen:
                        seen.add(port)
                        yield port

    def _classify_nodes(self, edges):
        incoming = defaultdict(int)
        outgoing = defaultdict(int)
        nodes = set()
        for src, targets in edges.items():
            nodes.add(src)
            outgoing[src] += len(targets)
            for tgt, *_ in targets:
                nodes.add(tgt)
                incoming[tgt] += 1
        entries = {n for n in nodes if incoming[n] == 0}
        exits = {n for n in nodes if outgoing[n] == 0}
        return entries, exits, nodes

    @staticmethod
    def _schema_for_port(agent: "ConnectedAgent", port_obj: "ToolPort") -> Optional[Type]:
        for attr in ("_output_ports", "output_ports"):
            if hasattr(agent, attr):
                for schema_type, p in getattr(agent, attr).items():
                    if p is port_obj:
                        return schema_type
        if hasattr(agent, "output_schemas"):
            out_schemas = getattr(agent, "output_schemas") or []
            if len(out_schemas) == 1:
                return out_schemas[0]
        return getattr(agent, "output_schema", None)

    @staticmethod
    def _port_key_to_label(key) -> str:
        import inspect
        return (PipelinePrinter._get_schema_alias(key)
                if inspect.isclass(key) else str(key))

    @staticmethod
    def _find_port_name(agent: "ConnectedAgent", port_obj: "ToolPort") -> str:
        if hasattr(agent, "_input_ports"):
            for name, p in agent._input_ports.items():
                if p is port_obj:
                    return f"@{PipelinePrinter._port_key_to_label(name)}"
        return ""

    # ------------------------------------------------------------------ renderers
    def _ascii(self, edges):
        for src, targets in edges.items():
            clean_src = " ".join(src.splitlines()).strip()
            if not targets:
                continue  # skip if no outgoing edges
            print(clean_src)
            for i, (tgt, tgt_suf, src_suf, *_rest) in enumerate(targets):
                connector = "└─▶" if i == len(targets) - 1 else "├─▶"
                clean_tgt = " ".join(tgt.splitlines()).strip()

                # fallback for empty suffixes
                clean_src_suf = src_suf.replace("@", "").strip() if src_suf else "-"
                clean_tgt_suf = tgt_suf.replace("@", "").strip() if tgt_suf else "-"

                # Compose label with fallback parts
                msg = f"{clean_src_suf} → {clean_tgt_suf}"
                label = f"[{msg}]"
                print(f"  {connector} {clean_tgt}: {label}")


    def _dot(self, edges) -> str:
        entries, exits, _ = self._classify_nodes(edges)

        lines: List[str] = [
            "digraph pipeline {",
            f"  rankdir={self.direction};",
            "  bgcolor=transparent;",
        ]
        if self.is_ortho:
            lines.append("  splines=ortho;")

        if self.fillcolor:
            lines.append(
                '  node [shape=box, style="rounded,filled", '
                f'fillcolor="{self.node_fill}", color="{self.node_border}", '
                f'fontcolor="{self.node_fontcolor}"];'
            )
        else:
            lines.append("  node [shape=box, style=rounded];")

        for node in entries | exits:
            fill = self.entry_exit_node_fill or self.node_fill or "white"
            bord = self.entry_exit_node_border or self.node_border or "black"
            lines.append(f'  "{node}" [fillcolor="{fill}", color="{bord}"];')

        # no more mid_counter since we don’t split edges any more

        for src, targets in edges.items():
            for tgt, tgt_suf, src_suf, out_schema, in_schema in targets:
                if self.show_schemas:
                    # (unchanged schema‐note logic…)
                    safe = tgt_suf or "@"
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
                    lines.append(
                        f'  "{src}" -> "{note_id}" [color={colour}, arrowsize=0.75];'
                    )
                    lines.append(
                        f'  "{note_id}" -> "{tgt}" [color={colour}, arrowsize=0.75];'
                    )
                else:
                    # ———— single edge with inline xlabel ————
                    #arrow = "\n↓\n"
                    #arrow = " →\n→ "
                    arrow = " →\n→ " if self.direction in ("TB", "BT") else "\n↓\n"
                    if src_suf or tgt_suf:
                        # Prefer schema class names for clarity (fallback to port suffixes)
                        if out_schema and in_schema:
                            if out_schema == in_schema:
                                label = out_schema.__name__
                            else:
                                label = f"{out_schema.__name__}{arrow}{in_schema.__name__}"
                        elif src_suf or tgt_suf:
                            label_parts = [p[1:] for p in (src_suf, tgt_suf) if p]
                            if len(label_parts) == 2 and label_parts[0] != label_parts[1]:
                                label = f"{label_parts[0]}{arrow}{label_parts[1]}"
                            else:
                                label = label_parts[0] if label_parts else ""
                        else:
                            label = ""

                        attrs = [
                            f'xlabel="{label}"',
                            f'fontsize={self.edge_label_fontsize}',
                            'fontcolor="blue4"',
                            'labeldistance=2.5',
                            'labelangle=0',
                        ]
                        lines.append(f'  "{src}" -> "{tgt}" [{", ".join(attrs)}];')
                    else:
                        lines.append(f'  "{src}" -> "{tgt}";')

        lines.append("}")
        return "\n".join(lines)


    # Mermaid renderer unchanged (centred labels) ---------------------------
    def _mermaid(self, edges) -> str:
        entries, exits, _ = self._classify_nodes(edges)
        orient_map = {"TB": "TD", "BT": "BT", "LR": "LR", "RL": "RL"}
        orientation = orient_map.get(self.direction, "TD")

        def _safe(node: str) -> str:
            import re
            return re.sub(r"\W+", "_", node)

        lines: List[str] = [f"graph {orientation}"]

        for src, targets in edges.items():
            src_id = _safe(src)
            lines.append(f'    {src_id}["{src}"]')
            if src in entries:
                lines.append(f"    class {src_id} entry_exit;")

            for tgt, tgt_suf, src_suf, out_schema, in_schema in targets:
                tgt_id = _safe(tgt)
                if f'{tgt_id}["' not in " ".join(lines):
                    lines.append(f'    {tgt_id}["{tgt}"]')
                    if tgt in exits:
                        lines.append(f"    class {tgt_id} entry_exit;")

                lab_parts = [p[1:] for p in (src_suf, tgt_suf) if p]
                elabel = f' |{" → ".join(lab_parts)}| ' if lab_parts else " "

                if self.show_schemas:
                    safe = tgt_suf or "@"
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
        if self.show_schemas:
            fill = self.schema_fillcolor or "#FFE"
            lines.insert(
                2,
                f'classDef schema fill:{fill},stroke:#333,stroke-width:1px,font-size:10px;',
            )
        fill = self.entry_exit_node_fill or "#EDF2FF"
        border = self.entry_exit_node_border or "#547DDE"
        lines.append(f'classDef entry_exit fill:{fill},stroke:{border},stroke-width:2px;')

        return "\n".join(lines)

    # ------------------------------------------------------------------ helpers
    def save_as_dot(
        self,
        agents: Iterable["ConnectedAgent"],
        dot_path: Union[str, Path],
    ) -> None:
        Path(dot_path).write_text(self.to_dot(agents), encoding="utf-8")
        print(f"DOT file generated: {dot_path}")

    def save_as_mermaid(
        self,
        agents: Iterable["ConnectedAgent"],
        mmd_path: Union[str, Path],
    ) -> None:
        Path(mmd_path).write_text(self.to_mermaid(agents), encoding="utf-8")
        print(f"Mermaid file generated: {mmd_path}")
