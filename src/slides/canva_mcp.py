"""Generate alternative slide design versions via the Canva MCP server.

Spawns ``mcp-remote`` as a subprocess to handle the full MCP protocol
(including OAuth 2.1 PKCE browser login) and communicates via stdio
using JSON-RPC 2.0.  No Canva API keys are needed — authentication is
handled entirely by the MCP server on first use (opens a browser).

Requires Node.js (>= 18) and npx available on PATH.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import queue
from typing import Any

logger = logging.getLogger(__name__)

MCP_REMOTE_URL = "https://mcp.canva.com/mcp"


# ── MCP stdio client ─────────────────────────────────────────────────────────


class CanvaMcpClient:
    """Lightweight MCP client that talks to mcp-remote over stdio."""

    def __init__(self) -> None:
        npx = shutil.which("npx")
        if not npx:
            raise RuntimeError(
                "npx not found on PATH. Install Node.js (>= 18) to use Canva MCP."
            )

        self._id = 0
        self._pending: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()

        self._proc = subprocess.Popen(
            [npx, "-y", "mcp-remote@latest", MCP_REMOTE_URL],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

        self._stderr_lines: list[str] = []

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._err_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._err_reader.start()

    # ── Protocol I/O ──────────────────────────────────────────────────────

    def _read_stderr(self) -> None:
        """Capture mcp-remote stderr for debugging."""
        assert self._proc.stderr is not None
        for raw_line in self._proc.stderr:
            line = raw_line.strip()
            if line:
                self._stderr_lines.append(line)
                logger.debug("mcp-remote stderr: %s", line)

    def get_stderr(self) -> str:
        """Return all captured stderr output."""
        return "\n".join(self._stderr_lines)

    def _read_loop(self) -> None:
        """Read JSON-RPC messages from stdout in a background thread."""
        assert self._proc.stdout is not None
        for raw_line in self._proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Non-JSON from mcp-remote: %s", line[:200])
                continue

            msg_id = msg.get("id")
            if msg_id is not None:
                with self._lock:
                    q = self._pending.get(msg_id)
                if q:
                    q.put(msg)
            # Notifications (no id) are silently ignored

    def _send(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: int = 120,
    ) -> dict:
        """Send a JSON-RPC request and wait for the response."""
        self._id += 1
        msg_id = self._id
        payload = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        resp_queue: queue.Queue[dict] = queue.Queue()
        with self._lock:
            self._pending[msg_id] = resp_queue

        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

        try:
            return resp_queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(
                f"Canva MCP: no response for '{method}' after {timeout}s"
            )
        finally:
            with self._lock:
                self._pending.pop(msg_id, None)

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

    # ── MCP lifecycle ─────────────────────────────────────────────────────

    def initialize(self) -> dict:
        """Perform the MCP initialize handshake."""
        resp = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "posting-app", "version": "1.0.0"},
        })
        self._notify("notifications/initialized")

        result = resp.get("result", {})
        logger.info(
            "MCP initialized: server=%s, version=%s",
            result.get("serverInfo", {}).get("name", "?"),
            result.get("protocolVersion", "?"),
        )
        return result

    def list_tools(self) -> list[dict]:
        """Discover available tools on the server."""
        resp = self._send("tools/list")
        return resp.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict:
        """Invoke an MCP tool and return the result."""
        resp = self._send("tools/call", {
            "name": name,
            "arguments": arguments or {},
        })
        if "error" in resp:
            err = resp["error"]
            raise RuntimeError(
                f"MCP tool '{name}' failed ({err.get('code', '?')}): "
                f"{err.get('message', 'unknown error')}"
            )
        return resp.get("result", {})

    def close(self) -> None:
        """Shut down the mcp-remote subprocess."""
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()

    def __enter__(self) -> "CanvaMcpClient":
        self.initialize()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_text_content(result: dict) -> str:
    """Pull the text out of an MCP tool result's content blocks."""
    for block in result.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


def _parse_json_content(result: dict) -> Any:
    """Try to parse the text content of an MCP result as JSON."""
    text = _extract_text_content(result)
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text or result


# ── High-level: generate alternative slides ──────────────────────────────────


def generate_alternative_slides(
    slides: list[dict],
    topic: str,
    aspect_ratio: str = "9:16",
    colors: dict | None = None,
    num_alternatives: int = 2,
) -> list[dict]:
    """Generate alternative Canva design versions of the given slides via MCP.

    Spawns mcp-remote, connects to mcp.canva.com, and uses the
    ``create_design`` tool to generate multiple style variations.
    On first run the user will be prompted to log in via a browser.

    Args:
        slides: List of slide dicts with "title", "body", and optional "footer".
        topic: The topic / headline for the slides.
        aspect_ratio: "9:16" (vertical) or "16:9" (landscape).
        colors: Dict with "background", "accent", etc.
        num_alternatives: How many alternative versions to generate (2-4).

    Returns:
        List of dicts, each with:
            - "version": version label
            - "style": style description
            - "design_id": Canva design ID
            - "edit_url": Canva editor link
            - "export_url": PNG download URL (or "" if export failed)
    """
    colors = colors or {}
    bg = colors.get("background", "#0D1117")
    accent = colors.get("accent", "#58A6FF")

    slides_text = "\n".join(
        f"Slide {i + 1}: Title: \"{s['title']}\" | Body: \"{s['body']}\" | "
        f"Footer: \"{s.get('footer', '')}\""
        for i, s in enumerate(slides)
    )

    dims = "1080x1920" if aspect_ratio == "9:16" else "1920x1080"
    base_context = (
        f"{len(slides)}-page finance carousel for TikTok/Instagram. "
        f"Dimensions: {dims}px. "
        f"Topic: {topic}. "
        f"Content for each slide:\n{slides_text}"
    )

    style_variations = [
        {
            "label": "Alternative A",
            "style": "Clean Minimal",
            "prompt": (
                f"Create a clean, minimal {base_context} "
                f"Dark background ({bg}), accent ({accent}). "
                f"Large bold sans-serif headlines, generous whitespace, "
                f"subtle gradients. No clutter."
            ),
        },
        {
            "label": "Alternative B",
            "style": "Bold & Vibrant",
            "prompt": (
                f"Create a bold, high-contrast {base_context} "
                f"Dark background ({bg}), neon accent ({accent}). "
                f"Impactful typography with color highlights on key numbers. "
                f"Geometric shapes and strong visual hierarchy."
            ),
        },
        {
            "label": "Alternative C",
            "style": "Editorial",
            "prompt": (
                f"Create a sophisticated editorial-style {base_context} "
                f"Dark background ({bg}), muted accent ({accent}). "
                f"Mix of serif and sans-serif fonts, thin dividers, "
                f"newspaper/magazine layout feel."
            ),
        },
        {
            "label": "Alternative D",
            "style": "Data-Forward",
            "prompt": (
                f"Create a data-visualization-focused {base_context} "
                f"Dark background ({bg}), accent ({accent}). "
                f"Emphasize numbers with large counters, mini charts, "
                f"progress indicators. Clean tech aesthetic."
            ),
        },
    ]

    alternatives: list[dict] = []

    client = CanvaMcpClient()
    try:
        client.initialize()

        # Discover available tools
        tools = client.list_tools()
        tool_names = [t.get("name", "") for t in tools]
        logger.info("Canva MCP tools available: %s", tool_names)

        if not tool_names:
            return [{
                "version": "Error",
                "style": "",
                "design_id": "",
                "edit_url": "",
                "export_url": "",
                "error": (
                    "MCP connected but no tools found. Check your Canva login.\n"
                    f"stderr: {client.get_stderr()}"
                ),
            }]

        for variation in style_variations[:num_alternatives]:
            try:
                alt = _create_one_alternative(
                    client=client,
                    tool_names=tool_names,
                    prompt=variation["prompt"],
                    title=f"{topic} — {variation['label']}",
                )
                alternatives.append({
                    "version": variation["label"],
                    "style": variation["style"],
                    "tools_available": tool_names,
                    **alt,
                })
            except Exception as exc:
                logger.error("Failed to create %s: %s", variation["label"], exc)
                alternatives.append({
                    "version": variation["label"],
                    "style": variation["style"],
                    "design_id": "",
                    "edit_url": "",
                    "export_url": "",
                    "error": str(exc),
                    "tools_available": tool_names,
                })
    except Exception as exc:
        stderr = client.get_stderr()
        raise RuntimeError(
            f"Canva MCP connection failed: {exc}\nmcp-remote output: {stderr}"
        ) from exc
    finally:
        client.close()

    return alternatives


def _find_urls_recursive(obj: Any, urls: dict[str, str] | None = None) -> dict[str, str]:
    """Walk a nested dict/list and collect anything that looks like a URL or ID."""
    if urls is None:
        urls = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                if "url" in k.lower() or "link" in k.lower():
                    urls[k] = v
                elif k.lower() in ("id", "design_id"):
                    urls[k] = v
            else:
                _find_urls_recursive(v, urls)
    elif isinstance(obj, list):
        for item in obj:
            _find_urls_recursive(item, urls)
    return urls


def _create_one_alternative(
    client: CanvaMcpClient,
    tool_names: list[str],
    prompt: str,
    title: str,
) -> dict:
    """Create a single alternative design via MCP tool calls."""
    # Call create_design
    create_result = client.call_tool("create_design", {
        "title": title,
        "description": prompt,
    })
    logger.info("create_design raw result: %s", json.dumps(create_result)[:500])

    parsed = _parse_json_content(create_result)

    # Try to extract URLs and IDs from the response, regardless of structure
    design_id = ""
    edit_url = ""

    if isinstance(parsed, dict):
        # Direct field access
        design_id = (
            parsed.get("id", "")
            or parsed.get("design_id", "")
            or parsed.get("designId", "")
        )
        edit_url = (
            parsed.get("edit_url", "")
            or parsed.get("editUrl", "")
            or parsed.get("url", "")
            or parsed.get("urls", {}).get("edit_url", "")
            or parsed.get("urls", {}).get("editUrl", "")
        )
        # Recursive search as last resort
        if not edit_url:
            found = _find_urls_recursive(parsed)
            edit_url = found.get("edit_url", found.get("editUrl", found.get("url", "")))
            if not design_id:
                design_id = found.get("id", found.get("design_id", ""))
    elif isinstance(parsed, str):
        if parsed.startswith("http"):
            edit_url = parsed

    # Try to export as PNG
    export_url = ""
    if design_id and "export_design" in tool_names:
        try:
            export_result = client.call_tool("export_design", {
                "design_id": design_id,
                "format": "png",
            })
            export_parsed = _parse_json_content(export_result)
            if isinstance(export_parsed, dict):
                export_url = (
                    export_parsed.get("url", "")
                    or export_parsed.get("download_url", "")
                    or export_parsed.get("downloadUrl", "")
                )
                if not export_url:
                    found = _find_urls_recursive(export_parsed)
                    export_url = next(
                        (v for v in found.values() if isinstance(v, str) and v.startswith("http")),
                        "",
                    )
            elif isinstance(export_parsed, str) and export_parsed.startswith("http"):
                export_url = export_parsed
        except Exception as exc:
            logger.warning("Export failed for '%s': %s", title, exc)

    return {
        "design_id": design_id,
        "edit_url": edit_url,
        "export_url": export_url,
        "raw_response": _extract_text_content(create_result) or str(create_result),
    }
