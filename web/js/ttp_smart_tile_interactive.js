import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "TTP_Smart_Tile_Interactive_Crop_Experimental";
const LOOP_SOURCE_NAME = "TTP_Smart_Tile_Loop_Source_Experimental";
const MAX_TILES = 64;
const MAX_GRID_AXIS = 8;
const dragThresholdPx = 3;
const snapGuideThresholdPx = 10;
const STORAGE_PREFIX = "ttp_smart_tile_interactive_layout";

function widgetByName(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function setWidgetVisible(widget, visible) {
    if (!widget) {
        return;
    }
    widget.hidden = !visible;
    widget.computeSize = visible ? undefined : () => [0, -4];
}

function scheduleCanvas(node) {
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function resizeNode(node) {
    if (node.computeSize) {
        const size = node.computeSize();
        node.size = [Math.max(node.size?.[0] ?? 520, size[0]), size[1]];
    }
    scheduleCanvas(node);
}

function clampNumber(value, min, max) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
        return min;
    }
    return Math.max(min, Math.min(max, number));
}

function roundRatio(value) {
    return Number(clampNumber(value, 0, 1).toFixed(6));
}

function imageSourceSize(node) {
    if (!node) {
        return null;
    }
    const image = node.ttpSmartTileImage;
    if (image?.width > 0 && image?.height > 0) {
        return { width: image.width, height: image.height };
    }
    return null;
}

function snapStep(node) {
    const value = Number(widgetByName(node, "round_to")?.value ?? 8);
    return Math.max(1, Math.min(128, Math.round(value)));
}

function minTileRatio(node, axis) {
    const size = imageSourceSize(node);
    const dimension = axis === "x" ? size?.width : size?.height;
    if (Number(dimension) > 0) {
        return Math.max(1 / Number(dimension), Math.min(0.2, 24 / Number(dimension)));
    }
    return 0.04;
}

function snapRatioToImage(node, ratio, axis) {
    const size = imageSourceSize(node);
    const dimension = axis === "x" ? size?.width : size?.height;
    if (!dimension || dimension <= 1) {
        return roundRatio(ratio);
    }
    const step = snapStep(node);
    if (ratio <= 0) {
        return 0;
    }
    if (ratio >= 1) {
        return 1;
    }
    const px = Math.round((ratio * dimension) / step) * step;
    return roundRatio(px / dimension);
}

function normalizeAxis(node, start, end, axis) {
    let first = clampNumber(start, 0, 1);
    let second = clampNumber(end, 0, 1);
    if (second < first) {
        [first, second] = [second, first];
    }
    first = snapRatioToImage(node, first, axis);
    second = snapRatioToImage(node, second, axis);
    if (second < first) {
        [first, second] = [second, first];
    }
    const minSpan = minTileRatio(node, axis);
    if (second - first < minSpan) {
        const center = (first + second) / 2;
        first = center - minSpan / 2;
        second = center + minSpan / 2;
        if (first < 0) {
            second -= first;
            first = 0;
        }
        if (second > 1) {
            first -= second - 1;
            second = 1;
        }
    }
    return [roundRatio(first), roundRatio(second)];
}

function normalizeTile(node, raw) {
    const metadata = {};
    for (const key of ["name", "source", "label", "score", "layer", "object_id", "occlusion_priority", "priority", "importance", "pad", "blend", "object_mask"]) {
        if (raw && raw[key] !== undefined) {
            metadata[key] = raw[key];
        }
    }
    const x0 = Number(raw?.x0 ?? raw?.x ?? 0);
    const y0 = Number(raw?.y0 ?? raw?.y ?? 0);
    const x1 = Number(raw?.x1 ?? (Number(raw?.x ?? 0) + Number(raw?.w ?? raw?.width ?? 0.5)));
    const y1 = Number(raw?.y1 ?? (Number(raw?.y ?? 0) + Number(raw?.h ?? raw?.height ?? 0.5)));
    const [left, right] = normalizeAxis(node, x0, x1, "x");
    const [top, bottom] = normalizeAxis(node, y0, y1, "y");
    return { ...metadata, x0: left, y0: top, x1: right, y1: bottom };
}

function defaultTiles(node) {
    return gridTiles(node, 2, 2);
}

function gridTiles(node, columns, rows, bounds = { x0: 0, y0: 0, x1: 1, y1: 1 }) {
    const cols = Math.max(1, Math.min(MAX_GRID_AXIS, Math.round(Number(columns) || 1)));
    const rowCount = Math.max(1, Math.min(MAX_GRID_AXIS, Math.round(Number(rows) || 1)));
    const box = normalizeTile(node, bounds);
    const width = box.x1 - box.x0;
    const height = box.y1 - box.y0;
    const tiles = [];
    for (let row = 0; row < rowCount; row += 1) {
        for (let col = 0; col < cols; col += 1) {
            tiles.push(normalizeTile(node, {
                x0: box.x0 + width * (col / cols),
                y0: box.y0 + height * (row / rowCount),
                x1: box.x0 + width * ((col + 1) / cols),
                y1: box.y0 + height * ((row + 1) / rowCount),
            }));
        }
    }
    return tiles;
}

function sourceStorageKey(node) {
    const size = imageSourceSize(node);
    const sizeKey = size ? `${Math.round(size.width)}x${Math.round(size.height)}` : "unknown";
    return `${STORAGE_PREFIX}:${node.id ?? node.title ?? "node"}:${sizeKey}`;
}

function readStoredLayout(node) {
    try {
        const raw = globalThis.localStorage?.getItem(sourceStorageKey(node));
        if (!raw) {
            return null;
        }
        const value = JSON.parse(raw);
        if (!Array.isArray(value?.tiles) || !value.tiles.length) {
            return null;
        }
        return value.tiles.slice(0, MAX_TILES).map((tile) => normalizeTile(node, tile));
    } catch {
        return null;
    }
}

function storeLayout(node, tiles) {
    try {
        const size = imageSourceSize(node);
        globalThis.localStorage?.setItem(
            sourceStorageKey(node),
            JSON.stringify({
                source_size: size ? [Math.round(size.width), Math.round(size.height)] : null,
                tiles,
            })
        );
    } catch {}
}

function parseLayout(node) {
    const widget = widgetByName(node, "layout_json");
    let value = {};
    try {
        value = JSON.parse(widget?.value || "{}");
    } catch {
        value = {};
    }
    if (Array.isArray(value?.tiles) && value.tiles.length) {
        return value.tiles.slice(0, MAX_TILES).map((tile) => normalizeTile(node, tile));
    }
    const stored = readStoredLayout(node);
    return stored ?? defaultTiles(node);
}

function layoutDefaults(node) {
    return {
        pad: Math.max(0, Math.round(Number(widgetByName(node, "default_pad")?.value ?? 128))),
        blend: Math.max(0, Math.round(Number(widgetByName(node, "default_blend")?.value ?? 64))),
        priority: 50,
        importance: 1.0,
    };
}

function writeLayout(node, tiles, selectedIndex = 0) {
    const normalizedTiles = (Array.isArray(tiles) && tiles.length ? tiles : defaultTiles(node))
        .slice(0, MAX_TILES)
        .map((tile, index) => ({
            name: `tile_${index + 1}`,
            ...normalizeTile(node, tile),
        }));
    const size = imageSourceSize(node);
    const layout = {
        version: 1,
        type: "ttp_smart_tile_interactive_layout",
        source_size: size ? [Math.round(size.width), Math.round(size.height)] : null,
        defaults: layoutDefaults(node),
        tiles: normalizedTiles,
    };
    const serialized = JSON.stringify(layout);
    const widget = widgetByName(node, "layout_json");
    if (widget && widget.value !== serialized) {
        widget.value = serialized;
    }
    node.ttpSmartTileLayout = layout;
    node.ttpSmartTileSelectedIndex = Math.max(0, Math.min(normalizedTiles.length - 1, Number(selectedIndex ?? 0)));
    storeLayout(node, normalizedTiles);
}

function uniqueSorted(values) {
    const sorted = values
        .map((value) => roundRatio(value))
        .filter((value) => Number.isFinite(value))
        .sort((a, b) => a - b);
    const result = [];
    for (const value of sorted) {
        if (!result.length || Math.abs(value - result[result.length - 1]) > 1e-6) {
            result.push(value);
        }
    }
    return result;
}

function analyzeCoverage(tiles) {
    const normalized = (Array.isArray(tiles) ? tiles : []).map((tile) => normalizeTile(null, tile));
    const xEdges = uniqueSorted([0, 1, ...normalized.flatMap((tile) => [tile.x0, tile.x1])]);
    const yEdges = uniqueSorted([0, 1, ...normalized.flatMap((tile) => [tile.y0, tile.y1])]);
    const rowRuns = [];
    let uncoveredArea = 0;
    for (let yi = 0; yi < yEdges.length - 1; yi += 1) {
        const y0 = yEdges[yi];
        const y1 = yEdges[yi + 1];
        let run = null;
        for (let xi = 0; xi < xEdges.length - 1; xi += 1) {
            const x0 = xEdges[xi];
            const x1 = xEdges[xi + 1];
            const cx = (x0 + x1) / 2;
            const cy = (y0 + y1) / 2;
            const covered = normalized.some(
                (tile) => cx >= tile.x0 - 1e-6 && cx <= tile.x1 + 1e-6 && cy >= tile.y0 - 1e-6 && cy <= tile.y1 + 1e-6
            );
            if (covered) {
                if (run) {
                    rowRuns.push(run);
                    run = null;
                }
                continue;
            }
            uncoveredArea += (x1 - x0) * (y1 - y0);
            if (run && Math.abs(run.x1 - x0) <= 1e-6) {
                run.x1 = x1;
            } else {
                if (run) {
                    rowRuns.push(run);
                }
                run = { x0, y0, x1, y1 };
            }
        }
        if (run) {
            rowRuns.push(run);
        }
    }
    const gaps = [];
    const activeBySpan = new Map();
    for (const run of rowRuns.sort((a, b) => a.x0 - b.x0 || a.x1 - b.x1 || a.y0 - b.y0 || a.y1 - b.y1)) {
        const key = `${run.x0}:${run.x1}`;
        const previous = activeBySpan.get(key);
        if (previous && Math.abs(previous.y1 - run.y0) <= 1e-6) {
            previous.y1 = run.y1;
        } else {
            const gap = { ...run };
            gaps.push(gap);
            activeBySpan.set(key, gap);
        }
    }
    gaps.sort((a, b) => (b.x1 - b.x0) * (b.y1 - b.y0) - (a.x1 - a.x0) * (a.y1 - a.y0));
    return {
        gaps: gaps.map((gap) => ({
            x0: roundRatio(gap.x0),
            y0: roundRatio(gap.y0),
            x1: roundRatio(gap.x1),
            y1: roundRatio(gap.y1),
        })),
        uncoveredArea,
    };
}

function fillTileGaps(node, tiles, maxTiles = MAX_TILES) {
    const nextTiles = (Array.isArray(tiles) ? tiles : []).map((tile) => ({ ...tile }));
    let coverageState = analyzeCoverage(nextTiles);
    let added = 0;
    while (coverageState.gaps.length && nextTiles.length < maxTiles) {
        const previousArea = coverageState.uncoveredArea;
        const gapTile = normalizeTile(node, coverageState.gaps[0]);
        nextTiles.push({
            ...gapTile,
            name: `auto_gap_${added + 1}`,
            source: "auto_gap",
            label: "background gap",
            priority: 5,
            importance: 0.35,
            layer: 0,
            object_id: 0,
            occlusion_priority: 0,
        });
        added += 1;
        coverageState = analyzeCoverage(nextTiles);
        if (coverageState.uncoveredArea >= previousArea - 1e-8) {
            nextTiles.pop();
            added -= 1;
            break;
        }
    }
    return {
        tiles: nextTiles,
        coverage: coverageState,
        added,
    };
}

function tilePixelRect(node, tile) {
    const size = imageSourceSize(node);
    if (!size) {
        return null;
    }
    const x0 = Math.round(tile.x0 * size.width);
    const y0 = Math.round(tile.y0 * size.height);
    const x1 = Math.round(tile.x1 * size.width);
    const y1 = Math.round(tile.y1 * size.height);
    return { x0, y0, width: Math.max(1, x1 - x0), height: Math.max(1, y1 - y0) };
}

function createButton(label, onClick, disabled = false) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.disabled = disabled;
    button.style.cssText = [
        "font:inherit",
        "font-size:12px",
        "color:" + (disabled ? "rgba(15,23,42,.45)" : "#0f172a"),
        "background:#e2e8f0",
        "border:1px solid #cbd5e1",
        "border-radius:4px",
        "padding:4px 7px",
        "cursor:" + (disabled ? "default" : "pointer"),
    ].join(";");
    if (!disabled) {
        button.onclick = onClick;
    }
    return button;
}

function createNumberInput(value, min, max) {
    const input = document.createElement("input");
    input.type = "number";
    input.min = String(min);
    input.max = String(max);
    input.step = "1";
    input.value = String(value);
    input.style.cssText = [
        "box-sizing:border-box",
        "width:54px",
        "font:inherit",
        "font-size:12px",
        "color:#e5e7eb",
        "background:#172033",
        "border:1px solid rgba(226,232,240,.35)",
        "border-radius:4px",
        "padding:4px 5px",
    ].join(";");
    return input;
}

function parseAnnotatedImageName(value) {
    let name = String(value ?? "").trim();
    let type = "input";
    for (const candidate of ["input", "output", "temp"]) {
        const suffix = ` [${candidate}]`;
        if (name.endsWith(suffix)) {
            name = name.slice(0, -suffix.length).trim();
            type = candidate;
            break;
        }
    }
    name = name.replace(/\\/g, "/");
    const slash = name.lastIndexOf("/");
    const subfolder = slash >= 0 ? name.slice(0, slash) : "";
    const filename = slash >= 0 ? name.slice(slash + 1) : name;
    return { filename, subfolder, type };
}

function inputImageUrl(value) {
    const { filename, subfolder, type } = parseAnnotatedImageName(value);
    if (!filename) {
        return "";
    }
    const params = new URLSearchParams({ filename, subfolder, type });
    return api.apiURL(`/view?${params.toString()}`);
}

async function loadSelectedInputImage(node, force = false) {
    const value = String(widgetByName(node, "image")?.value ?? "");
    if (!value) {
        node.ttpSmartTileImage = null;
        return;
    }
    if (!force && value === node.ttpSmartTileLoadedImageName) {
        return;
    }
    node.ttpSmartTileLoadedImageName = value;
    const url = inputImageUrl(value);
    try {
        await new Promise((resolve, reject) => {
            const image = new Image();
            image.onload = () => {
                node.ttpSmartTileImage = {
                    url,
                    filename: value,
                    width: image.naturalWidth,
                    height: image.naturalHeight,
                    element: image,
                };
                const layout = readStoredLayout(node) ?? parseLayout(node);
                writeLayout(node, layout, 0);
                resolve();
            };
            image.onerror = reject;
            image.src = url;
        });
        node.ttpSmartTileStatus = "";
    } catch {
        node.ttpSmartTileStatus = "Selected input image could not be previewed.";
    }
    renderEditor(node);
}

async function inferSmartTileLayout(node) {
    const mode = String(widgetByName(node, "auto_detect_mode")?.value ?? "none");
    if (mode === "none") {
        node.ttpSmartTileStatus = "Set auto detect mode to sam3.1 or qwenvl3 before inference.";
        renderEditor(node);
        return;
    }
    const requestWidget = widgetByName(node, "auto_detect_request");
    if (requestWidget) {
        requestWidget.value = Number(requestWidget.value ?? 0) + 1;
    }
    node.ttpSmartTileStatus = `Queued ${mode} inference...`;
    renderEditor(node);
    try {
        await app.queuePrompt(0);
    } catch (error) {
        node.ttpSmartTileStatus = error?.message || "Could not queue inference.";
        renderEditor(node);
    }
}

function applyInferenceResult(detail) {
    if (!detail?.node_id || !app.graph) {
        return;
    }
    const node = app.graph.getNodeById?.(Number(detail.node_id)) ?? app.graph._nodes_by_id?.[detail.node_id];
    if (!node || (node.comfyClass ?? node.type) !== NODE_NAME) {
        return;
    }
    const statusParts = [];
    if (detail.message) {
        statusParts.push(detail.message);
    } else {
        statusParts.push(detail.ok ? "Inference finished." : "Inference failed.");
    }
    if (detail.layout_json) {
        try {
            const layout = JSON.parse(detail.layout_json);
            if (Array.isArray(layout?.tiles) && layout.tiles.length) {
                const filled = fillTileGaps(node, layout.tiles);
                writeLayout(node, filled.tiles, 0);
                if (filled.added > 0) {
                    statusParts.push(`Auto Tile added ${filled.added} gap tile(s).`);
                }
                if (filled.coverage.gaps.length > 0) {
                    statusParts.push(`${filled.coverage.gaps.length} gap(s) remain.`);
                }
            }
        } catch (error) {
            node.ttpSmartTileStatus = error?.message || "Inference returned invalid layout.";
            renderEditor(node);
            return;
        }
    }
    node.ttpSmartTileStatus = statusParts.join(" ");
    renderEditor(node);
}

function bumpWidget(node, name) {
    const widget = widgetByName(node, name);
    if (widget) {
        widget.value = Number(widget.value ?? 0) + 1;
    }
}

function setLoopSourceStatus(node, status) {
    node.ttpSmartTileLoopStatus = status;
    const widget = widgetByName(node, "loop_status");
    if (widget) {
        widget.value = status;
    }
    scheduleCanvas(node);
}

async function queueSmartTileLoop(node, restart = false) {
    if (!node) {
        return;
    }
    if (restart) {
        bumpWidget(node, "restart_request");
    }
    bumpWidget(node, "loop_request");
    node.ttpSmartTileLoopActive = true;
    setLoopSourceStatus(node, restart ? "Starting tile loop..." : "Queueing next tile...");
    try {
        await app.queuePrompt(0);
    } catch (error) {
        node.ttpSmartTileLoopActive = false;
        setLoopSourceStatus(node, error?.message || "Could not queue tile loop.");
    }
}

function applyLoopEvent(detail) {
    if (!detail?.source_node_id || !app.graph) {
        return;
    }
    const node = app.graph.getNodeById?.(Number(detail.source_node_id)) ?? app.graph._nodes_by_id?.[detail.source_node_id];
    if (!node || (node.comfyClass ?? node.type) !== LOOP_SOURCE_NAME || !node.ttpSmartTileLoopActive) {
        return;
    }
    const count = Number(detail.count ?? 0);
    const index = Number(detail.index ?? 0);
    if (detail.done) {
        node.ttpSmartTileLoopActive = false;
        setLoopSourceStatus(node, detail.message || `Done ${count}/${count}`);
        return;
    }
    setLoopSourceStatus(node, detail.message || `Next tile ${index + 1}/${count}`);
    queueSmartTileLoop(node, false);
}

function editorHeight(node) {
    const size = imageSourceSize(node);
    const aspect = size ? Math.max(0.25, Math.min(4, size.height / size.width)) : 0.72;
    const width = Math.max(360, Math.round(Number(node.size?.[0] ?? 560) - 20));
    return Math.max(360, Math.round(width * aspect) + 156);
}

function ensureEditor(node) {
    if (node.ttpSmartTileContainer) {
        return node.ttpSmartTileContainer;
    }
    setWidgetVisible(widgetByName(node, "layout_json"), false);

    const container = document.createElement("div");
    container.className = "ttp-smart-tile-editor";
    container.tabIndex = 0;
    container.style.cssText = [
        "box-sizing:border-box",
        "width:100%",
        "padding:8px",
        "border:1px solid rgba(140,148,160,.35)",
        "border-radius:6px",
        "background:#101827",
        "color:#e5e7eb",
        "font:12px/1.35 system-ui,-apple-system,BlinkMacSystemFont,sans-serif",
        "outline:none",
    ].join(";");
    if (node.addDOMWidget) {
        const widget = node.addDOMWidget("smart_tile_editor", "Smart Tile Editor", container, {
            serialize: false,
            hideOnZoom: false,
        });
        widget.computeSize = () => [Math.max(520, node.size?.[0] ?? 520), editorHeight(node)];
    } else {
        node.addWidget("text", "smart_tile_editor", "Smart Tile Editor", () => {}, { serialize: false });
    }
    node.ttpSmartTileContainer = container;
    return container;
}

function renderEditor(node) {
    const container = ensureEditor(node);
    const currentTiles = node.ttpSmartTileLayout?.tiles ?? parseLayout(node);
    const tiles = currentTiles.map((tile) => normalizeTile(node, tile));
    const selectedIndex = Math.max(0, Math.min(tiles.length - 1, Number(node.ttpSmartTileSelectedIndex ?? 0)));
    writeLayout(node, tiles, selectedIndex);

    const sourceSize = imageSourceSize(node);
    const selectedRect = tilePixelRect(node, tiles[selectedIndex]);
    const coverage = analyzeCoverage(tiles);
    container.replaceChildren();

    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;";
    const title = document.createElement("div");
    title.textContent = "Smart Tile Image Editor";
    title.style.cssText = "font-weight:700;font-size:13px;";
    const value = document.createElement("div");
    value.textContent = sourceSize
        ? `${sourceSize.width}x${sourceSize.height} / ${tiles.length} tiles`
        : `${tiles.length} tiles`;
    value.style.cssText = "opacity:.78;text-align:right;";
    header.append(title, value);

    const stage = document.createElement("div");
    stage.style.cssText = [
        "position:relative",
        "width:100%",
        "aspect-ratio:" + (sourceSize ? `${sourceSize.width} / ${sourceSize.height}` : "16 / 9"),
        "min-height:220px",
        "max-height:720px",
        "border:1px solid rgba(226,232,240,.45)",
        "border-radius:6px",
        "overflow:hidden",
        "background:#172033",
        "touch-action:none",
        "cursor:crosshair",
    ].join(";");

    if (node.ttpSmartTileImage?.element) {
        const image = node.ttpSmartTileImage.element.cloneNode();
        image.draggable = false;
        image.style.cssText = [
            "position:absolute",
            "inset:0",
            "width:100%",
            "height:100%",
            "object-fit:fill",
            "user-select:none",
            "pointer-events:none",
        ].join(";");
        stage.append(image);
    } else {
        const empty = document.createElement("div");
        empty.textContent = "Select or upload an image with the official image widget above.";
        empty.style.cssText = [
            "position:absolute",
            "inset:0",
            "display:flex",
            "align-items:center",
            "justify-content:center",
            "color:rgba(226,232,240,.72)",
            "pointer-events:none",
        ].join(";");
        stage.append(empty);
    }

    for (const gap of coverage.gaps) {
        const overlay = document.createElement("div");
        overlay.title = "Uncovered area";
        overlay.style.cssText = [
            "position:absolute",
            "box-sizing:border-box",
            `left:${gap.x0 * 100}%`,
            `top:${gap.y0 * 100}%`,
            `width:${Math.max(0.1, (gap.x1 - gap.x0) * 100)}%`,
            `height:${Math.max(0.1, (gap.y1 - gap.y0) * 100)}%`,
            "background:rgba(239,68,68,.28)",
            "border:1px dashed rgba(254,202,202,.85)",
            "pointer-events:none",
        ].join(";");
        stage.append(overlay);
    }

    const pointFromEvent = (event) => {
        const rect = stage.getBoundingClientRect();
        return {
            x: clampNumber((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1),
            y: clampNumber((event.clientY - rect.top) / Math.max(1, rect.height), 0, 1),
        };
    };
    const tileContainsPoint = (tile, point) => (
        point.x >= tile.x0 - 1e-6 &&
        point.x <= tile.x1 + 1e-6 &&
        point.y >= tile.y0 - 1e-6 &&
        point.y <= tile.y1 + 1e-6
    );
    const resizeHit = (tile, point) => {
        const rect = stage.getBoundingClientRect();
        const padX = Math.max(0.01, 18 / Math.max(1, rect.width));
        const padY = Math.max(0.01, 18 / Math.max(1, rect.height));
        return point.x >= tile.x1 - padX && point.x <= tile.x1 + padX && point.y >= tile.y1 - padY && point.y <= tile.y1 + padY;
    };
    const setRegionStyle = (element, tile) => {
        element.style.left = `${tile.x0 * 100}%`;
        element.style.top = `${tile.y0 * 100}%`;
        element.style.width = `${Math.max(0.1, (tile.x1 - tile.x0) * 100)}%`;
        element.style.height = `${Math.max(0.1, (tile.y1 - tile.y0) * 100)}%`;
    };
    const snapTargets = (axis, skipIndex) => {
        const values = [0, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1];
        for (const [tileIndex, tile] of tiles.entries()) {
            if (tileIndex === skipIndex) {
                continue;
            }
            if (axis === "x") {
                values.push(tile.x0, tile.x1, (tile.x0 + tile.x1) / 2);
            } else {
                values.push(tile.y0, tile.y1, (tile.y0 + tile.y1) / 2);
            }
        }
        return [...new Set(values.map((value) => roundRatio(value)))].sort((a, b) => a - b);
    };
    const nearestSnap = (value, axis, skipIndex) => {
        const rect = stage.getBoundingClientRect();
        const dimension = axis === "x" ? rect.width : rect.height;
        const threshold = snapGuideThresholdPx / Math.max(1, dimension);
        let best = value;
        let bestDistance = threshold;
        for (const target of snapTargets(axis, skipIndex)) {
            const distance = Math.abs(value - target);
            if (distance <= bestDistance) {
                best = target;
                bestDistance = distance;
            }
        }
        return best;
    };
    const snapMovedTile = (tile, skipIndex) => {
        const width = tile.x1 - tile.x0;
        const height = tile.y1 - tile.y0;
        const leftSnap = nearestSnap(tile.x0, "x", skipIndex);
        const rightSnap = nearestSnap(tile.x1, "x", skipIndex);
        const dx = Math.abs(leftSnap - tile.x0) <= Math.abs(rightSnap - tile.x1)
            ? leftSnap - tile.x0
            : rightSnap - tile.x1;
        const topSnap = nearestSnap(tile.y0, "y", skipIndex);
        const bottomSnap = nearestSnap(tile.y1, "y", skipIndex);
        const dy = Math.abs(topSnap - tile.y0) <= Math.abs(bottomSnap - tile.y1)
            ? topSnap - tile.y0
            : bottomSnap - tile.y1;
        return normalizeTile(node, {
            ...tile,
            x0: Math.max(0, Math.min(1 - width, tile.x0 + dx)),
            y0: Math.max(0, Math.min(1 - height, tile.y0 + dy)),
            x1: Math.max(width, Math.min(1, tile.x0 + dx + width)),
            y1: Math.max(height, Math.min(1, tile.y0 + dy + height)),
        });
    };
    const snapResizedTile = (tile, skipIndex) => normalizeTile(node, {
        ...tile,
        x1: nearestSnap(tile.x1, "x", skipIndex),
        y1: nearestSnap(tile.y1, "y", skipIndex),
    });
    const snapCreatedTile = (tile, skipIndex) => normalizeTile(node, {
        ...tile,
        x0: nearestSnap(tile.x0, "x", skipIndex),
        y0: nearestSnap(tile.y0, "y", skipIndex),
        x1: nearestSnap(tile.x1, "x", skipIndex),
        y1: nearestSnap(tile.y1, "y", skipIndex),
    });

    const renderOrder = tiles.map((_tile, index) => index).filter((index) => index !== selectedIndex);
    renderOrder.push(selectedIndex);
    const regionElements = new Map();

    const beginDrag = (event, tileIndex, mode, region) => {
        event.preventDefault();
        event.stopPropagation();
        const point = pointFromEvent(event);
        const selected = renderOrder.filter((index) => tileContainsPoint(tiles[index], point)).pop() ?? tileIndex;
        const index = Math.max(0, Math.min(tiles.length - 1, selected));
        const dragMode = resizeHit(tiles[index], point) ? "resize" : mode;
        const targetRegion = regionElements.get(index) ?? region;
        node.ttpSmartTileSelectedIndex = index;
        const startPoint = pointFromEvent(event);
        const startClientX = event.clientX;
        const startClientY = event.clientY;
        const startTiles = tiles.map((tile) => ({ ...tile }));
        const startTile = startTiles[index];
        let hasDragged = false;
        const apply = (moveEvent) => {
            moveEvent.preventDefault();
            moveEvent.stopPropagation();
            const movedPx = Math.hypot(moveEvent.clientX - startClientX, moveEvent.clientY - startClientY);
            if (!hasDragged && movedPx < dragThresholdPx) {
                return;
            }
            if (!hasDragged) {
                hasDragged = true;
                if (targetRegion) {
                    targetRegion.style.zIndex = "120";
                }
            }
            const nextPoint = pointFromEvent(moveEvent);
            const dx = nextPoint.x - startPoint.x;
            const dy = nextPoint.y - startPoint.y;
            const nextTiles = startTiles.map((tile) => ({ ...tile }));
            if (dragMode === "resize") {
                nextTiles[index] = snapResizedTile(normalizeTile(node, {
                    ...startTile,
                    x1: startTile.x1 + dx,
                    y1: startTile.y1 + dy,
                }), index);
            } else {
                const width = startTile.x1 - startTile.x0;
                const height = startTile.y1 - startTile.y0;
                const x0 = Math.max(0, Math.min(1 - width, startTile.x0 + dx));
                const y0 = Math.max(0, Math.min(1 - height, startTile.y0 + dy));
                nextTiles[index] = snapMovedTile(normalizeTile(node, { ...startTile, x0, y0, x1: x0 + width, y1: y0 + height }), index);
            }
            writeLayout(node, nextTiles, index);
            if (targetRegion) {
                setRegionStyle(targetRegion, nextTiles[index]);
            }
            scheduleCanvas(node);
        };
        const cleanup = () => {
            window.removeEventListener("pointermove", apply, true);
            window.removeEventListener("pointerup", finish, true);
            window.removeEventListener("pointercancel", finish, true);
            window.removeEventListener("blur", finish, true);
        };
        const finish = (endEvent) => {
            endEvent?.preventDefault?.();
            endEvent?.stopPropagation?.();
            cleanup();
            renderEditor(node);
        };
        window.addEventListener("pointermove", apply, true);
        window.addEventListener("pointerup", finish, true);
        window.addEventListener("pointercancel", finish, true);
        window.addEventListener("blur", finish, true);
    };

    for (const index of renderOrder) {
        const tile = tiles[index];
        const selected = index === selectedIndex;
        const region = document.createElement("div");
        region.textContent = String(index + 1);
        region.title = `Tile ${index + 1}`;
        region.style.cssText = [
            "position:absolute",
            "box-sizing:border-box",
            "display:flex",
            "align-items:center",
            "justify-content:center",
            `left:${tile.x0 * 100}%`,
            `top:${tile.y0 * 100}%`,
            `width:${Math.max(0.1, (tile.x1 - tile.x0) * 100)}%`,
            `height:${Math.max(0.1, (tile.y1 - tile.y0) * 100)}%`,
            "border:" + (selected ? "2px solid #f8fafc" : "1px solid rgba(226,232,240,.55)"),
            "background:" + (selected ? "rgba(14,165,233,.28)" : "rgba(15,23,42,.20)"),
            "color:#f8fafc",
            "font-weight:800",
            "letter-spacing:0",
            "box-shadow:" + (selected ? "0 0 0 1px rgba(15,23,42,.85),0 0 16px rgba(14,165,233,.55)" : "none"),
            "z-index:" + (selected ? "100" : String(10 + index)),
            "cursor:move",
            "user-select:none",
        ].join(";");
        regionElements.set(index, region);
        region.addEventListener("pointerdown", (event) => beginDrag(event, index, "move", region));
        const handle = document.createElement("div");
        handle.title = `Resize tile ${index + 1}`;
        handle.style.cssText = [
            "position:absolute",
            "right:0",
            "bottom:0",
            "width:14px",
            "height:14px",
            "background:#f8fafc",
            "border-left:1px solid rgba(15,23,42,.8)",
            "border-top:1px solid rgba(15,23,42,.8)",
            "cursor:nwse-resize",
        ].join(";");
        handle.addEventListener("pointerdown", (event) => beginDrag(event, index, "resize", region));
        region.append(handle);
        stage.append(region);
    }

    stage.addEventListener("pointerdown", (event) => {
        if (event.target !== stage) {
            return;
        }
        event.preventDefault();
        event.stopPropagation();
        const point = pointFromEvent(event);
        const hits = renderOrder.filter((index) => tileContainsPoint(tiles[index], point));
        if (hits.length || tiles.length >= MAX_TILES) {
            node.ttpSmartTileSelectedIndex = hits.pop() ?? selectedIndex;
            renderEditor(node);
            return;
        }
        const startPoint = point;
        const startClientX = event.clientX;
        const startClientY = event.clientY;
        const newIndex = tiles.length;
        let hasDragged = false;
        let tempRegion = null;
        const apply = (moveEvent) => {
            moveEvent.preventDefault();
            moveEvent.stopPropagation();
            const movedPx = Math.hypot(moveEvent.clientX - startClientX, moveEvent.clientY - startClientY);
            if (!hasDragged && movedPx < dragThresholdPx) {
                return;
            }
            if (!hasDragged) {
                hasDragged = true;
                tempRegion = document.createElement("div");
                tempRegion.textContent = String(newIndex + 1);
                tempRegion.style.cssText = [
                    "position:absolute",
                    "box-sizing:border-box",
                    "display:flex",
                    "align-items:center",
                    "justify-content:center",
                    "border:2px solid #f8fafc",
                    "background:rgba(14,165,233,.28)",
                    "color:#f8fafc",
                    "font-weight:800",
                    "letter-spacing:0",
                    "box-shadow:0 0 0 1px rgba(15,23,42,.85),0 0 16px rgba(14,165,233,.55)",
                    "z-index:130",
                    "pointer-events:none",
                ].join(";");
                stage.append(tempRegion);
            }
            const nextPoint = pointFromEvent(moveEvent);
            const nextTiles = [...tiles, snapCreatedTile(normalizeTile(node, {
                x0: startPoint.x,
                y0: startPoint.y,
                x1: nextPoint.x,
                y1: nextPoint.y,
            }), newIndex)];
            writeLayout(node, nextTiles, newIndex);
            setRegionStyle(tempRegion, nextTiles[newIndex]);
            scheduleCanvas(node);
        };
        const cleanup = () => {
            window.removeEventListener("pointermove", apply, true);
            window.removeEventListener("pointerup", finish, true);
            window.removeEventListener("pointercancel", finish, true);
            window.removeEventListener("blur", finish, true);
        };
        const finish = (endEvent) => {
            endEvent?.preventDefault?.();
            endEvent?.stopPropagation?.();
            cleanup();
            renderEditor(node);
        };
        window.addEventListener("pointermove", apply, true);
        window.addEventListener("pointerup", finish, true);
        window.addEventListener("pointercancel", finish, true);
        window.addEventListener("blur", finish, true);
    });

    const controls = document.createElement("div");
    controls.style.cssText = "display:flex;flex-direction:column;gap:8px;margin-top:8px;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;gap:6px;align-items:center;flex-wrap:wrap;";

    const gridControls = document.createElement("div");
    gridControls.style.cssText = "display:flex;gap:6px;align-items:center;flex-wrap:wrap;";
    const gridLabel = document.createElement("span");
    gridLabel.textContent = "Grid";
    gridLabel.style.cssText = "opacity:.78;";
    const gridColumns = createNumberInput(node.ttpSmartTileGridColumns ?? 2, 1, MAX_GRID_AXIS);
    const gridRows = createNumberInput(node.ttpSmartTileGridRows ?? 2, 1, MAX_GRID_AXIS);
    const updateGridValues = () => {
        node.ttpSmartTileGridColumns = Math.max(1, Math.min(MAX_GRID_AXIS, Math.round(Number(gridColumns.value) || 1)));
        node.ttpSmartTileGridRows = Math.max(1, Math.min(MAX_GRID_AXIS, Math.round(Number(gridRows.value) || 1)));
        gridColumns.value = String(node.ttpSmartTileGridColumns);
        gridRows.value = String(node.ttpSmartTileGridRows);
    };
    const refreshGridValues = () => {
        updateGridValues();
        renderEditor(node);
    };
    gridColumns.onchange = refreshGridValues;
    gridRows.onchange = refreshGridValues;
    updateGridValues();
    const gridCount = node.ttpSmartTileGridColumns * node.ttpSmartTileGridRows;
    const subdivideCount = tiles.length - 1 + gridCount;
    gridControls.append(
        gridLabel,
        gridColumns,
        document.createTextNode("x"),
        gridRows,
        createButton("Replace grid", () => {
            updateGridValues();
            const nextTiles = gridTiles(node, node.ttpSmartTileGridColumns, node.ttpSmartTileGridRows);
            node.ttpSmartTileStatus = "";
            writeLayout(node, nextTiles, 0);
            renderEditor(node);
        }, gridCount > MAX_TILES),
        createButton(`Grid in T${selectedIndex + 1}`, () => {
            updateGridValues();
            const selectedTile = tiles[selectedIndex] ?? { x0: 0, y0: 0, x1: 1, y1: 1 };
            const replacement = gridTiles(node, node.ttpSmartTileGridColumns, node.ttpSmartTileGridRows, selectedTile);
            if (tiles.length - 1 + replacement.length > MAX_TILES) {
                node.ttpSmartTileStatus = `Grid would create ${tiles.length - 1 + replacement.length} tiles; max is ${MAX_TILES}.`;
                renderEditor(node);
                return;
            }
            const nextTiles = [
                ...tiles.slice(0, selectedIndex),
                ...replacement,
                ...tiles.slice(selectedIndex + 1),
            ];
            node.ttpSmartTileStatus = "";
            writeLayout(node, nextTiles, selectedIndex);
            renderEditor(node);
        }, subdivideCount > MAX_TILES)
    );
    controls.append(gridControls);

    for (const [index] of tiles.entries()) {
        const button = createButton(`T${index + 1}`, () => {
            node.ttpSmartTileSelectedIndex = index;
            renderEditor(node);
        });
        if (index === selectedIndex) {
            button.style.background = "#38bdf8";
            button.style.borderColor = "#7dd3fc";
            button.style.color = "#082f49";
        }
        actions.append(button);
    }

    actions.append(
        createButton("Auto Tile", () => {
            inferSmartTileLayout(node);
        }),
        createButton("Add tile", () => {
            const base = tiles[selectedIndex] ?? { x0: 0.25, y0: 0.25, x1: 0.75, y1: 0.75 };
            const width = Math.max(0.18, Math.min(0.48, base.x1 - base.x0));
            const height = Math.max(0.18, Math.min(0.48, base.y1 - base.y0));
            const offset = Math.min(0.14, 0.03 * tiles.length);
            const x0 = Math.min(1 - width, Math.max(0, base.x0 + offset));
            const y0 = Math.min(1 - height, Math.max(0, base.y0 + offset));
            writeLayout(node, [...tiles, { x0, y0, x1: x0 + width, y1: y0 + height }], tiles.length);
            renderEditor(node);
        }, tiles.length >= MAX_TILES),
        createButton("Delete", () => {
            writeLayout(node, tiles.filter((_tile, index) => index !== selectedIndex), Math.max(0, selectedIndex - 1));
            renderEditor(node);
        }, tiles.length <= 1),
        createButton("Fill gaps", () => {
            const filled = fillTileGaps(node, tiles);
            node.ttpSmartTileStatus = filled.coverage.gaps.length
                ? `${filled.coverage.gaps.length} gap(s) remain.`
                : "";
            writeLayout(node, filled.tiles, Math.min(tiles.length, filled.tiles.length - 1));
            renderEditor(node);
        }, !coverage.gaps.length || tiles.length >= MAX_TILES),
        createButton("Reset 2x2", () => {
            writeLayout(node, defaultTiles(node), 0);
            renderEditor(node);
        })
    );
    controls.append(actions);

    const status = document.createElement("div");
    status.style.cssText = "display:flex;gap:10px;flex-wrap:wrap;opacity:.76;";
    const selectedLabel = selectedRect
        ? `T${selectedIndex + 1}: x${selectedRect.x0}, y${selectedRect.y0}, ${selectedRect.width}x${selectedRect.height}`
        : `T${selectedIndex + 1}`;
    const coverageLabel = coverage.gaps.length
        ? `${coverage.gaps.length} gap(s), ${(coverage.uncoveredArea * 100).toFixed(2)}%`
        : "covered";
    const extra = node.ttpSmartTileStatus ? ` / ${node.ttpSmartTileStatus}` : "";
    status.textContent = `${selectedLabel} / ${coverageLabel} / snap ${snapStep(node)} px / max ${MAX_TILES}${extra}`;
    controls.append(status);

    container.append(header, stage, controls);
    resizeNode(node);
}

function attachWidgetRefresh(node) {
    setWidgetVisible(widgetByName(node, "auto_detect_request"), false);

    const imageWidget = widgetByName(node, "image");
    if (imageWidget && !imageWidget.ttpSmartTileWrapped) {
        const original = imageWidget.callback;
        imageWidget.callback = function () {
            original?.apply(this, arguments);
            loadSelectedInputImage(node, true);
        };
        imageWidget.ttpSmartTileWrapped = true;
    }

    for (const name of ["default_pad", "default_blend", "round_to"]) {
        const widget = widgetByName(node, name);
        if (!widget || widget.ttpSmartTileWrapped) {
            continue;
        }
        const original = widget.callback;
        widget.callback = function () {
            original?.apply(this, arguments);
            writeLayout(node, node.ttpSmartTileLayout?.tiles ?? parseLayout(node), node.ttpSmartTileSelectedIndex ?? 0);
            renderEditor(node);
        };
        widget.ttpSmartTileWrapped = true;
    }
}

api.addEventListener("ttp-smart-tile-layout", (event) => {
    applyInferenceResult(event.detail);
});

api.addEventListener("ttp-smart-tile-loop", (event) => {
    applyLoopEvent(event.detail);
});

app.registerExtension({
    name: "ttp.smart_tile.interactive_crop",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === LOOP_SOURCE_NAME) {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                originalOnNodeCreated?.apply(this, arguments);
                setWidgetVisible(widgetByName(this, "restart_request"), false);
                setWidgetVisible(widgetByName(this, "loop_request"), false);
                if (!widgetByName(this, "process_all_tiles")) {
                    this.addWidget("button", "process_all_tiles", "Process All Tiles", () => {
                        queueSmartTileLoop(this, true);
                    }, { serialize: false });
                }
                if (!widgetByName(this, "stop_tile_loop")) {
                    this.addWidget("button", "stop_tile_loop", "Stop Tile Loop", () => {
                        this.ttpSmartTileLoopActive = false;
                        setLoopSourceStatus(this, "Stopped.");
                    }, { serialize: false });
                }
                if (!widgetByName(this, "loop_status")) {
                    this.addWidget("text", "loop_status", this.ttpSmartTileLoopStatus || "Idle.", () => {}, { serialize: false });
                }
            };
            const originalOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                originalOnConfigure?.apply(this, arguments);
                requestAnimationFrame(() => {
                    setWidgetVisible(widgetByName(this, "restart_request"), false);
                    setWidgetVisible(widgetByName(this, "loop_request"), false);
                });
            };
            return;
        }

        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);
            ensureEditor(this);
            attachWidgetRefresh(this);
            writeLayout(this, parseLayout(this), this.ttpSmartTileSelectedIndex ?? 0);
            loadSelectedInputImage(this);
            renderEditor(this);
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            originalOnConfigure?.apply(this, arguments);
            requestAnimationFrame(() => {
                ensureEditor(this);
                attachWidgetRefresh(this);
                loadSelectedInputImage(this);
                writeLayout(this, parseLayout(this), this.ttpSmartTileSelectedIndex ?? 0);
                renderEditor(this);
            });
        };
    },
});
