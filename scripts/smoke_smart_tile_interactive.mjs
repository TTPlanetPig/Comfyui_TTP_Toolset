import fs from "node:fs";

const source = fs.readFileSync("web/js/ttp_smart_tile_interactive.js", "utf8");

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function sliceBetween(startText, endText) {
    const start = source.indexOf(startText);
    assert(start >= 0, `missing start marker: ${startText}`);
    const end = source.indexOf(endText, start);
    assert(end > start, `missing end marker: ${endText}`);
    return source.slice(start, end);
}

const beginDrag = sliceBetween("const beginDrag = (event, tileIndex, mode, region) =>", "for (const index of renderOrder)");
const createTile = sliceBetween("stage.addEventListener(\"pointerdown\", (event) =>", "const controls = document.createElement(\"div\")");
const createTileBeforeMove = createTile.slice(0, createTile.indexOf("const apply ="));

assert(/const dragThresholdPx = 3/.test(source), "tile editor should use a small drag threshold");
assert(!/apply\(event\);/.test(beginDrag), "clicking an existing tile must not immediately apply a drag");
assert(/if \(!hasDragged && movedPx < dragThresholdPx\)/.test(beginDrag), "existing tile drag should wait until movement crosses the threshold");
assert(/renderEditor\(node\);/.test(beginDrag), "existing tile click should still re-render selection on pointerup");
assert(/pointercancel/.test(beginDrag), "existing tile drag should clean up pointercancel");
assert(/blur/.test(beginDrag), "existing tile drag should clean up window blur");
assert(!/setPointerCapture/.test(beginDrag), "existing tile drag should not rely on pointer capture");
assert(!/releasePointerCapture/.test(beginDrag), "existing tile drag should not rely on pointer capture release");
assert(!/writeLayout\(node, nextTiles, newIndex\);/.test(createTileBeforeMove), "empty-stage pointerdown should not create a tile before dragging");
assert(/if \(!hasDragged && movedPx < dragThresholdPx\)/.test(createTile), "new tile creation should wait until movement crosses the threshold");

console.log("smoke_smart_tile_interactive: ok");
