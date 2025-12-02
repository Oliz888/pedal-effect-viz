// components/realtime.js

let startTime = null;
let playing = false;

function sendTime() {
    if (!playing) return;
    const now = Date.now();
    const elapsed = (now - startTime) / 1000.0;  // seconds
    window.parent.postMessage(
        {
            isStreamlitMessage: true,
            type: "realtime_tick",
            elapsed: elapsed
        },
        "*"
    );
}

window.addEventListener("message", (event) => {
    if (event.data === "start_timer") {
        startTime = Date.now();
        playing = true;

        // ~8 fps updates (fast enough but not CPU heavy)
        setInterval(sendTime, 120);
    }
});
