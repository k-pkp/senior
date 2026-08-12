import { chromium } from "playwright";

const OUT = process.env.SHOT_DIR || "/tmp/shots";
const URL = "http://localhost:3111";

const browser = await chromium.launch({
  args: [
    "--no-sandbox",
    // three.js needs real GL; software rasterisation is what makes WebGL work
    // in a headless container.
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
  ],
});
const page = await browser.newPage({ viewport: { width: 1440, height: 950 } });

const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));

async function shot(name) {
  await page.screenshot({ path: `${OUT}/${name}.png` });
  console.log("shot", name);
}

await page.goto(URL, { waitUntil: "networkidle" });
await page.waitForSelector("text=Volume from photographs");
await shot("1-samples");

// Result screen — the 3D centrepiece.
await page.click("text=Open result");
await page.waitForSelector("text=Measured volume");
await page.waitForTimeout(3500); // PLY load + first WebGL frames
await shot("2-result");

// With the reference cube alongside.
await page.click("text=With reference cube");
await page.waitForTimeout(2500);
await shot("3-result-scene");

// Review screen.
await page.click("text=Review");
await page.waitForSelector("text=Confirm cut");
await page.waitForTimeout(3500);
await shot("4-review");

// Drag a slider to prove the live split updates.
const slider = page.locator('input[type="range"]').first();
await slider.evaluate((el) => {
  el.value = "18";
  el.dispatchEvent(new Event("change", { bubbles: true }));
});
await page.waitForTimeout(1200);
await shot("5-review-moved");

// Dark theme.
await page.click('button[aria-label="Instrument"]');
await page.waitForTimeout(900);
await shot("6-instrument-theme");

console.log(errors.length ? "CONSOLE ERRORS:\n" + errors.join("\n") : "no console errors");
await browser.close();
