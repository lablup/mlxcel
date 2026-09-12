# Screenshot renderer provenance

The `linux` baselines target Ubuntu 24.04 with Chromium 153.0.8010.12 (Playwright 1.63.0, revision 1243), DejaVu Sans regular/bold for Latin text, and WenQuanYi Zen Hei for Korean fallback. They do not target every distribution's `system-ui` fallback. Production CSS continues to use the native system font stack; no test font is bundled or injected.

Install the same font packages after `playwright install --with-deps chromium`, then run the normal browser suite without `--update-snapshots`:

```sh
sudo apt-get install -y --no-install-recommends fonts-dejavu-core=2.37-8 fonts-dejavu-extra=2.37-8 fonts-wqy-zenhei=0.9.45-8
fc-cache -f
fc-match system-ui
fc-match 'system-ui:weight=bold'
MLXCEL_WEBUI_FONT_DIAGNOSTICS=1 pnpm --dir webui run browser
```

The expected fontconfig matches are `DejaVu Sans` / `Book` and `DejaVu Sans` / `Bold`. The opt-in diagnostic test also records actual Chromium platform fonts, rather than assuming that `fc-match` alone proves browser rendering. The CI WebUI job pins Ubuntu 24.04 and these font packages; project Node 26.5.1 and pnpm 11.18.0 remain unchanged.

## Baseline correction provenance

All 12 Linux images below were captured by [hosted run 34714466511](https://github.com/lablup/mlxcel/actions/runs/34714466511/job/103609073941) at PR head `1e6a6b0689c8cff2d76a3c509bf6b5d8d37a62be` (artifact 10304422126). Each actual image received visual review before adoption. The unchanged approved source digest is `c4df0326df35998f03e315ab1382867150bb20b3ad516aebe184e722cbf20748`.

The prior Linux baselines came from the stock Playwright Noble container, which lacks DejaVu and resolves even Latin `system-ui` text to WenQuanYi Regular. Both amd64 and arm64 stock containers exhibit that font-set difference; CPU architecture alone is not the cause. [Diagnostic hosted run 34717701357](https://github.com/lablup/mlxcel/actions/runs/34717701357/job/103617826248) confirmed DejaVu regular/bold through Chromium CDP.

A native arm64 reproduction using `mcr.microsoft.com/playwright:v1.63.0-noble` (manifest digest `sha256:eff16c30e6f3f4af0a03fa4b706120d5e9b0891c344a27d64559aff5900a4a27`) plus the pinned fonts passed all 20 tests against these hosted images: 12 strict screenshots, seven behavioral cases, and the font diagnostic. It served the unchanged committed bundle and used the container's Node 24.20.0; this establishes renderer compatibility, not canonical build-toolchain equivalence. An attempted local amd64 run under QEMU failed to launch Chromium GPU processes before rendering and is not a test pass.

No screenshot thresholds, geometry, appearance fixtures, production fonts, assets, or browser rendering flags were changed. Darwin baselines are unchanged. Native Safari/VoiceOver/browser-zoom manual acceptance remains separate; the 200% text-scale image is not native browser zoom evidence.

| File | SHA-256 |
|---|---|
| `1024-tinted-gallery-data-inspector.png` | `d495fd8d26c02f5d51255c08f7da741c97dec9099cd2cdc88c982e3b5e8d294c` |
| `1440-dark-gallery-controls.png` | `c9be84d374149a297a1a6970692f435a007d4b3f30835dcf1aa4e53796e344cc` |
| `1440-dark-gallery-data.png` | `04709f3244d92d9bfa42cacf04e7a71ec543b892a90993881a273232b07b4cac` |
| `1440-highcontrast-gallery-states.png` | `1717c0dd1ff41a6cb486ef0489ddfe0bec421926819f9c4d0107cbf9171df9b4` |
| `1440-light-gallery-controls.png` | `00585e6670bf66b1bb947ba2a749501dcc39f8d9a23789d3b66427ee29dee37e` |
| `1440-light-gallery-data.png` | `d482a388ac7733f9e62feb12b9fa601000ab25357e2d259586343a818983dee1` |
| `1440-light-product-login.png` | `17e42a8a721cdda0eb16c50da319380b4696447ce9208e4e682c93431ad4891f` |
| `1440-light-product-signed-in.png` | `81991105e8cb0c988ffcc1b649ff4f45d56e0650fe1bfaf7c0b2dac9373b4ccc` |
| `390-dark-opaque-textscale200-gallery-controls.png` | `b4d552fed75d6af42e25a651fb110f992c933b19191e68a862514c23f9a78a08` |
| `390-dark-product-login.png` | `c5526ec17fa6d6a8da67666098bddf0ab05e812d554171370f596f9714be7c0c` |
| `390-dark-product-signed-in.png` | `e86838ae6afd7b827a8da373fc3a4e732d4c89c7627433e6b4b42bf3ece89971` |
| `390-opaque-gallery-controls-drawer-cjk.png` | `25af05a349715cb2fae649ecfa0069539f21916e4081a3975d6ff62b3526d6cc` |
