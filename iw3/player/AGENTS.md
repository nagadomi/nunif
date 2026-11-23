# Agent Guidelines (iw3/player)

Guidelines specific to frontend development, JavaScript tooling, and player utilities under `iw3/player`. For general Python environment standards, code formatting (Ruff, mypy), and repository-wide conventions, refer to the root `AGENTS.md`.

## Execution

Always execute tools and modules from the project root:

- **GUI**:
  ```bash
  python -m iw3.player.gui
  ```
- **CLI**:
  ```bash
  python -m iw3.player [options]
  # or
  python -m iw3.player.cli [options]
  ```
- **Tests**:
  ```bash
  pytest iw3/player/tests
  ```

## Frontend & JavaScript Development

### Architecture Overview
- Application JavaScript code is located under `public/js/` and is served as standard browser ES modules. Editing files in `public/js/` requires no build step—simply reload the browser to see updates.
- Third-party vendor dependencies (e.g., `three`, `@pmndrs/uikit`) are defined in `package.json` and bundled into `public/lib/vendor.bundle.js` via `src/vendor.js`.

### Node.js & npm Tooling
Run npm commands within `iw3/player`:

```bash
cd iw3/player
npm install
```

- **Linting**:
  ```bash
  npm run lint
  ```
  Runs ESLint across `public/js/`. Ensure all errors (e.g., undefined variables or syntax issues) are fixed.
- **Vendor Bundle Build**:
  ```bash
  npm run dev      # Build vendor bundle with sourcemap and DEBUG_MODE=true
  npm run build    # Production minified bundle with DEBUG_MODE=false
  npm run watch    # Watch mode for vendor bundle rebuilds
  ```
- **Draco Decoder Setup**:
  ```bash
  npm run setup-draco
  ```
  Copies Draco decoder binaries from `node_modules/three/` into `public/lib/draco/`. This is automatically triggered during `npm run build` and `npm run dev`.

### Frontend Guidelines
- **Logging**:
  Avoid cluttering production frontend code with direct `console.log`, `console.warn`, or `console.error` calls. Prefer using the `DebugLog` instance (e.g., `this.uiManager.debugLogInstance.log(...)`) so logs can be displayed on the in-VR HUD during headset debugging.
- **Persistence & Storage**:
  Use `storage.js` (IndexedDB with Web Crypto API encryption) for user settings, playback states, and file history. Do not use plain `localStorage` for per-file configurations.
- **Screen Architecture**:
  All stereo projection screens inherit from `StereoBaseScreen` (`StereoScreen` for flat/curved screens, `StereoDomeScreen` for VR180 mesh, `StereoDomeXRScreen` for WebXR Layers). Keep shader logic separated from geometry and UI controls.

## Utilities & Assets

- **Asset Packaging for Releases**:
  Create downloadable asset zip archives (`iw3_player_{name}_{date}.zip` under `release/`):
  ```bash
  python -m iw3.player.build_dist
  # or specify an output directory:
  python -m iw3.player.build_dist -o /path/to/output
  ```
- **Asset Download**:
  Download and install pre-packaged assets (fonts, default LUTs, skybox environments, example media):
  ```bash
  python -m iw3.player.download_assets
  ```
  > [!WARNING]
  > Running this command may overwrite locally modified or actively developed assets under `public/lib/`, `public/fonts/`, `public/environments/`, `public/lut/`, and `example/` with downloaded release archives. Use caution when working on local assets. To suppress automatic downloads on server startup, create an empty file at `version/ignore`.
- **Icon Color Conversion**:
  Update or batch-convert SVG icon colors under `public/icons/`:
  ```bash
  python -m iw3.player.convert_icon_color "#cccccc"
  ```
- **HaldCLUT to 3D LUT Conversion**:
  Convert HaldCLUT images (PNG) to Adobe `.cube` format:
  ```bash
  python -m iw3.player.haldclut2cube -i input.png -o iw3/player/public/lut/
  ```
