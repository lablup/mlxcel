# mlxcel download site

The marketing and download site for mlxcel. It is a [Next.js](https://nextjs.org/)
app built as a **static export** (`output: "export"` and `trailingSlash: true` in
`next.config.ts`), so the deployable artifact is the `out/` directory, not a
running server. The site is bilingual: `app/[lang]` is generated for `en` and
`ko` (`generateStaticParams`), with copy in `dictionaries/en.ts` and
`dictionaries/ko.ts`.

The build and deploy steps live in the repository `Makefile` and
`scripts/deploy_webpage.sh`. Run `make` commands from the **repository root**,
not from this directory.

## Prerequisites

| Tool | Needed for | Notes |
| --- | --- | --- |
| Node.js | every build | A version supported by the pinned Next.js (`next` in `package.json`). `package.json` declares no `engines` field. |
| [pnpm](https://pnpm.io/) | every build | `pnpm-lock.yaml` is lockfile v9. `pnpm-workspace.yaml` leaves the `unrs-resolver` build script declined (`allowBuilds`), so `pnpm install` does not stop on it. |
| [uv](https://docs.astral.sh/uv/) and `zensical` | `make webpage-build` only | Builds the bundled manual. See [The bundled manual](#the-bundled-manual). |
| Network access | every build | The home page fetches the latest `lablup/mlxcel-releases` release from the GitHub API **at build time** (`lib/release.ts`). Set `GITHUB_TOKEN` to authenticate and avoid rate limits; it is optional. If the request fails, the page is still built and the downloads section falls back to a link to the releases page. |

## Entry points

`package.json` only defines the plain Next.js scripts (`dev`, `build`, `start`,
`lint`). Use the Makefile targets for anything that is part of the real flow:

| Command (repo root) | What it does |
| --- | --- |
| `make webpage-dev` | `pnpm install`, then `pnpm dev` in this directory. Does not build the manual. |
| `make webpage-build` | Gated by `docs-guard`. Rebuilds the manual into `public/{en,ko}/manual`, then runs `pnpm install && pnpm build`. Output lands in `webpage/site/out/`. |
| `make webpage-deploy` | Runs `scripts/deploy_webpage.sh`. Not gated by `docs-guard`. Warns, but does not stop, if `public/{en,ko}/manual` is missing. |
| `make clean` | Removes `webpage/site/.next` and `webpage/site/out`, but also runs `cargo clean` for the whole Rust workspace. It does not remove `public/{en,ko}/manual`. |

`pnpm lint` (from this directory) runs ESLint.

## The bundled manual

`make webpage-build` runs, from the repository root:

```bash
uv run zensical build -f mkdocs.yml    -d webpage/site/public/en/manual
uv run zensical build -f mkdocs.ko.yml -d webpage/site/public/ko/manual
```

before `pnpm build`, after removing any previous `public/{en,ko}/manual`.

The MkDocs manual sources (`docs/en`, `docs/ko`, `docs/shared`,
`docs/requirements.txt`, `docs/scripts`) are **not part of this repository**.
`docs-guard` fails `webpage-build` with an explanation when `docs/en` is absent,
and nothing in this checkout provisions `zensical` for `uv run`. Building the
manual therefore needs the separate documentation tree. See `docs/README.md` for
the split.

`public/{en,ko}/manual` are build outputs, but they are **not** listed in
`.gitignore` (only `webpage/site/out/`, `.next/` and `node_modules/` are), so do
not commit them by accident.

## Deploying

`make webpage-deploy` runs `scripts/deploy_webpage.sh`, which:

1. runs `pnpm install` and `pnpm run build` in this directory,
2. adds `.nojekyll` to `out/` and **replaces `out/index.html`** with a small
   page that redirects to `./ko` or `./en` by browser language,
3. creates a fresh git repository in `out/`, commits, and **force-pushes** it to
   the `gh-pages` branch of `git@github.com:lablup/mlxcel-releases.git`,
   overwriting that branch's previous content.

Because the script rebuilds with plain `pnpm run build`, the manual is **not**
rebuilt by `webpage-deploy`. It ships whatever `public/{en,ko}/manual` already
holds, which is why the Makefile warns when those directories are missing. Run
`make webpage-build` first. Deploying needs push access to `lablup/mlxcel-releases`
over SSH.

## Layout

| Path | Contents |
| --- | --- |
| `app/(root)` | `/` , redirects to `/en`. |
| `app/[lang]` | The localized site (`en`, `ko`). |
| `components/` | Page sections and UI building blocks. |
| `dictionaries/` | Per-language copy (`en.ts`, `ko.ts`). |
| `lib/` | `release.ts` (latest-release fetch), `dictionary.ts`, `utils.ts`. |
| `public/` | Static assets (`brands/`) and the generated `{en,ko}/manual`. |
