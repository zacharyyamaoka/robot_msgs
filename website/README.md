# robotmsgs-website

robotmsgs.com (lives in robot_msgs/website since 2026-11-11): a landing page and a docs browser for the [robot_msgs](https://github.com/zacharyyamaoka/robot_msgs) Python package.
Every page is generated from the installed package (`build/model.py` introspects it); nothing is written per type.

CI (`.github/workflows/site.yml`) builds and tests it on every push, and deploys it after each PyPI release (needs the `CLOUDFLARE_API_TOKEN` repo secret). By hand:

```bash
pnpm run build    # dist/
pnpm run test     # rebuilds, then checks pages, links and sizes
pnpm run serve    # http://127.0.0.1:4492
pnpm run deploy   # build + test + wrangler deploy (Cloudflare static-assets Worker "robotmsgs")
```
