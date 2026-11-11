# robotmsgs-website

robotmsgs.com: a landing page and a docs browser for the [robot_msgs](https://github.com/zacharyyamaoka/robot_msgs) Python package.
Every page is generated from the installed package (`build/model.py` introspects it); nothing is written per type.

```bash
pnpm run build    # dist/
pnpm run test     # rebuilds, then checks pages, links and sizes
pnpm run serve    # http://127.0.0.1:4492
pnpm run deploy   # build + test + wrangler deploy (Cloudflare static-assets Worker "robotmsgs")
```
