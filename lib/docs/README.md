# mGPLVM Documentation

## Building documentation locally

1. Install dependencies
```sh
pip install sphinx shpinx_rtd_theme sphinx_autodoc_typehints
```

2. Build

```sh
cd docs
make api
make html
```

3. Open `build/html/index.html` in your browser.
