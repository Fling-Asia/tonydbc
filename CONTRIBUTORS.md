### Contributing

If you are a contributor and want an [editable install](https://peps.python.org/pep-0660/), rather than in `site-packages`:

1. Clone the tonydbc repo from GitHub
2. Navigate to the tonydbc root directory
3. Run `python -m pip install -e .` to install the repo as an editable install.

### Publishing

To publish a new version of tonydbc to PyPI:

1. Get an API token at PyPI
2. Save the token to `C:\Users\<username>\.pyprc`:

```
[pypi]
  username = __token__
  password = <password>
```

3. Navigate to the tonydbc root directory
4. Run these commands:

```bash
python -m pip install build twine bumpver
bumpver update --no-push -n --patch
python -m build
twine check dist/*
twine upload -r testpypi dist/*
twine upload dist/*
```
