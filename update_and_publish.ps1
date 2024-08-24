# Update the version
bumpver update --no-push -n --patch

# Build the package
python -m build

# Check the distribution files
twine check dist/*

# Upload to PyPI (production)
twine upload dist/*

# Uncomment the following line and comment out the
# above line if you want to upload to TestPyPI instead
# twine upload -r testpypi dist/*

Write-Host "Script completed. Please check the output for any errors."