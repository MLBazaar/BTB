# Release Process

## 0. Pre-Release Checklist

Before starting the release process, verify the following:

- [GitHub Action for Unit Tests are green on main](https://github.com/MLBazaar/BTB/actions/workflows/tests.yml?query=branch:main)
- [GitHub Action for Install Tests are green on main](https://github.com/MLBazaar/BTB/actions/workflows/install_test.yaml?query=branch:main)
- Get agreement on the version number to use for the release.

#### Version Numbering

BTB uses [semantic versioning](https://semver.org/). Every release has a major, minor and patch version number, and are displayed like so: `<majorVersion>.<minorVersion>.<patchVersion>`.

## 1. Create BTB release on GitHub

#### Create Release Branch

1. Branch off of BTB main. For the branch name, please use "release_vX.Y.Z" as the naming scheme (e.g. "release_v0.3.1").

#### Bump Version Number

1. Bump `__version__` in `baytune/version.py`, and `tests/test_version.py`.

#### Update Changelog

1. Replace top most "What’s new in" in `docs/changelog.rst` with the current date

    ```
    What’s new in 0.3.1 (January 4, 2023)
    =====================================
    ```

2. Remove any unused sections for this release (e.g. Enhancements, Fixes, Changes)
3. The release PR does not need to be mentioned in the list of changes

#### Create Release PR

A release PR should have **the version number as the title** and the notes for that release as the PR body text.

Checklist before merging:

- The title of the PR is the version number.
- All tests are currently green on checkin and on `main`.
- PR has been reviewed and approved.

## 2. Create GitHub Release

After the release pull request has been merged into the `main` branch, it is time draft [the GitHub release](https://github.com/HDI-Project/BTB/releases/new)

- The target should be the `main` branch
- The tag should be the version number with a v prefix (e.g. v0.3.1)
- Release title is the same as the tag
- This is not a pre-release
- Publishing the release will automatically upload the package to PyPI