# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - UNRELEASED

Transformers are now based on [pnspy](https://pypi.org/project/pnspy/) package.

### Added

- `InverseExtrinsicPNS` class is added.

### Removed

- `PNS` alias for `ExtrinsicPNS` is removed.
- `ExtrinsicPNS.to_hypersphere()` is removed. Use `inverse_transform()` method instead.

- `skpns.pns` module is removed. Use `PnsPy` package (`pns.pns`, `pns.pss`, `pns.base`) instead.
- `skpns.util` module is removed. Use `PnsPy` package (`pns.util`) instead.

## [1.3.0] - 2025-11-26

### Added

- `pns.pss()` function now takes `maxiter` argument.
- `pns.pns()` function now takes `maxiter` argument.

- `IntrinsicPNS()` class now takes `maxiter` argument.
- `ExtrinsicPNS()` class now takes `maxiter` argument.

### Fixed

- When lowest subsphere has multiple Fréchet means, `[1, 0]` is now set as the principal axis instead of raising `ZeroDivisionError`.

## [1.2.0] - 2025-11-09

### Added

- `IntrinsicPNS` class is added.

### Changed

- `PNS()` is renamed to `ExtrinsicPNS()`. Old name is still supported as alias.
- `PNS.to_hypersphere()` is renamed to `PNS.inverse_transform()`. Old name is still supported as alias.

- `pns.to_unit_sphere()` is renamed to `embed()`. Old name is still supported as alias.
- `pns.from_unit_sphere()` is renamed to `reconstruct()`. Old name is still supported as alias.
- `pns.pns()` now takes `residual` argument.
- `pns.proj()` now returns residuals.

### Removed

- `pns.residual()` is removed.

## [1.1.0] - 2025-10-17

### Added

- `PNS` can now be saved as ONNX.

## [1.0.1] - 2025-07-04

### Fixed

- Allow transformation to the same dimension.

## [1.0.0] - 2025-06-16

### Added

- `PNS` fitting and transformation.
