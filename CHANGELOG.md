## [0.3.1] - 2025-09-09
### Added
### Changed
### Deprecated
### Removed
### Fixed
* Resolved bug in normalization, resulting in uint8 spectra instead of float64. 

### Security


## [0.3.0] - 2025-08-21
### Added
* Lazy computation of interrogation window intensities and spectra for large video problems.
  This should solve any memory issues that a user may have.
* Improved parallelization of optimization.
* Moving to "spawn" as parallelization method.

### Changed
* README improved with math and symbols and modified to reflect the new API changes.

### Deprecated
### Removed
* `Iwave.get_spectra` is not longer needed, as spectra are a lazy property of any instance.
  Spectra become available as soon as images are set on `Iwave.imgs`.
### Fixed
### Security


## [0.2.0] - 2025-06-24
### Added
* Two step approach to optimization with a new `twosteps` argument for `iwave.velocimetry`. 
* Optional depth reconstruction during optimization (experimental!)
* Quality score [0-1] for quality of the velocity reconstruction

### Changed
### Deprecated
### Removed
### Fixed
* Improved convergence of optimization

### Security


## [0.1.0] - 2024-11-01
### Added
* First release of IWaVE

### Changed
### Deprecated
### Removed
### Fixed
### Security
