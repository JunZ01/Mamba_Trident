# Mamba-Trident: Terrain-guided Tri-modal Mamba Network for Mapping Little Ice Age Glaciers

Official implementation of the paper published in *<期刊名>*.

## Contents
- `src/models/` — proposed model architectures
- `src/ablation_engine/` — ablation framework
- `src/utils/` — loss functions and evaluation metrics

## Dependencies
- PyTorch 2.x
- [mamba-ssm](https://github.com/state-spaces/mamba) >= 2.3.1
- [VMamba](https://github.com/MzeroMiko/VMamba) (backbone)

## Data
Sentinel-1/2 imagery: Copernicus Data Space Ecosystem and Google Earth Engine.
SRTM 30 m DEM: USGS EarthExplorer. Glacier outlines: RGI v6.0.

## Citation


## License
Apache-2.0. See [LICENSE](LICENSE).
