`marigold-relight.py` : uses the marigold model -> loads  unet into cache, goes upto 1gb in net size

`depth-anything-relight.py` : uses the depth map to compute normals by applying various filters and not just gradient. if depth map works accruately, the relighting works. 

## TODO:
1. better depth estimation models
2. convert to repo structure
3. add ui toggles for az and el computation