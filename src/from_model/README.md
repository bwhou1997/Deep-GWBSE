The way to generate datasets is discussed [here](../../examples/README.md).

# `e2vaetrainer.py`

To run [`e2vaetrainer.py`](e2vaetrainer.py),
enter the GPU interactive environment with a sufficiently large GPU memory
(a login node probably isn't enough),
and then `conda activate deep-gwbse` and `python e2vaetrainer.py`.

Key points:
- Data preprocessing. Because different materials have different G-grids,
  when wave functions are treated as images,
  it is not possible to make the image size identical for 
  wave functions from different materials. 
  Thus, a batch should contain and can only contain 
  all wave functions from a single material.
  This is why we use `collate_fn` to manipulate the training dataset.
- The wave function is periodic, and we should only present them in unit cells.
  Currently wave functions are stored in W-S unit cells.
  The empty spaces around these cells in images should be masked,
  or otherwise the loss function overestimates the accuracy of the model
  (as long as it successfully predicts that there are indeed empty spaces around W-S cells).
- For plotting, the empty spaces around the W-S cells should be filled by NaN.

# `gwtrainer.py`

To run [`gwtrainer.py`](gwtrainer.py),
enter the GPU interactive environment with a sufficiently large GPU memory
(a login node probably isn't enough),
and then `conda activate deep-gwbse` and `python gwtrainer.py`.
