We present a way to generate a DFT-GW-BSE dataset here.


# The crystal structure dataset

The [`nc300.tar`](nc300.tar) is created by 
[selecting materials with at most 6 atoms in the primitive unit cell and a gap larger than 0.4](../src/from_c2db/from_c2db.py)
and then packaging the folders into a `tar` file.

We first get the material info folders.
```shell
tar -xvf nc300.tar
```
This results in creation of the [`nc300`](nc300) folder.

# Creating the workflow

Create a folder, like `flows-nc300`.
Copy `fpconfig.json` from [`nc300.tar`](nc300.tar) into [that folder](flows-nc300/fpconfig.json).
Pay attention to 
- Where you want the calculation to happen (see `dirname`)
- The positions of the software packages (`BGW_path`, etc.)
- What calculations are being done (e.g. `"DFT_band": true`). This is also related to how the executable paths are set: for instance the Siesta path isn't needed when SIESTA runs are not being performed.


```shell
flows.py -c fpconfig.json 
```

Folders containing workflows of various materials are created in [`flows-nc300`](flows-nc300).

# High-thoroughput DFT run

```shell
salloc --nodes 4 --qos interactive --time 01:00:00 --constraint cpu --account m3571
```

When the interactive session starts, run 

```shell
bash run.sh
```

Note that it's possible that the whole folder can't be finished within one hour.
But in this case, we can simply remove finished materials from [`run.sh`](flows-nc300/run.sh)
and do the same thing again, until all materials are finished.

After the DFT run finishes, use the following comment to check if the materials are metal in [`flows-nc300`](flows-nc300):

```shell
collect_tool.py st -flow=./
```

Open [`_status.json`](flows-nc300/_status.json),
we find that most DFT calculations are successfully finished.

Now we do a metallic check. Run

```shell
collect_tool.py metalseek -flow=./
```

The result is in [`metal_seek.json`](flows-nc300/metal_seek.json).

# Preparing GW-BSE data for insulators

Run [`list_semiconductor.py`](list_semiconductor.py) to copy crystal files of insulating band structures into [`nc300-semiconductor`](nc300-semiconductor).
Copy [`flows-nc300/fpconfig.json`](flows-nc300/fpconfig.json) into [`flows-nc300-semiconductor`](flows-nc300-semiconductor),
and [make adjustments to enable GW-BSE calculations](flows-nc300-semiconductor/fpconfig.json).
Then run 

```shell
flows.py -c fpconfig.json 
```

in [`flows-nc300-semiconductor`](flows-nc300-semiconductor).
